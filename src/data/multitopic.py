"""CAIS-style per-topic alternating dataset for joint unlearning.

Wraps N per-topic `ForgetRetainDataset` objects so that K consecutive global
indices belong to topic 0, the next K belong to topic 1, ..., cycling. With
K = per_device_batch * grad_accum * world_size, one optimizer step is fed
exclusively from one topic — matching CAIS RMU's
`topic_idx = step_idx % num_topics` schedule
(https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py).

Per-topic order is randomized ONCE at init via a seeded permutation, then
held fixed for the rest of the run — so within a topic the order is
shuffled, but across topics the alternation pattern is preserved. This is
necessary because the matching sampler must be `SequentialSampler` (any
RNG shuffle on the global index range would break the K-block routing).

Resume safety: the per-topic permutations are seeded, so two runs with the
same `seed` produce identical orderings (pairs cleanly with the existing
`trainer.args.seed=0` + `FLASH_ATTENTION_DETERMINISTIC=1` pinning).

The `topic_id` is attached to each emitted forget batch (not retain — RMU
only needs to route the control vector on the forget side). The
DataCollatorForSupervisedDataset propagates `topic_id` through unchanged.
"""

import torch
from torch.utils.data import Dataset


class AlternatingMultiTopicDataset(Dataset):
    # __getitem__ routes global idx → (topic, local), so the K-block topic
    # alternation only holds if the dataloader traverses in index order. Any
    # dataset with this flag makes UnlearnTrainer force a SequentialSampler.
    requires_sequential_sampler = True

    def __init__(
        self,
        topics,
        block_size: int,
        length: int | None = None,
        seed: int = 0,
        subset_seed: int | None = None,
        subset_consumed: int | None = None,
        capture_path: str | None = None,
    ):
        """
        Args:
            topics: list of ForgetRetainDataset, one per topic (e.g. [bio, cyber]).
            block_size: K consecutive global indices fed from the same topic.
                Set to per_device_batch * grad_accum * world_size to make one
                optimizer step exactly one topic (CAIS-style).
            length: total number of items the dataset advertises. Defaults to
                `n_topics * max(len(t) for t in topics)` so HF Trainer can drive
                `max_steps` independently. Truncating to `max_steps * K` is fine.
            seed: RNG seed for the per-topic init shuffle.
            capture_path: if set, every __getitem__ appends the exact consumed
                tuple "idx topic forget_local retain_local" to this file (in call
                order). Used to record a reference run's exact (forget, retain)
                batches so AlternatingReplayDataset can replay the SAME batches
                reordered under other seeds. Requires num_workers=0. Does not
                change what is returned, so the captured run stays bit-identical.
            subset_seed: if set, enables FIXED-SUBSET mode. Only the first N_t
                entries of each topic's permutation are ever consumed by a run
                (N_t is fixed by the block schedule, not the seed), so different
                `seed`s otherwise draw DIFFERENT documents from a large corpus.
                In this mode the consumed document SUBSET is pinned to the one
                `subset_seed` would consume, and `seed` only reorders WITHIN that
                subset. `seed == subset_seed` gives the identity reorder, i.e. it
                reproduces the `subset_seed` run bit-for-bit. Requires
                `subset_consumed`. Leave None for the original per-seed behavior.
            subset_consumed: total number of samples the run will read
                (= max_steps * per_device_batch * grad_accum * world_size). Used
                only in fixed-subset mode to compute the per-topic subset size N_t.
        """
        assert len(topics) >= 1, "Need at least one topic"
        assert block_size > 0, "block_size must be positive"
        self.topics = list(topics)
        self.n = len(self.topics)
        self.K = int(block_size)
        self.total = int(length) if length is not None else self.n * max(len(t) for t in self.topics)

        # One permutation per topic, sampled once at init. Deterministic given `seed`.
        g = torch.Generator().manual_seed(int(seed))
        if subset_seed is None:
            self.perms = [
                torch.randperm(len(t), generator=g).tolist() for t in self.topics
            ]
            self.subset_sizes = None
        else:
            if subset_consumed is None:
                raise ValueError(
                    "AlternatingMultiTopicDataset: subset_seed requires subset_consumed "
                    "(= max_steps * per_device_batch * grad_accum * world_size)."
                )
            # Per-topic consumed count N_t from the block schedule (seed-independent):
            # each full K-block adds K to its topic; the final partial block adds the
            # remainder. Mirrors the routing in __getitem__ exactly.
            M = int(subset_consumed)
            full_blocks, rem = divmod(M, self.K)
            N = [0] * self.n
            for b in range(full_blocks):
                N[b % self.n] += self.K
            if rem:
                N[full_blocks % self.n] += rem
            # `subset_seed`'s consumed SEQUENCE per topic is the fixed data: the exact
            # length-N_t list of docs it feeds, in order. When N_t > corpus size the
            # sequence WRAPS (docs repeat) — e.g. the WMDP cyber forget corpus (1000
            # docs) is cycled ~2.6x over 2560 consumed samples, so the fixed "data" is
            # the whole corpus with subset_seed's repeat pattern. Other seeds get a
            # PERMUTATION of that same sequence (identical multiset, different order);
            # seed == subset_seed is the identity → bit-identical reproduction.
            gb = torch.Generator().manual_seed(int(subset_seed))
            base = [torch.randperm(len(t), generator=gb).tolist() for t in self.topics]
            self.perms = []
            for i, t in enumerate(self.topics):
                lt = len(t)
                seq0 = [base[i][r % lt] for r in range(N[i])]  # subset_seed consumed seq
                if int(seed) == int(subset_seed):
                    self.perms.append(seq0)
                else:
                    sigma = torch.randperm(N[i], generator=g).tolist()
                    self.perms.append([seq0[j] for j in sigma])
            self.subset_sizes = N

        # Capture sink (opened lazily on first __getitem__ so a fork/pickle before
        # any read doesn't carry an open handle). num_workers must be 0.
        self.capture_path = capture_path
        self._cap_fh = None

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        # K-block routing: which topic owns this K-block, and how far into the topic's pass we are.
        block = idx // self.K
        topic = block % self.n
        topic_block = block // self.n
        offset = idx % self.K
        # Modulo the PERM length: in fixed-subset mode perms[topic] is the length-N_t
        # consumed sequence (may be < or > corpus size); in normal mode it is a full
        # corpus permutation, so len(perms[topic]) == len(topics[topic]) and this is
        # identical to the original behavior.
        local_raw = (topic_block * self.K + offset) % len(self.perms[topic])
        local = self.perms[topic][local_raw]

        item = self.topics[topic][local]
        # Capture the exact consumed (forget, retain) pair for later replay. `local`
        # is the forget-corpus index; the retain index was just chosen inside the
        # ForgetRetainDataset and stashed on it. Logged in call order.
        if self.capture_path is not None:
            if self._cap_fh is None:
                self._cap_fh = open(self.capture_path, "w")
            rl = getattr(self.topics[topic], "last_retain_idx", None)
            self._cap_fh.write(f"{idx}\t{topic}\t{local}\t{rl}\n")
            self._cap_fh.flush()
        # Attach topic_id to the forget side only — RMU reads it from forget_inputs
        # to pick the matching control vector. Retain doesn't need it. The collator
        # gathers per-sample `topic_id` into a tensor of shape (batch,); all entries
        # in a single batch will be equal since each K-block is a single topic.
        if "forget" in item and isinstance(item["forget"], dict):
            item["forget"]["topic_id"] = topic
        return item
