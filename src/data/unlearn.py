import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    # https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget",
                 retain_replacement=True, retain_seed=0):
        """Wraps the forget retain dataset into unlearning dataset.

        Args:
            forget (Dataset): Forget Dataset
            retain (Dataset): Retain Dataset
            anchor (str, optional): Specifies which dataset to anchor while randomly sampling from the other dataset. Defaults to 'forget'.
            retain_replacement (bool): how the non-anchor (retain, when
                anchor='forget') index is drawn. True (default) preserves the
                original behavior — an i.i.d. ``torch.randint`` draw WITH
                replacement per item, so some retain docs repeat and ~1/e are
                never seen in a pass. False draws WITHOUT replacement: iterate a
                seeded permutation of the retain indices, reshuffling with a new
                seed only once the permutation is exhausted. Deterministic given
                ``retain_seed`` and independent of the global RNG, so it neither
                perturbs nor is perturbed by the rest of the pipeline. The
                default path is byte-identical to before this option existed.
            retain_seed (int): seed for the without-replacement permutation.
        """
        self.forget = forget
        self.retain = retain
        self.anchor = anchor
        self.retain_replacement = bool(retain_replacement)
        self._retain_seed = int(retain_seed)
        self._retain_perm = None
        self._retain_cursor = 0
        self._retain_reshuffles = 0
        # Last indices actually returned by __getitem__ — read by
        # AlternatingMultiTopicDataset's capture mode to log the exact (forget,
        # retain) pair consumed. num_workers must be 0 for this to be meaningful.
        self.last_forget_idx = None
        self.last_retain_idx = None

    def _draw_other(self, n):
        """Index into the non-anchor dataset (size ``n``), with or without
        replacement per ``retain_replacement``."""
        if self.retain_replacement:
            return torch.randint(0, n, (1,)).item()
        if self._retain_perm is None or self._retain_cursor >= len(self._retain_perm):
            # Reshuffle on exhaustion with a fresh seed so a run that consumes
            # more than one pass (draws > corpus size) keeps sampling without
            # replacement WITHIN each pass rather than reverting to i.i.d.
            g = torch.Generator().manual_seed(self._retain_seed + self._retain_reshuffles)
            self._retain_perm = torch.randperm(n, generator=g).tolist()
            self._retain_cursor = 0
            self._retain_reshuffles += 1
        idx = self._retain_perm[self._retain_cursor]
        self._retain_cursor += 1
        return idx

    def __len__(self):
        """Ensures the sampled dataset matches the anchor dataset's length."""
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            self.last_forget_idx = idx
            if self.retain:
                retain_idx = self._draw_other(len(self.retain))
                self.last_retain_idx = retain_idx
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            item["retain"] = self.retain[idx]
            self.last_retain_idx = idx
            if self.forget:
                forget_idx = self._draw_other(len(self.forget))
                self.last_forget_idx = forget_idx
                item["forget"] = self.forget[forget_idx]
        return item
