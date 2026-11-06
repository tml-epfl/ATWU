from typing import Dict, Any, Union
from omegaconf import DictConfig

from data.qa import (
    QADataset,
    QAwithIdkDataset,
    QAwithAlternateDataset,
    ParaphraseQADataset,
)
from data.hf_dataset import HFDataset
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.multitopic import AlternatingMultiTopicDataset
from data.pretraining import (
    PretrainingDataset,
    CompletionDataset,
    PartitionedPretrainingDataset,
)

DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args
    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return mapping to multiple datasets
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    # Retain-sampling policy (see ForgetRetainDataset). Popped before the split
    # loop so they are not mistaken for dataset splits. Defaults reproduce the
    # original with-replacement behavior exactly.
    retain_replacement = bool(data_cfg.pop("retain_replacement", True))
    retain_sample_seed = int(data_cfg.pop("retain_sample_seed", 0))
    # Optional CAIS-style per-topic alternation: pairs N forget datasets with N retain
    # datasets BY KEY ORDER (must match length and match by position) and wraps each
    # pair in a ForgetRetainDataset, then wraps the list in AlternatingMultiTopicDataset.
    # Cfg example (under `data:`):
    #     alternating_topics:
    #       block_size: 32         # K consecutive samples per topic (= bs * grad_accum * world)
    #       seed: 0                # per-topic init shuffle seed
    #       length: null           # advertised dataset length; null → n_topics * max(len)
    #       first_topic: bio       # which forget/retain key becomes topic_id=0 (visited
    #                              # first). If unset, YAML order is honored as-is.
    alt = data_cfg.pop("alternating_topics", None)
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        if alt is not None:
            alt = dict(alt)
            forget_map = unlearn_splits.get("forget")
            retain_map = unlearn_splits.get("retain")
            if not isinstance(forget_map, dict) or not isinstance(retain_map, dict):
                raise ValueError(
                    "alternating_topics requires multiple datasets under both "
                    "data.forget and data.retain (matched by key order)."
                )
            if len(forget_map) != len(retain_map):
                raise ValueError(
                    f"alternating_topics: forget has {len(forget_map)} topics but "
                    f"retain has {len(retain_map)} — must match."
                )
            # Optional: reorder forget/retain so the named topic is at index 0
            # (becomes topic_id=0 and is visited first by AlternatingMultiTopicDataset).
            # No-op if first_topic is already at index 0.
            first_topic = alt.pop("first_topic", None)
            if first_topic is not None:
                if first_topic not in forget_map:
                    raise ValueError(
                        f"alternating_topics.first_topic={first_topic!r} not found in "
                        f"data.forget keys {list(forget_map.keys())}."
                    )
                if first_topic not in retain_map:
                    raise ValueError(
                        f"alternating_topics.first_topic={first_topic!r} not found in "
                        f"data.retain keys {list(retain_map.keys())}."
                    )

                def _reorder(d, first):
                    return {first: d[first], **{k: v for k, v in d.items() if k != first}}

                forget_map = _reorder(forget_map, first_topic)
                retain_map = _reorder(retain_map, first_topic)
            # Offset the retain seed per topic so two topics drawing from the
            # SAME general retain file do not march through an identical
            # without-replacement permutation in lockstep.
            topics = [
                ForgetRetainDataset(
                    forget=f, retain=r, anchor=anchor,
                    retain_replacement=retain_replacement,
                    retain_seed=retain_sample_seed + i,
                )
                for i, (f, r) in enumerate(zip(forget_map.values(), retain_map.values()))
            ]
            _subset_seed = alt.get("subset_seed")
            _subset_consumed = alt.get("subset_consumed")
            unlearn_dataset = AlternatingMultiTopicDataset(
                topics=topics,
                block_size=int(alt["block_size"]),
                length=alt.get("length"),
                seed=int(alt.get("seed", 0)),
                subset_seed=None if _subset_seed is None else int(_subset_seed),
                subset_consumed=None if _subset_consumed is None else int(_subset_consumed),
            )
        else:
            unlearn_dataset = ForgetRetainDataset(
                **unlearn_splits, anchor=anchor,
                retain_replacement=retain_replacement,
                retain_seed=retain_sample_seed,
            )
        data["train"] = unlearn_dataset
        for split in unlearn_splits:
            data.pop(split)
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(PartitionedPretrainingDataset)
_register_data(QAwithAlternateDataset)
_register_data(ParaphraseQADataset)
_register_data(HFDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)
_register_data(AlternatingMultiTopicDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
