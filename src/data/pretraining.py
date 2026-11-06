# import torch
import random
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        raw_text = "\n\n".join(raw_text)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )


class PartitionedPretrainingDataset(Dataset):
    """Like PretrainingDataset but chunks within each partition (e.g., subject).

    Groups samples by ``partition_key``, concatenates texts within each group
    with ``\\n\\n``, and chunks into fixed-length blocks.  This avoids mixing
    content across partitions in a single chunk.

    When ``per_sample=True``, skips concatenation and chunking — each original
    sample is kept as an independent training example (truncated to
    ``max_length``).  This matches per-sample processing used by other
    frameworks like LLaMA-Factory.
    """

    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text",
        partition_key="subject", max_length=512, random_crop=False,
        per_sample=False, random_window=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.random_crop = random_crop
        self.per_sample = per_sample
        # random_window: per-sample mode only. Instead of always tokenizing each
        # doc from word 0, pick a random word index that still leaves max_length
        # words ahead (full-window guarantee); fall back to start=0 when the doc
        # has fewer than max_length words. Activates only with per_sample=True.
        self.random_window = random_window
        if random_window and not per_sample:
            raise ValueError(
                "PartitionedPretrainingDataset: random_window=True requires per_sample=True"
            )

        data = load_hf_dataset(**hf_args)

        if per_sample:
            self.samples = [sample[text_key] for sample in data]
        else:
            groups = defaultdict(list)
            for sample in data:
                groups[sample[partition_key]].append(sample[text_key])

            if random_crop:
                self.partitions = []
                self.index_map = []
                for texts in groups.values():
                    raw_text = "\n\n".join(texts)
                    tokens = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
                    part_idx = len(self.partitions)
                    self.partitions.append(tokens)
                    num_chunks = max(1, len(tokens) // max_length + 1)
                    self.index_map.extend([part_idx] * num_chunks)
            else:
                self.chunks = []
                for texts in tqdm(groups.values(), desc='Chunking Dataset'):
                    self.chunks.extend(self._chunk_raw_text(texts))

    def _chunk_raw_text(self, raw_texts):
        raw_text = "\n\n".join(raw_texts)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        if self.per_sample:
            return len(self.samples)
        if self.random_crop:
            return len(self.index_map)
        return len(self.chunks)

    def __getitem__(self, idx):
        if self.per_sample:
            text = self.samples[idx]
            if self.random_window:
                # Word-space random start. 1 word ≥ 1 token (BPE never under-
                # tokenizes English), so requiring max_length words ahead
                # guarantees the truncated window is full max_length tokens
                # whenever the doc is long enough.
                words = text.split()
                n_words = len(words)
                if n_words > self.max_length:
                    start = random.randint(0, n_words - self.max_length)
                    text = " ".join(words[start:])
                # else: doc too short → start=0 (use full text; truncation handles it)
            return preprocess_pretraining_instance(
                self.tokenizer, "", text, self.max_length
            )
        if self.random_crop:
            tokens = self.partitions[self.index_map[idx]]
            if len(tokens) <= self.max_length:
                start = 0
            else:
                start = random.randint(0, len(tokens) - self.max_length)
            crop = tokens[start : start + self.max_length]
            text = self.tokenizer.decode(crop)
            return preprocess_pretraining_instance(
                self.tokenizer, "", text, self.max_length
            )
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )
