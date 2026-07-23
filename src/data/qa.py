import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        question_prefix=None,
        system_prompt_suffix=None,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = dict(template_args)
        if system_prompt_suffix is not None:
            base = self.template_args.get("system_prompt")
            if not base:
                raise ValueError(
                    "system_prompt_suffix requires a base system_prompt in template_args. "
                    "Models without a system prompt (e.g. base models) cannot use this feature."
                )
            self.template_args["system_prompt"] = base + " " + system_prompt_suffix
        self.question_key = question_key
        self.answer_key = answer_key
        self.question_prefix = question_prefix or ""
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        if self.question_prefix:
            prompt_msgs[0] = self.question_prefix + prompt_msgs[0]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class ParaphraseQADataset(QADataset):
    """QADataset extended with paraphrase support for training and evaluation.

    In **training mode** (``eval_mode=False``), the dataset is flattened: each
    original sample becomes ``1 + num_train_paraphrases`` items (original
    question plus the first *N* paraphrases), all paired with the same answer.

    In **evaluation mode** (``eval_mode=True``), each item contains all
    held-out question variants (original + paraphrases after position *N*)
    processed via ``_process_sample``, keyed by question ID, plus raw
    ``answer`` and ``index`` metadata.
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        paraphrases_key="paraphrases",
        num_train_paraphrases=0,
        eval_mode=False,
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        question_prefix=None,
        system_prompt_suffix=None,
    ):
        super().__init__(
            hf_args=hf_args,
            template_args=template_args,
            tokenizer=tokenizer,
            question_key=question_key,
            answer_key=answer_key,
            few_shot_dataset_hf_args=few_shot_dataset_hf_args,
            max_length=max_length,
            predict_with_generate=predict_with_generate,
            question_prefix=question_prefix,
            system_prompt_suffix=system_prompt_suffix,
        )
        self.paraphrases_key = paraphrases_key
        self.num_train_paraphrases = num_train_paraphrases
        self.eval_mode = eval_mode
        self._build_index()

    def _build_index(self):
        """Precompute flat index mapping virtual idx to (sample_idx, question_text, question_id)."""
        self._items = []
        for sample_idx in range(len(self.data)):
            sample = self.data[sample_idx]
            original_question = sample[self.question_key]
            paraphrases = sample[self.paraphrases_key]

            if not self.eval_mode:
                # Training: original + first num_train_paraphrases
                self._items.append((sample_idx, original_question, "question"))
                for para_idx in range(min(self.num_train_paraphrases, len(paraphrases))):
                    self._items.append(
                        (sample_idx, paraphrases[para_idx], f"para_{para_idx}")
                    )
            else:
                # Eval: original + held-out paraphrases
                self._items.append((sample_idx, original_question, "question"))
                for para_idx in range(self.num_train_paraphrases, len(paraphrases)):
                    eval_para_idx = para_idx - self.num_train_paraphrases
                    self._items.append(
                        (sample_idx, paraphrases[para_idx], f"q_para_{eval_para_idx}")
                    )

        # For eval_mode, build a per-sample lookup for grouped __getitem__
        if self.eval_mode:
            self._sample_items = {}
            for sample_idx, question_text, question_id in self._items:
                self._sample_items.setdefault(sample_idx, []).append(
                    (question_text, question_id)
                )

    def __len__(self):
        if self.eval_mode:
            return len(self.data)
        return len(self._items)

    def __getitem__(self, idx):
        if not self.eval_mode:
            # Training: return a single tokenized (question, answer) pair
            sample_idx, question_text, _question_id = self._items[idx]
            answer = self.data[sample_idx][self.answer_key]
            index = self.data[sample_idx]["index"]
            return self._process_sample(
                question=question_text, answer=answer, index=index,
            )

        # Eval: return all question variants for sample idx
        sample = self.data[idx]
        answer = sample[self.answer_key]
        index = sample["index"]

        item = {}
        for question_text, question_id in self._sample_items[idx]:
            processed = self._process_sample(
                question=question_text, answer=answer, index=index,
            )
            processed["question_text"] = question_text
            item[question_id] = processed
        item["answer"] = answer
        item["index"] = index
        return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]
