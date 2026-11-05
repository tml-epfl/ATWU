import copy
from typing import Any

import torch
from transformers import PreTrainedModel

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import (
    compute_kl_divergence,
    jensun_multitok_loss,
    jensun_retain_loss
)

TOK_EQ = {
    'JensUn': [[2822, 4623], [1939, 2969]],
    'JensUnEOT': [[2822, 4623, 128009], [1939, 2969, 32000]],
    'JensUnHash': [[11], [396]],
    'JensUnComma': [[2], [1919]],
    'JensUnWhiteSpace': [[220], [259]],
}

class GradJSDiff(UnlearnTrainer):
    def __init__(
        self,
        gamma : float = 1.0,
        alpha : float = 1.0,
        retain_loss_type: str = "JensUn",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type

        self.fix_toks()

        self.ref_model: PreTrainedModel = None # type: ignore
        if retain_loss_type == "KL" or retain_loss_type=='JensUn':
            self.ref_model = self._prepare_ref_model(self.model) # type: ignore

    def fix_toks(self):
        # Tokens are different for Phi/Llama Tokenizers
        idx = int("Phi" in self.model.config._name_or_path)
        self.tok_id = TOK_EQ[self.__class__.__name__][idx]

    def _prepare_ref_model(self, model: PreTrainedModel) -> PreTrainedModel:
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model # type: ignore

    def compute_retain_loss(
        self,
        model: PreTrainedModel,
        retain_inputs: dict[str, Any]
    ) -> torch.Tensor:
        retain_outputs = model(**retain_inputs)

        if self.retain_loss_type == "NLL":
            retain_loss = retain_outputs.loss
        elif self.retain_loss_type == "KL":
            retain_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
        elif self.retain_loss_type == "JensUn":
            retain_loss, retain_outputs = jensun_retain_loss(
                self.model, self.ref_model, retain_inputs # type: ignore
            )
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )

        return retain_loss

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = jensun_multitok_loss(model, forget_inputs, self.tok_id)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss

class JensUn(GradJSDiff):
    """A specialized GradJSDiff trainer using "JensUn" for the retain loss.
       The default JensUn target tokens: 'No Idea'
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)

class JensUnEOT(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning based on End-Of-Text (EOT) tokens.
       The JensUn target tokens: 'No Idea <EOT>'
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)

class JensUnHash(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning for random tokens.
       The JensUn target tokens: '#'
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)

class JensUnComma(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning for random tokens.
       The JensUn target tokens: ','
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn",  *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)

class JensUnWhiteSpace(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning for random tokens.
       The JensUn target tokens: ' '
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)
