"""Simple HuggingFace Dataset wrapper that returns raw data without tokenization."""
from torch.utils.data import Dataset
from data.utils import load_hf_dataset


class HFDataset(Dataset):
    """Wrapper for HuggingFace datasets that returns raw data without any preprocessing.

    This is useful for evaluation metrics that need access to raw text (questions, answers)
    rather than tokenized data. For example, LKF evaluation needs raw text for generation
    and judging, not pre-tokenized training data.

    Args:
        hf_args: Dictionary with HuggingFace dataset loading arguments (path, name, split, etc.)
        **kwargs: Additional arguments (ignored, for compatibility with dataset registry)
    """

    def __init__(self, hf_args, **kwargs):
        super(HFDataset, self).__init__()
        self.data = load_hf_dataset(**hf_args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return raw data dictionary from HuggingFace dataset without any processing."""
        return dict(self.data[idx])