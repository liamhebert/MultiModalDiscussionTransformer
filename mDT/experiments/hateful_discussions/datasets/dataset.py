from src.data import register_dataset
from .hateful_discussions import HatefulDiscussions
import numpy as np
import os


@register_dataset("hateful_discussions")
def create_hatespeech_dataset():
    dataset = HatefulDiscussions(root="processed_graphs")

    path = os.path.expandvars("$SLURM_TMPDIR")
    train_valid_idx = []
    with open(path + "/train-idx-many.txt", "r") as file:
        for line in file:
            train_valid_idx.append(int(line[:-1]))

    test_idx = []
    with open(path + "/test-idx-many.txt", "r") as file:
        for line in file:
            test_idx.append(int(line[:-1]))

    return {
        "dataset": dataset,
        "train_idx": np.array(train_valid_idx),
        "valid_idx": np.array(test_idx),
        "test_idx": np.array(test_idx),
        "source": "pyg",
    }


if __name__ == "__main__":
    create_hatespeech_dataset()
