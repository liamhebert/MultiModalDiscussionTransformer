from src.data import register_dataset
from .multi_modal import MultiModalReddit
import numpy as np
import os 
import random 

@register_dataset("multi_modal_reddit")
def create_customized_dataset():
    dataset = MultiModalReddit(root='processed_graphs')
    num_graphs = len(dataset)
    # path = os.path.expandvars('$SLURM_TMPDIR')
    # train_valid_idx = []
    # with open(path + '/train-idx-many.txt', 'r') as file:
    #     for line in file:
    #         train_valid_idx.append(int(line[:-1]))
    
    # test_idx = []
    # with open(path + '/test-idx-many.txt', 'r') as file:
    #     for line in file:
    #         test_idx.append(int(line[:-1]))
    
    return {
        "dataset": dataset,
        "train_idx": np.arange(int(num_graphs*0.9)),
        "valid_idx": np.arange(int(num_graphs*0.9), num_graphs),
        "test_idx": np.arange(int(num_graphs*0.9), num_graphs),
        "train_idx": None,
        "valid_idx": None,
        "test_idx": None,
        "source": "pyg"
    }

@register_dataset("multi_modal_reddit-unseen")
def create_customized_dataset():
    dataset = MultiModalReddit(root='processed_graphs-unseen', unseen=True)
    num_graphs = len(dataset)
    # path = os.path.expandvars('$SLURM_TMPDIR')
    # train_valid_idx = []
    # with open(path + '/train-idx-many.txt', 'r') as file:
    #     for line in file:
    #         train_valid_idx.append(int(line[:-1]))
    
    # test_idx = []
    # with open(path + '/test-idx-many.txt', 'r') as file:
    #     for line in file:
    #         test_idx.append(int(line[:-1]))
    
    return {
        "dataset": dataset,
        "train_idx": np.arange(int(num_graphs)),
        "valid_idx": np.arange(int(num_graphs)),
        "test_idx": np.arange(int(num_graphs)),
        "source": "pyg"
    }


if __name__ == '__main__':
    create_customized_dataset()
