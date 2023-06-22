from graphormer.data import register_dataset
from .multi_modal import MultiModalReddit
from .multi_modal_clip import MultiModalRedditCLIP 
import numpy as np
from sklearn.model_selection import train_test_split
import os 

@register_dataset("multi_modal_reddit")
def create_customized_dataset():
    dataset = MultiModalReddit(root='processed_graphs')
    
    # # customized dataset split
    # train_valid_idx, test_idx = train_test_split(
    #     np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
    # )
    # train_idx, valid_idx = train_test_split(
    #     train_valid_idx, test_size=num_graphs // 5, random_state=0
    # )
    path = os.path.expandvars('$SLURM_TMPDIR')
    train_valid_idx = []
    with open(path + '/train-idx-many.txt', 'r') as file:
        for line in file:
            train_valid_idx.append(int(line[:-1]))
    
    test_idx = []
    with open(path + '/test-idx-many.txt', 'r') as file:
        for line in file:
            test_idx.append(int(line[:-1]))
    
    return {
        "dataset": dataset,
        "train_idx": np.array(train_valid_idx),
        "valid_idx": np.array(test_idx),
        "test_idx": np.array(test_idx),
        "source": "pyg"
    }

# @register_dataset("multi_modal_reddit_clip")
# def create_customized_dataset():
#     dataset = MultiModalRedditCLIP(root='processed_graphs')
#     num_graphs = len(dataset)

#     # customized dataset split
#     train_valid_idx, test_idx = train_test_split(
#         np.arange(num_graphs), test_size=num_graphs // 10, random_state=0
#     )
#     train_idx, valid_idx = train_test_split(
#         train_valid_idx, test_size=num_graphs // 5, random_state=0
#     )
#     return {
#         "dataset": dataset,
#         "train_idx": train_valid_idx,
#         "valid_idx": test_idx,
#         "test_idx": test_idx,
#         "source": "pyg"
#     }

if __name__ == '__main__':
    create_customized_dataset()
