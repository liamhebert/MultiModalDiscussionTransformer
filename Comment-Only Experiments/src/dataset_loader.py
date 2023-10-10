import os
from datasets import Dataset, DatasetDict


class DatasetLoader:
    def __init__(self, data_dir, dname, modalities, split):
        self.data_dir = data_dir
        self.modalities = modalities
        if dname == "mm-reddit":
            train_fname = os.path.join(data_dir, f"HatefulDiscussions_dataset_train-split-{split}.parquet")
            valid_fname = os.path.join(data_dir, f"HatefulDiscussions_dataset_test-split-{split}.parquet")
            self.ds = DatasetDict(
                train=Dataset.from_parquet(path_or_paths=train_fname),
                validation=Dataset.from_parquet(path_or_paths=valid_fname),
            )

        print(self.ds)
        print(f"Train labels: {len(set(self.ds['train']['label']))}")
        # print(f"Test labels: {len(set(self.ds['test']['label']))}")


if __name__ == "__main__":
    DatasetLoader("./data/hateful_memes/", ["text"])
