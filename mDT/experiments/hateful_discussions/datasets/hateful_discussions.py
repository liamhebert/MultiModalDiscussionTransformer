from torch_geometric.data import Dataset
import json
from torch_geometric.utils import from_networkx
import copy
from transformers import (
    AutoTokenizer,
    ViTImageProcessor,
    CLIPImageProcessor,
    CLIPTextModel,
)
from PIL import Image
import re
import networkx
import torch
from tqdm import tqdm
from typing import Optional, Callable
from glob import glob
import os
import pandas as pd


class HatefulDiscussions(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.k = 0
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        path = os.path.expandvars("$SLURM_TMPDIR/")
        return [path + "/raw_graphs.json"]

    @property
    def processed_file_names(self):
        path = os.path.expandvars("$SLURM_TMPDIR/processed_graphs/processed")
        # TODO: this will be the total number of labels in the dataset, will have to update manually
        return [path + f"/graph-{i}.pt" for i in range(19328)]

    def process(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        markdown_regex = re.compile(
            "^\[([\w\s\d]+)\]\(((?:\/|https?:\/\/)[\w\d./?=#]+)\)$"
        )
        all_url_regex = re.compile(
            "https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
        )

        # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def clean_urls(x):
            x = markdown_regex.sub(
                "[LINK1] \g<1> [LINK2]", x
            )  # replace markdown links with [LINK1] title [LINK2], start and finish
            return all_url_regex.sub("", x)

        def extract_text(x):
            if "title" in x[0]:  # x[0] is data
                if "selftext" in x[0]:
                    body = (
                        "\n" + clean_urls(x[0]["selftext"])
                        if x[0]["selftext"] != ""
                        else ""
                    )
                else:
                    body = (
                        "\n" + clean_urls(x[0]["body"]) if x[0]["body"] != "NA" else ""
                    )
                # if len(x[1]) != 0:
                #     return '[IMG1] ' + x[0]['title'] + ' [IMG2] ' + body + '\n'
                return x[0]["title"] + body
            else:
                return clean_urls(x[0]["body"])
                # return x[0]['body']

        path = os.path.expandvars("$SLURM_TMPDIR/processed_graphs/processed")
        path_slurm = os.path.expandvars("$SLURM_TMPDIR")
        known_files = glob(path + "/*")
        total = 0
        pre_fix = 0
        self.k = 0
        valid_idx = []
        train_idx = []
        with open(path_slurm + "/train-idx.txt") as f:
            for line in f:
                train_idx += [int(line)]
        with open(path_slurm + "/test-idx.txt") as f:
            for line in f:
                valid_idx += [int(line)]

        duped = pd.read_parquet(path_slurm + "/duped.parquet")["text"].unique()

        # with open(self.raw_file_names[0], 'r') as file, open(os.environ['SLURM_TMPDIR'] + '/train-idx-many.txt', 'w') as train, open(os.environ['SLURM_TMPDIR'] + '/test-idx-many.txt', 'w') as valid:
        with open(self.raw_file_names[0], "r") as file, open(
            path_slurm + "/train-idx-many.txt", "w"
        ) as train, open(path_slurm + "/test-idx-many.txt", "w") as valid:
            for graph_num, line in tqdm(enumerate(file), total=8927):
                raw_data = json.loads(line)
                self.get_relative_depth(raw_data)
                self.spread_downwards(raw_data)
                data = {}
                self.collapse_tree(raw_data, data, [])

                g = networkx.Graph()
                adj = [
                    (
                        x[0]["parent_id"]
                        if x[0]["parent_id"] != x[0]["link_id"]
                        else "top_level",
                        x[0]["id"],
                    )
                    for x in data.values()
                    if "parent_id" in x[0]
                ]

                # print(adj)
                def make_features(x):
                    is_duped = (
                        x[3] == "NA"
                        or (
                            "link_id" not in x[0]
                            and x[0]["title"] + "\n" + x[0]["body"] in duped
                        )
                        or ("body" in x[0] and x[0]["body"] in duped)
                    )

                    if "parent_id" not in x[0]:
                        return (
                            "top_level",
                            {"x": x[0]["id"], "y": x[3] if not is_duped else "NA"},
                        )
                    return (
                        x[0]["id"],
                        {"x": x[0]["id"], "y": x[3] if not is_duped else "NA"},
                    )

                g.add_nodes_from([make_features(x) for x in data.values()])
                g.add_edges_from(adj)
                # g = g.to_undirected()
                graph = from_networkx(g)

                graph.x_images = [
                    [path_slurm + "/" + y for y in data[x][1]] for x in graph.x
                ]
                graph.x_text = [data[x] for x in graph.x]
                order = graph.x
                matrix = []
                for key in order:
                    distances = data[key][2]
                    matrix += [[distances[y] for y in order]]
                graph.distance_matrix = matrix
                graph.x = tokenizer(
                    [extract_text(x) for x in graph.x_text],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=100,
                )  # this is a dictionary

                graph.x_image_index = torch.Tensor(
                    [True if len(x) != 0 else False for x in graph.x_images]
                )
                # graph.x_images = [extractor(Image.open([Image.open(for x in graph.x_images if len(x) != 0), return_tensors='pt')]
                images = [
                    Image.open(x[0]).convert(mode="RGB")
                    for x in graph.x_images
                    if len(x) != 0
                ]
                if len(images) != 0:
                    graph.x_images = extractor(images, return_tensors="pt")[
                        "pixel_values"
                    ]
                else:
                    graph.x_images = torch.zeros((1, 3, 224, 224))
                    # print(graph.x_images.size())
                ys = graph.y
                hate_labels = [
                    "DEG",
                    "lti_hate",
                    "IdentityDirectedAbuse",
                    "AffiliationDirectedAbuse",
                ]
                good_labels = ["Neutral", "lti_normal", "NDG", "HOM"]
                true_ys = [x for x in ys if x in hate_labels or x in good_labels]
                # all_ys = graph.y
                for i in range(len(true_ys)):
                    z = 0
                    graph_y = ["NA" for _ in ys]
                    flag = False
                    for k, label in enumerate(ys):
                        if label != "NA":
                            if z == i:
                                graph_y[k] = label
                                flag = True
                                break
                            z += 1
                    if not flag:
                        print("missing label!!")
                        continue

                    graph.y_mask = torch.Tensor(
                        [True if x != "NA" else False for x in graph_y]
                    ).bool()
                    graph.y = [x for x in graph_y if x != "NA"]
                    graph.y = torch.Tensor(
                        [1 if x in hate_labels else 0 for x in graph.y]
                    )

                    torch.save(graph, path + f"/graph-{self.k}.pt")
                    if graph_num in valid_idx:
                        total += 1
                        valid.write(str(self.k) + "\n")
                    elif graph_num in train_idx:
                        total += 1
                        train.write(str(self.k) + "\n")
                    self.k += 1

        print("FINAL K", self.k)
        print("TOTAL Ys", total)
        print("pre total", pre_fix)

    def len(self):
        return len(self.processed_file_names)

    # algorithm, go depth first, then do a second pass
    def get_relative_depth(self, node, depths={}) -> dict:
        distances = copy.deepcopy(depths)
        for key in distances.keys():
            distances[key][0] += 1
        distances[node["id"]] = [0, 0]

        for x in node["tree"]:
            val = self.get_relative_depth(x, distances)
            for key, value in val.items():
                if key not in distances:
                    value[1] = value[1] + 1
                    distances[key] = value
        node["distances"] = distances
        return copy.deepcopy(distances)

    def spread_downwards(self, node, depths={}):
        dists = copy.deepcopy(depths)
        for key, value in dists.items():
            if key not in node["distances"]:
                value[0] += 1
                node["distances"][key] = value
        for x in node["tree"]:
            self.spread_downwards(x, node["distances"])

    def collapse_tree(self, comment, data, root_images):
        if "id" not in comment["data"]:
            comment["data"]["id"] = comment["id"]
        comment["data"]["id"] = comment["id"]

        id = comment["data"]["id"]
        i = 0
        if comment["data"]["id"] in data:
            if comment["data"]["body"] != data[comment["data"]["id"]][0]["body"]:
                if data[comment["data"]["id"]][0]["body"] == "[deleted]":
                    if len(comment["images"]) == 0:
                        comment["images"] = root_images
                    data[comment["data"]["id"]] = (
                        comment["data"],
                        comment["images"],
                        comment["distances"],
                        comment["data"]["label"],
                    )
                    print("updated!")
        else:
            if len(comment["images"]) == 0:
                comment["images"] = root_images
            data[comment["data"]["id"]] = (
                comment["data"],
                comment["images"],
                comment["distances"],
                comment["data"]["label"],
            )
        for child in comment["tree"]:
            self.collapse_tree(child, data, root_images)

    def get(self, idx):
        path = self.processed_file_names[idx]
        return torch.load(path)


if __name__ == "__main__":
    HatefulDiscussions(root="processed_graphs")
