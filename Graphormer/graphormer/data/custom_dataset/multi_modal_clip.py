from torch_geometric.data import Dataset
import json 
from torch_geometric.utils import from_networkx
import copy 
from transformers import AutoTokenizer, ViTImageProcessor, CLIPImageProcessor, CLIPTextModel
from PIL import Image
import re
import networkx 
import torch
from tqdm import tqdm
from typing import Optional, Callable
from glob import glob
import os 

class MultiModalRedditCLIP(Dataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        path = os.path.expandvars('$SLURM_TMPDIR/')
        return [path + '/raw_graphs_fixed-2.json']
    
    @property
    def processed_file_names(self):
        #return ['giga-data.pt']
        path = os.path.expandvars('$SLURM_TMPDIR/processed_graphs/processed')
        return [path + f'/graph-{i}.pt' for i in range(8288)]
        
    
    # def len(self) -> int:
    #     return len(self.processed_file_names)
    
    def process(self):
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        website_pattern = '[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        regex = re.compile('\[([a-zA-Z0-9 ]*)\]\(https?:\/\/(\S+?(?:jpe?g|png|gif|svg))\)')
        image_regex = re.compile(website_pattern)
        def extract_text(x):
            if 'title' in x[0]:
                if 'selftext' in x[0]:
                    body = '\n' + x[0]['selftext'] if x[0]['selftext'] != '' else ''
                else:
                    body = '\n' + x[0]['body'] if x[0]['body'] != 'NA' else ''
                if len(x[1]) != 0:
                    return '[IMG1] ' + x[0]['title'] + ' [IMG2] ' + body + '\n'
                return x[0]['title'] + body
            else:
                x = regex.sub('[IMG1] \g<1> [IMG2]', x[0]['body'])
                return image_regex.sub('', x)
        path = os.path.expandvars('$SLURM_TMPDIR/processed_graphs/processed')
        path_images = os.path.expandvars('$SLURM_TMPDIR')
        known_files = glob(path + '/*')
        data_graphs = []
        k = 0
        with open(self.raw_file_names[0], 'r') as file:
            for i, line in tqdm(enumerate(file), total=8288):
                if path + f'/graph-{i}.pt' in known_files:
                    continue
                raw_data = json.loads(line)
                self.get_relative_depth(raw_data)
                self.spread_downwards(raw_data)
                data = {}
                self.collapse_tree(raw_data, data)
                g = networkx.Graph() 
                adj = [(x[0]['parent_id'], x[0]['id']) for x in data.values() if 'parent_id' in x[0]]
             
                def make_features(x):
                    if 'parent_id' not in x[0]:
                        return ('top_level', {'x': x[0]['id'], 'y': 'NA'})
                    return (x[0]['id'], {'x': x[0]['id'], 'y': x[3]})
                
                g.add_nodes_from([
                        make_features(x) for x in data.values()
                    ]
                )
                g.add_edges_from(adj)
                graph = from_networkx(g)

                graph.x_images = [[path_images + '/' + y for y in data[x][1]] for x in graph.x]
                graph.x_text = [extract_text(data[x]) for x in graph.x]
                order = graph.x
                matrix = []
                for key in order:
                    distances = data[key][2]
                    matrix += [[distances[y] for y in order]]
                graph.distance_matrix = matrix
                graph.x = tokenizer(graph.x_text, padding='max_length', truncation=True, return_tensors='pt') # this is a dictionary


                del graph.x_text
                graph.x_image_index = torch.Tensor([True if len(x) != 0 else False for x in graph.x_images])
                #graph.x_images = [extractor(Image.open([Image.open(for x in graph.x_images if len(x) != 0), return_tensors='pt')]
                images = [Image.open(x[0]).convert(mode='RGB') for x in graph.x_images if len(x) != 0]
                if len(images) != 0:
                    graph.x_images = extractor(images, return_tensors='pt')['pixel_values']
                else:
                    graph.x_images = torch.zeros((1, 3, 224, 224))
                    #print(graph.x_images.size())
                graph.y_mask = torch.Tensor([True if x != 'NA' else False for x in graph.y]).bool()
                hate_labels = ['DEG', 
                               'lti_hate', 
                               'AffiliationDirectedAbuse', 
                               'Slur', 
                               'IdentityDirectedAbuse', 
                               'PersonDirectedAbuse']
               
                graph.y = torch.Tensor([0 if x in hate_labels else 1 for x in graph.y])
              
                #data_graphs += [graph]
                k += 1
                torch.save(graph, path + f'/graph-{i}.pt')
                
        #data, slices = self.collate(data_graphs)
        #torch.save((data, slices), self.processed_paths[0])
    
    def len(self):
        return len(self.processed_file_names)

    
    # algorithm, go depth first, then do a second pass 
    def get_relative_depth(self, node, depths={}) -> dict:
        distances = copy.deepcopy(depths)
        for key in distances.keys():
            distances[key][0] += 1
        distances[node['id']] = [0, 0]

        for x in node['tree']:
            val = self.get_relative_depth(x, distances)
            for key, value in val.items():
                if key not in distances:
                    value[1] = value[1] + 1
                    distances[key] = value
        node['distances'] = distances
        return copy.deepcopy(distances)
    
    def spread_downwards(self, node, depths={}):
        dists = copy.deepcopy(depths)
        for key, value in dists.items():
            if key not in node['distances']:
                value[0] += 1
                node['distances'][key] = value
        for x in node['tree']:
            self.spread_downwards(x, node['distances'])
    
    def collapse_tree(self, comment, data):
        if 'id' not in comment['data']:
            comment['data']['id'] = comment['id']
        data[comment['id']] = (comment['data'], comment['images'], comment['distances'], comment['data']['label'])
        for child in comment['tree']:
            self.collapse_tree(child, data)
   
    def get(self, idx):
        path = self.processed_file_names[idx]
        return torch.load(path)
   



