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
import pandas as pd


class MultiModalReddit(Dataset):
    def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, unseen=False):
        if unseen:
            test = ['sanfrancisco', 'bayarea', 'toronto', 'vancouver', 'waterloo', 'texas', 'newyorkcity']
            self.groups = {'Test-LocalCity': {}}
            for i, key in enumerate(test):
                self.groups['Test-LocalCity'][i] = [key]
        else:
            self.groups = {
                'Partisan A': {
                    'a': ['democrats', 'OpenChristian', 'GamerGhazi', 'excatholic', 'EnoughLibertarianSpam', 'AskAnAmerican', 'lastweektonight'],
                    'b': ['Conservative', 'progun', 'TrueChristian', 'Catholicism', 'askaconservative', 'AskTrumpSupporters', 'CGPGrey']
                },
                'Partisan B': {
                    'a': ['hillaryclinton', 'SandersForPresident', 'askhillarysupporters', 'BlueMidterm2018', 'badwomensanatomy', 'PoliticalVideo', 'liberalgunowners', 'GrassrootsSelect', 'GunsAreCool'],
                    'b': ['The_Donald', 'KotakuInAction', 'HillaryForPrison', 'AskThe_Donald', 'PoliticalHumor', 'ChoosingBeggars', 'uncensorednews', 'Firearms', 'DNCleaks', 'dgu']
                },
                'Affluence': {
                    'a': ['vagabond', 'hitchhiking', 'DumpsterDiving', 'almosthomeless', 'AskACountry', 'KitchenConfidential', 'Nightshift', 'alaska', 'fuckolly', 'FolkPunk'],
                    'b': ['backpacking', 'hiking', 'Frugal', 'personalfinance', 'travel', 'Cooking', 'fitbit', 'CampingandHiking', 'gameofthrones', 'IndieFolk']
                },
                'Gender': {
                    'a': ['AskMen', 'TrollYChromosome', 'AskMenOver30', 'OneY', 'TallMeetTall', 'daddit', 'ROTC', 'FierceFlow', 'malelivingspace', 'predaddit'],
                    'b': ['AskWomen', 'CraftyTrolls', 'AskWomenOver30', 'women', 'bigboobproblems', 'Mommit', 'USMilitarySO', 'HaircareScience', 'InteriorDesign', 'BabyBumps']  
                },
                'Age': {
                    'a': ['teenagers', 'youngatheists', 'teenrelationships', 'AskMen', 'saplings', 'hsxc', 'trackandfield', 'bapccanada', 'RedHotChiliPeppers'],
                    'b': ['RedditForGrownups', 'TrueAtheism', 'relationship_advice', 'AskMenOver30', 'eldertrees', 'running', 'trailrunning', 'MaleFashionMarket', 'canadacordcutters', 'pearljam'] 
                },
                'Edgy': {
                    'a': ['memes', 'watchpeoplesurvive', 'MissingPersons', 'twinpeaks', 'pickuplines', 'texts', 'startrekgifs', 'subredditoftheday', 'peeling', 'rapbattles'],
                    'b': ['ImGoingToHellForThis', 'watchpeopledie', 'MorbidReality', 'TrueDetective', 'MeanJokes', 'FiftyFifty', 'DaystromInstitute', 'SRSsucks', 'bestofworldstar']
                }
            }
        self.unseen = unseen
        self.root = root
        
            
            
        
        # map each key to a number (ie: Partisan A, a -> 0, Partisan A, b -> 1, Partisan B, a -> 2, etc.) using a for loop
        self.idx_map = {}
        idx = 0
        for key, value in self.groups.items():
            self.idx_map[key] = {}
            for side, files in value.items():
                self.idx_map[key][side] = idx 
                idx += 1
        if not self.unseen:
            self.idx_map['Partisan B']['a'] = self.idx_map['Partisan A']['a']
            self.idx_map['Partisan B']['b'] = self.idx_map['Partisan A']['b']
        
        self.data = {}
        super().__init__(root, transform, pre_transform, pre_filter)

        self.skip = [] # put topics not to include in the loss
        
       
    @property
    def raw_file_names(self):
        path = os.path.expandvars('$SLURM_TMPDIR')
        return list(glob(path + '/added_images/*/*.json'))

    @property
    def processed_file_names(self):
        #return ['giga-data.pt']
        path = os.path.expandvars(f'$SLURM_TMPDIR/{self.root}/processed')
        # TODO: this will be the total number of labels in the dataset, will have to update manually 
        for graph in tqdm(glob(path + f'/graph-*.pt')):
            # make sure graphs are appropriately sorted
            idx = int(graph.split('-')[-1][: -3])
            self.data[idx] = graph
            
        return [os.path.basename(x) for x in list(glob(path + f'/graph-*.pt'))]


    # def len(self) -> int:
    #     return len(self.processed_file_names)

    def process(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        markdown_regex = re.compile('^\[([\w\s\d]+)\]\(((?:\/|https?:\/\/)[\w\d./?=#]+)\)$')
        all_url_regex = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)")

        # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def clean_urls(x):
            x = markdown_regex.sub('[LINK1] \g<1> [LINK2]', x)
            return all_url_regex.sub('', x)

        def extract_text(x):
            if 'title' in x[0]: # x[0] is data
                if 'selftext' in x[0]:
                    body = '\n' + clean_urls(x[0]['selftext']) if x[0]['selftext'] != '' else ''
                else:
                    body = '\n' + clean_urls(x[0]['body']) if x[0]['body'] != 'NA' else ''
                # if len(x[1]) != 0:
                #     return '[IMG1] ' + x[0]['title'] + ' [IMG2] ' + body + '\n'
                return x[0]['title'] + body
            else:
                return clean_urls(x[0]['body'])
                #return x[0]['body']
        path = os.path.expandvars(f'$SLURM_TMPDIR/{self.root}/processed')
        path_slurm = os.path.expandvars('$SLURM_TMPDIR')
        known_files = glob(path + '/*')
        total = 0
        k = 0
        # with open(path_slurm + '/train-idx.txt') as f:
        #     for line in f:
        #         train_idx += [int(line)]
        # with open(path_slurm + '/test-idx.txt') as f:
        #     for line in f:
        #         valid_idx += [int(line)]

        # duped = pd.read_parquet(path_slurm + '/duped.parquet')['text'].unique()
        duped = []
        file_dir = path_slurm + '/added_images/'

        #with open(self.raw_file_names[0], 'r') as file, open(os.environ['SLURM_TMPDIR'] + '/train-idx-many.txt', 'w') as train, open(os.environ['SLURM_TMPDIR'] + '/test-idx-many.txt', 'w') as valid:
        for topic, value in tqdm(self.groups.items(), desc = 'Topics', position=0):
            for side, files in tqdm(value.items(), desc='Files', position=1, leave=False): # a, b
                for file in files:
               
                    with open(file_dir + topic + '/' + file + '.json', 'r') as file:
                        for graph_num, line in tqdm(enumerate(file), desc='Graphs', position=2, leave=False):
                            if path + f'/graph-{k}.pt' in known_files:
                                k += 1
                                continue
                            
                            raw_data = json.loads(line)
                            self.get_relative_depth(raw_data)
                            self.spread_downwards(raw_data)
                            data = {}
                            self.collapse_tree(raw_data, data, [])

                            # TODO: remove networkx operations, just use DGL
                            g = networkx.Graph()
                            adj = [(x[0]['parent_id'] if x[0]['parent_id'] != x[0]['link_id'] else 'top_level', x[0]['id']) for x in data.values() if 'parent_id' in x[0]]
                            # print(adj)
                            def make_features(x):
                                

                                if 'parent_id' not in x[0]:
                                    return (x[0]['id'], {'x': x[0]['id'], 'y': 'NA'})
                                return (x[0]['id'], {'x': x[0]['id'], 'y': 'NA'})

                            g.add_nodes_from([
                                    make_features(x) for x in data.values()
                                ]
                            )
                            g.add_edges_from(adj)
                            #g = g.to_undirected()
                            # print(g.nodes.data())
                            # print(adj)
                            graph = from_networkx(g)

                            graph.x_images = [[path_slurm + '/' + y for y in data[x][1]] for x in graph.x]
                            graph.x_text = [data[x] for x in graph.x]
                            order = graph.x
                            matrix = []
                            for key in order:
                                distances = data[key][2]
                                matrix += [[distances[y] for y in order]]
                            graph.distance_matrix = matrix
                            graph.x = tokenizer([extract_text(x) for x in graph.x_text], padding='max_length', truncation=True, return_tensors='pt', max_length=100) # this is a dictionary


                            graph.x_image_index = torch.tensor([True if len(x) != 0 else False for x in graph.x_images], dtype=torch.bool)
                            #graph.x_images = [extractor(Image.open([Image.open(for x in graph.x_images if len(x) != 0), return_tensors='pt')]
                            images = [Image.open(x[0]).convert(mode='RGB') for x in graph.x_images if len(x) != 0]
                            if len(images) != 0:
                                graph.x_images = extractor(images, return_tensors='pt')['pixel_values']
                            else:
                                graph.x_images = torch.zeros((1, 3, 224, 224))
                                #print(graph.x_images.size())
                        

                            # path.split('/)[-2] will be the topic name, which we can use contrastive loss
                            graph.y = torch.tensor([self.idx_map[topic][side]]) # todo: update this with the correct contrastive label
                            if not self.unseen:
                                if side == 'a':
                                    graph.hard_y = torch.tensor([self.idx_map[topic]['b']])
                                else:
                                    graph.hard_y = torch.tensor([self.idx_map[topic]['a']])
                            else:
                                graph.hard_y = torch.tensor([0])
                            
                            torch.save(graph, path + f'/graph-{k}.pt')
                            # if graph_num in valid_idx:
                            #     total += 1
                            #     valid.write(str(self.k) + '\n')
                            # elif graph_num in train_idx:
                            #     total += 1
                            #     train.write(str(self.k) + '\n')
                            k += 1
        
        
        

        print("FINAL K", k)
        print("TOTAL Ys", total)
        
        print("loading into memory")
       
        for path in tqdm(self.processed_file_names):
            # make sure graphs are appropriately sorted
            idx = int(path.split('-')[-1][: -3])
            
            self.data[idx] = path
            
            # print the dtype of all attributes
           
            #print(self.data[idx].x_images.dtype)
        #data, slices = self.collate(data_graphs)
        #torch.save((data, slices), self.processed_paths[0])

    def len(self):
        return len(self.data)


    # algorithm, go depth first, then do a second pass
    def get_relative_depth(self, node, depths={}) -> dict:
        distances = copy.deepcopy(depths)
        for key in distances.keys():
            distances[key] = (distances[key][0] + 1, distances[key][1])
        distances[node['id']] = [0, 0]

        for x in node['tree']:
            val = self.get_relative_depth(x, distances)
            for key, value in val.items():
                if key not in distances:
                    distances[key] = (value[0], value[1] + 1)
        node['distances'] = distances
        return distances

    def spread_downwards(self, node, depths={}):
        dists = depths
        for key, value in dists.items():
            if key not in node['distances']:
                node['distances'][key] = (value[0] + 1, value[1])
        for x in node['tree']:
            self.spread_downwards(x, node['distances'])

    def collapse_tree(self, comment, data, root_images):
        if 'id' not in comment['data']:
            comment['data']['id'] = comment['id']
        comment['data']['id'] = comment['id']

        id = comment['data']['id']
        i = 0
        if comment['data']['id'] in data:
            if comment['data']['body'] != data[comment['data']['id']][0]['body']: # if we already got the comment, and the contents are different (ie: deleted in one dump, but available in another)
                if data[comment['data']['id']][0]['body'] == '[deleted]':
                    if len(comment['images']) == 0: # if the new comment has no images, then we can just use the images of the post
                        comment['images'] = root_images #TODO: do we want this??
                    data[comment['data']['id']] = (comment['data'], comment['images'], comment['distances'])
                    print("updated!")


        # while comment['data']['id'] in data:
        #     comment['data']['id'] = id + '-' + str(i)
        #     i += 1
        else:
            if len(comment['images']) == 0:
                comment['images'] = root_images
            data[comment['data']['id']] = (comment['data'], comment['images'], comment['distances'])
        for child in comment['tree']:
            self.collapse_tree(child, data, root_images)

    def get(self, idx):
        return torch.load(self.data[idx])


if __name__ == '__main__':
    MultiModalReddit(root='processed_graphs')