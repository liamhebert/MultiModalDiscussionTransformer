import json
from tqdm import tqdm
from glob import glob
import pandas as pd
from joblib import Parallel, delayed
def main():
    # output should be year-month-combined.json
    # json with
    """
    {
        data: {} (including "label": <>)
        id: ""
        tree: [<>]
    }
    """
    df = pd.concat([pd.read_parquet(x) for x in glob('data/*-processed.parquet')])
    df = df.set_index('id')
    #labels = 0
    files_we_have = list(glob('complete-graphs-*.json'))
    def process_file(file, num):
        if f'complete-graphs-{str(num)}.json' in files_we_have:
            return 0
        with open(f'complete-graphs-{str(num)}.json', 'w') as write:
            labels = 0
            #for file in tqdm(list(glob('*-*-comments.json'))):
            date = file[:7]
            sub = date + "-submissions.json"
            comment = date + "-comments.json"
            
            graph = {}
            counts = {}
            for line in open(sub, 'r'):
                if line == '\n':
                    continue
                data = json.loads(line)
                link_id = data['id']
                 # note: this can be made much more efficient if the df was filtered by date, like it is in the previous step
                data['label'] = df.loc[link_id]['label'] if link_id in df.index else 'NA'
                if type(data['label']) == pd.Series:
                    #print(data['label'])
                    #print(df.loc[link_id]['label'])
                    data['label'] = 'NA'
                if data['label'] != 'NA':
                    labels += 1

                graph[link_id] = {link_id: {'data': data, 'tree': [], 'id': link_id}}
                counts[link_id] = 1
            missing = []
            for line in open(comment, 'r'):
                if line == '\n':
                    continue
                node = json.loads(line)
                parent_id = node['parent_id'][3:]
                node['parent_id'] = node['parent_id'][3:]
                link_id = node['link_id'][3:]
                if link_id not in graph:
                    continue
                node['label'] = df.loc[node['id']]['label'] if node['id'] in df.index else 'NA'
                if type(node['label']) == pd.Series:
                    #print(node['label'])
                    #print(df.loc[node['id']]['label'])
                    node['label'] = 'NA'
                if node['label'] != 'NA':
                    labels += 1

                graph[link_id][node['id']] = {'data': node, 'tree': [], 'id': node['id']}
                if parent_id in graph[link_id]:
                    graph[link_id][parent_id]['tree'] += [graph[link_id][node['id']]]
                    counts[link_id] += 1
                else:
                    missing += [(link_id, parent_id, graph[link_id][node['id']])]
            
            for link_id, parent_id, data in missing:
                if parent_id in graph[link_id]:
                    graph[link_id][parent_id]['tree'] += [data]
                    counts[link_id] += 1
                
                        
            for key, data in graph.items():
                write.write(json.dumps(data[key]) + '\n')
                if counts[key] != count_size_of_tree(data[key]):
                    print(counts[key], count_size_of_tree(data[key]))
        return labels
    res = Parallel(n_jobs=-1)(delayed(process_file)(file, i) for i, file in tqdm(enumerate(list(glob('*-*-comments.json'))), total=144))
    with open('complete-graphs.json', 'w') as file:
        for graph_file in glob('complete-graphs-*.json'):
            with open(graph_file, 'r') as read:
                for line in read:
                    file.write(line)
    print('labels: ', sum(res))

def count_size_of_tree(x):
    return sum([count_size_of_tree(y) for y in x['tree']]) + 1
            
    

if __name__ == '__main__':
    main()