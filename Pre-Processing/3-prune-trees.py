import math 
import json
from glob import glob 
from tqdm import tqdm 

def main():
    with open('complete-graphs.json', 'r') as read, open('pruned-graphs.json', 'w') as write:
        for line in tqdm(read, total=33192):
            data = json.loads(line)
            trim_and_get_size(data)
            write.write(json.dumps(data) + '\n')

def count_size_of_tree(x):
    return sum([count_size_of_tree(y) for y in x['tree']]) + 1
            
def trim_and_get_size(comment: dict, depth=0):
    sizes = []  # (size, index)
    infs = 0
    for i, child in enumerate(comment['tree']):
        if child['data']['label'] != 'NA':
            trim_and_get_size(child, depth + 1)
            # else:
            #     child['tree'] = []
            infs += 1
            sizes += [(math.inf, i)]
        else:
            if depth + 1 < 7:
                res = trim_and_get_size(child, depth + 1)
                sizes += [(res, i)]
                if res == math.inf:
                    infs += 1
            else:
                child['tree'] = []
                sizes += [(0, i)]
    trimed_size = max(2, infs)
    sizes = sorted(sizes, key=lambda x: x[0], reverse=True)[:trimed_size]
    new_size = sum([s[0] for s in sizes])
    comment['tree'] = [comment['tree'][x[1]] for x in sizes]
    return new_size + 1

if __name__ == '__main__':
    main()