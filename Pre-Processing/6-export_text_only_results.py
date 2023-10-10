import json 
import pandas as pd
from tqdm import tqdm

duped = pd.read_parquet('duped.parquet')['text'].unique()
hate_labels = ['DEG', 'lti_hate', 'IdentityDirectedAbuse', 'AffiliationDirectedAbuse']
good_labels = ['Neutral', 'lti_normal', 'NDG', 'HOM']

def collapse_tree(comment, data=[], root_images=[]):
    #is_duped = (comment['data']['label'] not in hate_labels and comment['data']['label'] not in good_labels) or (comment['data']['body'] in duped)
    is_duped = (comment['data']['body'] in duped)
    
    
    if not is_duped:
        comment['root_images'] = root_images
        data += [comment]

    for x in comment['tree']:
        collapse_tree(x, data, root_images)

def extract_relevent_bits(comment):
    id = comment['data']['id']
    if 'link_id' in comment['data']:
        link_id = comment['data']['link_id']
        text = comment['data']['body']
    else:
        link_id = comment['data']['id']
        text = comment['data']['title'] + '\n' + comment['data']['body']
    root_images = comment['root_images']
    images = comment['images']
    label = comment['data']['label']

    if label in hate_labels:
        primary_label = True
    else:
        primary_label = False
    
    return id, link_id, text, images, root_images, label, primary_label


def main():
    total = 0
    data = []
    with open('pruned-with-images.json', 'r') as file:
        for line in tqdm(file, total=33192):
            yx = json.loads(line)
            collapse_tree(yx, data, yx['images'])
    print(len(data))
    for split_idx in range(7):
        valid_ids = []
        with open(f'test_index-{split_idx}.txt', 'r') as file:
            for line in file:
                valid_ids += [int(line)]

        train_ids = []
        with open(f'train_index-{split_idx}.txt', 'r') as file:
            for line in file:
                train_ids += [int(line)]

        files = ['pruned-with-images.json']
        total = 0
        counted = 0
        for y in files:
            for split in ['train', 'test']:
                #print(split)
                data = []
                
                with open(y, 'r') as file:
                    for z, line in enumerate(file):
                        if (split == 'train' and z in train_ids) or (split == 'test' and z in valid_ids):
                            x = json.loads(line)
                            collapse_tree(x, data, x['images'])
                        
                        # elif z not in train_ids and z not in valid_ids:
                        #     print('MISSING', z)
                    total += len(data)
                    #print(len(data))
                fields = {}
                fields['id'] = []
                fields['link_id'] = []
                fields['text'] = []
                fields['images'] = []
                fields['label'] = []
                fields['primary_label'] = []
                fields['root_images'] = []
                for row in data:
                    id, link_id, text, images, root_images, label, primary_label = extract_relevent_bits(row)
                    fields['id'] += [id]
                    fields['link_id'] += [link_id]
                    fields['text'] += [text]
                    fields['images'] += [images]
                    fields['root_images'] += [root_images]
                    fields['label'] += [label]
                    fields['primary_label'] += [primary_label]
                
                data = pd.DataFrame.from_dict(fields)
                data['image_count'] = data['images'].apply(len)
                data.to_parquet(f'HatefulDiscussions_dataset_{split}-split-{split_idx}.parquet')

        print('TOTAL', total)

if __name__ == '__main__':
    main()