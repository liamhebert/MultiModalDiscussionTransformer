import pandas as pd
import re
import requests
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
from time import sleep
from tqdm import tqdm 

def allowed_gai_family():
    """
     https://github.com/shazow/urllib3/blob/master/urllib3/util/connection.py
    """
    family = socket.AF_INET
    # if urllib3_cn.HAS_IPV6:
    #     family = socket.AF_INET6 # force ipv6 only if it is available
    return family

urllib3_cn.allowed_gai_family = allowed_gai_family

def get_comment_linkid(id: list):
    try:
        url = f"https://api.pushshift.io/reddit/comment/search/?ids={','.join(id)}&size=900"
        data = requests.get(url).json()['data']
        res = {x: y['link_id'] for x, y in zip(id, data)}
    except Exception:
        print('sleeping')
        sleep(10)
        url = f"https://api.pushshift.io/reddit/comment/search/?ids={','.join(id)}&size=900"
        data = requests.get(url).json()['data']
        res = {x: y['link_id'] for x, y in zip(id, data)}
    return res

def main():
    # things I need:
    # - Label
    # - id
    # - comment id
    df = pd.read_csv('LTI/reddit.csv')
    df['id'] = df.id.str.split('\n?\d+\. \t*')
    # df.id = df.id.str.split('\n?\d+\. ')
    # df.id = df.id.apply(lambda x: x[1:])
    # def func(x):
    #     x[-1] = x[-1][:-1]
    #     return x
    # df.id = df.id.apply(func)

    df.id = df.id.apply(lambda x: x[1:])
    def func(x):
        x[-1] = x[-1][:-1]
        return x
    df.id = df.id.apply(func)
    df['size'] = df['id'].apply(len)
    def func(x):
        return [False] * x
    df['hate_mask'] = df['size'].apply(func)
    df['hate_speech_idx'] = df['hate_speech_idx'].fillna('[]')
    def func(x):
        #print('idx ', x['hate_speech_idx'])
        try:
            for y in x['hate_speech_idx'].strip('][').split(', '):
                if y == '':
                    break
                #print('size', len(x['hate_mask']))
                if int(y) > len(x['hate_mask']):
                    x['hate_mask'][-1] = True
                else:
                    x['hate_mask'][int(y) - 1] = True
        except Exception:
            print('ERROR', x.name)
        return x['hate_mask']

    df['hate_mask'] = df[['hate_speech_idx', 'hate_mask']].apply(func, axis=1)
    df = df[['id', 'hate_mask']]
    df = df.explode(column=['id', 'hate_mask'])
    
    ids = df['id'].unique()
    mapping_link_ids = {}
    last = 0
    for i in tqdm(range(900, len(ids), 900)):
        id_set = list(ids[last:i])
        link_ids = get_comment_linkid(id_set)
        for key, value in link_ids.items():
            mapping_link_ids[key] = [value[3:]]
        last = i
    
    link_ids = pd.DataFrame(mapping_link_ids).T
    link_ids = link_ids.rename({0: 'link_id'}, axis=1)
   
    df = df.join(link_ids, on='id')
    df = df.dropna()
    print(df)
    df = df.rename({'hate_mask': 'label'}, axis=1)
    
    df.to_parquet('lti-processed.parquet')

if __name__ == '__main__':
    main()