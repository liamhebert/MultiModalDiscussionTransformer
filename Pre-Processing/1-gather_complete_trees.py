import pandas as pd
import re
import requests
import socket
import requests.packages.urllib3.util.connection as urllib3_cn
from time import sleep
from tqdm import tqdm 
from glob import glob 
from datetime import datetime
import os 

def allowed_gai_family():
    """
     https://github.com/shazow/urllib3/blob/master/urllib3/util/connection.py
    """
    family = socket.AF_INET

    return family

urllib3_cn.allowed_gai_family = allowed_gai_family

def get_comment_time(id: list):
    try:
        url = f"https://api.pushshift.io/reddit/comment/search/?ids={','.join(id)}&size=900"
        data = requests.get(url).json()['data']
        res = {x: (y['created_utc'], y['link_id']) for x, y in zip(id, data)}
    except Exception:
        print('sleeping')
        sleep(15)
        url = f"https://api.pushshift.io/reddit/comment/search/?ids={','.join(id)}&size=900"
        try:
            data = requests.get(url).json()['data']
        except Exception:
            return {}
        res = {x: (y['created_utc'], y['link_id']) for x, y in zip(id, data)}
    return res

def main():
    df = pd.concat([pd.read_parquet(x) for x in glob('data/*-processed.parquet')])
    # assume id, link_id, label
    
    times_mapping = {'id': [], 'created_utc': [], 'link_id': []}
    ids = df['id'].unique()
    last = 0
    print('getting created_utc times')
    for i in tqdm(range(900, len(ids), 900)):
        id_set = list(ids[last:i])
        times = get_comment_time(id_set)
        for key, value in times.items():
            
            times_mapping['created_utc'] += [value[0]]
            times_mapping['id'] += [key]
            times_mapping['link_id'] += [value[1][3:]]
        last = i
    
    times = pd.DataFrame(times_mapping)
    times = times.set_index('id')
    #times = times.rename({0: 'created_utc'}, axis=1)
    df = df.drop('link_id', axis=1).set_index('id').join(times)
    df = df.dropna()
    df['created_utc'] = df['created_utc'].apply(datetime.fromtimestamp)
    df['formatted_date'] = df['created_utc'].apply(lambda x: str(x.year) + '-' + str(x.month))
    df.to_parquet('complete_dataframe.parquet')
    print('downloading dumps...')
    total = len(df.groupby('formatted_date'))
    files_we_have = list(glob('*-comments.json'))
    for i, (date, x) in tqdm(enumerate(df.groupby('formatted_date')), total=total):
        date = str(date)
        
        if len(date) == 6:
            date = date[:5] + '0' + date[5:]
        if date + '-comments.json' in files_we_have:
            continue
        ids_to_find = list(x['link_id'].unique())
        ids_found = []
        os.system(f'wget files.pushshift.io/reddit/submissions/RS_{date}.zst > /dev/null')
        os.system(f'sftp files.pushshift.ioreddit/comments/RC_{date}.zst > /dev/null')
        os.system(f'unzstd RS_{date}.zst --memory=2048MB > /dev/null')
        os.system(f'unzstd RC_{date}.zst --memory=2048MB > /dev/null')
        regex_sub_id = re.compile('\"id\":\"([a-zA-Z0-9]*)\"')
        regex_comment_id = re.compile('\"link_id\":\"t3_([a-zA-Z0-9]*)\"')
        found = 0
        with open(f'RS_{date}', 'r') as read:
            with open(date + '-submissions.json', 'w') as write:
                for line in read:
                    id_test = next(regex_sub_id.finditer(line)).group(1)

                    #print("\"" + id_test + "\"")
                    if id_test in ids_to_find:
                        found += 1
                        ids_found += [str(id_test)]
                        ids_to_find.remove(str(id_test))
                        write.write(line)
                        write.write('\n')

        with open(f'RC_{date}', 'r') as read:
            with open(date + '-comments.json', 'w') as write:
                for line in read:
                    id_test = next(regex_comment_id.finditer(line)).group(1)
                    #print("\"" + id_test + "\"")
                    if id_test in ids_found:
                        #ids_to_find.remove(str(id_test))
                        write.write(line)
                        write.write('\n')
        os.system(f'rm RS_{date}.zst')
        os.system(f'rm RC_{date}.zst')
        os.system(f'rm RS_{date}')
        os.system(f'rm RC_{date}')

        if i % 25 == 0:
            os.system(f'notify "processed {i}/{total} ({i / total})"')

    os.system('notify \"done processing!\"')

if __name__ == '__main__':
    main()
