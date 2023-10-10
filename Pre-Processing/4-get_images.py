import json 
from tqdm import tqdm 
import re 
from PIL import Image
from io import BytesIO
from datetime import datetime
from pathlib import Path
import socket
import urllib3.util.connection as urllib_cn
from glob import glob 

def allowed_gai_family(): return socket.AF_INET
urllib_cn.allowed_gai_family = allowed_gai_family

from requests_futures.sessions import FuturesSession
from requests_ratelimiter import LimiterAdapter

from requests.adapters import HTTPAdapter, Retry
import requests

s = requests.Session()

retries = Retry(total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504 ])

imgur_adaptor = LimiterAdapter(per_hour=500)

s.mount('http://', HTTPAdapter(max_retries=retries))
s.mount('https://', HTTPAdapter(max_retries=retries))
s.mount('https://i.imgur.com/', imgur_adaptor)
image_session = FuturesSession(max_workers=24, session=s) # one worker per cpu core
deleted_img_url = "https://i.redd.it/EwG5Emc9PelOE-9TdeB2JlFwOK47ilV_bv0OWJXbpeY.jpg?auto=webp&amp;s=87d44a717eba5831e219c9a88b6d91c9f74cc333"
    
deleted_img = Image.open(BytesIO(image_session.get(deleted_img_url, timeout = 3).result().content)).getdata()
other_deleted_img = Image.open('deleted_imgur.png').getdata()

def main(): 
    datas = []
    image_links = []
    print('finding images')
    with open('pruned-graphs.json', 'r') as read:
        for line in tqdm(read, total=33192):
            data = json.loads(line)
            link_id = data['id']
            image_links += get_images(link_id, data)
            datas += [data]
    
    futures = []
    images_known = list(glob('images/*/*'))
    valid_formats = ['.jpg', '.jpeg', '.png', '.svg'] 
    print('queuing downloads')
    with open('imgs_to_download.txt', 'w') as file:
      for parent_id, id, images in tqdm(image_links):
          images = [x for x in images if any([y in x for y in valid_formats])]
          for i, image in enumerate(images):
              path = 'images/' + parent_id
              if (path + f'/{id}-{i}.png' not in images_known): 
                  #futures += [(id, path, i)]
                  file.write(','.join([image, id, path, str(i)]) + '\n')
                  #futures += [image_session.get(image, hooks={'response': hook_factory(id, path, i)}, timeout=5)]
  
    print('waiting for results...')
    progress = tqdm(futures, miniters=1)
    stat = {
       'got': 0,
       'failed': 0
    }
    for future in progress:
        try:
            res = future.result()
            print(res.succ)
            if res.succ[0]:
               stat['got'] += 1
            else:
               stat['failed'] += 1
        except Exception:
            stat['failed'] += 1
            continue
        progress.set_postfix(stat)

    images_known = list(glob('images/*/*'))
    def check_images(comment):
        # flag = False
        # if len(comment['images']) != 0:
        #     #print(comment['images'])
        #     flag = True
        comment['images'] = [x for x in comment['images'] if x in images_known]
        # if len(comment['images']) == 0 and flag:
        #     print('none there!')
        for x in comment['tree']:
            check_images(x)
    
    print('writing results')
    with open('pruned-with-images.json', 'w') as write:
       for x in tqdm(datas):
          check_images(x)
          write.write(json.dumps(x) + '\n') 
    
    print('done!')
    
def hook_factory(name, path, i):
  def format_save_image(response, *args, **args_kwargs):
    #print("doing the thing")
    # generate image path
    
    # print("try to open image")
    
    try:
      if not response.ok:
         response.succ = (False, name, path, response.status_code, response.url)
         return response
      img = Image.open(BytesIO(response.content))
    

    # check for deleted image

    # if it is same as deleted image
      if list(img.getdata()) == list(deleted_img):
        return 

      # resize image
      max_width, max_height = 256, 256
      height = int(img.height * max_width / img.width)

      # one dimension with max width/height of 256
      if height > max_width:
        width = int(max_height * img.width / img.height)

        img = img.resize((width, max_height), Image.Resampling.LANCZOS)
      else:
        img = img.resize((max_width, height), Image.Resampling.LANCZOS)

      if list(img.getdata()) == list(other_deleted_img):
        return
      # image name = name + iterator
      image_name = name + "-" + str(i) + ".png"
      Path(path).mkdir(parents=True, exist_ok=True)
      img.save(f"{path}/{image_name}")
      response.succ = (True, name, path, None, response.url)
    except Exception as e:
      response.succ = (False, name, path, e, response.url)
      
    return response
   
  return format_save_image

image_pattern = 'https?:\/\/(\S+?(?:jpe?g|png|gif|svg))'
https = "https://"

# prepend https:// to all urls in match_urls
def parse_images(body):
    image_urls = re.findall(image_pattern, body)
    image_urls = [https + url for url in image_urls]
    return image_urls

def get_images(link_id, comment):
    if 'body' in comment['data']:
        image_urls = parse_images(comment['data']['body'])
    else:
        image_urls = []
        comment['data']['body'] = 'NA'

    if 'url' in comment['data']:
        image_urls += parse_images(comment['data']['url'])
    image_urls = [x for x in image_urls if 'i.imgur.com' in x]
    if len(image_urls) != 0:
        res = [(link_id, comment['id'], image_urls)]
        id = comment['id']
        comment['images'] = [f'images/{link_id}/{id}-{i}.png' for i, x in enumerate(res)]
    else:
        res = []
        comment['images'] = []

    for child in comment['tree']:
        res += get_images(link_id, child)
    #res = [x for x in res if 'i.imgur.com' not in x[-1]]
    # for x in tree['tree']['children']:
    #     res = res + get_images(parent_id, x)
    return res

if __name__ == '__main__':
    main()