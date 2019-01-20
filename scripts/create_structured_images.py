import glob
import json
import os
from pathlib import Path

base_dir = "/home/ubuntu/"
# base_dir = "\data\street2shop"
meta_dir = os.path.join(base_dir, "street2shop_crop", "meta", "json")
image_dir = os.path.join(base_dir, "street2shop_images")
structured_dir = os.path.join(base_dir, "street2shop_crop", "structured_images")
# all_pair_file_paths = glob.glob(meta_dir + "/retrieval_*.json")

files = ["retrieval_dresses.json", "retrieval_skirts.json", "retrieval_tops.json"]
all_pair_file_paths = [str(meta_dir+'\\'+x) for x in files]


for path in all_pair_file_paths:
    vertical = path.split("_")[-1].split(".")[0]
    query_dir = os.path.join(structured_dir, vertical+"_query")
    if not os.path.exists(query_dir):
        print("nahi hai naa")
        os.mkdir(query_dir)
    catalog_dir = os.path.join(structured_dir, vertical)

    print('catalog_dir   ', catalog_dir) 
    
    if not os.path.exists(catalog_dir):
        os.mkdir(catalog_dir)
    
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    product_ids = set()
    
    for item in data:
        ## getting individual data item   {'photo': 415547, 'product': 35465}
        product_ids.add(item["photo"])
    print("Symlinking catalog ids for {}".format(vertical))
    
    for product_id in product_ids:
        img_path = os.path.join(image_dir, str(product_id)+".jpg")
        dst_path = os.path.join(catalog_dir, str(product_id)+".jpg")
        if os.path.exists(img_path):
            os.symlink(img_path, dst_path)
    
    query_ids = set()
    
    for partition in ["train", "test"]:
        partition_file = partition+"_pairs_"+vertical+".json"
        print(partition_file)
        with open(os.path.join(meta_dir, partition_file), 'r') as jsonFile:
            pairs = json.load(jsonFile)
        for pair in pairs:
            query_ids.add(pair["photo"])
    print("Symlinking query ids for {}".format(vertical))
    
    for query_id in query_ids:
        img_path = os.path.join(image_dir, str(query_id)+".jpg")
        dst_path = os.path.join(query_dir, str(query_id)+".jpg")
        if os.path.exists(img_path):
            os.symlink(img_path, dst_path)