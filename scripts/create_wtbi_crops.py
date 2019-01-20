import glob
import json
import os
import cv2


base_dir = os.getcwd()
meta_dir = os.path.join(base_dir, "meta", "json")
image_dir = os.path.join(base_dir, "images")
structured_dir = os.path.join(base_dir, "structured_images")

query_files = glob.glob(meta_dir + "/*_pairs_*.json")

print(query_files)
counter = 0
for path in query_files:
    vertical = path.split("_")[-1].split(".")[0]
    wtbi_crop_dir = os.path.join(structured_dir, "wtbi_"+vertical+"_query_crop")
    if not os.path.exists(wtbi_crop_dir):
        print('wtbi crop dir', wtbi_crop_dir)
        os.mkdir(wtbi_crop_dir)
    query_dir = os.path.join(structured_dir, vertical+"_query")
    print("Processing path %s"%(path))
    with open(path) as jsonFile:
        pairs = json.load(jsonFile)
    for pair in pairs:
        query_id = pair["photo"]
        bbox = pair["bbox"]
        query_path = os.path.join(query_dir, str(query_id)+".jpg")
        if not os.path.exists(query_path):
            continue
        img = cv2.imread(query_path, cv2.IMREAD_COLOR)
        x, w, y, h = bbox["left"], bbox["width"], bbox["top"], bbox["height"]
        if type(x) != int:
            x = int(x)
            w = int(w)
            y = int(y)
            h = int(h)
        try:
            crop_img = img[y:y+h, x:x+w]
        except Exception as e:
            print('exception encountered', e)
            continue
        
        cv2.imwrite(os.path.join(wtbi_crop_dir, str(query_id)+".jpg"), crop_img)
        counter += 1
        if counter % 100 == 0:
            print(counter)