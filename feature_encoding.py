import torch 
import skimage.io as io
import clip 
from PIL import Image 
import pickle 
import json 
import os 
from tqdm import tqdm 

import ssl
ssl._create_default_https_context = ssl._create_unverified_context 


def image_encpoding(): 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    out_put_path = './data/features/train.pkl'
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open('./data/caption.json', 'r') as f:
        data = json.load(f)

    all_embeddings = [] 
    all_captions = [] 
    for i in tqdm(range(len(data))): 
        d = data[i] 
        img_id = d['image_id'] 
        file_name = f"./data/image/{int(img_id)}.jpg" 
        image = io.imread(file_name) 
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d) #[1, 512]
        
    
    with open(out_put_path, 'wb') as f:
        pickle.dump({'clip_embedding': torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)
    





if __name__ == '__main__': 
    image_encpoding()







