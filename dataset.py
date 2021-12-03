import torch 
from torch.utils.data import Dataset 
import pickle

class ClipCocoDataset(Dataset): 
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f) 
        self.clip_embs = all_data['clip_embedding']
        captions_raw = all_data['captions'] 
        self.img_ids = [caption['image_id'] for caption in captions_raw] 
        self.captions = [caption['txt'] for caption in captions_raw] 
        self.toeknizer = tokenizer 

        self.captions_tokens = [] 
        self.caption2embedding = []
        self.max_seq_len = 0 
        for caption in captions_raw: 
            self.captions_tokens.append(torch.tensor(self.toeknizer.encode(caption['txt']), dtype=torch.int64))
            self.caption2embedding.append(caption['clip_embedding'])
            self.max_seq_len = max(self.max_seq_len, self.captions_tokens[-1].size(0)) 
        
    def __len__(self):
        return len(self.captions_tokens) 
    
    def pad_tokens(self, index):
        tokens = self.captions_tokens[index] 
        padding = self.max_seq_len - tokens.size(0) 
        

