import torch 
from torch.utils.data import Dataset 
import pickle

class ClipCocoDataset(Dataset): 
    def __init__(self, data_path, tokenizer, img_length):
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f) 
        self.clip_embs = all_data['clip_embedding']
        captions_raw = all_data['captions'] 
        self.img_ids = [caption['image_id'] for caption in captions_raw] 
        self.captions = [caption['txt'] for caption in captions_raw] 
        self.toeknizer = tokenizer 
        self.img_length = img_length 

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
        if padding > 0: 
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)-1)) 
            self.captions_tokens[index] = tokens 
        else:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[index] = tokens 
        mask = tokens.ge(0) 
        tokens[~mask] = 0 
        mask = mask.float() 
        mask = torch.cat((torch.ones(self.img_length), mask), dim=0) 
        return tokens, mask 
    
    def __getitem__(self, index):
        tokens, mask = self.pad_tokens(index) 
        clip_emb = self.clip_embs[self.caption2embedding[index]]
        return tokens, mask, clip_emb 

