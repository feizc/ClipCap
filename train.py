import torch 
import torch.nn as nn 
from torch.nn import functional as F
import argparse 
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from dataset import ClipCocoDataset 
from model import ClipCaptionModel, MappingType
import os 
from tqdm import tqdm 
from torch.utils.data import DataLoader 


def train(dataset, model, args):
    epochs = 5
    lr = 1e-5
    warmup_steps = 5000
    use_device = torch.cuda.is_available()
    device = torch.device('cuda' if use_device else 'cpu') 
    batch_size = 1 
    output_dir = args.out_dir
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir) 
    
    model = model.to(device) 
    model.train() 
    optimizer = AdamW(model.parameters(), lr=lr) 
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs): 
        progress = tqdm(total=len(train_dataloader))
        for idx, (tokens, mask, clip_emb) in enumerate(train_dataloader):
            model.zero_grad() 
            tokens, mask, clip_emb = tokens.to(device), mask.to(device), clip_emb.to(device) 
            outputs = model(tokens, clip_emb, mask) 
            logits = outputs.logits[:, dataset.img_length-1:-1] 
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0) 
            loss.backward()
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
        progress.close() 
        torch.save(
            model.state_dict(), 
            os.path.join(output_dir, 'model.bin')
        )
        

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', type=str, default='./data/features/train.pkl') 
    parser.add_argument('--out_dir', type=str, default='./ckpt/clipcap')
    args = parser.parse_args() 

    img_length = 10 # full mask 
    clip_constant = 10
    clip_size = 512 
    number_layer = 12 
    use_mlp = False 

    tokenizer = GPT2Tokenizer.from_pretrained('./ckpt/gpt2') 
    dataset = ClipCocoDataset(args.data_path, tokenizer, img_length) 
    if use_mlp: 
        model = ClipCaptionModel(img_length, clip_constant, clip_size, number_layer) 
    else:
        model = ClipCaptionModel(img_length, clip_constant, clip_size, number_layer, MappingType.Transformer) 
    
    train(dataset, model, args)


if __name__ == '__main__': 
    main()

