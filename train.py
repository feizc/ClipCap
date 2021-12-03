import torch 
import torch.nn as nn 
import argparse 
from transformers import GPT2Tokenizer 
from dataset import ClipCocoDataset 




def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_path', type=str, default='./data/features/train.pkl') 
    args = parser.parse_args() 

    tokenizer = GPT2Tokenizer.from_pretrained('./ckpt/gpt2') 
    dataset = ClipCocoDataset(args.data_path, tokenizer)




if __name__ == '__main__': 
    main()

