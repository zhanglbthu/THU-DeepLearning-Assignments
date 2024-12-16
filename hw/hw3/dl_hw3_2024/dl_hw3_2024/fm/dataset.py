from torch.utils.data import Dataset
import torch
import random
import json
from tokenizer import TiktokenTokenizer
import pandas as pd


class EYLSFTStaticDataset(Dataset):

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None) -> None:
        super().__init__()
        if split == "train":
            with open("./data/sft/sft_train.json") as fp:
                dataset = json.load(fp)
        else:
            with open("./data/sft/sft_test.json") as fp:
                dataset = json.load(fp)
        self.tokens = []
        self.block_size = block_size
        
        tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading EYLSFTStaticDataset {split} split")
        for chosen in dataset:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y


class DahoasRMStaticDataset(Dataset):
    """
    https://huggingface.co/datasets/Dahoas/rm-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None) -> None:
        super().__init__()

        splits = {'train': 'data/train-00000-of-00001-2a1df75c6bce91ab.parquet',
                  'test': 'data/test-00000-of-00001-8c7c51afc6d45980.parquet'}
        dataset = pd.read_parquet("./data/rm-static/" + splits[split])
        self.pairs = []
        self.masks = []

        tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasRMStaticDataset {split} split")
        for _, data in dataset.iterrows():
            cnt += 1
            prompt = data['prompt']

            positive_text = prompt + data['chosen'] + "<|endoftext|>"
            positive = tokenizer(positive_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            negative_text = prompt + data['rejected'] + "<|endoftext|>"
            negative = tokenizer(negative_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            self.pairs.append(
                torch.stack((positive['input_ids'], negative['input_ids']),
                            dim=0))

            self.masks.append(
                torch.stack(
                    (positive['attention_mask'], negative['attention_mask']),
                    dim=0))
            if max_examples and cnt >= max_examples:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)
