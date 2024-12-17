from gpt import GPT
from configs import get_configs
import torch
import tiktoken
import click


def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode


def generate_gpt2(model, prompt, device, samples=2):
    model.eval()
    model.to(device)
    max_new_tokens = 50
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    for k in range(samples):
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')


@click.command()
@click.option('--task', '-t', default=0)
@click.option('--ckpt', '-c', default="./checkpoints/gpt2_pretrained.pt")
def main(task, ckpt):
    device = 'cuda'
    cfg = get_configs("gpt2")

    if task == 0:
        prompt = """Human: Hello, my name is Kate. What is your name?

Assitant:"""
    elif task == 1:
        prompt = """Human: You are an asshole! You are an idiot!

Assitant:"""

    model = GPT.from_pretrained(cfg, ckpt)
    
    generate_gpt2(model, prompt, device, samples=10)


if __name__ == "__main__":
    main()
