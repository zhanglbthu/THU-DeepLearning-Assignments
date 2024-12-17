import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from configs import TrainingConfig
from torch.utils.checkpoint import checkpoint
from tokenizer import TiktokenTokenizer
from attention import multi_head_self_attention


class MaskedMultiheadSelfAttention(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg: TrainingConfig = cfg
        self.qkv_projection = nn.Linear(cfg.embedding_dim,
                                        3 * cfg.embedding_dim,
                                        bias=cfg.use_bias)
        self.output_projection = nn.Linear(cfg.embedding_dim,
                                            cfg.embedding_dim,
                                            bias=cfg.use_bias)
        self.attention_dropout = nn.Dropout(cfg.dropout_rate)
        self.output_dropout = nn.Dropout(cfg.dropout_rate)

        ############################ Your code here ############################
        # TODO: construct a mask like this
        # [[1, 0, 0]
        #  [1, 1, 0]]
        #  [1, 1, 1]] when block_size is 3
        ############################ Your code here ############################
        mask = torch.empty(cfg.block_size, cfg.block_size)
        ########################################################################

        # insert (B, T) dimension for broadcasting later
        mask = mask.view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask)  # (1, 1, block_size, block_size)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: shape of (B, T, C)
        """
        B, T, C = x.size()
        x3 = self.qkv_projection(x)  # (B, T, 3 x C)
        Q, K, V = x3.split(self.cfg.embedding_dim, dim=2)  # (B, T, C)
        
        Q = Q.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(1, 2)  # (B, h, T, h_dim)
        K = K.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(1, 2)  # (B, h, T, h_dim)
        V = V.view(B, T, self.cfg.n_heads, C // self.cfg.n_heads).transpose(1, 2)  # (B, h, T, h_dim)
        weighted_value = multi_head_self_attention(Q, K, V, self.mask[:, :, :T, :T], self.attention_dropout, attention_mask)
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(B, T, C)

        y = self.output_projection(weighted_value)
        y = self.output_dropout(y)
        return y


class FeedForwardNetworks(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.embedding_dim,
                                4 * cfg.embedding_dim,
                                bias=cfg.use_bias)
        self.fc2 = nn.Linear(4 * cfg.embedding_dim,
                                cfg.embedding_dim,
                                bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        y = self.dropout(x)
        return y


class TransformerDecoderBlock(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg: TrainingConfig = cfg
        self.ln1 = nn.LayerNorm(cfg.embedding_dim,
                                elementwise_affine=cfg.use_bias)
        self.mmsa = MaskedMultiheadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embedding_dim,
                                elementwise_affine=cfg.use_bias)
        self.ffn = FeedForwardNetworks(cfg)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        identity1 = x
        x = self.ln1(x)
        x = self.mmsa(x, attention_mask)
        x = identity1 + x

        identity2 = x
        x = self.ln2(x)
        x = self.ffn(x)
        y = identity2 + x
        return y


class TransformerDecoder(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding_layer = nn.Embedding(
            cfg.vocab_size, cfg.embedding_dim)  # (Vocab, d)
        self.postion_embedding_layer = nn.Embedding(cfg.block_size,
                                                    cfg.embedding_dim)
        self.input_dropout = nn.Dropout(cfg.dropout_rate)
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.embedding_dim,
                               elementwise_affine=cfg.use_bias)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        B, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long,
                           device=x.device).unsqueeze(0)  # (1, T)
        token_embeddings = self.token_embedding_layer(x)  # (B, T, d)
        pos_embeddings = self.postion_embedding_layer(pos)  # (B, T, d)
        x = self.input_dropout(token_embeddings + pos_embeddings)

        # N decoder blocks
        for block in self.decoder_blocks:
            if self.cfg.activation_checkpointing:
                x = checkpoint(block, x, attention_mask)
            else:
                x = block(x, attention_mask)

        y = self.ln(x)
        return y


class GPT(nn.Module):

    def __init__(self, cfg: TrainingConfig) -> None:
        super().__init__()
        self.cfg: TrainingConfig = cfg
        self.tokenizer = TiktokenTokenizer("gpt2")

        self.transformer = TransformerDecoder(cfg)
        # Final linear layer as language model head w/o softmax
        self.lm_head = nn.Linear(cfg.embedding_dim,
                                    cfg.vocab_size,
                                    bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: Shape of (B, T)
        """
        x = self.transformer(x, attention_mask)  # x = (B, T, embedding_dim)
        logits = self.lm_head(x)  # logits = (B, T, voca_size)
        return logits
    
    def get_log_p(self, x: Tensor, attention_mask: Tensor = None):
        """
        x: Shape of (B, T)
        """
        x, y, attention_mask = x[:, :-1], x[:, 1:], attention_mask[:, 1:]
        logits = self(x, attention_mask)
        log_prob_all_vocab = F.log_softmax(logits, dim=2)
        log_prob_output = torch.gather(log_prob_all_vocab, dim=2, index=y.unsqueeze(2)).squeeze(2)
        return (log_prob_output * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1)
    
    @classmethod
    def from_pretrained(cls,
                        cfg: TrainingConfig,
                        ckpt: str = "./checkpoints/gpt2_pretrained.pt"):
        model = GPT(cfg)
        pretrained_states = torch.load(ckpt, map_location="cpu")
        if "model_state_dict" in pretrained_states:
            pretrained_states = pretrained_states["model_state_dict"]
        model.load_state_dict(pretrained_states, strict=False)
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L343
    
        Take a conditioning sequence of idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            next_id = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, next_id), dim=1)

        return idx
