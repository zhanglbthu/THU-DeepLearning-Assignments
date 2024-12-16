from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    # Transformer
    n_layers: int
    n_heads: int
    embedding_dim: int
    dropout_rate: float
    use_bias: bool
    block_size: int
    vocab_size: int
    grad_clip: float = 1.0
    exp_name: str = ""
    batch_size: int = 1
    lr: float = 1e-6
    lora_rank: int = 0
    ckpt: str = "./checkpoints/gpt2_pretrained.pt"
    activation_checkpointing: bool = False
    # SFT
    num_steps: int = 20000
    # DPO
    num_epochs: int = 1
    kl_beta: float = 0.1

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def get_configs(name) -> TrainingConfig:
    if name == "gpt2":
        return TrainingConfig(
            n_layers=12,
            n_heads=12,
            embedding_dim=768,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257
        )
    elif name == "gpt2/dropout":
        return TrainingConfig(
            n_layers=12,
            n_heads=12,
            embedding_dim=768,
            dropout_rate=0.2,
            use_bias=True,
            block_size=1024,
            vocab_size=50257
        )