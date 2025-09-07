import torch
from typing import Protocol, Sequence


class Config(Protocol):
    rope_theta: float
    head_dim: int
    max_seq_len: int


class Loadable(Protocol):
    def load(self, tensor: torch.Tensor) -> None:
        ...

class AttrLoads(Protocol):
    def __getattr__(self, name: str) -> Loadable:
        ...

class MLP(Protocol):
    experts: Sequence[AttrLoads]
    gate: Loadable

class Layer(AttrLoads, Protocol):
    self_attn: AttrLoads
    mlp: MLP

class Model(Protocol):
    embed_tokens: Loadable
    lm_head: Loadable
    norm: Loadable
    layers: Sequence[Layer]
