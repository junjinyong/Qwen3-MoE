from typing import List, Dict
from pathlib import Path
import torch
import torch.nn as nn
import ttnn
from safetensors.torch import safe_open
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

PATTERN_LN = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.(?P<norm_name>input_layernorm|post_attention_layernorm)\.weight$')
PATTERN_ATTN = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.(?P<attn_name>q_norm|k_norm|q_proj|k_proj|v_proj|o_proj)\.weight$')
PATTERN_MLP_PROJ = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.mlp\.experts\.(?P<expert_index>\d+)\.(?P<proj_name>gate_proj|up_proj|down_proj)\.weight$')
PATTERN_MOE_GATE = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.mlp\.gate\.weight$')


def owner_of(i: int, n_items: int, n_parts: int) -> int:
    if not (0 <= i < n_items):
        raise IndexError()
    if n_parts <= 0:
        raise ValueError()

    q, r = divmod(n_items, n_parts)
    cutoff = (q + 1) * r
    return (i // (q + 1)) if i < cutoff else (r + (i - cutoff) // q)


def load_shard(ckpt_path: Path, model: nn.Module) -> None:
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            source: torch.Tensor = f.get_tensor(key)

            if key == "model.embed_tokens.weight":
                model.embed_tokens.load(source)
            elif key == "lm_head.weight":
                model.lm_head.load(source)
            elif key == "model.norm.weight":
                model.norm.load(source)
            elif m := PATTERN_LN.fullmatch(key):
                layer_index, norm_name = int(m['layer_index']), m['norm_name']
                getattr(model.layers[layer_index], norm_name).load(source)
            elif m:= PATTERN_ATTN.fullmatch(key):
                layer_index, attn_name = int(m['layer_index']), m['attn_name']
                getattr(model.layers[layer_index].self_attn, attn_name).load(source)
            elif m := PATTERN_MLP_PROJ.fullmatch(key):
                layer_index, expert_index, proj_name = int(m['layer_index']), int(m['expert_index']), m['proj_name']
                getattr(model.layers[layer_index].mlp.experts[expert_index], proj_name).load(source)
            elif m:= PATTERN_MOE_GATE.fullmatch(key):
                layer_index = int(m['layer_index'])
                model.layers[layer_index].mlp.gate.load(source)
            else:
                assert False


def load(ckpt_dir: str, model: nn.Module, io_workers: int = 4, blas_workers: int = 2) -> None:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.safetensors"))

    num_threads = torch.get_num_threads()
    torch.set_num_threads(blas_workers)

    with torch.no_grad():
        if io_workers == 1:
            for ckpt_path in ckpt_paths:
                load_shard(ckpt_path, model)
        else:
            with ThreadPoolExecutor(max_workers=io_workers) as ex:
                futures = [ex.submit(load_shard, ckpt_path, model) for ckpt_path in ckpt_paths]
                for fut in as_completed(futures):
                    _ = fut.result()

    torch.set_num_threads(num_threads)
