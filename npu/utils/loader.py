from typing import List, Dict
from pathlib import Path
import torch
import torch.nn as nn
import ttnn
from safetensors.torch import safe_open
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

PATTERN_1 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.input_layernorm\.weight$')
PATTERN_2 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.post_attention_layernorm\.weight$')
PATTERN_3 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.q_norm.weight$')
PATTERN_4 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.k_norm.weight$')
PATTERN_5 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.q_proj.weight$')
PATTERN_6 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.k_proj.weight$')
PATTERN_7 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.v_proj.weight$')
PATTERN_8 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.self_attn\.o_proj.weight$')
PATTERN_9 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.mlp\.experts\.(?P<expert_index>\d+)\.gate_proj\.weight$')
PATTERN_10 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.mlp\.experts\.(?P<expert_index>\d+)\.up_proj\.weight$')
PATTERN_11 = re.compile(r'^model\.layers\.(?P<layer_index>\d+)\.mlp\.experts\.(?P<expert_index>\d+)\.down_proj\.weight$')



def owner_of(i: int, n_items: int, n_parts: int) -> int:
    if not (0 <= i < n_items):
        raise IndexError()
    if n_parts <= 0:
        raise ValueError()

    q, r = divmod(n_items, n_parts)
    cutoff = (q + 1) * r
    return (i // (q + 1)) if i < cutoff else (r + (i - cutoff) // q)


def load_shard(ckpt_path: Path, model: nn.Module, devices: Dict[int, ttnn.MeshDevice]) -> None:
    assert len(devices) == 8

    with torch.no_grad():
        state_dict = dict(model.named_parameters())

    def partially_applied_owner_of(i: int) -> int:
        return owner_of(i, 48, 8)

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            source: torch.Tensor = f.get_tensor(key)

            if key == "model.embed_tokens.weight":
                model.embed_tokens.load(source, devices[0])
            elif key == "lm_head.weight":
                model.lm_head.load(source, devices[7])
            elif key == "model.norm.weight":
                model.norm.load(source, devices[7])
            elif m := PATTERN_1.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].input_layernorm.load(source, device)
            elif m:= PATTERN_2.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].post_attention_layernorm.load(source, device)
            elif m:= PATTERN_3.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].self_attn.q_norm.load(source, device)
            elif m := PATTERN_4.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].self_attn.k_norm.load(source, device)
            elif m := PATTERN_5.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].self_attn.q_proj.load(source, device)
            elif m := PATTERN_6.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].self_attn.k_proj.load(source, device)
            elif m := PATTERN_7.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].self_attn.v_proj.load(source, device)
            elif m := PATTERN_8.fullmatch(key):
                layer_index = int(m['layer_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].self_attn.o_proj.load(source, device)
            elif m := PATTERN_9.fullmatch(key):
                layer_index, expert_index = int(m['layer_index']), int(m['expert_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].mlp.experts[expert_index].gate_proj.load(source, device)
            elif m := PATTERN_10.fullmatch(key):
                layer_index, expert_index = int(m['layer_index']), int(m['expert_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].mlp.experts[expert_index].up_proj.load(source, device)
            elif m := PATTERN_11.fullmatch(key):
                layer_index, expert_index = int(m['layer_index']), int(m['expert_index'])
                device = devices[partially_applied_owner_of(layer_index)]
                model.layers[layer_index].mlp.experts[expert_index].down_proj.load(source, device)
            else:
                key = key[len("model."):] if key.startswith("model.") else key
                target: torch.Tensor = state_dict[key]

                assert source.shape == target.shape
                with torch.no_grad():
                    target.copy_(source.to(dtype=torch.float16))




def load(ckpt_dir: str, model: nn.Module, devices: Dict[int, ttnn.MeshDevice], io_workers: int = 4, blas_workers: int = 2) -> None:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.safetensors"))

    num_threads = torch.get_num_threads()
    torch.set_num_threads(blas_workers)

    with torch.no_grad():
        if io_workers == 1:
            for ckpt_path in ckpt_paths:
                load_shard(ckpt_path, model, devices)
        else:
            with ThreadPoolExecutor(max_workers=io_workers) as ex:
                futures = [ex.submit(load_shard, ckpt_path, model, devices) for ckpt_path in ckpt_paths]
                for fut in as_completed(futures):
                    _ = fut.result()

    torch.set_num_threads(num_threads)


def materialize(model: nn.Module) -> None:
    seen_param_map = dict()

    def _recurse(m: nn.Module) -> None:
        for name, parameter in m._parameters.items():
            if parameter is None:
                continue
            if not parameter.is_meta:
                continue

            parameter.grad = None
            key = id(parameter)
            if key in seen_param_map:
                m._parameters[name] = seen_param_map[key]
            else:
                new_parameter = nn.Parameter(torch.empty_like(parameter, device=torch.device("cpu")), requires_grad=False)
                seen_param_map[key] = new_parameter
                m._parameters[name] = new_parameter

        for child in m.children():
            _recurse(child)

    _recurse(model)
