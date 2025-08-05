import fire
import json
from typing import Optional
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

from tokenizers import Tokenizer

from qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

def main(
        ckpt_dir: str = "/home/junjinyong/Desktop/Qwen3-30B-A3B/",
        tokenizer_path: str = "/home/junjinyong/Desktop/Qwen3-30B-A3B/tokenizer.json",
        config_path: Optional[str] = None,
):
    data = None
    if config_path is not None:
        with open(config_path, "r") as f:
            data = json.load(f)
    config = Qwen3MoeConfig.from_dict(data)

    ckpt_paths = sorted(Path(ckpt_dir).glob("*.safetensors"))
    state_dict = {
        (key[len("model."): ] if key.startswith("model.") else key): value
        for ckpt_path in ckpt_paths
        for key, value in load_file(ckpt_path, device="cpu").items()
    }

    torch.set_default_dtype(torch.float16)
    model = Qwen3MoeModel(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    tokenizer = Tokenizer.from_file(tokenizer_path)

    prompt = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal."
    prompt = "Four score and seven years ago our fathers brought"
    print("prompt:", prompt)

    prompts = [prompt]
    tokens = torch.tensor([tokenizer.encode(prompt).ids for prompt in prompts], dtype=torch.int64, device=torch.device("cpu"))
    logits = model(tokens)
    save_file({"logits": logits}, "result.safetensors")

    next_token_logits = logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)
    topk = torch.topk(probs, k=5)
    top_tokens = list(map(tokenizer.id_to_token, topk.indices.tolist()))
    top_probs = topk.values.tolist()
    for token, prob in zip(top_tokens, top_probs):
        print(f"{token!r}: {prob:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
