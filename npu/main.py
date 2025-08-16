import fire
from typing import Optional

from npu.generation import Qwen3MoE


def main(
        ckpt_dir: str = "/home/jinyong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        tokenizer_path: str = "/home/jinyong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/tokenizer.json",
        config_path: Optional[str] = None,
):
    qwen3_moe = Qwen3MoE(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path)
    prompts = [
        "Four score and seven years ago our fathers brought",
        "We hold these truths to be",
        "I have a dream that one day this nation will",
        "The only thing we have to fear is"
    ]
    responses = qwen3_moe.generate(prompts, max_gen_len=32, temperature=0.4, top_p=0.8)

    for prompt, completion in responses:
        print("\033[31m" + prompt + "\033[0m" + completion + "\n")


if __name__ == "__main__":
    fire.Fire(main)
