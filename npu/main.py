import fire
from typing import List, Dict, Optional

import ttnn

from npu.generation import Qwen3MoE


def main(
        ckpt_dir: str = "/home/jinyong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        tokenizer_path: str = "/home/jinyong/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/tokenizer.json",
        config_path: Optional[str] = None,
):
    print("\033[31m \033[43m" + f"Available devices: {ttnn.GetNumAvailableDevices()}" + "\033[0m")
    print("\033[31m \033[43m" + f"PCIe devices: {ttnn.GetNumPCIeDevices()}" + "\033[0m")
    assert ttnn.GetNumPCIeDevices() == 4
    assert ttnn.GetNumAvailableDevices() == 8

    device_ids: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]
    devices: Dict[int, ttnn.MeshDevice] = ttnn.CreateDevices(device_ids)

    qwen3_moe = Qwen3MoE(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, config_path=config_path, devices=devices)
    prompts = [
        "Four score and seven years ago our fathers brought",
        "We hold these truths to be",
        "I have a dream that one day this nation will",
        "The only thing we have to fear is"
    ]
    responses = qwen3_moe.generate(prompts, max_gen_len=4, temperature=0.0, top_p=0.8)

    for prompt, completion in responses:
        print("\033[31m" + prompt + "\033[0m" + completion + "\n")

    ttnn.CloseDevices(devices)


if __name__ == "__main__":
    fire.Fire(main)
