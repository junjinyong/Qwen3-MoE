# Qwen3-MoE

This repository provides a **standalone implementation** of the Qwen3-MoE model that runs without the `transformers` package. 
It uses only the PyTorch standard API and a minimal set of external libraries (`tokenizers`, `safetensors`) to support model loading, inference, and text generation. 


## Features

* Transformers-free
    + Removed all dependencies on the `transformers` library by reimplementing the required components locally.

* Mixture of Experts (MoE)
    + Expert gating, top-k routing, and weighted combination based on the Tenstorrent Qwen3-MoE reference.

* SDPA Attention 
    + Directly uses PyTorch’s `scaled_dot_product_attention`.
    + Supports Grouped Query Attention (GQA).

* Rotary Position Embedding (RoPE)
    + Complex-number based implementation of RoPE.

* Nucleus Sampling
    + Implements Top-p sampling for natural text generation.

* KV Cache
    + Efficient memory usage for long sequence generation.


## Project Structure
```
.
├── qwen3_moe
│  ├── config.json                       # Example configuration file
│  ├── configuration_qwen3_moe.py        # Model hyperparameters and configuration
│  ├── modeling_qwen3_moe.py             # Qwen3-MoE model architecture
│  ├── README.md                         # Module-specific documentation
│  └── rope_helpers.py                   # RoPE precomputation and application
├── transformers
│  └── sdpa_attention.py                 # SDPA attention wrapper
├── utils
│  └── structural_types.py               # Structural type definitions
├── generation.py                         # Model loading and text generation
├── main.py                               # CLI entry point
├── output.txt                            # Sample generation output
├── README.md                             # Main project documentation
└── requirements.txt                      # Python dependencies

```


## Installation
```bash
git clone https://github.com/junjinyong/Qwen3-MoE
cd Qwen3-MoE

pip install torch tokenizers safetensors fire
```

Obtain Qwen3-MoE `.safetensors` checkpoint files and `tokenizer.json` from the Hugging Face Hub or another source. 
`ckpt_dir` should contain all safetensor files, and `tokenizer_path` should point to the tokenizer JSON.
```bash
git clone https://huggingface.co/Qwen/Qwen3-30B-A3B
cd Qwen3-30B-A3B

git lfs install
git lfs pull
```

## Usage
```bash
python3 main.py \
  --ckpt_dir /path/to/Qwen3-30B-A3B \
  --tokenizer_path /path/to/Qwen3-30B-A3B/tokenizer.json \
  --config_path /path/to/config.json
```

## Example
| Prompt                                             | Completion                                                                                                                                                            |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Four score and seven years ago our fathers brought | forth on this continent, a new nation, conceived in liberty, and dedicated to the proposition that all men are created equal. But, we are now engaged in a ...        |
| We hold these truths to be                         | self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable rights, that among them, that among these are, that ... |
| I have a dream that one day this nation will       | rise up and live out the true meaning of the American dream. I have a dream that one day, I will be able to live in a country where I ...                             |
| The only thing we have to fear is                  | fear itself. That's the line that's been repeated so often that it's become a cliché, but it's true. Fear is the only thing we have to ...                            |


## Reference
This implementation was derived from the following sources:
1. [Qwen3-MoE](https://github.com/tenstorrent/tt-metal/blob/ilkoo/qwen3_moe/models/demos/qwen3_moe/reference/modeling_qwen3_moe.py#L180): Model Architecture
2. [LLaMA](https://github.com/meta-llama/llama): Positional Embeddings, KV Cache, Generation
3. [Transformers](https://github.com/huggingface/transformers)
