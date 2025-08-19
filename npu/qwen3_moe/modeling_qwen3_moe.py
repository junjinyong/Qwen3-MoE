import torch
import torch.nn.functional as F
from torch import nn

from typing import Union, Dict

import ttnn

from npu.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from npu.transformers.sdpa_attention import sdpa_attention_forward
from npu.qwen3_moe.rope_helpers import precompute_freqs_cis, apply_rotary_emb


class Embedding:
    def __init__(self):
        super().__init__()
        self.weight = None
        self.device = None

    def load(self, torch_weight: torch.Tensor, device: ttnn.Device):
        self.weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.device = device

    def __call__(self, x: Union[torch.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        if isinstance(x, torch.Tensor):
            x = ttnn.as_tensor(x, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(ttnn.embedding(x, self.weight), layout=ttnn.TILE_LAYOUT)


class Linear:
    def __init__(self):
        super().__init__()
        self.weight = None
        self.device = None

    def load(self, torch_weight: torch.Tensor, device: ttnn.Device):
        self.weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.device = device

    def __call__(self, x: Union[torch.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        if isinstance(x, torch.Tensor):
            x = ttnn.as_tensor(x, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(x, self.weight, transpose_b=True, bias=None)


class RMSNorm:
    def __init__(self):
        super().__init__()
        self.weight = None
        self.epsilon = None
        self.device = None

    def load(self, torch_weight: torch.Tensor, device: ttnn.Device):
        self.weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.epsilon = 1e-6
        self.device = device

    def __call__(self, x: Union[torch.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        if isinstance(x, torch.Tensor):
            x = ttnn.as_tensor(x, dtype=ttnn.float32, device=self.device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.rms_norm(x, epsilon=self.epsilon, weight=self.weight)


class Attention:
    def __init__(self, config: Qwen3MoeConfig, device: ttnn.Device):
        super().__init__()
        self.config = config
        self.device = device
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = pow(self.head_dim, -0.5)
        self.attention_dropout = config.attention_dropout

        self.q_proj = Linear()
        self.k_proj = Linear()
        self.v_proj = Linear()
        self.o_proj = Linear()
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self.sliding_window = None

        cache_shape = (config.max_batch_size, config.max_seq_len, self.num_key_value_heads, self.head_dim)
        self.cache_k = torch.zeros(cache_shape, dtype=torch.float16, device=torch.device("cpu"), requires_grad=False)
        self.cache_v = torch.zeros(cache_shape, dtype=torch.float16, device=torch.device("cpu"), requires_grad=False)
        #self.cache_k = ttnn.zeros(cache_shape, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        #self.cache_v = ttnn.zeros(cache_shape, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None
        assert not config.attention_bias


    def forward(
        self,
        hidden_states: Union[torch.Tensor, ttnn.Tensor],
        start_pos: int,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.to_torch(hidden_states, dtype=torch.float16)

        batch_size, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim) # [batch_size, seq_len, -1, head_dim]

        query_states = self.q_norm(ttnn.to_torch(self.q_proj(hidden_states), dtype=torch.float16).view(hidden_shape))
        key_states = self.k_norm(ttnn.to_torch(self.k_proj(hidden_states), dtype=torch.float16).view(hidden_shape))
        value_states = ttnn.to_torch(self.v_proj(hidden_states), dtype=torch.float16).view(hidden_shape)

        query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)

        #query_states = ttnn.as_tensor(query_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        #key_states = ttnn.as_tensor(key_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = key_states
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = value_states

        key_states = self.cache_k[:batch_size, :start_pos + seq_len]
        value_states = self.cache_v[:batch_size, :start_pos + seq_len]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = ttnn.as_tensor(query_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT)
        key_states = ttnn.as_tensor(key_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT)
        value_states = ttnn.as_tensor(value_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT)

        attn_output = sdpa_attention_forward(query_states, key_states, value_states, attention_mask)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3MoeMLP:
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = Linear()
        self.up_proj = Linear()
        self.down_proj = Linear()
        self.act_fn = ttnn.silu
        assert config.hidden_act == "silu"

    def forward(self, x: Union[torch.Tensor, ttnn.Tensor]) -> ttnn.Tensor:
        return self.down_proj(ttnn.multiply(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]

        self.layer_idx = layer_idx

    def forward(self, hidden_states: Union[torch.Tensor, ttnn.Tensor]) -> torch.Tensor:
        if isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.to_torch(hidden_states, dtype=torch.float16)

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = torch.mul(ttnn.to_torch(expert_layer.forward(current_state), dtype=torch.float16), routing_weights[top_x, idx, None])

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, device: ttnn.Device):
        super().__init__()
        self.device = device
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config, device)

        assert (config.mlp_only_layers is None) or (layer_idx not in config.mlp_only_layers)
        assert config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx)

        self.input_layernorm = RMSNorm()
        self.post_attention_layernorm = RMSNorm()

    def forward(
        self,
        hidden_states: Union[torch.Tensor, ttnn.Tensor],
        start_pos: int,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.to_torch(hidden_states, dtype=torch.float16)
        hidden_states = hidden_states + ttnn.to_torch(self.self_attn.forward(
            hidden_states=self.input_layernorm(hidden_states),
            start_pos=start_pos,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        ), dtype=torch.float16)

        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, devices: Dict[int, ttnn.MeshDevice]):
        super().__init__()
        self.config = config
        self.devices = devices

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding()
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, layer_idx, devices[layer_idx // 6]) for layer_idx in range(config.num_hidden_layers)])  # temp
        self.norm = RMSNorm()
        self.lm_head = Linear()

        position_embeddings = precompute_freqs_cis(config)
        self.register_buffer('position_embeddings', position_embeddings, persistent=False)

        assert config.sliding_window is None

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, input_ids: torch.LongTensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        position_embeddings = self.position_embeddings[start_pos: start_pos + seq_len]

        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            device = decoder_layer.device

            shape = [batch_size, 1, seq_len, start_pos + seq_len]
            padded_shape = [batch_size, 1, ((seq_len + 31) // 32) * 32, ((start_pos + seq_len + 31) // 32) * 32]
            attention_mask = ttnn.full(shape=shape, fill_value=float("-inf"), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attention_mask = ttnn.tilize_with_val_padding(attention_mask, padded_shape, float("-inf"), dtype=ttnn.bfloat16)
            attention_mask = ttnn.triu(attention_mask, diagonal=start_pos + 1)
            attention_mask = ttnn.pad(attention_mask, [(0, 0), (0, 0), (0, padded_shape[2] - shape[2]), (0, padded_shape[3] - shape[3])], float("-inf"))
            attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                start_pos=start_pos,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if isinstance(logits, ttnn.Tensor):
            logits = ttnn.to_torch(logits, dtype=torch.float16)
        return logits

__all__ = ["Qwen3MoeModel"]
