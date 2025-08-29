import torch
import torch.nn.functional as F
from torch import nn

from typing import Union, Dict, Tuple

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


    def __call__(
        self,
        hidden_states: Union[torch.Tensor, ttnn.Tensor],
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
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

        #print(query_states.shape)
        #print(key_states.shape)
        #print(position_embeddings[0].shape)
        #print(position_embeddings[1].shape)

        query_states = ttnn.to_layout(query_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        key_states = ttnn.to_layout(key_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        
        query_states = ttnn.to_layout(query_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        key_states = ttnn.to_layout(key_states, layout=ttnn.ROW_MAJOR_LAYOUT)

        query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)

        query_states = ttnn.to_torch(query_states, dtype=torch.float16)
        key_states = ttnn.to_torch(key_states, dtype=torch.float16)

        #query_states = ttnn.as_tensor(query_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        #key_states = ttnn.as_tensor(key_states, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = key_states
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = value_states

        key_states = self.cache_k[:batch_size, :start_pos + seq_len]
        value_states = self.cache_v[:batch_size, :start_pos + seq_len]

        #key_states = repeat_kv(key_states, self.num_key_value_groups)
        #value_states = repeat_kv(value_states, self.num_key_value_groups)

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

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.down_proj(ttnn.multiply(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class Qwen3MoeSparseMoeBlock:
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, device: ttnn.Device):
        super().__init__()
        self.device = device

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = Linear()
        self.experts = [Qwen3MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]

        self.layer_idx = layer_idx

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        N = batch_size * sequence_length
        E = self.num_experts
        K = self.top_k


        hidden_states = ttnn.reshape(hidden_states, (N, hidden_dim))
        router_logits = self.gate(hidden_states)
        router_logits = ttnn.to_layout(router_logits, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
        routing_weights = ttnn.softmax(router_logits, dim=-1)
        routing_weights = ttnn.reshape(routing_weights, (1, 1, N, E))
        routing_weights, selected_experts = ttnn.topk(routing_weights, k=K, dim=-1, largest=True, sorted=True)

        if self.norm_topk_prob:
            denom = ttnn.sum(routing_weights, dim=-1, keepdim=True)
            routing_weights = ttnn.div(routing_weights, denom)

        routing_weights = ttnn.to_layout(routing_weights, layout=ttnn.ROW_MAJOR_LAYOUT)
        selected_experts = ttnn.to_layout(selected_experts, layout=ttnn.ROW_MAJOR_LAYOUT)
        routing_weights = ttnn.to_layout(routing_weights, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
        selected_experts = ttnn.to_layout(selected_experts, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)


        weights_full = ttnn.zeros(
            (1, 1, N, E),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        weights_4d = ttnn.scatter(weights_full, dim=-1, index=selected_experts, src=routing_weights)

        final_hidden_states = ttnn.zeros(
            (1, 1, N, hidden_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        expert_hitted = range(self.num_experts)
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]

            current_hidden_states = expert_layer(hidden_states)
            current_hidden_states = ttnn.reshape(current_hidden_states, (1, 1, N, hidden_dim))
            current_hidden_states = ttnn.to_layout(current_hidden_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

            mask_e = ttnn.slice(weights_4d, [0, 0, 0, expert_idx], [1, 1, N, expert_idx + 1])
            mask_e = ttnn.repeat(mask_e, [1, 1, 1, hidden_dim])

            final_hidden_states = ttnn.add(final_hidden_states, ttnn.multiply(current_hidden_states, mask_e))

        final_hidden_states = ttnn.reshape(final_hidden_states, (N, hidden_dim))
        final_hidden_states = ttnn.reshape(final_hidden_states, (batch_size, sequence_length, hidden_dim))

        return ttnn.to_torch(final_hidden_states, dtype=torch.float16)


class Qwen3MoeDecoderLayer:
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, device: ttnn.Device):
        super().__init__()
        self.device = device
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config, device)

        assert (config.mlp_only_layers is None) or (layer_idx not in config.mlp_only_layers)
        assert config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx, device)

        self.input_layernorm = RMSNorm()
        self.post_attention_layernorm = RMSNorm()

    def __call__(
        self,
        hidden_states: Union[torch.Tensor, ttnn.Tensor],
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.to_torch(hidden_states, dtype=torch.float16)
        hidden_states = hidden_states + ttnn.to_torch(self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            start_pos=start_pos,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        ), dtype=torch.float16)

        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3MoeModel:
    def __init__(self, config: Qwen3MoeConfig, devices: Dict[int, ttnn.MeshDevice]):
        super().__init__()
        self.config = config
        self.devices = devices

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding()
        self.layers = [Qwen3MoeDecoderLayer(config, layer_idx, devices[layer_idx // 6]) for layer_idx in range(config.num_hidden_layers)]
        self.norm = RMSNorm()
        self.lm_head = Linear()

        pos_embs_cos, pos_embs_sin = precompute_freqs_cis(config)
        self.pos_embs_cos = pos_embs_cos
        self.pos_embs_sin = pos_embs_sin

        assert config.sliding_window is None

        # Initialize weights and apply final processing
        # self.post_init()

    def __call__(self, input_ids: torch.LongTensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        pos_embs_cos = self.pos_embs_cos[start_pos: start_pos + seq_len]
        pos_embs_sin = self.pos_embs_sin[start_pos: start_pos + seq_len]

        hidden_states = self.embed_tokens(input_ids)

        for decoder_layer in self.layers:
            device = decoder_layer.device

            cos = ttnn.from_torch(pos_embs_cos, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sin = ttnn.from_torch(pos_embs_sin, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            position_embeddings = cos, sin

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
