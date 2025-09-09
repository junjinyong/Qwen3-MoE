import torch
import ttnn

from typing import Union, Dict, Tuple

from npu.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from npu.transformers.sdpa_attention import sdpa_attention_forward
from npu.qwen3_moe.rope_helpers import precompute_freqs_cis, apply_rotary_emb
from npu.utils.loader import owner_of


class Embedding:
    def __init__(self, device: ttnn.MeshDevice):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.weight = None
        self.device = device

    def load(self, torch_weight: torch.Tensor):
        assert isinstance(torch_weight, torch.Tensor)
        self.weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, x: torch.LongTensor) -> ttnn.Tensor:
        x = ttnn.as_tensor(x, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(ttnn.embedding(x, self.weight), layout=ttnn.TILE_LAYOUT)


class Linear:
    def __init__(self, device: ttnn.MeshDevice):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.weight = None
        self.device = device

    def load(self, torch_weight: torch.Tensor):
        assert isinstance(torch_weight, torch.Tensor)
        self.weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.clone(ttnn.linear(x, self.weight, transpose_b=True, bias=None), dtype=ttnn.bfloat16)


class RMSNorm:
    def __init__(self, device: ttnn.MeshDevice, flag: bool = False):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.weight = None
        self.epsilon = None
        self.device = device
        self.flag = flag

    def load(self, torch_weight: torch.Tensor):
        assert isinstance(torch_weight, torch.Tensor)
        self.weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        self.epsilon = 1e-6

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.rms_norm(x, epsilon=self.epsilon, weight=self.weight)
        if self.flag:
            return ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            return ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class Attention:
    def __init__(self, config: Qwen3MoeConfig, device: ttnn.MeshDevice):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.config = config
        self.device = device

        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len

        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = pow(self.head_dim, -0.5)
        self.attention_dropout = config.attention_dropout

        self.q_proj = Linear(device=self.device)
        self.k_proj = Linear(device=self.device)
        self.v_proj = Linear(device=self.device)
        self.o_proj = Linear(device=self.device)
        self.q_norm = RMSNorm(device=self.device, flag=True)
        self.k_norm = RMSNorm(device=self.device, flag=True)
        self.sliding_window = None

        cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
        self.cache_k = ttnn.zeros(cache_shape, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)
        self.cache_v = ttnn.zeros(cache_shape, dtype=ttnn.bfloat16, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT)

        assert config._attn_implementation == "sdpa"
        assert config.num_attention_heads % config.num_key_value_heads == 0
        assert config.sliding_window is None
        assert not config.attention_bias

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: ttnn.Tensor
    ) -> ttnn.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_norm(ttnn.reshape(self.q_proj(hidden_states), hidden_shape))
        key_states = self.k_norm(ttnn.reshape(self.k_proj(hidden_states), hidden_shape))
        value_states = ttnn.reshape(self.v_proj(hidden_states), hidden_shape)

        query_states, key_states = apply_rotary_emb(query_states, key_states, position_embeddings)

        if start_pos == 0:
            is_decode_mode = False
        elif seq_len == 1:
            is_decode_mode = True
        else:
            raise Exception("Neither Decode mode nor Prefill mode")

        key_states = ttnn.permute(key_states, (0, 2, 1, 3))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        for batch_index in range(batch_size):
            key = ttnn.slice(key_states, [batch_index, 0, 0, 0], [batch_index + 1, self.num_key_value_heads, seq_len, self.head_dim])
            value = ttnn.slice(value_states, [batch_index, 0, 0, 0], [batch_index + 1, self.num_key_value_heads, seq_len, self.head_dim])
            key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

            if is_decode_mode:
                ttnn.kv_cache.update_cache_for_token_(self.cache_k, key, update_index=start_pos + self.max_seq_len * self.num_key_value_heads * batch_index)
                ttnn.kv_cache.update_cache_for_token_(self.cache_v, value, update_index=start_pos + self.max_seq_len * self.num_key_value_heads * batch_index)
            else:
                ttnn.kv_cache.fill_cache_for_user_(self.cache_k, key, batch_index=batch_index)
                ttnn.kv_cache.fill_cache_for_user_(self.cache_v, value, batch_index=batch_index)


        key_states = ttnn.slice(self.cache_k, [0, 0, 0, 0], [batch_size, self.num_key_value_heads, start_pos + seq_len, self.head_dim])
        value_states = ttnn.slice(self.cache_v, [0, 0, 0, 0], [batch_size, self.num_key_value_heads, start_pos + seq_len, self.head_dim])

        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        attn_output = sdpa_attention_forward(query_states, key_states, value_states, attention_mask)

        attn_output = ttnn.reshape(attn_output, (batch_size, seq_len, -1))
        attn_output = self.o_proj(attn_output)
        return attn_output


class MoeMLP:
    def __init__(self, config: Qwen3MoeConfig, device: ttnn.MeshDevice):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.config = config
        self.device = device

        self.hidden_size = config.hidden_size
        self.gate_proj = Linear(device=self.device)
        self.up_proj = Linear(device=self.device)
        self.down_proj = Linear(device=self.device)
        self.act_fn = ttnn.silu
        assert config.hidden_act == "silu"

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.down_proj(ttnn.multiply(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


class SparseMoeBlock:
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, device: ttnn.MeshDevice):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.device = device

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = Linear(device=self.device)
        self.experts = [MoeMLP(config, device=self.device) for _ in range(self.num_experts)]

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

        routing_weights = ttnn.clone(routing_weights, dtype=ttnn.bfloat16)

        weights_full = ttnn.zeros(
            (1, 1, N, E),
            dtype=ttnn.bfloat16,
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
            final_hidden_states = ttnn.add(final_hidden_states, ttnn.multiply(current_hidden_states, mask_e))

        final_hidden_states = ttnn.reshape(final_hidden_states, (N, hidden_dim))
        final_hidden_states = ttnn.reshape(final_hidden_states, (batch_size, sequence_length, hidden_dim))
        return final_hidden_states


class DecoderLayer:
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, device: ttnn.MeshDevice):
        super().__init__()
        assert isinstance(device, ttnn.MeshDevice)
        self.device = device
        self.hidden_size = config.hidden_size

        self.self_attn = Attention(config, device=self.device)

        assert (config.mlp_only_layers is None) or (layer_idx not in config.mlp_only_layers)
        assert config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        self.mlp = SparseMoeBlock(config, layer_idx, device=self.device)

        self.input_layernorm = RMSNorm(device=self.device)
        self.post_attention_layernorm = RMSNorm(device=self.device)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        start_pos: int,
        position_embeddings: Tuple[ttnn.Tensor, ttnn.Tensor],
        attention_mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if hidden_states.device() != self.device:
            hidden_states = ttnn.to_device(ttnn.from_device(hidden_states), device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = hidden_states + self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            start_pos=start_pos,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask
        )
        hidden_states = ttnn.add(hidden_states, self.mlp(self.post_attention_layernorm(hidden_states)))
        return hidden_states


class Model:
    def __init__(self, config: Qwen3MoeConfig, devices: Dict[int, ttnn.MeshDevice]):
        super().__init__()
        assert list(devices.keys()) == list(range(8))
        assert all(isinstance(device, ttnn.MeshDevice) for device in devices.values())
        self.config = config
        self.devices = devices
        self.num_devices = len(devices)


        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(devices[0])
        get_device = lambda layer_idx: self.devices[owner_of(layer_idx, config.num_hidden_layers, self.num_devices)]
        self.layers = [DecoderLayer(config, layer_idx, device=get_device(layer_idx)) for layer_idx in range(config.num_hidden_layers)]
        self.norm = RMSNorm(devices[7])
        self.lm_head = Linear(devices[7])

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

        for i, decoder_layer in enumerate(self.layers):
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
        return ttnn.to_torch(logits, dtype=torch.float16)

__all__ = ["Model"]
