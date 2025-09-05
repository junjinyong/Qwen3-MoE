import torch
import ttnn

from typing import Union, Optional


__all__ = ["_patch"]


def to_(self: Union[torch.Tensor, ttnn.Tensor], layout: Optional[ttnn.Layout] = None, dtype: Optional[ttnn.DataType] = None, device: Optional[ttnn.MeshDevice] = None, memory_config: Optional[ttnn.MemoryConfig] = None):
    if isinstance(self, torch.Tensor):
        return ttnn.as_tensor(self, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    elif isinstance(self, ttnn.Tensor):
        if device is not None:
            self = ttnn.to_device(self, device=device, memory_config=memory_config)
        if layout is not None:
            self = ttnn.to_layout(self, layout=layout, dtype=dtype, memory_config=memory_config)
        return self
    else:
        raise Exception(f"Unsupported type {type(self)}")


def _patch():
    assert not hasattr(torch.Tensor, "to_")
    assert not hasattr(ttnn.Tensor, "to_")
    ttnn.Tensor.to_ = to_
    torch.Tensor.to_ = to_
