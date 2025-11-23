"""
Integration module for multimodal dataset creation
"""
try:
    from .multimodal_dataset import MultimodalMIMICDataset, multimodal_collate_fn
except ImportError:
    MultimodalMIMICDataset = None
    multimodal_collate_fn = None

try:
    from .precompiled_dataset import PrecompiledMultimodalDataset, precompiled_collate_fn
except ImportError:
    PrecompiledMultimodalDataset = None
    precompiled_collate_fn = None

__all__ = [
    'MultimodalMIMICDataset',
    'multimodal_collate_fn',
    'PrecompiledMultimodalDataset',
    'precompiled_collate_fn'
]
