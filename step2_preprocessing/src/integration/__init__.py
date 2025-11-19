"""
Integration module for multimodal dataset creation
"""
from .multimodal_dataset import MultimodalMIMICDataset, multimodal_collate_fn

__all__ = ['MultimodalMIMICDataset', 'multimodal_collate_fn']
