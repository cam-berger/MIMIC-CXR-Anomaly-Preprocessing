"""
Training-loop helpers shared by train_mae.py and train_classifier.py.

- set_seed / is_cuda_device: reproducibility and device checks
- build_warmup_cosine_scheduler: warmup + cosine LR with optional per-group delays
- backward_and_step: backward, clip over the optimizer's params, skip non-finite steps
- check_trainable_params_in_optimizer / assert_params_finite: fail-fast invariants
- save_checkpoint_files / load_checkpoint_states: checkpoint I/O
"""

from .loop import (
    set_seed,
    is_cuda_device,
    build_warmup_cosine_scheduler,
    warmup_cosine_factor,
    backward_and_step,
    optimizer_params,
    check_trainable_params_in_optimizer,
    assert_params_finite,
    ConsecutiveSkipGuard,
    save_checkpoint_files,
    load_checkpoint_states,
)

__all__ = [
    "set_seed",
    "is_cuda_device",
    "build_warmup_cosine_scheduler",
    "warmup_cosine_factor",
    "backward_and_step",
    "optimizer_params",
    "check_trainable_params_in_optimizer",
    "assert_params_finite",
    "ConsecutiveSkipGuard",
    "save_checkpoint_files",
    "load_checkpoint_states",
]
