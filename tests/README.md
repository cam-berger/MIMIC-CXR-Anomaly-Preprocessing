# NaN/Inf Stability Test Suite

## Overview

This test suite provides comprehensive unit and integration tests for diagnosing and validating fixes for NaN/Inf gradient issues and weight corruption during multimodal classifier training.

## Test Structure

```
tests/
├── conftest.py                    # pytest fixtures and utilities
├── test_nan_handling.py           # Unit tests for all 4 fixes
├── test_multimodal_stability.py   # Integration and stability tests
├── utils/                          # Test utilities
│   └── __init__.py
└── README.md                       # This file
```

## Test Coverage

### Unit Tests (test_nan_handling.py) - 22 tests

#### Fix #1: GradScaler Reset (3 tests)
- `test_gradscaler_new_instantiation_loses_state` - Demonstrates current bug
- `test_gradscaler_manual_reset_preserves_object` - Validates fix
- `test_gradscaler_recovery_after_corruption` - End-to-end recovery scenario

#### Fix #2: CrossAttentionFusion NaN Guards (5 tests)
- `test_zero_vector_inputs` - Zero embedding handling
- `test_large_magnitude_inputs` - Large value stability
- `test_nan_inputs` - NaN sanitization
- `test_inf_inputs` - Inf sanitization
- `test_attention_weights_no_overflow` - Softmax overflow prevention

#### Fix #3: Safe Normalization (5 tests)
- `test_normalize_zero_vector` - Zero-norm vector handling
- `test_normalize_tiny_norm_vector` - Small-norm stability
- `test_normalize_large_norm_vector` - Large-norm handling
- `test_normalize_preserves_direction` - Correctness validation
- `test_clip_loss_with_zero_embeddings` - CLIP loss integration

#### Fix #4: MAE Epsilon (4 tests)
- `test_mae_constant_image` - Zero-variance patch handling
- `test_mae_low_variance_image` - Low-variance stability
- `test_mae_blank_image` - Extreme case (all zeros)
- `test_mae_epsilon_prevents_division_by_tiny_variance` - Epsilon comparison

#### Integration Tests (2 tests)
- `test_multimodal_classifier_edge_cases` - Full forward pass with edge cases
- `test_loss_computation_with_edge_cases` - All loss functions with edge cases

#### Regression Tests (3 tests)
- `test_mae_reconstruction_quality` - Ensure epsilon doesn't degrade MAE
- `test_cross_attention_capacity` - Ensure guards don't reduce capacity
- `test_normalization_unit_norm` - Ensure safe_normalize correctness

### Integration Tests (test_multimodal_stability.py)

#### Training Loop Stability
- `test_training_step_with_nan_batch` - NaN batch skipping
- `test_weight_corruption_detection_and_recovery` - Weight integrity fuse
- `test_gradscaler_cascade_failure_prevention` - Scaler reset mechanism
- `test_100_batches_with_forced_corruption` - Full protection stack (CRITICAL TEST)

#### Data Quality
- `test_blacklisted_study_filtering` - Blacklist validation
- `test_image_sanitization` - Image NaN/Inf handling
- `test_structured_features_sanitization` - Structured data handling

#### Performance Regression
- `test_forward_pass_speed` - Performance benchmark
- `test_model_capacity_not_reduced` - Capacity validation

## Running Tests

### Run all tests
```bash
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_nan_handling.py -v
```

### Run specific test class
```bash
python -m pytest tests/test_nan_handling.py::TestGradScalerReset -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=src/models --cov-report=html
```

### Run only tests that don't require CUDA
```bash
python -m pytest tests/ -v -m "not cuda"
```

## Test Fixtures

### Model Fixtures
- `mae_model` - MaskedAutoencoder for testing
- `multimodal_classifier` - MultimodalClassifier for testing
- `cross_attention_fusion` - Standalone CrossAttentionFusion
- `loss_functions` - All loss functions (CLIP, SupCon, Focal, MultiTask)

### Data Fixtures
- `normal_image` - Standard test image
- `edge_case_images` - Zero, constant, low-variance, high-value images
- `normal_embeddings` - Standard test embeddings
- `edge_case_embeddings` - Zero, tiny, large, FP16-max, NaN, Inf embeddings
- `chexpert_labels` - Test labels with uncertain (-1) and missing (NaN) values

### Utility Fixtures
- `assert_finite` - Assert tensor contains no NaN/Inf
- `assert_no_nan` - Assert tensor contains no NaN
- `assert_no_inf` - Assert tensor contains no Inf
- `inject_nan` - Inject NaN into tensor
- `inject_inf` - Inject Inf into tensor

## Expected Test Behavior (Before Fixes)

### Currently Passing
- GradScaler tests (demonstrates bug but doesn't fail)
- Edge-case data fixtures
- Utility functions

### Currently Failing / Skipped
- `test_normalize_zero_vector` - Skipped (safe_normalize not yet implemented)
- `test_nan_inputs` - May fail (CrossAttention doesn't sanitize yet)
- `test_clip_loss_with_zero_embeddings` - May fail (F.normalize returns NaN)

### Will Pass After Fixes
All tests should pass after implementing the 4 fixes in the plan.

## Integration with CI/CD

### Recommended GitHub Actions Workflow
```yaml
name: NaN Stability Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --cov=src/models
```

## Debugging Failed Tests

### If tests fail unexpectedly:

1. **Check device** - Some tests require CUDA
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

2. **Check model imports** - Ensure src.models is importable
   ```bash
   python -c "from src.models.multimodal import MultimodalClassifier"
   ```

3. **Run with verbose output**
   ```bash
   python -m pytest tests/test_nan_handling.py::TestGradScalerReset::test_gradscaler_manual_reset_preserves_object -vv -s
   ```

4. **Use pytest debugger**
   ```bash
   python -m pytest tests/ --pdb  # Drop into debugger on failure
   ```

## Test Metrics and Success Criteria

### Round 1: Test Infrastructure (COMPLETED)
- ✅ 22 unit tests created
- ✅ 8 integration tests created
- ✅ All tests collected successfully
- ✅ Fixtures comprehensive

### Round 2-5: Fix Implementation
After each fix:
- Unit tests for that fix should pass
- No regression in other tests
- 5-epoch training test completes

### Round 6: Full Validation
- Zero test failures
- 100-batch stability test passes
- 50-epoch training completes with:
  - NaN rate < 0.5%
  - Zero weight corruptions
  - No circuit breaker triggers

## Contributing

When adding new tests:
1. Follow existing test structure (Arrange-Act-Assert)
2. Use descriptive test names (`test_<what>_<scenario>_<expected>`)
3. Add fixtures to conftest.py if reusable
4. Document edge cases in docstrings
5. Use parametrize for multiple similar test cases
