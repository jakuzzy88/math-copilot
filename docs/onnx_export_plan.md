# ONNX Export Plan — CNN-CTC Model

## Sprint C: Export Readiness

This document describes the ONNX export strategy for deploying the CNN-CTC
handwriting recognition model to mobile (React Native via ONNX Runtime Mobile).

---

## 1. Model Architecture Summary

| Property          | Value              |
|-------------------|--------------------|
| Input shape       | `(B, 1, 128, 512)` |
| Output shape      | `(T, B, 19)`       |
| Time steps (T)    | `W // 16 = 32`     |
| Vocabulary size   | 19 (blank + 18 chars) |
| Parameters        | ~300K              |
| Estimated size    | ~1.2 MB (float32)  |
| BiLSTM            | Yes (128 hidden)   |

## 2. Export Configuration

```python
torch.onnx.export(
    model,
    dummy_input,           # (1, 1, 128, 512)
    "model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["log_probs"],
    dynamic_axes={
        "image":     {0: "batch_size", 3: "width"},
        "log_probs": {0: "time_steps", 1: "batch_size"},
    },
)
```

### Dynamic Axes

- **Batch size** (`dim 0` of input, `dim 1` of output): Allows batched inference.
- **Width** (`dim 3` of input): Allows variable-width equation images.
  - The width must be a multiple of 16 (due to 4 pooling layers).
  - The time steps T in the output scale proportionally: `T = W // 16`.

### Opset Version

Opset **17** is chosen for broad compatibility:
- Supported by ONNX Runtime Mobile 1.16+
- Covers all operators used (Conv, BatchNorm, LSTM, Linear, LogSoftmax)

## 3. Post-Export Validation

```bash
# Validate model structure
python -c "import onnx; m = onnx.load('model.onnx'); onnx.checker.check_model(m); print('OK')"

# Run inference with onnxruntime
python -c "
import numpy as np
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx')
out = sess.run(None, {'image': np.random.randn(1, 1, 128, 512).astype(np.float32)})
print('Output shape:', out[0].shape)  # Expected: (32, 1, 19)
"
```

## 4. Mobile Deployment Path

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ PyTorch      │ ──> │ ONNX         │ ──> │ ONNX Runtime   │
│ CNN-CTC      │     │ model.onnx   │     │ Mobile (React  │
│ (training)   │     │ (~1.2 MB)    │     │  Native)       │
└─────────────┘     └──────────────┘     └────────────────┘
```

### Mobile Runtime Options

| Runtime           | Size Impact | Platform      | Notes                        |
|-------------------|-------------|---------------|------------------------------|
| ONNX Runtime Mobile | ~5 MB     | iOS + Android | Best compatibility           |
| TFLite (via ONNX→TF) | ~3 MB   | iOS + Android | Requires extra conversion    |
| CoreML (iOS only) | ~2 MB       | iOS           | Best iOS performance         |

**Recommended**: ONNX Runtime Mobile for cross-platform support.

## 5. Quantisation (Future)

For production mobile deployment, consider quantisation:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8,
)
# Expected: ~300 KB (4× reduction)
```

## 6. Pre-Export Checklist

- [x] Model shape inspection (`export_onnx.py --inspect`)
- [x] Smoke test with dummy tensors (`export_onnx.py --smoke-test`)
- [x] Export smoke tests pass (`pytest training/tests/test_export_onnx.py`)
- [ ] Training reaches ≥70% test accuracy (Decision Gate)
- [ ] Export from trained checkpoint (`export_onnx.py --checkpoint ...`)
- [ ] Validate with onnxruntime
- [ ] Test on mobile device

## 7. Commands Reference

```bash
# Inspect model shapes (no checkpoint needed)
python training/export_onnx.py --inspect

# Run smoke test with dummy tensors
python training/export_onnx.py --smoke-test

# Run export tests
python -m pytest training/tests/test_export_onnx.py -v

# Export from checkpoint (after training)
python training/export_onnx.py \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --out training/runs/synthetic_v2_full/model.onnx
```
