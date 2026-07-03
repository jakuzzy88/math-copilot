# Mobile Inference Plan

Sprint 5D: ONNX Runtime Mobile Static Inference.

## Overview

This document describes the integration path for running the trained CNN-CTC
ONNX model inside a React Native app via ONNX Runtime Mobile. Sprint 5D
implements **real static ONNX inference** using `onnxruntime-react-native` with
mocked runtime for Jest and a full pipeline test pathway.

---

## 1. Runtime Package

**Chosen package:** `onnxruntime-react-native`

| Package | Status | Notes |
|---------|--------|-------|
| `onnxruntime-react-native` | ✅ Installed | Official ONNX Runtime binding for React Native. Supports iOS & Android. Uses ONNX Runtime Mobile under the hood. |
| `onnxruntime-web` | ❌ Not used | WebAssembly based; not ideal for native mobile performance. |
| `onnxruntime-node` | ❌ Not used | Node.js only; useful for tests but not mobile deployment. |

**Installation:**
```bash
cd app
npm install onnxruntime-react-native
```

> **Status (Sprint 5D):** Installed as a production dependency.
> Jest tests use a **mock ONNX Runtime** injected via `OnnxEquationRecognizer.create(modelPath, mockOrt)`.

---

## 2. Safe Runtime Abstraction

**Module:** `src/inference/onnxRuntimeProvider.ts`

The ONNX Runtime native module is not available in Node.js (Jest).  The
provider module uses a dynamic `require()` wrapped in a try-catch:

```typescript
import { getOnnxRuntime, isOnnxAvailable } from './onnxRuntimeProvider';

const ort = getOnnxRuntime();  // null in Node/Jest
if (ort === null) {
  // Handle unavailable runtime
}
```

**Test utilities:**
- `_setOnnxRuntime(mock)` — inject a mock for testing.
- `_resetOnnxRuntimeCache()` — clear cached runtime between tests.

---

## 3. Model Asset Location

### Where `model.onnx` lives in the app

```
app/
  assets/
    models/
      equation_recognizer_v1.onnx    ← production model (5.5 MB)
      model_manifest.json            ← version metadata
```

### Model versioning

```json
{
  "modelId": "equation_recognizer",
  "version": "1.0.0",
  "exportDate": "2026-07-03",
  "trainingRun": "synthetic_v2_full_50ep",
  "exactAccuracy": 0.914,
  "charAccuracy": 0.9796,
  "inputShape": [1, 1, 128, 512],
  "outputShape": [32, 1, 19],
  "opsetVersion": 17,
  "vocabSize": 19,
  "vocabulary": ["<blank>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "x", "+", "-", "=", "/", "(", ")", " "]
}
```

- Models are versioned by semantic version in the manifest.
- The manifest allows the app to validate model compatibility at load time.
- Model files are **git-ignored** (`*.onnx` in `.gitignore`). Distribution
  via Git LFS or release assets is a future decision.

### Copying model from training

```bash
cp training/runs/synthetic_v2_full_50ep/model.onnx \
   app/assets/models/equation_recognizer_v1.onnx
```

---

## 4. Model I/O Specification

### Input

| Property | Value |
|----------|-------|
| Tensor name | `image` |
| Shape | `[batch_size, 1, 128, 512]` (NCHW) |
| Dtype | `float32` |
| Value range | `[0.0, 1.0]` |
| Channel | Grayscale (1 channel) |

### Output

| Property | Value |
|----------|-------|
| Tensor name | `log_probs` |
| Shape | `[32, batch_size, 19]` (T, N, C) |
| Dtype | `float32` |
| Content | Log-probabilities per time step per class |

### Constants

```
MODEL_INPUT_HEIGHT  = 128
MODEL_INPUT_WIDTH   = 512
MODEL_CHANNELS      = 1
MODEL_TIME_STEPS    = 32   (= 512 / 16, from CNN stride)
MODEL_NUM_CLASSES   = 19   (blank + 18 printable chars)
```

---

## 5. Grayscale Preprocessing Steps

Given a cropped equation image (already isolated from the camera frame):

1. **Convert to grayscale** (single channel).
2. **Resize** to 128 (height) × 512 (width) using bilinear interpolation.
3. **Normalise to [0, 1]**: divide each pixel by 255.0.
4. **Reshape to Float32Array** of length 128 × 512 = 65,536.
5. **Tensor layout**: NCHW = `[1, 1, 128, 512]`.

> **Note:** The training pipeline uses `PIL.Image.open().convert("L")`
> followed by `.resize((512, 128), Image.BILINEAR)` and divides by 255.0.
> The app-side preprocessing must produce identical normalisation.

### What is NOT done (matches training)

- No mean subtraction or standardisation (no ImageNet-style normalisation).
- No histogram equalisation.
- No inversion — dark text on light background is the expected input.

---

## 6. CTC Greedy Decoding

### Vocabulary (must match `training/models/vocabulary.py` exactly)

```
Index 0:  <blank>   (CTC blank token)
Index 1:  '0'
Index 2:  '1'
Index 3:  '2'
Index 4:  '3'
Index 5:  '4'
Index 6:  '5'
Index 7:  '6'
Index 8:  '7'
Index 9:  '8'
Index 10: '9'
Index 11: 'x'
Index 12: '+'
Index 13: '-'
Index 14: '='
Index 15: '/'
Index 16: '('
Index 17: ')'
Index 18: ' '   (space)
```

### Decoding algorithm

```
function ctcGreedyDecode(logProbs: number[][]): string
  1. For each time step t, take argmax(logProbs[t]) → index
  2. Collapse consecutive repeated indices
  3. Remove all blank tokens (index 0)
  4. Map remaining indices to characters using the vocabulary
  5. Return joined string
```

---

## 7. Decoded String → OcrCandidate[]

After CTC decoding produces a raw string (e.g. `"3x+4=10"`), it is
wrapped into the pipeline's `OcrCandidate` type:

```typescript
const candidates: OcrCandidate[] = [
  {
    text: decodedText,      // e.g. "3x+4=10"
    confidence: avgProb,    // average max-probability across time steps
  }
];
```

For static single-image inference, we produce exactly one candidate.
For future multi-beam decoding, multiple candidates with varying
confidence scores would be generated.

---

## 8. OcrCandidate[] → processCandidates()

The candidate array is fed directly into the existing deterministic
pipeline:

```typescript
import { processCandidates } from '../pipeline/candidatePipeline';

const result: PipelineResult = processCandidates(candidates);
```

`processCandidates()` performs:
1. OCR text normalisation (fix common misreads)
2. Grammar validation
3. AST parsing
4. Linear equation solving
5. Explanation generation
6. Candidate scoring and ranking

Returns either `PipelineAcceptedResult` or `PipelineRejectedResult`.

---

## 9. OnnxEquationRecognizer — Implementation

**Module:** `src/inference/staticImageRecognizer.ts`

### Factory pattern

```typescript
import { OnnxEquationRecognizer } from './staticImageRecognizer';

// On device:
const recognizer = await OnnxEquationRecognizer.create(
  'assets/models/equation_recognizer_v1.onnx'
);

// For tests (with mock):
const recognizer = await OnnxEquationRecognizer.create(
  'model.onnx',
  mockOrtApi
);
```

### Pipeline flow

```
recognize(input)
  ├── prepareGrayscaleInput() → Float32Array [0,1]
  ├── new ort.Tensor('float32', data, [1,1,128,512])
  ├── session.run({ image: tensor })
  ├── reshapeOutputToLogProbs(flat, 32, 19)
  ├── ctcGreedyDecodeFromLogProbs(logProbs) → rawText
  ├── computeConfidence(logProbs) → confidence
  └── return { rawText, candidates: [{ text: rawText, confidence }] }
```

### Lifecycle

```typescript
const recognizer = await OnnxEquationRecognizer.create(modelPath);
try {
  const result = await recognizer.recognize(input);
  // use result
} finally {
  await recognizer.dispose();
}
```

---

## 10. Full recognizeAndSolve() Pipeline

**Module:** `src/inference/recognizeAndSolve.ts`

```typescript
import { recognizeAndSolve } from './recognizeAndSolve';

const result = await recognizeAndSolve(recognizer, {
  grayscalePixels: pixels,
  width: 512,
  height: 128,
});

if (result.pipeline.accepted) {
  console.log(result.pipeline.equation);         // "3x+4=10"
  console.log(result.pipeline.solution);          // "x=2"
  console.log(result.pipeline.explanationSteps);  // step-by-step
}
```

---

## 11. Architecture Diagram

```
┌──────────────┐
│ Cropped Image│  (Uint8Array, 128×512 grayscale)
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ imagePreprocessor.ts │  normalize → Float32Array [0,1]
└──────┬───────────────┘
       │
       ▼
┌───────────────────────────────┐
│ OnnxEquationRecognizer        │  ONNX Runtime inference
│   ort.Tensor [1,1,128,512]    │  → session.run()
│   output: "log_probs" [32,1,19]│
└──────┬────────────────────────┘
       │
       ▼
┌──────────────────┐
│ ctcDecoder.ts    │  CTC greedy decode → raw string
└──────┬───────────┘
       │
       ▼
┌─────────────────────────┐
│ OcrCandidate[]          │  { text, confidence }
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ processCandidates()     │  normalize → validate → parse → solve
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ PipelineResult          │  accepted: equation + solution + steps
│                         │  rejected: reason + errors
└──────┬──────────────────┘
       │  (future: live camera)
       ▼
┌─────────────────────────┐
│ StabilityAggregator     │  sliding window → stable equation
└─────────────────────────┘
```

---

## 12. Device / Manual Testing Instructions

### Prerequisites

1. React Native environment set up (iOS or Android).
2. Model file placed in `app/assets/models/equation_recognizer_v1.onnx`.
3. Metro bundler configured to bundle `.onnx` files (or use a file path resolver).

### Quick device test (React Native)

```typescript
import { OnnxEquationRecognizer } from './inference/staticImageRecognizer';
import { recognizeAndSolve } from './inference/recognizeAndSolve';

async function runDeviceTest() {
  // 1. Load recognizer with the bundled model.
  const modelPath = 'path/to/assets/models/equation_recognizer_v1.onnx';
  const recognizer = await OnnxEquationRecognizer.create(modelPath);

  // 2. Create a test image (all-white, should produce blank/empty output).
  const pixels = new Uint8Array(128 * 512);
  pixels.fill(255);

  // 3. Run inference.
  const result = await recognizeAndSolve(recognizer, {
    grayscalePixels: pixels,
    width: 512,
    height: 128,
  });

  console.log('Raw text:', result.rawText);
  console.log('Accepted:', result.pipeline.accepted);
  if (result.pipeline.accepted) {
    console.log('Equation:', result.pipeline.equation);
    console.log('Solution:', result.pipeline.solution);
  }

  // 4. Cleanup.
  await recognizer.dispose();
}

runDeviceTest().catch(console.error);
```

### Running Jest tests (Node)

```bash
cd app
npm test
```

All 205 tests should pass, including:
- `onnxRuntimeProvider.test.ts` — safe provider abstraction
- `onnxRecognizer.test.ts` — mock ONNX inference pipeline
- `staticInferencePipeline.test.ts` — end-to-end with recognizeAndSolve()

### Running on Android (future, when RN project is scaffolded)

```bash
# 1. Copy model to android assets
cp app/assets/models/equation_recognizer_v1.onnx \
   android/app/src/main/assets/equation_recognizer_v1.onnx

# 2. Run the app
npx react-native run-android

# 3. Check logcat for inference output
adb logcat | grep "Raw text"
```

### Running on iOS (future, when RN project is scaffolded)

```bash
# 1. Add model to Xcode project resources
# Drag equation_recognizer_v1.onnx into Xcode project navigator

# 2. Run the app
npx react-native run-ios

# 3. Check Xcode console for inference output
```

---

## 13. Sprint Boundaries

| Sprint | Scope | Status |
|--------|-------|--------|
| 5C | TypeScript types, decoder, preprocessor, interfaces, fake tests | ✅ Complete |
| 5D | Install `onnxruntime-react-native`, real inference implementation, mock tests | ✅ Complete |
| 5E | Camera frame capture + crop + resize pipeline | ✅ Complete |
| 5F | Live OCR with StabilityAggregator integration | 🔜 Next |
| 6A | TFLite conversion evaluation | 🔜 Planned |

---

## 14. Files Added/Modified in Sprint 5D

| File | Action | Description |
|------|--------|-------------|
| `src/inference/onnxRuntimeProvider.ts` | **New** | Safe ONNX Runtime abstraction (null in Node) |
| `src/inference/staticImageRecognizer.ts` | **Rewritten** | Full ONNX inference pipeline (was placeholder) |
| `src/inference/syntheticTestImages.ts` | **New** | Deterministic test image generators |
| `src/__tests__/onnxRuntimeProvider.test.ts` | **New** | Provider abstraction tests |
| `src/__tests__/onnxRecognizer.test.ts` | **New** | Mock ONNX inference pipeline tests |
| `src/__tests__/staticInferencePipeline.test.ts` | **New** | End-to-end pipeline integration tests |
| `src/__tests__/recognizeAndSolve.test.ts` | **Modified** | Updated dispose() signature |
| `app/assets/models/model_manifest.json` | **New** | Model version metadata |
| `app/assets/models/equation_recognizer_v1.onnx` | **Copied** | Production model (5.5 MB, git-ignored) |
| `tsconfig.test.json` | **Modified** | Added `"node"` to types for require() |
| `package.json` | **Modified** | Added `onnxruntime-react-native` dependency |
| `docs/mobile_inference_plan.md` | **Updated** | This document |

---

## 15. Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline

### Overview

Sprint 5E implements the complete image preprocessing chain that converts
a raw camera frame (RGBA, any resolution) into a model-ready
`StaticRecognizerInput` (128×512, Float32Array, normalised [0,1]).

### Architecture

```
┌──────────────────────┐
│ CameraFrame          │  RGBA 640×480 (or any size)
│ (cameraFrameProvider)│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ colorConversion.ts   │  rgbaToGrayscale() — BT.601 luminance
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ cropRegion.ts        │  cropGrayscale() — explicit or auto 4:1 crop
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ imageResizer.ts      │  resizeBilinear() → 128×512
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ imagePreprocessor.ts │  prepareGrayscaleInput() → Float32Array [0,1]
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ StaticRecognizerInput│  Ready for OnnxEquationRecognizer
└──────────────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| `colorConversion.ts` | RGBA/RGB → grayscale (BT.601 luminance, matches PIL) |
| `cropRegion.ts` | ROI cropping + centered aspect-ratio crop computation |
| `imageResizer.ts` | Bilinear interpolation resize (pure TypeScript) |
| `cameraFrameProvider.ts` | Frame provider interface + StaticFrameProvider + synthetic frame factories |
| `framePipeline.ts` | Complete pipeline: `processFrame()` chains all steps |

### Usage

```typescript
import { processFrame } from './inference/framePipeline';
import { createSyntheticRgbaFrame } from './inference/cameraFrameProvider';

// From a camera frame:
const frame = await cameraProvider.captureFrame();
const result = processFrame(frame);

// result.input → ready for recognizer.recognize(result.input)
// result.appliedCrop → the crop rectangle used
// result.processingTimeMs → performance tracking

// With explicit crop:
const result2 = processFrame(frame, {
  cropRect: { x: 50, y: 100, width: 400, height: 100 },
});
```

### Files Added/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/inference/colorConversion.ts` | **New** | RGBA/RGB → grayscale (BT.601) |
| `src/inference/cropRegion.ts` | **New** | ROI cropping + centered crop computation |
| `src/inference/imageResizer.ts` | **New** | Bilinear interpolation resize |
| `src/inference/cameraFrameProvider.ts` | **New** | Frame provider interface + synthetic factories |
| `src/inference/framePipeline.ts` | **New** | Full frame → model input pipeline |
| `src/__tests__/colorConversion.test.ts` | **New** | 14 tests for color conversion |
| `src/__tests__/cropRegion.test.ts` | **New** | 19 tests for cropping |
| `src/__tests__/imageResizer.test.ts` | **New** | 15 tests for bilinear resize |
| `src/__tests__/cameraFrameProvider.test.ts` | **New** | 10 tests for frame provider |
| `src/__tests__/framePipeline.test.ts` | **New** | 19 tests for end-to-end pipeline |
| `docs/mobile_inference_plan.md` | **Updated** | This document |

