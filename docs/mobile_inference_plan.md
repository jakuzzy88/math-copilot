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
| 5F | Live OCR loop with StabilityAggregator integration | ✅ Complete |
| 6A | UI overlay integration | ✅ Complete |
| 6B | React Native camera shell + demo overlay | ✅ Complete |
| 7A | Android native project scaffolding + demo mode phone run | ✅ Complete |

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

---

## 16. Sprint 5F: Live OCR Loop with StabilityAggregator Integration

### Overview

Sprint 5F wires the frame preprocessing pipeline (Sprint 5E) into a live-style
recognition loop that connects all previously-built components end-to-end:

```
CameraFrameProvider → processFrame() → recognizeAndSolve() → StabilityAggregator → stable UI-ready result
```

The `LiveRecognitionController` orchestrates this entire pipeline at a
configurable frame interval, with built-in busy-frame protection, error
isolation, and comprehensive diagnostics.

### Architecture

```
┌──────────────────────┐
│ CameraFrameProvider  │  RGBA frames at configurable interval
│ (device / synthetic) │
└──────┬───────────────┘
       │ captureFrame()
       ▼
┌──────────────────────┐
│ framePipeline.ts     │  RGBA → grayscale → crop → resize → normalize
│ processFrame()       │  → StaticRecognizerInput (128×512 Float32)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ recognizeAndSolve()  │  ONNX inference → CTC decode → solve pipeline
│ recognizer + pipeline│  → PipelineResult (accepted/rejected)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────┐
│ StabilityAggregator      │  sliding window → agreement check
│ addFrame(pipelineResult) │  → StableRecognitionResult
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ onStableResult callback  │  → UI overlay (Sprint 6+)
└──────────────────────────┘
```

### Interval Strategy

- **Default interval:** 250 ms (4 fps).
- **Configurable** via `intervalMs` option.
- The interval is a timer-based loop using `setInterval`.
- On slow devices, the interval can be increased (e.g. 500 ms) to reduce
  CPU/GPU pressure.
- On fast devices, it can be decreased (e.g. 100 ms) for more responsive
  recognition.

### Why Busy Frames Are Skipped (Not Queued)

When a frame is still being processed and a new interval fires, the new
frame is **skipped** rather than queued. Reasons:

1. **Backlog prevention:** Queuing frames on a slow device would create an
   ever-growing backlog, increasing latency and memory pressure.
2. **Freshness:** Older queued frames are stale — the user may have moved
   the camera. Processing the newest available frame is always better.
3. **Predictability:** The controller processes at most one frame at a
   time, making timing diagnostics reliable and reproducible.
4. **Simplicity:** No queue management, no priority logic, no memory
   bounds to configure.

Skipped frames are tracked in `diagnostics.framesSkippedBusy` for
monitoring.

### How Stability Prevents Flicker

The `StabilityAggregator` (Sprint E) maintains a sliding window of recent
pipeline results. A stable equation is only emitted when:

1. The **same equation string** appears at least `minAgreement` times in
   the window (default: 3).
2. The **average confidence** across those frames meets the threshold
   (default: 0.65).
3. If the current window does not meet stability criteria, the **last
   stable result is retained** (flicker prevention).

This prevents the UI from jumping between different equations or showing
transient misreads.

### Diagnostics Fields

| Field | Type | Description |
|-------|------|-------------|
| `framesSeen` | number | Total frames seen (including skipped) |
| `framesProcessed` | number | Frames fully processed through the pipeline |
| `framesSkippedBusy` | number | Frames skipped due to busy processing |
| `framesFailedPreprocessing` | number | Frames that failed during crop/resize |
| `framesFailedRecognition` | number | Frames that failed during ONNX inference |
| `framesRejectedByPipeline` | number | Frames rejected by the solve pipeline |
| `stableResultsEmitted` | number | Stable results sent to the callback |
| `lastRawText` | string \| null | Raw OCR text from the last recognition |
| `lastStableEquation` | string \| null | Last stable equation string |
| `averagePreprocessMs` | number | Running average preprocessing time |
| `averageRecognitionMs` | number | Running average recognition time |
| `averageTotalMs` | number | Running average total frame time |

### Usage

```typescript
import { LiveRecognitionController } from './inference/liveRecognitionController';
import { StaticFrameProvider } from './inference/cameraFrameProvider';
import { OnnxEquationRecognizer } from './inference/staticImageRecognizer';

// 1. Create dependencies.
const frameProvider = new StaticFrameProvider([...frames]);
const recognizer = await OnnxEquationRecognizer.create(modelPath);

// 2. Create and start controller.
const controller = new LiveRecognitionController({
  frameProvider,
  recognizer,
  intervalMs: 250,
  onStableResult: (result) => {
    if (result.stable) {
      console.log(`Stable: ${result.equation} → ${result.solution}`);
      updateUI(result);
    }
  },
  onError: (error, context) => {
    console.warn(`[${context}] ${error.message}`);
  },
});

controller.start();

// 3. Check diagnostics periodically.
setInterval(() => {
  const diag = controller.getDiagnostics();
  console.log(`Processed: ${diag.framesProcessed}, Stable: ${diag.lastStableEquation}`);
}, 2000);

// 4. Clean up.
controller.stop();
await controller.dispose();
```

### Error Isolation

Errors at each pipeline stage are caught individually:

| Stage | Diagnostic counter | Callback context |
|-------|-------------------|-----------------|
| Frame capture | `framesFailedPreprocessing` | `'captureFrame'` |
| Preprocessing | `framesFailedPreprocessing` | `'processFrame'` |
| Recognition | `framesFailedRecognition` | `'recognizeAndSolve'` |

Errors never crash the controller. The loop continues processing
subsequent frames.

### Files Added/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/inference/liveRecognitionController.ts` | **New** | Live recognition loop controller |
| `src/__tests__/liveRecognitionController.test.ts` | **New** | 29 tests for controller lifecycle, processing, stability, errors |
| `docs/mobile_inference_plan.md` | **Updated** | This document |

### Next Step: Sprint 6B — Real Camera Integration

The next sprint will connect the `LiveRecognitionController` to a real device
camera via `expo-camera` or `react-native-camera`, replacing the
`StaticFrameProvider` with live camera frames.

---

## 17. Sprint 6A: UI Overlay Integration

### Overview

Sprint 6A connects the live recognition architecture (Sprint 5F) to a
UI-friendly state layer and composable overlay components. The overlay
renders directly on top of a camera preview placeholder and includes:

- Equation guide-box overlay
- Recognized equation + solution display
- First explanation step preview
- Toggleable diagnostics/debug panel
- Demo/static mode for testing without real camera or ONNX runtime

### Architecture

```
┌──────────────────────────┐
│ LiveRecognitionScreen    │  Screen controller
│   ├── LiveRecognitionController (from Sprint 5F)
│   ├── RecognitionUiState adapter
│   └── State listeners
└──────┬───────────────────┘
       │ state updates
       ▼
┌──────────────────────────┐
│ RecognitionOverlay       │  Composition component
│   ├── GuideBoxOverlay    │  4:1 aspect ratio target box
│   ├── SolutionCard       │  equation + solution + step
│   └── DiagnosticsPanel   │  debug info (toggleable)
└──────────────────────────┘
```

### UI State Mapping

The `recognitionUiState.ts` module converts internal types into a flat,
UI-friendly `RecognitionUiState` object:

| Internal State | UI Mode | Status Message |
|---------------|---------|---------------|
| Controller not started | `idle` | "Point camera at a handwritten equation…" |
| Frames processing, no result yet | `scanning` | "Scanning for equation…" |
| Stable result, confidence ≥ 0.65 | `stable` | "Equation recognized: 3x+4=10" |
| Stable result but low confidence | `uncertain` | "Low confidence (55.0%)" |
| Processing error | `error` | "Error: ONNX session crashed" |

### Guide Box Behavior

The guide box is a bordered rectangle with a 4:1 aspect ratio (matching
the model's 512×128 input), displayed at 85% container width by default.

| Mode | Border Color | Opacity |
|------|-------------|---------|
| Active (scanning/uncertain) | `#00E676` (green) | 1.0 |
| Inactive (stable/idle/error) | `rgba(255,255,255,0.4)` | 0.6 |

### Solution Card

The solution card appears below the guide box and adapts its background
color based on the recognition mode and confidence level:

| Condition | Background |
|-----------|-----------|
| Stable, confidence ≥ 0.85 | Green |
| Stable, confidence < 0.85 | Blue |
| Uncertain | Orange |
| Error | Red |
| Scanning/Idle | Dark |

### Diagnostics Panel

A toggleable debug panel showing:

- Frames seen / processed / skipped / rejected
- Preprocessing and recognition failure counts
- Stable results emitted
- Last raw OCR text and last stable equation
- Average preprocessing, recognition, and total processing times

### Demo/Static Mode

A `DemoRecognizer` class returns predefined equation responses
("3x+4=10" → "x=2") for UI testing without real camera or ONNX runtime:

```typescript
const screen = new LiveRecognitionScreen({ demoMode: true });
screen.start();
// UI transitions: idle → scanning → uncertain → stable
```

This allows full end-to-end UI testing in Jest with:
- `StaticFrameProvider` (synthetic RGBA frames)
- `DemoRecognizer` (fake recognition output)
- `StabilityAggregator` (real aggregation logic)

### Files Added/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/ui/recognitionUiState.ts` | **New** | UI state adapter with mode mapping and formatting |
| `src/ui/components/GuideBoxOverlay.tsx` | **New** | Guide box overlay component |
| `src/ui/components/SolutionCard.tsx` | **New** | Solution card overlay component |
| `src/ui/components/DiagnosticsPanel.tsx` | **New** | Diagnostics panel component |
| `src/ui/components/RecognitionOverlay.tsx` | **New** | Overlay composition component |
| `src/ui/LiveRecognitionScreen.ts` | **New** | Screen controller with demo mode |
| `src/__tests__/recognitionUiState.test.ts` | **New** | 28 tests for UI state mapping |
| `src/__tests__/recognitionOverlay.test.ts` | **New** | 18 tests for overlay components |
| `src/__tests__/liveRecognitionScreen.test.ts` | **New** | 16 tests for screen controller |
| `jest.config.ts` | **Modified** | Added .tsx support and @ui/* path |
| `tsconfig.json` | **Modified** | Added jsx and @ui/* path |
| `docs/mobile_inference_plan.md` | **Updated** | This document |

### What Is Still Missing Before Real Device Camera OCR

1. **React Native App Shell** — No `App.tsx` / metro bundler is scaffolded yet.
   The overlay components are written as data-driven descriptions, not JSX.
2. **Real Camera Provider** — `expo-camera` or `react-native-camera`
   integration to replace `StaticFrameProvider` with live frames.
3. **ONNX Runtime on Device** — The `OnnxEquationRecognizer` needs a real
   `.onnx` model loaded via `onnxruntime-react-native` on an Android/iOS device.
4. **Frame Rate Tuning** — The 250 ms interval needs benchmarking on real
   devices to find the optimal capture rate.
5. ~~**UI Rendering** — The serializable render data needs to be mapped to
   actual React Native `<View>` / `<Text>` primitives.~~ ✅ Done in Sprint 6B.
6. ~~**Permissions** — Camera permissions (Android/iOS) are not handled yet.~~ ✅ Done in Sprint 6B.
7. **Focus/Exposure Control** — Auto-focus and exposure compensation for
   different lighting conditions.

---

## 18. Sprint 6B: React Native Camera Shell

### Overview

Sprint 6B converts the app from a TypeScript-only test package into a
phone-testable React Native application shell. The screen renders actual
React Native components (`View`, `Text`, `Pressable`, `StyleSheet`) and
handles camera permission flow, guide box overlay, equation/solution
display, and a toggleable diagnostics panel.

### Key Decisions

- **React Native CLI (bare)** — The project uses bare React Native, not
  Expo, to maintain compatibility with `onnxruntime-react-native` and
  `react-native-vision-camera` native modules.
- **Demo mode first** — `DEMO_MODE = true` allows full UI testing on a
  phone without requiring ONNX Runtime or live camera frames.
- **Conditional camera import** — `react-native-vision-camera` is
  imported via `try/catch` so that Jest tests run without native modules.

### Architecture

```
┌──────────────────────────┐
│ App.tsx                  │  Root entry
│   └── MathCameraScreen   │  Single-screen app
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ MathCameraScreen.tsx     │  Real React Native component
│   ├── Camera permission  │  useSafeCameraPermission()
│   ├── Camera preview     │  VisionCamera or placeholder
│   ├── Guide box overlay  │  4:1 aspect ratio, green when active
│   ├── Status text        │  Mode-specific message
│   ├── Solution card      │  Equation + solution + step
│   ├── Diagnostics toggle │  Pressable toggle
│   └── Diagnostics panel  │  Debug rows from controller
└──────┬───────────────────┘
       │ uses
       ▼
┌──────────────────────────┐
│ LiveRecognitionScreen    │  Sprint 6A screen controller
│   └── DemoRecognizer     │  Fake OCR for demo mode
└──────────────────────────┘
```

### Demo Mode Behavior

When `DEMO_MODE = true`:

1. Camera permission is auto-granted (no system prompt).
2. A dark placeholder replaces the camera preview, with "📸 Camera
   preview (demo mode)" text and an orange "DEMO MODE" badge.
3. The LiveRecognitionScreen controller uses `DemoRecognizer` which
   emits the equation `3x+4=10` with varying confidence scores.
4. After ~3 frames, the `StabilityAggregator` reaches stability and the
   solution card shows:
   - Equation: `3x+4=10`
   - Solution: `→ x=2`
   - Confidence: `~90.0% · HIGH`
   - First explanation step
5. Mode indicator dot transitions: gray (idle) → blue (scanning) →
   green (stable).
6. Diagnostics panel shows frame counts, average timing, and stability
   metrics.

### What the User Sees on Phone

| Area | Content |
|------|---------|
| Top-left | Mode indicator (dot + "STABLE") |
| Top-right | "DEMO MODE" orange badge |
| Center | 4:1 guide box with green border |
| Below guide box | Status message |
| Below camera | Solution card (green background) |
| Bottom | "▶ Show Diagnostics" toggle |

### Camera Permission Flow

| State | Screen |
|-------|--------|
| Checking | Loading spinner + "Checking camera permission…" |
| Denied | Title "📷 Camera Access Required" + explanation + "Grant Camera Access" button |
| Granted | Full camera screen with guide box |

### Android Phone Testing Instructions

#### Prerequisites

1. Node.js 18+ installed.
2. Android SDK installed (Android Studio recommended).
3. `ANDROID_HOME` environment variable set.
4. USB debugging enabled on the Android device.
5. Device connected via USB or wireless ADB.

#### Setup (first time only)

```bash
# 1. Navigate to the app directory.
cd math-copilot/app

# 2. Install JavaScript dependencies.
npm install

# 3. Generate the Android project (if android/ directory is missing).
#    NOTE: The android/ directory must be generated using react-native
#    init or manually. See "Generating Android Project" below.

# 4. Add camera permission to AndroidManifest.xml.
#    Add the following lines inside <manifest>:
#      <uses-permission android:name="android.permission.CAMERA" />
#    And inside <application>:
#      <meta-data android:name="com.google.android.gms.vision.DEPENDENCIES"
#                 android:value="barcode" />
```

#### Generating Android Project

If the `android/` directory does not exist yet:

```bash
# Option A: Use @react-native-community/cli to generate native projects.
npx -y @react-native-community/cli init MathCopilot --directory ./temp-rn
# Copy android/ from temp-rn into app/android/.
# Update android/app/build.gradle with the correct app name.

# Option B: Manual setup (advanced).
# Create android/ directory structure manually following RN docs.
```

#### Running

```bash
# 1. Start Metro bundler.
cd math-copilot/app
npm start

# 2. In a separate terminal, build and install on device.
npm run android

# 3. The app should launch showing the demo UI.
```

#### Expected Demo UI Behavior

1. App launches with a dark camera placeholder.
2. Orange "DEMO MODE" badge appears top-right.
3. Mode indicator shows "SCANNING" (blue dot).
4. After ~1 second, transitions to "STABLE" (green dot).
5. Solution card turns green, showing:
   - `3x+4=10`
   - `→ x=2`
   - `Confidence: 90.0% · HIGH`
6. Tapping "▶ Show Diagnostics" reveals frame processing stats.

#### How to Know It Works

- [x] App installs and launches without crash.
- [x] Demo mode badge is visible.
- [x] Guide box renders with 4:1 aspect ratio.
- [x] Equation and solution appear in the solution card.
- [x] Diagnostics panel toggles on/off.
- [x] Mode indicator transitions from idle → scanning → stable.

### Known Limitations

1. **No android/ directory generated** — The bare React Native android
   project must be scaffolded separately (Sprint 7 or manual).
2. **Camera preview is a placeholder** — Real camera preview requires
   completing the Android project setup and linking vision-camera.
3. **No real OCR** — `DEMO_MODE = true` uses a fake recognizer. Setting
   `DEMO_MODE = false` requires the ONNX model and native frame extraction.
4. **Frame extraction is stubbed** — `VisionCameraFrameProvider.updateFrame()`
   throws "not yet implemented". A native Frame Processor plugin is needed.
5. **No iOS support** — iOS requires Xcode project setup and CocoaPods.

### Integration Path to Real OCR

| Step | Sprint | Status |
|------|--------|--------|
| React Native shell + demo UI | 6B | ✅ Complete |
| Generate `android/` project | 7 | TODO |
| Link `react-native-vision-camera` | 7 | TODO |
| Implement `VisionCameraFrameProvider` | 7 | TODO |
| Test camera preview on device | 7 | TODO |
| Load ONNX model on device | 7+ | TODO |
| Wire real `OnnxEquationRecognizer` | 7+ | TODO |
| End-to-end live OCR on device | 7+ | TODO |

### Files Added/Modified

| File | Action | Description |
|------|--------|-------------|
| `src/screens/MathCameraScreen.tsx` | **New** | Main camera screen with real RN components |
| `src/screens/realModeStubs.ts` | **New** | VisionCameraFrameProvider stub + TODO docs |
| `src/App.tsx` | **New** | Root app component |
| `index.js` | **New** | React Native entry point |
| `metro.config.js` | **New** | Metro bundler config (.onnx asset support) |
| `babel.config.js` | **New** | Babel config with RN preset |
| `src/__tests__/mathCameraScreen.test.ts` | **New** | 24 tests for screen adapter logic |
| `package.json` | **Modified** | Added react, react-native, vision-camera deps + scripts |
| `tsconfig.json` | **Modified** | Added @screens/* path alias |
| `jest.config.ts` | **Modified** | Added @screens/* module mapping |
| `docs/mobile_inference_plan.md` | **Updated** | This document |

---

## 19. Sprint 7A: Android Native Project Scaffolding + Demo Mode Phone Run

### Overview

Sprint 7A adds the Android native project to the React Native app,
making it possible to build, install, and run the Math Copilot demo
on a real Android phone.

### Key Decisions

- **Generated from RN CLI template** — Used `@react-native-community/cli init`
  with `--version 0.86.0 --package-name com.mathcopilot` to generate a
  compatible Android project, then copied `android/` into the app.
- **Native modules disabled for demo** — Both `onnxruntime-react-native`
  and `react-native-vision-camera` are excluded from autolinking via
  `react-native.config.js` because:
  1. `onnxruntime-react-native` has a Gradle 9.x compatibility issue.
  2. Neither is needed for demo mode.
  3. They will be re-enabled when real OCR integration begins.
- **DEMO_MODE = true** — The app runs with a fake recognizer and
  placeholder camera preview. No real OCR or camera frames.

### Android Project Details

| Property | Value |
|----------|-------|
| Package ID | `com.mathcopilot` |
| App name | `MathCopilot` |
| Min SDK | 24 (Android 7.0) |
| Target SDK | 36 |
| Compile SDK | 36 |
| Kotlin version | 2.1.20 |
| Gradle version | 9.3.1 |
| New Architecture | Enabled |
| Hermes engine | Enabled |

### Permissions

| Permission | Purpose |
|-----------|--------|
| `INTERNET` | Metro bundler debug connection (RN default) |
| `CAMERA` | Prepared for future camera preview |

### Build & Install Verification

| Step | Result |
|------|--------|
| `./gradlew tasks` | ✅ BUILD SUCCESSFUL |
| `./gradlew assembleDebug` | ✅ BUILD SUCCESSFUL (1m 54s) |
| APK size | 112 MB (debug, all architectures) |
| `./gradlew installDebug` | ✅ Installed on moto g56 5G |
| Metro bundle served | ✅ 779 modules bundled |
| App launched on device | ✅ Activity started |
| Jest tests | ✅ 387 passed, 24 suites |

### How to Run on Android Phone

#### Prerequisites

1. **Node.js 18+** installed.
2. **Android Studio** with Android SDK installed.
3. **Java 17** (OpenJDK recommended).
4. **`ANDROID_HOME`** environment variable set to SDK location.
5. **USB debugging** enabled on the Android device.
6. Device connected via USB and visible in `adb devices`.

#### First-Time Setup

```bash
# 1. Install JS dependencies.
cd math-copilot/app
npm install

# 2. Verify device is connected.
adb devices
# Should show your device as "device" (not "unauthorized").
```

#### Build & Run

```bash
# Terminal 1: Start Metro bundler.
cd math-copilot/app
npm start

# Terminal 2: Build, install, and launch.
cd math-copilot/app
npm run android

# Alternative: Manual build + install.
cd math-copilot/app/android
./gradlew installDebug
adb reverse tcp:8081 tcp:8081
adb shell am start -n com.mathcopilot/.MainActivity
```

#### Troubleshooting

| Problem | Solution |
|---------|----------|
| `adb devices` shows nothing | Enable USB debugging in Developer Options |
| Device shows "unauthorized" | Accept the USB debugging prompt on the phone |
| App shows white screen | Run `adb reverse tcp:8081 tcp:8081` and restart |
| Metro not found by device | Ensure `adb reverse tcp:8081 tcp:8081` is run |
| Gradle version mismatch | Use Java 17 (`java -version` to check) |
| Build fails at autolinking | Check `react-native.config.js` disables problematic modules |

### Expected Demo UI on Phone

When the app loads in demo mode:

1. Black background with dark purple camera placeholder.
2. "📸 Camera preview (demo mode)" text centered.
3. Orange "DEMO MODE" badge in the top-right corner.
4. Mode indicator (green dot + "STABLE") in the top-left.
5. 4:1 green-bordered guide box overlay in the center.
6. "Equation recognized: 3x+4=10" status text below the guide box.
7. Green solution card showing:
   - `3x+4=10`
   - `→ x=2`
   - `Confidence: 90.0% · HIGH`
8. "▶ Show Diagnostics" toggle at the bottom.

### Known Limitations

1. **Demo mode only** — `DEMO_MODE = true`, no real camera or OCR.
2. **Native modules disabled** — `onnxruntime-react-native` and
   `react-native-vision-camera` are excluded from the Android build.
3. **No production signing** — Uses the default debug keystore.
4. **Debug APK is large** — 112 MB includes all 4 CPU architectures.
   Production release with ABI splitting would be ~30 MB.
5. **No iOS project** — Only Android is scaffolded.
6. **Metro DevTools error** — The Chrome sandbox error in Metro output
   is cosmetic and does not affect the app.

### Files Added/Modified

| File | Action | Description |
|------|--------|-------------|
| `android/` (entire directory) | **New** | Android native project (Gradle, Kotlin, manifests) |
| `android/app/src/main/AndroidManifest.xml` | **Modified** | Added CAMERA permission |
| `android/settings.gradle` | **Modified** | Set rootProject.name to 'MathCopilot' |
| `react-native.config.js` | **New** | Disables ONNX and VisionCamera autolinking |
| `app.json` | **New** | App name configuration for RN CLI |
| `package.json` | **Modified** | Added RN CLI dev dependencies |
| `.gitignore` | **Modified** | Added Android build artifact patterns |
| `docs/mobile_inference_plan.md` | **Updated** | This document |

### Next Sprint: 7B — Real Camera Preview

| Task | Description |
|------|-------------|
| Re-enable `react-native-vision-camera` | Fix autolinking, resolve native build |
| Live camera preview | Replace placeholder with real camera feed |
| Camera permission flow | Test the permission request on device |
| Frame rate benchmarking | Measure capture interval on real hardware |
| Guide box positioning | Verify 4:1 overlay aligns with camera frame |
