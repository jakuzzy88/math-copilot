/**
 * End-to-end static inference pipeline test.
 *
 * Sprint 5D: ONNX Runtime Mobile Static Inference.
 *
 * Tests the full pathway:
 *   synthetic image → OnnxEquationRecognizer → recognizeAndSolve() → solution.
 *
 * Uses a mock ONNX Runtime that returns known log-probs, so the test
 * verifies pipeline integration, NOT model accuracy.
 */

import { OnnxEquationRecognizer } from '../inference/staticImageRecognizer';
import type { StaticRecognizerInput } from '../inference/staticImageRecognizer';
import type {
  OnnxRuntimeApi,
  OnnxInferenceSession,
  OnnxTensor,
} from '../inference/onnxRuntimeProvider';
import { recognizeAndSolve } from '../inference/recognizeAndSolve';
import {
  MODEL_INPUT_HEIGHT,
  MODEL_INPUT_WIDTH,
  MODEL_TIME_STEPS,
  MODEL_NUM_CLASSES,
  ONNX_OUTPUT_NAME,
} from '../inference/modelIO';
import { charToIndex } from '../inference/ctcVocabulary';
import {
  createWhiteImage,
  createBlackImage,
  createStripedImage,
  createGradientImage,
  getTestImageDimensions,
} from '../inference/syntheticTestImages';

// ---------------------------------------------------------------------------
// Mock helpers
// ---------------------------------------------------------------------------

function buildMockLogProbsForText(text: string): Float32Array {
  const flat = new Float32Array(MODEL_TIME_STEPS * MODEL_NUM_CLASSES);
  flat.fill(-10);
  for (let t = 0; t < MODEL_TIME_STEPS; t++) {
    const base = t * MODEL_NUM_CLASSES;
    if (t < text.length) {
      flat[base + charToIndex(text[t])] = 0.0;
    } else {
      flat[base + 0] = 0.0; // blank
    }
  }
  return flat;
}

function createMockOrt(outputData: Float32Array): OnnxRuntimeApi {
  const mockSession: OnnxInferenceSession = {
    run: jest.fn().mockResolvedValue({
      [ONNX_OUTPUT_NAME]: {
        type: 'float32',
        data: outputData,
        dims: [MODEL_TIME_STEPS, 1, MODEL_NUM_CLASSES],
      } as OnnxTensor,
    }),
    release: jest.fn().mockResolvedValue(undefined),
  };

  return {
    InferenceSession: {
      create: jest.fn().mockResolvedValue(mockSession),
    },
    Tensor: jest.fn().mockImplementation(
      (type: string, data: Float32Array, dims: number[]) => ({
        type,
        data,
        dims,
      }),
    ) as unknown as OnnxRuntimeApi['Tensor'],
  };
}

const dims = getTestImageDimensions();

function makeInput(pixels: Uint8Array): StaticRecognizerInput {
  return {
    grayscalePixels: pixels,
    width: dims.width,
    height: dims.height,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('Static inference pipeline (end-to-end)', () => {
  it('recognizes "3x+4=10" and solves x=2', async () => {
    const ort = createMockOrt(buildMockLogProbsForText('3x+4=10'));
    const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

    const result = await recognizeAndSolve(recognizer, makeInput(createWhiteImage()));

    expect(result.rawText).toBe('3x+4=10');
    expect(result.pipeline.accepted).toBe(true);
    if (result.pipeline.accepted) {
      expect(result.pipeline.equation).toBe('3x+4=10');
      expect(result.pipeline.solution).toBe('x=2');
      expect(result.pipeline.explanationSteps.length).toBeGreaterThan(0);
    }

    await recognizer.dispose();
  });

  it('recognizes "x+5=12" and solves x=7', async () => {
    const ort = createMockOrt(buildMockLogProbsForText('x+5=12'));
    const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

    const result = await recognizeAndSolve(recognizer, makeInput(createBlackImage()));

    expect(result.pipeline.accepted).toBe(true);
    if (result.pipeline.accepted) {
      expect(result.pipeline.solution).toBe('x=7');
    }

    await recognizer.dispose();
  });

  it('recognizes "5x-2=18" and solves x=4', async () => {
    const ort = createMockOrt(buildMockLogProbsForText('5x-2=18'));
    const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

    const result = await recognizeAndSolve(
      recognizer,
      makeInput(createStripedImage()),
    );

    expect(result.pipeline.accepted).toBe(true);
    if (result.pipeline.accepted) {
      expect(result.pipeline.solution).toBe('x=4');
    }

    await recognizer.dispose();
  });

  it('recognizes "2x=8" and solves x=4', async () => {
    const ort = createMockOrt(buildMockLogProbsForText('2x=8'));
    const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

    const result = await recognizeAndSolve(
      recognizer,
      makeInput(createGradientImage()),
    );

    expect(result.pipeline.accepted).toBe(true);
    if (result.pipeline.accepted) {
      expect(result.pipeline.solution).toBe('x=4');
    }

    await recognizer.dispose();
  });

  it('rejects empty string from all-blank output', async () => {
    const allBlank = new Float32Array(MODEL_TIME_STEPS * MODEL_NUM_CLASSES);
    allBlank.fill(-10);
    for (let t = 0; t < MODEL_TIME_STEPS; t++) {
      allBlank[t * MODEL_NUM_CLASSES + 0] = 0.0;
    }

    const ort = createMockOrt(allBlank);
    const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

    const result = await recognizeAndSolve(recognizer, makeInput(createWhiteImage()));

    expect(result.rawText).toBe('');
    expect(result.pipeline.accepted).toBe(false);

    await recognizer.dispose();
  });

  it('rejects nonsensical decoded text', async () => {
    // "+++==" is grammatically invalid.
    const ort = createMockOrt(buildMockLogProbsForText('+++='));
    const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

    const result = await recognizeAndSolve(recognizer, makeInput(createWhiteImage()));

    expect(result.pipeline.accepted).toBe(false);

    await recognizer.dispose();
  });

  it('works with all four synthetic image types', async () => {
    const images = [
      createWhiteImage(),
      createBlackImage(),
      createStripedImage(),
      createGradientImage(),
    ];

    for (const pixels of images) {
      const ort = createMockOrt(buildMockLogProbsForText('x+1=2'));
      const recognizer = await OnnxEquationRecognizer.create('model.onnx', ort);

      const result = await recognizeAndSolve(recognizer, makeInput(pixels));

      expect(result.rawText).toBe('x+1=2');
      expect(result.pipeline.accepted).toBe(true);
      if (result.pipeline.accepted) {
        expect(result.pipeline.solution).toBe('x=1');
      }

      await recognizer.dispose();
    }
  });
});
