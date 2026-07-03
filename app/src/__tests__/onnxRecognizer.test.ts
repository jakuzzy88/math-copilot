/**
 * Tests for OnnxEquationRecognizer with mocked ONNX Runtime.
 *
 * Sprint 5D: ONNX Runtime Mobile Static Inference.
 *
 * These tests verify the full inference pipeline by injecting a mock
 * ONNX Runtime that returns known output tensors.  This allows us to
 * test:
 *   - preprocess → tensor creation
 *   - session.run() invocation
 *   - output reshaping
 *   - CTC decode → rawText
 *   - confidence calculation
 *   - candidate wrapping
 *   - dispose lifecycle
 *
 * without requiring native ONNX bindings.
 */

import {
  OnnxEquationRecognizer,
} from '../inference/staticImageRecognizer';
import type {
  StaticRecognizerInput,
} from '../inference/staticImageRecognizer';
import type {
  OnnxRuntimeApi,
  OnnxInferenceSession,
  OnnxTensor,
} from '../inference/onnxRuntimeProvider';
import {
  MODEL_INPUT_HEIGHT,
  MODEL_INPUT_WIDTH,
  MODEL_TIME_STEPS,
  MODEL_NUM_CLASSES,
  ONNX_INPUT_NAME,
  ONNX_OUTPUT_NAME,
} from '../inference/modelIO';
import { VOCAB_SIZE, charToIndex } from '../inference/ctcVocabulary';
import {
  createWhiteImage,
  getTestImageDimensions,
} from '../inference/syntheticTestImages';

// ---------------------------------------------------------------------------
// Mock ONNX Runtime factory
// ---------------------------------------------------------------------------

/**
 * Build a mock output tensor containing log-probs that decode to `targetText`.
 *
 * Strategy: for each time step, place high log-prob (0.0) at the target
 * character index, and low log-prob (-10) at all other indices.
 *
 * Steps beyond the target text length emit blank tokens (index 0).
 */
function buildMockLogProbsForText(text: string): Float32Array {
  const flat = new Float32Array(MODEL_TIME_STEPS * MODEL_NUM_CLASSES);
  flat.fill(-10); // default: very low log-prob

  for (let t = 0; t < MODEL_TIME_STEPS; t++) {
    const base = t * MODEL_NUM_CLASSES;
    if (t < text.length) {
      // Emit the character at this time step.
      const charIdx = charToIndex(text[t]);
      flat[base + charIdx] = 0.0; // log(1) = 0
    } else {
      // Emit blank for remaining time steps.
      flat[base + 0] = 0.0; // blank index
    }
  }

  return flat;
}

/**
 * Create a mock ONNX Runtime API that returns a fixed output tensor.
 */
function createMockOrt(
  outputTensorData: Float32Array,
): {
  ort: OnnxRuntimeApi;
  spies: {
    sessionCreate: jest.Mock;
    sessionRun: jest.Mock;
    sessionRelease: jest.Mock;
    tensorConstructor: jest.Mock;
  };
} {
  const sessionRun = jest.fn().mockResolvedValue({
    [ONNX_OUTPUT_NAME]: {
      type: 'float32',
      data: outputTensorData,
      dims: [MODEL_TIME_STEPS, 1, MODEL_NUM_CLASSES],
    } as OnnxTensor,
  });

  const sessionRelease = jest.fn().mockResolvedValue(undefined);

  const mockSession: OnnxInferenceSession = {
    run: sessionRun,
    release: sessionRelease,
  };

  const sessionCreate = jest.fn().mockResolvedValue(mockSession);

  const tensorConstructor = jest.fn().mockImplementation(
    (type: string, data: Float32Array, dims: number[]) => ({
      type,
      data,
      dims,
    }),
  );

  const ort: OnnxRuntimeApi = {
    InferenceSession: {
      create: sessionCreate,
    },
    Tensor: tensorConstructor as unknown as OnnxRuntimeApi['Tensor'],
  };

  return {
    ort,
    spies: { sessionCreate, sessionRun, sessionRelease, tensorConstructor },
  };
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const testDims = getTestImageDimensions();

function makeInput(pixels?: Uint8Array): StaticRecognizerInput {
  return {
    grayscalePixels: pixels ?? createWhiteImage(),
    width: testDims.width,
    height: testDims.height,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('OnnxEquationRecognizer', () => {
  describe('create()', () => {
    it('creates a recognizer and loads the ONNX session', async () => {
      const { ort, spies } = createMockOrt(new Float32Array(0));
      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );

      expect(spies.sessionCreate).toHaveBeenCalledWith('model.onnx');
      expect(recognizer).toBeDefined();

      await recognizer.dispose();
    });

    it('throws when ONNX Runtime is null', async () => {
      await expect(
        OnnxEquationRecognizer.create('model.onnx', null as unknown as OnnxRuntimeApi),
      ).rejects.toThrow(/ONNX Runtime is not available/);
    });
  });

  describe('recognize()', () => {
    it('decodes "3x+4=10" from mock output', async () => {
      const targetText = '3x+4=10';
      const mockOutput = buildMockLogProbsForText(targetText);
      const { ort } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      const result = await recognizer.recognize(makeInput());

      expect(result.rawText).toBe('3x+4=10');
      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].text).toBe('3x+4=10');
      expect(result.candidates[0].confidence).toBeGreaterThan(0.9);

      await recognizer.dispose();
    });

    it('decodes "x+5=12" from mock output', async () => {
      const mockOutput = buildMockLogProbsForText('x+5=12');
      const { ort } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      const result = await recognizer.recognize(makeInput());

      expect(result.rawText).toBe('x+5=12');
      expect(result.candidates[0].confidence).toBeGreaterThan(0.5);

      await recognizer.dispose();
    });

    it('decodes empty string from all-blank output', async () => {
      // All time steps produce blank (index 0).
      const allBlank = new Float32Array(MODEL_TIME_STEPS * MODEL_NUM_CLASSES);
      allBlank.fill(-10);
      for (let t = 0; t < MODEL_TIME_STEPS; t++) {
        allBlank[t * MODEL_NUM_CLASSES + 0] = 0.0;
      }

      const { ort } = createMockOrt(allBlank);
      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      const result = await recognizer.recognize(makeInput());

      expect(result.rawText).toBe('');
      expect(result.candidates[0].text).toBe('');

      await recognizer.dispose();
    });

    it('creates a Tensor with correct shape [1, 1, 128, 512]', async () => {
      const mockOutput = buildMockLogProbsForText('x=1');
      const { ort, spies } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      await recognizer.recognize(makeInput());

      expect(spies.tensorConstructor).toHaveBeenCalledWith(
        'float32',
        expect.any(Float32Array),
        [1, 1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
      );

      await recognizer.dispose();
    });

    it('passes input tensor with correct name to session.run()', async () => {
      const mockOutput = buildMockLogProbsForText('x=1');
      const { ort, spies } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      await recognizer.recognize(makeInput());

      expect(spies.sessionRun).toHaveBeenCalledWith(
        expect.objectContaining({
          [ONNX_INPUT_NAME]: expect.objectContaining({
            type: 'float32',
            dims: [1, 1, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
          }),
        }),
      );

      await recognizer.dispose();
    });

    it('normalises Uint8Array pixels to [0,1] before creating tensor', async () => {
      const mockOutput = buildMockLogProbsForText('x=1');
      const { ort, spies } = createMockOrt(mockOutput);

      const pixels = new Uint8Array(MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH);
      pixels.fill(255);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      await recognizer.recognize(makeInput(pixels));

      // The tensor data should be Float32Array with values close to 1.0.
      const tensorData = spies.tensorConstructor.mock.calls[0][1] as Float32Array;
      expect(tensorData[0]).toBeCloseTo(1.0, 5);

      await recognizer.dispose();
    });

    it('throws if output tensor name is missing', async () => {
      const sessionRun = jest.fn().mockResolvedValue({
        wrong_name: { type: 'float32', data: new Float32Array(0), dims: [] },
      });
      const mockSession: OnnxInferenceSession = {
        run: sessionRun,
        release: jest.fn().mockResolvedValue(undefined),
      };

      const ort: OnnxRuntimeApi = {
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

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );

      await expect(recognizer.recognize(makeInput())).rejects.toThrow(
        /output tensor "log_probs" not found/,
      );

      await recognizer.dispose();
    });
  });

  describe('dispose()', () => {
    it('releases the ONNX session on dispose', async () => {
      const mockOutput = buildMockLogProbsForText('x=1');
      const { ort, spies } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      await recognizer.dispose();

      expect(spies.sessionRelease).toHaveBeenCalledTimes(1);
    });

    it('throws if recognize() is called after dispose', async () => {
      const mockOutput = buildMockLogProbsForText('x=1');
      const { ort } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      await recognizer.dispose();

      await expect(recognizer.recognize(makeInput())).rejects.toThrow(
        /has been disposed/,
      );
    });

    it('is idempotent — calling dispose twice does not throw', async () => {
      const mockOutput = buildMockLogProbsForText('x=1');
      const { ort, spies } = createMockOrt(mockOutput);

      const recognizer = await OnnxEquationRecognizer.create(
        'model.onnx',
        ort,
      );
      await recognizer.dispose();
      await recognizer.dispose();

      // release() should only be called once.
      expect(spies.sessionRelease).toHaveBeenCalledTimes(1);
    });
  });
});
