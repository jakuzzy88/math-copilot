/**
 * Static image recognizer — ONNX-based equation recognition.
 *
 * Sprint 5D: ONNX Runtime Mobile Static Inference.
 *
 * Implements full inference pipeline:
 *   1. Preprocess grayscale pixels → Float32Array [0,1].
 *   2. Create ONNX tensor [1, 1, 128, 512].
 *   3. Run ONNX InferenceSession.
 *   4. Reshape output [32, 1, 19] → [32][19] log-probabilities.
 *   5. CTC greedy decode → raw text.
 *   6. Compute confidence score.
 *   7. Return { rawText, candidates }.
 */

import type { OcrCandidate } from '../pipeline/types';
import { prepareGrayscaleInput } from './imagePreprocessor';
import {
  ctcGreedyDecodeFromLogProbs,
  computeConfidence,
} from './ctcDecoder';
import {
  MODEL_INPUT_HEIGHT,
  MODEL_INPUT_WIDTH,
  MODEL_CHANNELS,
  MODEL_TIME_STEPS,
  MODEL_NUM_CLASSES,
  ONNX_INPUT_NAME,
  ONNX_OUTPUT_NAME,
  reshapeOutputToLogProbs,
} from './modelIO';
import type {
  OnnxRuntimeApi,
  OnnxInferenceSession,
  OnnxTensor,
} from './onnxRuntimeProvider';
import { getOnnxRuntime } from './onnxRuntimeProvider';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Input to the static recognizer. */
export interface StaticRecognizerInput {
  /** Grayscale pixel data (128×512, row-major). */
  grayscalePixels: Uint8Array | Float32Array;
  /** Image width (must be 512). */
  width: number;
  /** Image height (must be 128). */
  height: number;
}

/** Output from the static recognizer. */
export interface StaticRecognizerOutput {
  /** Raw decoded text from CTC greedy decode. */
  rawText: string;
  /** Candidate(s) for the pipeline. */
  candidates: OcrCandidate[];
}

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

/**
 * Equation recognition session interface.
 *
 * Implementations load an ONNX model and provide a `recognize` method
 * that takes preprocessed grayscale image data and returns decoded text
 * with candidate(s).
 */
export interface EquationRecognitionSession {
  /** Run inference on a single static image. */
  recognize(input: StaticRecognizerInput): Promise<StaticRecognizerOutput>;
  /** Release model resources. */
  dispose(): Promise<void>;
}

// ---------------------------------------------------------------------------
// ONNX implementation
// ---------------------------------------------------------------------------

/**
 * ONNX-based equation recognizer.
 *
 * Full inference pipeline:
 *   preprocess → ONNX session.run() → reshape → CTC decode → candidates.
 *
 * The ONNX session is created lazily on first `recognize()` call, or
 * eagerly via the static `create()` factory.
 */
export class OnnxEquationRecognizer implements EquationRecognitionSession {
  private disposed = false;
  private session: OnnxInferenceSession | null = null;
  private ort: OnnxRuntimeApi;

  /**
   * Private constructor — use `OnnxEquationRecognizer.create()` instead.
   */
  private constructor(
    private readonly modelPath: string,
    ort: OnnxRuntimeApi,
  ) {
    this.ort = ort;
  }

  // -----------------------------------------------------------------------
  // Factory
  // -----------------------------------------------------------------------

  /**
   * Create and initialise an ONNX equation recognizer.
   *
   * @param modelPath - Path/URI to the .onnx model asset.
   * @param ortOverride - Optional ONNX Runtime API override (for testing).
   * @returns Initialised recognizer with a loaded ONNX session.
   * @throws {Error} if ONNX Runtime is not available.
   */
  static async create(
    modelPath: string,
    ortOverride?: OnnxRuntimeApi,
  ): Promise<OnnxEquationRecognizer> {
    const ort = ortOverride ?? getOnnxRuntime();
    if (ort === null) {
      throw new Error(
        'OnnxEquationRecognizer: ONNX Runtime is not available. ' +
        'Ensure onnxruntime-react-native is installed and linked.',
      );
    }

    const recognizer = new OnnxEquationRecognizer(modelPath, ort);
    await recognizer.loadSession();
    return recognizer;
  }

  // -----------------------------------------------------------------------
  // Session management
  // -----------------------------------------------------------------------

  /**
   * Load the ONNX inference session from the model file.
   */
  private async loadSession(): Promise<void> {
    this.session = await this.ort.InferenceSession.create(this.modelPath);
  }

  // -----------------------------------------------------------------------
  // Inference
  // -----------------------------------------------------------------------

  async recognize(input: StaticRecognizerInput): Promise<StaticRecognizerOutput> {
    if (this.disposed) {
      throw new Error('OnnxEquationRecognizer has been disposed.');
    }
    if (this.session === null) {
      throw new Error('OnnxEquationRecognizer: session not loaded.');
    }

    // 1. Preprocess: normalise pixels to Float32Array [0,1].
    const normalised = prepareGrayscaleInput(
      input.grayscalePixels,
      input.width,
      input.height,
    );

    // 2. Create input tensor: [1, 1, 128, 512] (NCHW).
    const inputTensor = new this.ort.Tensor(
      'float32',
      normalised,
      [1, MODEL_CHANNELS, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH],
    );

    // 3. Run ONNX inference.
    const feeds: Record<string, OnnxTensor> = {
      [ONNX_INPUT_NAME]: inputTensor,
    };
    const results = await this.session.run(feeds);

    // 4. Extract output tensor.
    const outputTensor = results[ONNX_OUTPUT_NAME];
    if (!outputTensor) {
      throw new Error(
        `OnnxEquationRecognizer: output tensor "${ONNX_OUTPUT_NAME}" not found. ` +
        `Available outputs: ${Object.keys(results).join(', ')}`,
      );
    }

    // 5. Reshape flat output to [T][C] log-probability matrix.
    //    ONNX output shape is [32, 1, 19] (T, N=1, C).
    //    Flat array is T * N * C = 32 * 1 * 19 = 608 floats.
    const flatOutput = outputTensor.data as Float32Array;
    const logProbs = reshapeOutputToLogProbs(
      flatOutput,
      MODEL_TIME_STEPS,
      MODEL_NUM_CLASSES,
    );

    // 6. CTC greedy decode → raw text.
    const rawText = ctcGreedyDecodeFromLogProbs(logProbs);

    // 7. Compute confidence score.
    const confidence = computeConfidence(logProbs);

    // 8. Wrap as OcrCandidate(s).
    const candidates: OcrCandidate[] = [
      {
        text: rawText,
        confidence,
      },
    ];

    return { rawText, candidates };
  }

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    if (this.session !== null) {
      await this.session.release();
      this.session = null;
    }
  }
}
