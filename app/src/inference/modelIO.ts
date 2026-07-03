/**
 * Model I/O constants for ONNX Runtime inference.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 *
 * These constants match the trained CNN-CTC model exported via
 * `training/export_onnx.py`. The ONNX model was validated with
 * exact match against PyTorch on all 500 test samples.
 *
 * Input shape:  [batch_size, 1, 128, 512]  (NCHW, grayscale)
 * Output shape: [32, batch_size, 19]       (T, N, C, log-probabilities)
 */

// ---------------------------------------------------------------------------
// Model shape constants
// ---------------------------------------------------------------------------

/** Input image height in pixels. */
export const MODEL_INPUT_HEIGHT = 128;

/** Input image width in pixels. */
export const MODEL_INPUT_WIDTH = 512;

/** Number of input channels (1 = grayscale). */
export const MODEL_CHANNELS = 1;

/** Number of output time steps (width / 16 from CNN stride). */
export const MODEL_TIME_STEPS = 32;

/** Number of output classes (blank + 18 printable characters). */
export const MODEL_NUM_CLASSES = 19;

// ---------------------------------------------------------------------------
// ONNX tensor names (from export_onnx.py)
// ---------------------------------------------------------------------------

/** Name of the input tensor in the ONNX graph. */
export const ONNX_INPUT_NAME = 'image';

/** Name of the output tensor in the ONNX graph. */
export const ONNX_OUTPUT_NAME = 'log_probs';

// ---------------------------------------------------------------------------
// Derived constants
// ---------------------------------------------------------------------------

/** Total number of pixels in a single input image. */
export const MODEL_INPUT_PIXELS = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;

/** Total number of float32 values in a single input tensor (N=1). */
export const MODEL_INPUT_SIZE = MODEL_CHANNELS * MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;

/** Total number of float32 values in a single output tensor (N=1). */
export const MODEL_OUTPUT_SIZE = MODEL_TIME_STEPS * MODEL_NUM_CLASSES;

// ---------------------------------------------------------------------------
// ONNX opset and metadata
// ---------------------------------------------------------------------------

/** ONNX opset version used during export. */
export const ONNX_OPSET_VERSION = 17;

/**
 * Model version information.
 *
 * This should be updated when retraining or re-exporting the model.
 */
export const MODEL_VERSION = {
  version: '1.0.0',
  trainingRun: 'synthetic_v2_full_50ep',
  exactAccuracy: 0.914,
  charAccuracy: 0.9796,
} as const;

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/**
 * Reshape a flat ONNX output array into [T][C] log-probability matrix.
 *
 * The ONNX model outputs shape [T, N, C]. For single-image inference
 * (N=1), this flattens to T*C values. This helper reshapes them into
 * a 2-D array suitable for CTC decoding.
 *
 * @param flat - Flat Float32Array of length T * C.
 * @param timeSteps - Number of time steps (default: MODEL_TIME_STEPS).
 * @param numClasses - Number of classes (default: MODEL_NUM_CLASSES).
 * @returns 2-D array of shape [T][C].
 * @throws {Error} if flat array length does not match T * C.
 */
export function reshapeOutputToLogProbs(
  flat: Float32Array | number[],
  timeSteps: number = MODEL_TIME_STEPS,
  numClasses: number = MODEL_NUM_CLASSES,
): number[][] {
  const expectedLength = timeSteps * numClasses;
  if (flat.length !== expectedLength) {
    throw new Error(
      `Output array length ${flat.length} does not match expected ` +
      `${timeSteps} × ${numClasses} = ${expectedLength}.`,
    );
  }

  const result: number[][] = [];
  for (let t = 0; t < timeSteps; t++) {
    const row: number[] = [];
    for (let c = 0; c < numClasses; c++) {
      row.push(flat[t * numClasses + c]);
    }
    result.push(row);
  }
  return result;
}
