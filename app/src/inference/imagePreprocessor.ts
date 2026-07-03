/**
 * Image preprocessing for ONNX model inference.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 *
 * Preprocessing pipeline (matches training):
 *   1. Input: grayscale pixels [0,255] (Uint8Array) or [0,1] (Float32Array)
 *   2. Normalise to [0.0, 1.0] by dividing by 255
 *   3. Output: Float32Array of length H*W for ONNX tensor
 *
 * Tensor layout: NCHW = [1, 1, 128, 512], row-major order.
 * Camera frame resizing is NOT implemented (future Sprint 5E).
 */

import { MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH } from './modelIO';

const EXPECTED_PIXEL_COUNT = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH;

/**
 * Prepare a grayscale image for model inference.
 *
 * @param pixels - Grayscale pixel data (row-major, top-left origin).
 * @param width - Must be MODEL_INPUT_WIDTH (512).
 * @param height - Must be MODEL_INPUT_HEIGHT (128).
 * @returns Float32Array of normalised pixel values.
 * @throws {Error} if dimensions or pixel count are wrong.
 */
export function prepareGrayscaleInput(
  pixels: Uint8Array | Float32Array,
  width: number,
  height: number,
): Float32Array {
  if (width !== MODEL_INPUT_WIDTH || height !== MODEL_INPUT_HEIGHT) {
    throw new Error(
      `Image dimensions ${width}×${height} do not match model input ` +
      `${MODEL_INPUT_WIDTH}×${MODEL_INPUT_HEIGHT}. ` +
      `Resize the image before calling prepareGrayscaleInput().`,
    );
  }

  const expectedLength = width * height;
  if (pixels.length !== expectedLength) {
    throw new Error(
      `Pixel array length ${pixels.length} does not match ` +
      `${width} × ${height} = ${expectedLength}.`,
    );
  }

  if (pixels instanceof Float32Array) {
    return new Float32Array(pixels);
  }

  const normalised = new Float32Array(expectedLength);
  for (let i = 0; i < expectedLength; i++) {
    normalised[i] = pixels[i] / 255.0;
  }
  return normalised;
}

/**
 * Validate that a pixel buffer has the correct length for model input.
 */
export function isValidInputSize(pixels: Uint8Array | Float32Array): boolean {
  return pixels.length === EXPECTED_PIXEL_COUNT;
}

/**
 * Get the expected pixel count for model input.
 */
export function getExpectedPixelCount(): number {
  return EXPECTED_PIXEL_COUNT;
}
