/**
 * Synthetic test image generator for offline testing.
 *
 * Sprint 5D: ONNX Runtime Mobile Static Inference.
 *
 * Generates deterministic 128×512 grayscale pixel arrays for testing
 * the inference pipeline without requiring real handwriting images.
 *
 * These images are NOT expected to produce correct OCR results from
 * the real model — they are used to verify the pipeline mechanics
 * (preprocess → infer → decode → candidates) work end-to-end.
 */

import {
  MODEL_INPUT_HEIGHT,
  MODEL_INPUT_WIDTH,
  MODEL_INPUT_PIXELS,
} from './modelIO';

// ---------------------------------------------------------------------------
// Synthetic images
// ---------------------------------------------------------------------------

/**
 * Create an all-white image (background only, no text).
 *
 * The model expects dark text on a light background, so an all-white
 * image should produce all-blank or near-blank CTC output.
 */
export function createWhiteImage(): Uint8Array {
  const pixels = new Uint8Array(MODEL_INPUT_PIXELS);
  pixels.fill(255);
  return pixels;
}

/**
 * Create an all-black image.
 *
 * Solid black is an adversarial input — useful for verifying the
 * pipeline handles degenerate inputs without crashing.
 */
export function createBlackImage(): Uint8Array {
  return new Uint8Array(MODEL_INPUT_PIXELS);
}

/**
 * Create a horizontally-striped pattern image.
 *
 * Alternating rows of white (255) and gray (128), producing a pattern
 * that looks nothing like handwriting but exercises the full pipeline.
 */
export function createStripedImage(): Uint8Array {
  const pixels = new Uint8Array(MODEL_INPUT_PIXELS);
  for (let y = 0; y < MODEL_INPUT_HEIGHT; y++) {
    const val = y % 2 === 0 ? 255 : 128;
    const rowStart = y * MODEL_INPUT_WIDTH;
    for (let x = 0; x < MODEL_INPUT_WIDTH; x++) {
      pixels[rowStart + x] = val;
    }
  }
  return pixels;
}

/**
 * Create a gradient image (left=black, right=white).
 *
 * Useful for verifying normalisation produces a smooth 0→1 ramp.
 */
export function createGradientImage(): Uint8Array {
  const pixels = new Uint8Array(MODEL_INPUT_PIXELS);
  for (let y = 0; y < MODEL_INPUT_HEIGHT; y++) {
    const rowStart = y * MODEL_INPUT_WIDTH;
    for (let x = 0; x < MODEL_INPUT_WIDTH; x++) {
      pixels[rowStart + x] = Math.round((x / (MODEL_INPUT_WIDTH - 1)) * 255);
    }
  }
  return pixels;
}

/**
 * Return the standard image dimensions for test helpers.
 */
export function getTestImageDimensions(): { width: number; height: number } {
  return {
    width: MODEL_INPUT_WIDTH,
    height: MODEL_INPUT_HEIGHT,
  };
}
