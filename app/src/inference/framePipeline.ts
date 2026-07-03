/**
 * Camera frame preprocessing pipeline.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 *
 * Chains the individual processing steps into a single function:
 *
 *   CameraFrame (RGBA, arbitrary size)
 *     → grayscale conversion
 *       → ROI crop (or centred aspect-ratio crop)
 *         → bilinear resize to 128×512
 *           → normalise to Float32Array [0,1]
 *             → StaticRecognizerInput (ready for ONNX)
 *
 * Each step is individually testable.  This module composes them into
 * the complete pipeline for use in the live camera loop (Sprint 5F).
 */

import type { CameraFrame } from './cameraFrameProvider';
import type { CropRect } from './cropRegion';
import type { StaticRecognizerInput } from './staticImageRecognizer';
import { rgbaToGrayscale, rgbToGrayscale } from './colorConversion';
import { cropGrayscale, computeCenteredCropRect } from './cropRegion';
import { resizeBilinear } from './imageResizer';
import { prepareGrayscaleInput } from './imagePreprocessor';
import { MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH } from './modelIO';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Configuration for the frame processing pipeline. */
export interface FramePipelineConfig {
  /**
   * Optional explicit crop rectangle.
   *
   * If provided, the frame is cropped to this exact region before
   * resizing.  If omitted, a centered crop with the model's 4:1
   * aspect ratio is computed automatically.
   */
  cropRect?: CropRect;

  /**
   * Target aspect ratio for automatic crop (width / height).
   * Default: 4.0 (= 512 / 128).
   */
  targetAspect?: number;
}

/** Result from the frame processing pipeline. */
export interface FramePipelineResult {
  /** Recognizer-ready input (normalised 128×512 grayscale). */
  input: StaticRecognizerInput;
  /** The crop rectangle that was applied. */
  appliedCrop: CropRect;
  /** Processing time in milliseconds. */
  processingTimeMs: number;
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/**
 * Process a camera frame into model-ready input.
 *
 * Steps:
 *   1. Convert RGBA/RGB → grayscale.
 *   2. Crop to ROI (explicit or auto-centered).
 *   3. Resize to 128×512 via bilinear interpolation.
 *   4. Normalise to Float32Array [0,1].
 *
 * @param frame - Camera frame (RGBA, RGB, or grayscale).
 * @param config - Optional pipeline configuration.
 * @returns Recognizer-ready input + metadata.
 * @throws {Error} if the frame format is unsupported or dimensions are invalid.
 */
export function processFrame(
  frame: CameraFrame,
  config: FramePipelineConfig = {},
): FramePipelineResult {
  const startTime = performance.now();

  // 1. Convert to grayscale.
  let grayscale: Uint8Array;
  switch (frame.format) {
    case 'rgba':
      grayscale = rgbaToGrayscale(frame.data, frame.width, frame.height);
      break;
    case 'rgb':
      grayscale = rgbToGrayscale(frame.data, frame.width, frame.height);
      break;
    case 'grayscale':
      grayscale = frame.data;
      break;
    default:
      throw new Error(
        `Unsupported pixel format: ${frame.format as string}. ` +
        `Expected 'rgba', 'rgb', or 'grayscale'.`,
      );
  }

  // 2. Compute or use explicit crop rectangle.
  const targetAspect = config.targetAspect ?? 4.0;
  const cropRect = config.cropRect ??
    computeCenteredCropRect(frame.width, frame.height, targetAspect);

  // 3. Crop.
  const cropped = cropGrayscale(
    grayscale,
    frame.width,
    frame.height,
    cropRect,
  );

  // 4. Resize to model dimensions (128×512).
  const resized = resizeBilinear(
    cropped,
    cropRect.width,
    cropRect.height,
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT,
  );

  // 5. Normalise to Float32Array [0,1].
  const normalised = prepareGrayscaleInput(
    resized,
    MODEL_INPUT_WIDTH,
    MODEL_INPUT_HEIGHT,
  );

  const processingTimeMs = performance.now() - startTime;

  return {
    input: {
      grayscalePixels: normalised,
      width: MODEL_INPUT_WIDTH,
      height: MODEL_INPUT_HEIGHT,
    },
    appliedCrop: cropRect,
    processingTimeMs,
  };
}

/**
 * Convenience: process frame and return only the recognizer input.
 */
export function processFrameToInput(
  frame: CameraFrame,
  config?: FramePipelineConfig,
): StaticRecognizerInput {
  return processFrame(frame, config).input;
}
