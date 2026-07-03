/**
 * Tests for image preprocessor.
 *
 * Sprint 5C: Mobile Static Image Inference Preparation.
 */

import {
  prepareGrayscaleInput,
  isValidInputSize,
  getExpectedPixelCount,
} from '../inference/imagePreprocessor';
import { MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH } from '../inference/modelIO';

const EXPECTED_PIXELS = MODEL_INPUT_HEIGHT * MODEL_INPUT_WIDTH; // 65,536

describe('prepareGrayscaleInput', () => {
  it('accepts exactly 128×512 grayscale pixels (Uint8Array)', () => {
    const pixels = new Uint8Array(EXPECTED_PIXELS);
    pixels.fill(128);
    const result = prepareGrayscaleInput(pixels, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(EXPECTED_PIXELS);
  });

  it('normalises Uint8Array [0,255] to Float32Array [0,1]', () => {
    const pixels = new Uint8Array(EXPECTED_PIXELS);
    // Set a few known values.
    pixels[0] = 0;
    pixels[1] = 128;
    pixels[2] = 255;

    const result = prepareGrayscaleInput(pixels, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    expect(result[0]).toBeCloseTo(0.0, 6);
    expect(result[1]).toBeCloseTo(128 / 255, 6);
    expect(result[2]).toBeCloseTo(1.0, 6);
  });

  it('passes through Float32Array without re-normalising', () => {
    const pixels = new Float32Array(EXPECTED_PIXELS);
    pixels[0] = 0.5;
    pixels[1] = 0.75;

    const result = prepareGrayscaleInput(pixels, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    expect(result[0]).toBe(0.5);
    expect(result[1]).toBe(0.75);
  });

  it('returns a copy of Float32Array (not the same reference)', () => {
    const pixels = new Float32Array(EXPECTED_PIXELS);
    const result = prepareGrayscaleInput(pixels, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    expect(result).not.toBe(pixels);
  });

  it('rejects wrong dimensions (width mismatch)', () => {
    const pixels = new Uint8Array(100 * 100);
    expect(() => prepareGrayscaleInput(pixels, 100, 100)).toThrow(
      /do not match model input/,
    );
  });

  it('rejects wrong dimensions (height mismatch)', () => {
    const pixels = new Uint8Array(MODEL_INPUT_WIDTH * 64);
    expect(() => prepareGrayscaleInput(pixels, MODEL_INPUT_WIDTH, 64)).toThrow(
      /do not match model input/,
    );
  });

  it('rejects pixel array with wrong length', () => {
    // Right dimensions but wrong pixel count.
    const pixels = new Uint8Array(100);
    expect(() =>
      prepareGrayscaleInput(pixels, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT),
    ).toThrow(/does not match/);
  });
});

describe('isValidInputSize', () => {
  it('returns true for correct size', () => {
    expect(isValidInputSize(new Uint8Array(EXPECTED_PIXELS))).toBe(true);
  });

  it('returns false for wrong size', () => {
    expect(isValidInputSize(new Uint8Array(100))).toBe(false);
  });
});

describe('getExpectedPixelCount', () => {
  it('returns 65536 (128 × 512)', () => {
    expect(getExpectedPixelCount()).toBe(65536);
  });
});
