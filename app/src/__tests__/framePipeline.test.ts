/**
 * Tests for the full frame processing pipeline.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 *
 * Tests the complete chain:
 *   CameraFrame (RGBA) → grayscale → crop → resize → normalise → StaticRecognizerInput.
 */

import { processFrame, processFrameToInput } from '../inference/framePipeline';
import type { FramePipelineConfig } from '../inference/framePipeline';
import {
  createSyntheticRgbaFrame,
  createGradientRgbaFrame,
  createTextLikeRgbaFrame,
} from '../inference/cameraFrameProvider';
import type { CameraFrame } from '../inference/cameraFrameProvider';
import { MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH } from '../inference/modelIO';

// ---------------------------------------------------------------------------
// Pipeline output shape
// ---------------------------------------------------------------------------

describe('processFrame — output shape', () => {
  it('produces 128×512 Float32Array from a 640×480 RGBA frame', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const result = processFrame(frame);

    expect(result.input.width).toBe(MODEL_INPUT_WIDTH);
    expect(result.input.height).toBe(MODEL_INPUT_HEIGHT);
    expect(result.input.grayscalePixels.length).toBe(
      MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT,
    );
    expect(result.input.grayscalePixels).toBeInstanceOf(Float32Array);
  });

  it('produces correct output from a 1920×1080 frame', () => {
    const frame = createSyntheticRgbaFrame(1920, 1080);
    const result = processFrame(frame);

    expect(result.input.width).toBe(512);
    expect(result.input.height).toBe(128);
    expect(result.input.grayscalePixels.length).toBe(65536);
  });

  it('produces correct output from a small 100×100 frame', () => {
    const frame = createSyntheticRgbaFrame(100, 100);
    const result = processFrame(frame);

    expect(result.input.width).toBe(512);
    expect(result.input.height).toBe(128);
  });
});

// ---------------------------------------------------------------------------
// Automatic crop
// ---------------------------------------------------------------------------

describe('processFrame — auto crop', () => {
  it('computes a centered 4:1 crop from 640×480', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const result = processFrame(frame);

    const crop = result.appliedCrop;
    expect(crop.width).toBe(640);
    // 640/4 = 160
    expect(crop.height).toBe(160);
    // Centered: (480-160)/2 = 160
    expect(crop.y).toBe(160);
    expect(crop.x).toBe(0);
  });

  it('uses full image when aspect ratio matches', () => {
    // 800×200 = exactly 4:1.
    const frame = createSyntheticRgbaFrame(800, 200);
    const result = processFrame(frame);

    expect(result.appliedCrop.x).toBe(0);
    expect(result.appliedCrop.y).toBe(0);
    expect(result.appliedCrop.width).toBe(800);
    expect(result.appliedCrop.height).toBe(200);
  });

  it('uses custom target aspect ratio', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const config: FramePipelineConfig = { targetAspect: 2.0 };
    const result = processFrame(frame, config);

    // width=640, aspect 2.0 → height = 640/2 = 320.
    expect(result.appliedCrop.width).toBe(640);
    expect(result.appliedCrop.height).toBe(320);
  });
});

// ---------------------------------------------------------------------------
// Explicit crop
// ---------------------------------------------------------------------------

describe('processFrame — explicit crop', () => {
  it('uses provided crop rect', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const config: FramePipelineConfig = {
      cropRect: { x: 50, y: 100, width: 400, height: 100 },
    };
    const result = processFrame(frame, config);

    expect(result.appliedCrop).toEqual({ x: 50, y: 100, width: 400, height: 100 });
    expect(result.input.width).toBe(512);
    expect(result.input.height).toBe(128);
  });

  it('throws if explicit crop is out of bounds', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const config: FramePipelineConfig = {
      cropRect: { x: 600, y: 0, width: 100, height: 100 },
    };
    expect(() => processFrame(frame, config)).toThrow(/beyond/);
  });
});

// ---------------------------------------------------------------------------
// Pixel value integrity
// ---------------------------------------------------------------------------

describe('processFrame — pixel values', () => {
  it('normalises pixel values to [0, 1]', () => {
    const frame = createSyntheticRgbaFrame(800, 200, 128, 128, 128, 255);
    const result = processFrame(frame);

    const pixels = result.input.grayscalePixels as Float32Array;
    for (let i = 0; i < pixels.length; i++) {
      expect(pixels[i]).toBeGreaterThanOrEqual(0.0);
      expect(pixels[i]).toBeLessThanOrEqual(1.0);
    }
  });

  it('uniform gray frame produces uniform normalised output', () => {
    const frame = createSyntheticRgbaFrame(800, 200, 128, 128, 128, 255);
    const result = processFrame(frame);

    const pixels = result.input.grayscalePixels as Float32Array;
    // All pixels should be close to 128/255 ≈ 0.502.
    const expected = 128 / 255;
    for (let i = 0; i < Math.min(10, pixels.length); i++) {
      expect(pixels[i]).toBeCloseTo(expected, 1);
    }
  });

  it('white frame produces values near 1.0', () => {
    const frame = createSyntheticRgbaFrame(800, 200, 255, 255, 255, 255);
    const result = processFrame(frame);

    const pixels = result.input.grayscalePixels as Float32Array;
    expect(pixels[0]).toBeCloseTo(1.0, 2);
  });

  it('black frame produces values near 0.0', () => {
    const frame = createSyntheticRgbaFrame(800, 200, 0, 0, 0, 255);
    const result = processFrame(frame);

    const pixels = result.input.grayscalePixels as Float32Array;
    expect(pixels[0]).toBeCloseTo(0.0, 2);
  });
});

// ---------------------------------------------------------------------------
// Grayscale format passthrough
// ---------------------------------------------------------------------------

describe('processFrame — grayscale input', () => {
  it('handles already-grayscale frames', () => {
    const grayData = new Uint8Array(640 * 480);
    grayData.fill(100);
    const frame: CameraFrame = {
      data: grayData,
      width: 640,
      height: 480,
      format: 'grayscale',
      timestampMs: Date.now(),
    };

    const result = processFrame(frame);
    expect(result.input.width).toBe(512);
    expect(result.input.height).toBe(128);
  });
});

// ---------------------------------------------------------------------------
// RGB format
// ---------------------------------------------------------------------------

describe('processFrame — RGB input', () => {
  it('handles RGB frames', () => {
    const rgbData = new Uint8Array(100 * 100 * 3);
    rgbData.fill(128);
    const frame: CameraFrame = {
      data: rgbData,
      width: 100,
      height: 100,
      format: 'rgb',
      timestampMs: Date.now(),
    };

    const result = processFrame(frame);
    expect(result.input.width).toBe(512);
    expect(result.input.height).toBe(128);
  });
});

// ---------------------------------------------------------------------------
// Processing metadata
// ---------------------------------------------------------------------------

describe('processFrame — metadata', () => {
  it('reports processing time', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const result = processFrame(frame);

    expect(typeof result.processingTimeMs).toBe('number');
    expect(result.processingTimeMs).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// processFrameToInput convenience
// ---------------------------------------------------------------------------

describe('processFrameToInput', () => {
  it('returns only the StaticRecognizerInput', () => {
    const frame = createSyntheticRgbaFrame(640, 480);
    const input = processFrameToInput(frame);

    expect(input.width).toBe(512);
    expect(input.height).toBe(128);
    expect(input.grayscalePixels).toBeInstanceOf(Float32Array);
    expect(input.grayscalePixels.length).toBe(65536);
  });
});

// ---------------------------------------------------------------------------
// Synthetic frame variety
// ---------------------------------------------------------------------------

describe('processFrame — synthetic frame types', () => {
  it('processes gradient frame', () => {
    const frame = createGradientRgbaFrame(640, 480);
    const result = processFrame(frame);
    expect(result.input.grayscalePixels.length).toBe(65536);
  });

  it('processes text-like frame', () => {
    const frame = createTextLikeRgbaFrame(640, 480);
    const result = processFrame(frame);
    expect(result.input.grayscalePixels.length).toBe(65536);

    // The text-like frame has dark centre — check that some pixels
    // are dark and some are light.
    const pixels = result.input.grayscalePixels as Float32Array;
    let hasDark = false;
    let hasLight = false;
    for (let i = 0; i < pixels.length; i++) {
      if (pixels[i] < 0.3) hasDark = true;
      if (pixels[i] > 0.7) hasLight = true;
    }
    expect(hasDark).toBe(true);
    expect(hasLight).toBe(true);
  });
});
