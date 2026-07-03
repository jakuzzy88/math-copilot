/**
 * Tests for camera frame provider.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 */

import {
  StaticFrameProvider,
  createSyntheticRgbaFrame,
  createGradientRgbaFrame,
  createTextLikeRgbaFrame,
} from '../inference/cameraFrameProvider';
import type { CameraFrame } from '../inference/cameraFrameProvider';

// ---------------------------------------------------------------------------
// Synthetic frame factories
// ---------------------------------------------------------------------------

describe('createSyntheticRgbaFrame', () => {
  it('creates a frame with default 640×480 dimensions', () => {
    const frame = createSyntheticRgbaFrame();
    expect(frame.width).toBe(640);
    expect(frame.height).toBe(480);
    expect(frame.format).toBe('rgba');
    expect(frame.data.length).toBe(640 * 480 * 4);
  });

  it('creates a frame with custom dimensions', () => {
    const frame = createSyntheticRgbaFrame(320, 240);
    expect(frame.width).toBe(320);
    expect(frame.height).toBe(240);
    expect(frame.data.length).toBe(320 * 240 * 4);
  });

  it('fills with the specified colour', () => {
    const frame = createSyntheticRgbaFrame(2, 2, 100, 150, 200, 255);
    expect(frame.data[0]).toBe(100);  // R
    expect(frame.data[1]).toBe(150);  // G
    expect(frame.data[2]).toBe(200);  // B
    expect(frame.data[3]).toBe(255);  // A
  });

  it('has a valid timestamp', () => {
    const frame = createSyntheticRgbaFrame();
    expect(frame.timestampMs).toBeGreaterThan(0);
  });
});

describe('createGradientRgbaFrame', () => {
  it('creates a gradient from black to white', () => {
    const frame = createGradientRgbaFrame(100, 10);
    // Left edge should be dark (near 0).
    expect(frame.data[0]).toBe(0);    // R of first pixel
    expect(frame.data[1]).toBe(0);    // G of first pixel
    // Right edge should be bright (near 255).
    const lastPixelBase = (10 - 1) * 100 * 4 + (100 - 1) * 4;
    expect(frame.data[lastPixelBase]).toBe(255);
  });
});

describe('createTextLikeRgbaFrame', () => {
  it('has light background and dark middle band', () => {
    const frame = createTextLikeRgbaFrame(100, 100);
    // Top-left corner should be light (background).
    expect(frame.data[0]).toBeGreaterThan(200);
    // Centre pixel should be dark (text band).
    const centreIdx = (50 * 100 + 50) * 4;
    expect(frame.data[centreIdx]).toBeLessThan(50);
  });
});

// ---------------------------------------------------------------------------
// StaticFrameProvider
// ---------------------------------------------------------------------------

describe('StaticFrameProvider', () => {
  it('returns frames in order, cycling', async () => {
    const frame1 = createSyntheticRgbaFrame(10, 10, 100, 100, 100, 255);
    const frame2 = createSyntheticRgbaFrame(10, 10, 200, 200, 200, 255);
    const provider = new StaticFrameProvider([frame1, frame2]);

    const got1 = await provider.captureFrame();
    const got2 = await provider.captureFrame();
    const got3 = await provider.captureFrame(); // wraps around

    expect(got1).toBe(frame1);
    expect(got2).toBe(frame2);
    expect(got3).toBe(frame1); // cycled back

    await provider.dispose();
  });

  it('returns null after dispose', async () => {
    const frame = createSyntheticRgbaFrame(2, 2);
    const provider = new StaticFrameProvider([frame]);

    await provider.dispose();
    const result = await provider.captureFrame();
    expect(result).toBeNull();
  });

  it('isReady returns true initially, false after dispose', async () => {
    const provider = new StaticFrameProvider([createSyntheticRgbaFrame(2, 2)]);
    expect(provider.isReady()).toBe(true);
    await provider.dispose();
    expect(provider.isReady()).toBe(false);
  });

  it('throws if constructed with zero frames', () => {
    expect(() => new StaticFrameProvider([])).toThrow(/at least one/);
  });
});
