/**
 * Tests for colour-space conversion utilities.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 */

import { rgbaToGrayscale, rgbToGrayscale } from '../inference/colorConversion';

describe('rgbaToGrayscale', () => {
  it('converts a single white pixel (255,255,255,255) to 255', () => {
    const rgba = new Uint8Array([255, 255, 255, 255]);
    const gray = rgbaToGrayscale(rgba, 1, 1);
    expect(gray.length).toBe(1);
    expect(gray[0]).toBe(255);
  });

  it('converts a single black pixel (0,0,0,255) to 0', () => {
    const rgba = new Uint8Array([0, 0, 0, 255]);
    const gray = rgbaToGrayscale(rgba, 1, 1);
    expect(gray[0]).toBe(0);
  });

  it('uses BT.601 luminance weights (pure red → 76)', () => {
    // Y = 0.299*255 + 0.587*0 + 0.114*0 ≈ 76.245 → 76
    const rgba = new Uint8Array([255, 0, 0, 255]);
    const gray = rgbaToGrayscale(rgba, 1, 1);
    expect(gray[0]).toBe(76);
  });

  it('uses BT.601 luminance weights (pure green → 150)', () => {
    // Y = 0.299*0 + 0.587*255 + 0.114*0 ≈ 149.685 → 150
    const rgba = new Uint8Array([0, 255, 0, 255]);
    const gray = rgbaToGrayscale(rgba, 1, 1);
    expect(gray[0]).toBe(150);
  });

  it('uses BT.601 luminance weights (pure blue → 29)', () => {
    // Y = 0.299*0 + 0.587*0 + 0.114*255 ≈ 29.07 → 29
    const rgba = new Uint8Array([0, 0, 255, 255]);
    const gray = rgbaToGrayscale(rgba, 1, 1);
    expect(gray[0]).toBe(29);
  });

  it('ignores the alpha channel', () => {
    const rgba1 = new Uint8Array([100, 150, 200, 0]);
    const rgba2 = new Uint8Array([100, 150, 200, 255]);
    const gray1 = rgbaToGrayscale(rgba1, 1, 1);
    const gray2 = rgbaToGrayscale(rgba2, 1, 1);
    expect(gray1[0]).toBe(gray2[0]);
  });

  it('converts a 2×2 image correctly', () => {
    // 4 pixels: R, G, B, white
    const rgba = new Uint8Array([
      255, 0, 0, 255,    // red
      0, 255, 0, 255,    // green
      0, 0, 255, 255,    // blue
      255, 255, 255, 255, // white
    ]);
    const gray = rgbaToGrayscale(rgba, 2, 2);
    expect(gray.length).toBe(4);
    expect(gray[0]).toBe(76);  // red
    expect(gray[1]).toBe(150); // green
    expect(gray[2]).toBe(29);  // blue
    expect(gray[3]).toBe(255); // white
  });

  it('handles a larger image (10×10)', () => {
    const w = 10, h = 10;
    const rgba = new Uint8Array(w * h * 4);
    // Fill with mid-gray (128,128,128).
    for (let i = 0; i < w * h; i++) {
      rgba[i * 4] = 128;
      rgba[i * 4 + 1] = 128;
      rgba[i * 4 + 2] = 128;
      rgba[i * 4 + 3] = 255;
    }
    const gray = rgbaToGrayscale(rgba, w, h);
    expect(gray.length).toBe(100);
    // 0.299*128 + 0.587*128 + 0.114*128 = 128
    expect(gray[0]).toBe(128);
    expect(gray[99]).toBe(128);
  });

  it('throws if buffer length does not match dimensions', () => {
    const rgba = new Uint8Array(100);
    expect(() => rgbaToGrayscale(rgba, 10, 10)).toThrow(/does not match/);
  });
});

describe('rgbToGrayscale', () => {
  it('converts a single white pixel (255,255,255) to 255', () => {
    const rgb = new Uint8Array([255, 255, 255]);
    const gray = rgbToGrayscale(rgb, 1, 1);
    expect(gray[0]).toBe(255);
  });

  it('converts a single black pixel (0,0,0) to 0', () => {
    const rgb = new Uint8Array([0, 0, 0]);
    const gray = rgbToGrayscale(rgb, 1, 1);
    expect(gray[0]).toBe(0);
  });

  it('uses BT.601 luminance weights consistently', () => {
    const rgb = new Uint8Array([255, 0, 0]);
    const gray = rgbToGrayscale(rgb, 1, 1);
    expect(gray[0]).toBe(76); // Same as RGBA test
  });

  it('throws if buffer length does not match dimensions', () => {
    const rgb = new Uint8Array(100);
    expect(() => rgbToGrayscale(rgb, 10, 10)).toThrow(/does not match/);
  });

  it('converts 2×2 image', () => {
    const rgb = new Uint8Array([
      255, 0, 0,       // red
      0, 255, 0,       // green
      0, 0, 255,       // blue
      255, 255, 255,   // white
    ]);
    const gray = rgbToGrayscale(rgb, 2, 2);
    expect(gray.length).toBe(4);
    expect(gray[0]).toBe(76);
    expect(gray[1]).toBe(150);
    expect(gray[2]).toBe(29);
    expect(gray[3]).toBe(255);
  });
});
