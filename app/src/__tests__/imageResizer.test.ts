/**
 * Tests for bilinear image resizer.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 */

import { resizeBilinear, resizeToModelInput } from '../inference/imageResizer';

// ---------------------------------------------------------------------------
// resizeBilinear
// ---------------------------------------------------------------------------

describe('resizeBilinear', () => {
  it('returns a copy when dimensions are unchanged', () => {
    const src = new Uint8Array([10, 20, 30, 40]);
    const dst = resizeBilinear(src, 2, 2, 2, 2);
    expect(dst).toEqual(new Uint8Array([10, 20, 30, 40]));
    // Must be a copy, not the same reference.
    expect(dst).not.toBe(src);
  });

  it('upscales 1×1 → 2×2 (uniform fill)', () => {
    const src = new Uint8Array([128]);
    const dst = resizeBilinear(src, 1, 1, 2, 2);
    expect(dst.length).toBe(4);
    // All pixels should be 128 (uniform source).
    for (const val of dst) {
      expect(val).toBe(128);
    }
  });

  it('downscales 4×4 → 2×2', () => {
    // Uniform 4×4 → 2×2 should preserve value.
    const src = new Uint8Array(16).fill(200);
    const dst = resizeBilinear(src, 4, 4, 2, 2);
    expect(dst.length).toBe(4);
    for (const val of dst) {
      expect(val).toBe(200);
    }
  });

  it('upscales a 2×1 horizontal gradient', () => {
    // [0, 255] → upscale to 4×1.
    const src = new Uint8Array([0, 255]);
    const dst = resizeBilinear(src, 2, 1, 4, 1);
    expect(dst.length).toBe(4);
    // Should produce a smooth gradient: values increasing left to right.
    expect(dst[0]).toBeLessThan(dst[1]);
    expect(dst[1]).toBeLessThan(dst[2]);
    expect(dst[2]).toBeLessThan(dst[3]);
  });

  it('downscales a 4×1 gradient to 2×1', () => {
    const src = new Uint8Array([0, 85, 170, 255]);
    const dst = resizeBilinear(src, 4, 1, 2, 1);
    expect(dst.length).toBe(2);
    // Left should be darker, right should be lighter.
    expect(dst[0]).toBeLessThan(dst[1]);
  });

  it('preserves uniform image at any size', () => {
    const val = 42;
    const src = new Uint8Array(100).fill(val);
    const dst = resizeBilinear(src, 10, 10, 25, 7);
    expect(dst.length).toBe(175);
    for (const pixel of dst) {
      expect(pixel).toBe(val);
    }
  });

  it('handles extreme downscale (100×100 → 1×1)', () => {
    // Uniform → single pixel should equal source value.
    const src = new Uint8Array(10000).fill(180);
    const dst = resizeBilinear(src, 100, 100, 1, 1);
    expect(dst.length).toBe(1);
    expect(dst[0]).toBe(180);
  });

  it('handles extreme upscale (1×1 → 100×100)', () => {
    const src = new Uint8Array([77]);
    const dst = resizeBilinear(src, 1, 1, 100, 100);
    expect(dst.length).toBe(10000);
    for (const pixel of dst) {
      expect(pixel).toBe(77);
    }
  });

  it('outputs values in [0, 255]', () => {
    // Random-ish source.
    const src = new Uint8Array([0, 255, 128, 64, 192, 32]);
    const dst = resizeBilinear(src, 3, 2, 7, 5);
    for (const pixel of dst) {
      expect(pixel).toBeGreaterThanOrEqual(0);
      expect(pixel).toBeLessThanOrEqual(255);
    }
  });

  it('throws for zero source dimensions', () => {
    expect(() => resizeBilinear(new Uint8Array(0), 0, 0, 10, 10)).toThrow(/positive/);
  });

  it('throws for zero target dimensions', () => {
    expect(() => resizeBilinear(new Uint8Array(4), 2, 2, 0, 2)).toThrow(/positive/);
  });

  it('throws for mismatched source buffer length', () => {
    expect(() => resizeBilinear(new Uint8Array(5), 2, 2, 4, 4)).toThrow(/does not match/);
  });
});

// ---------------------------------------------------------------------------
// resizeToModelInput
// ---------------------------------------------------------------------------

describe('resizeToModelInput', () => {
  it('outputs exactly 128×512 = 65536 pixels', () => {
    const src = new Uint8Array(640 * 160).fill(100);
    const dst = resizeToModelInput(src, 640, 160);
    expect(dst.length).toBe(128 * 512);
  });

  it('preserves uniform value', () => {
    const src = new Uint8Array(200 * 50).fill(99);
    const dst = resizeToModelInput(src, 200, 50);
    for (const pixel of dst) {
      expect(pixel).toBe(99);
    }
  });

  it('works with very small input', () => {
    const src = new Uint8Array([128]);
    const dst = resizeToModelInput(src, 1, 1);
    expect(dst.length).toBe(65536);
    expect(dst[0]).toBe(128);
  });
});
