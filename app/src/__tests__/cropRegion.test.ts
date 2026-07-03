/**
 * Tests for ROI cropping.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 */

import {
  cropGrayscale,
  computeCenteredCropRect,
  validateCropRect,
} from '../inference/cropRegion';
import type { CropRect } from '../inference/cropRegion';

// ---------------------------------------------------------------------------
// validateCropRect
// ---------------------------------------------------------------------------

describe('validateCropRect', () => {
  it('accepts a valid crop within bounds', () => {
    expect(() => validateCropRect({ x: 0, y: 0, width: 100, height: 50 }, 100, 50)).not.toThrow();
  });

  it('accepts a small crop inside a large image', () => {
    expect(() => validateCropRect({ x: 10, y: 20, width: 30, height: 40 }, 640, 480)).not.toThrow();
  });

  it('rejects zero width', () => {
    expect(() => validateCropRect({ x: 0, y: 0, width: 0, height: 10 }, 100, 100)).toThrow(/positive/);
  });

  it('rejects negative height', () => {
    expect(() => validateCropRect({ x: 0, y: 0, width: 10, height: -5 }, 100, 100)).toThrow(/positive/);
  });

  it('rejects negative origin x', () => {
    expect(() => validateCropRect({ x: -1, y: 0, width: 10, height: 10 }, 100, 100)).toThrow(/non-negative/);
  });

  it('rejects crop extending beyond width', () => {
    expect(() => validateCropRect({ x: 90, y: 0, width: 20, height: 10 }, 100, 100)).toThrow(/beyond source width/);
  });

  it('rejects crop extending beyond height', () => {
    expect(() => validateCropRect({ x: 0, y: 95, width: 10, height: 10 }, 100, 100)).toThrow(/beyond source height/);
  });
});

// ---------------------------------------------------------------------------
// cropGrayscale
// ---------------------------------------------------------------------------

describe('cropGrayscale', () => {
  it('crops a 2×2 region from a 4×4 image', () => {
    // 4×4 image with row values: row0=[0,1,2,3], row1=[4,5,6,7], etc.
    const src = new Uint8Array([
      0, 1, 2, 3,
      4, 5, 6, 7,
      8, 9, 10, 11,
      12, 13, 14, 15,
    ]);
    const crop = cropGrayscale(src, 4, 4, { x: 1, y: 1, width: 2, height: 2 });
    expect(crop).toEqual(new Uint8Array([5, 6, 9, 10]));
  });

  it('crops the full image when rect covers entire source', () => {
    const src = new Uint8Array([1, 2, 3, 4]);
    const crop = cropGrayscale(src, 2, 2, { x: 0, y: 0, width: 2, height: 2 });
    expect(crop).toEqual(new Uint8Array([1, 2, 3, 4]));
  });

  it('crops a single pixel', () => {
    const src = new Uint8Array([10, 20, 30, 40]);
    const crop = cropGrayscale(src, 2, 2, { x: 1, y: 0, width: 1, height: 1 });
    expect(crop).toEqual(new Uint8Array([20]));
  });

  it('crops the bottom-right corner', () => {
    const src = new Uint8Array([
      0, 1, 2,
      3, 4, 5,
      6, 7, 8,
    ]);
    const crop = cropGrayscale(src, 3, 3, { x: 1, y: 1, width: 2, height: 2 });
    expect(crop).toEqual(new Uint8Array([4, 5, 7, 8]));
  });

  it('crops the first row', () => {
    const src = new Uint8Array([
      10, 20, 30,
      40, 50, 60,
    ]);
    const crop = cropGrayscale(src, 3, 2, { x: 0, y: 0, width: 3, height: 1 });
    expect(crop).toEqual(new Uint8Array([10, 20, 30]));
  });

  it('throws if source buffer length is wrong', () => {
    const src = new Uint8Array(10);
    expect(() => cropGrayscale(src, 4, 4, { x: 0, y: 0, width: 2, height: 2 }))
      .toThrow(/does not match/);
  });

  it('throws if crop extends out of bounds', () => {
    const src = new Uint8Array(16); // 4×4
    expect(() => cropGrayscale(src, 4, 4, { x: 3, y: 3, width: 2, height: 2 }))
      .toThrow(/beyond/);
  });
});

// ---------------------------------------------------------------------------
// computeCenteredCropRect
// ---------------------------------------------------------------------------

describe('computeCenteredCropRect', () => {
  it('returns full image when aspect ratio matches exactly', () => {
    // 800×200 → aspect 4.0 matches target 4.0.
    const rect = computeCenteredCropRect(800, 200, 4.0);
    expect(rect.x).toBe(0);
    expect(rect.y).toBe(0);
    expect(rect.width).toBe(800);
    expect(rect.height).toBe(200);
  });

  it('crops width for a wider-than-target image', () => {
    // 1000×200 → aspect 5.0 > target 4.0 → crop width to 800.
    const rect = computeCenteredCropRect(1000, 200, 4.0);
    expect(rect.width).toBe(800);
    expect(rect.height).toBe(200);
    expect(rect.x).toBe(100); // centered
    expect(rect.y).toBe(0);
  });

  it('crops height for a taller-than-target image', () => {
    // 640×480 → aspect ≈ 1.33 < target 4.0 → crop height.
    const rect = computeCenteredCropRect(640, 480, 4.0);
    expect(rect.width).toBe(640);
    expect(rect.height).toBe(160); // 640/4 = 160
    expect(rect.x).toBe(0);
    expect(rect.y).toBe(160); // (480-160)/2 = 160
  });

  it('handles square image', () => {
    // 400×400 → aspect 1.0 < target 4.0 → crop height.
    const rect = computeCenteredCropRect(400, 400, 4.0);
    expect(rect.width).toBe(400);
    expect(rect.height).toBe(100); // 400/4 = 100
    expect(rect.x).toBe(0);
    expect(rect.y).toBe(150); // (400-100)/2 = 150
  });

  it('uses default aspect ratio of 4.0', () => {
    const rect = computeCenteredCropRect(640, 480);
    expect(Math.abs(rect.width / rect.height - 4.0)).toBeLessThan(0.1);
  });

  it('throws for zero dimensions', () => {
    expect(() => computeCenteredCropRect(0, 480)).toThrow(/positive/);
    expect(() => computeCenteredCropRect(640, 0)).toThrow(/positive/);
  });
});
