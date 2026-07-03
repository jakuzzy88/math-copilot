/**
 * Bilinear image resizer for grayscale images.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 *
 * Implements bilinear interpolation to resize a grayscale image to
 * arbitrary target dimensions.  This matches the resize behaviour of
 * `PIL.Image.resize((w, h), Image.BILINEAR)` used during training.
 *
 * The implementation is pure TypeScript — no native dependencies — so
 * it works identically in Jest and on-device.
 */

// ---------------------------------------------------------------------------
// Bilinear resize
// ---------------------------------------------------------------------------

/**
 * Resize a grayscale image using bilinear interpolation.
 *
 * @param src - Source grayscale pixels (Uint8Array, row-major).
 * @param srcWidth - Width of the source image.
 * @param srcHeight - Height of the source image.
 * @param dstWidth - Desired output width.
 * @param dstHeight - Desired output height.
 * @returns New Uint8Array of size dstWidth × dstHeight.
 * @throws {Error} if source dimensions are invalid.
 */
export function resizeBilinear(
  src: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number,
): Uint8Array {
  // ---- Validation ----

  if (srcWidth <= 0 || srcHeight <= 0) {
    throw new Error(
      `Source dimensions must be positive: got ${srcWidth}×${srcHeight}.`,
    );
  }

  if (dstWidth <= 0 || dstHeight <= 0) {
    throw new Error(
      `Target dimensions must be positive: got ${dstWidth}×${dstHeight}.`,
    );
  }

  const expectedSrcLength = srcWidth * srcHeight;
  if (src.length !== expectedSrcLength) {
    throw new Error(
      `Source pixel array length ${src.length} does not match ` +
      `${srcWidth} × ${srcHeight} = ${expectedSrcLength}.`,
    );
  }

  // ---- No-op fast path ----

  if (srcWidth === dstWidth && srcHeight === dstHeight) {
    return new Uint8Array(src);
  }

  // ---- Bilinear interpolation ----

  const dst = new Uint8Array(dstWidth * dstHeight);

  // Scale factors: map destination pixel centre to source coordinate.
  // Uses the "align corners = false" convention (consistent with PIL).
  const scaleX = srcWidth / dstWidth;
  const scaleY = srcHeight / dstHeight;

  for (let dy = 0; dy < dstHeight; dy++) {
    // Source y coordinate (centre of destination pixel).
    const sy = (dy + 0.5) * scaleY - 0.5;
    const sy0 = Math.floor(sy);
    const sy1 = Math.min(sy0 + 1, srcHeight - 1);
    const fy = sy - sy0;
    const sy0Clamped = Math.max(sy0, 0);

    for (let dx = 0; dx < dstWidth; dx++) {
      // Source x coordinate (centre of destination pixel).
      const sx = (dx + 0.5) * scaleX - 0.5;
      const sx0 = Math.floor(sx);
      const sx1 = Math.min(sx0 + 1, srcWidth - 1);
      const fx = sx - sx0;
      const sx0Clamped = Math.max(sx0, 0);

      // Four neighbours.
      const topLeft = src[sy0Clamped * srcWidth + sx0Clamped];
      const topRight = src[sy0Clamped * srcWidth + sx1];
      const bottomLeft = src[sy1 * srcWidth + sx0Clamped];
      const bottomRight = src[sy1 * srcWidth + sx1];

      // Bilinear blend.
      const top = topLeft + fx * (topRight - topLeft);
      const bottom = bottomLeft + fx * (bottomRight - bottomLeft);
      const value = top + fy * (bottom - top);

      dst[dy * dstWidth + dx] = Math.round(Math.max(0, Math.min(255, value)));
    }
  }

  return dst;
}

/**
 * Resize a grayscale image to the model's expected input dimensions.
 *
 * Convenience wrapper around `resizeBilinear` that targets 128×512.
 *
 * @param src - Source grayscale pixels.
 * @param srcWidth - Source width.
 * @param srcHeight - Source height.
 * @returns Resized Uint8Array of exactly 128 × 512 = 65,536 pixels.
 */
export function resizeToModelInput(
  src: Uint8Array,
  srcWidth: number,
  srcHeight: number,
): Uint8Array {
  // Import avoided to prevent circular deps — constants inlined.
  const MODEL_W = 512;
  const MODEL_H = 128;
  return resizeBilinear(src, srcWidth, srcHeight, MODEL_W, MODEL_H);
}
