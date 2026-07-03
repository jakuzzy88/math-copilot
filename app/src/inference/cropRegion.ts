/**
 * ROI (Region of Interest) cropping from camera frames.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 *
 * Extracts a rectangular region from a flat grayscale pixel buffer.
 * The crop is defined by a bounding box (x, y, width, height) in pixel
 * coordinates.  The source buffer is assumed row-major, top-left origin.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Axis-aligned bounding box in pixel coordinates. */
export interface CropRect {
  /** Left edge (0-indexed, inclusive). */
  x: number;
  /** Top edge (0-indexed, inclusive). */
  y: number;
  /** Width of the crop region. */
  width: number;
  /** Height of the crop region. */
  height: number;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Validate that a crop rect fits within source dimensions.
 *
 * @throws {Error} if the crop rect is out of bounds or has zero/negative size.
 */
export function validateCropRect(
  rect: CropRect,
  srcWidth: number,
  srcHeight: number,
): void {
  if (rect.width <= 0 || rect.height <= 0) {
    throw new Error(
      `Crop dimensions must be positive: got ${rect.width}×${rect.height}.`,
    );
  }

  if (rect.x < 0 || rect.y < 0) {
    throw new Error(
      `Crop origin must be non-negative: got (${rect.x}, ${rect.y}).`,
    );
  }

  if (rect.x + rect.width > srcWidth) {
    throw new Error(
      `Crop extends beyond source width: ` +
      `x=${rect.x} + width=${rect.width} = ${rect.x + rect.width} > ${srcWidth}.`,
    );
  }

  if (rect.y + rect.height > srcHeight) {
    throw new Error(
      `Crop extends beyond source height: ` +
      `y=${rect.y} + height=${rect.height} = ${rect.y + rect.height} > ${srcHeight}.`,
    );
  }
}

// ---------------------------------------------------------------------------
// Cropping
// ---------------------------------------------------------------------------

/**
 * Extract a rectangular region from a grayscale pixel buffer.
 *
 * @param pixels - Source grayscale pixels (row-major, top-left origin).
 * @param srcWidth - Width of the source image.
 * @param srcHeight - Height of the source image.
 * @param rect - Bounding box to extract.
 * @returns New Uint8Array containing only the cropped pixels.
 * @throws {Error} if the source buffer size is wrong or the crop is out of bounds.
 */
export function cropGrayscale(
  pixels: Uint8Array,
  srcWidth: number,
  srcHeight: number,
  rect: CropRect,
): Uint8Array {
  // Validate source dimensions.
  const expectedSrcLength = srcWidth * srcHeight;
  if (pixels.length !== expectedSrcLength) {
    throw new Error(
      `Source pixel array length ${pixels.length} does not match ` +
      `${srcWidth} × ${srcHeight} = ${expectedSrcLength}.`,
    );
  }

  validateCropRect(rect, srcWidth, srcHeight);

  const cropped = new Uint8Array(rect.width * rect.height);
  for (let row = 0; row < rect.height; row++) {
    const srcOffset = (rect.y + row) * srcWidth + rect.x;
    const dstOffset = row * rect.width;
    cropped.set(pixels.subarray(srcOffset, srcOffset + rect.width), dstOffset);
  }

  return cropped;
}

/**
 * Compute a centered crop rect with a target aspect ratio.
 *
 * Useful for extracting the maximum-area region with the model's
 * expected aspect ratio (4:1 for 512×128) from an arbitrary frame.
 *
 * @param srcWidth - Source image width.
 * @param srcHeight - Source image height.
 * @param targetAspect - Desired width / height ratio (default: 4.0 for 512/128).
 * @returns A centered CropRect that fits within the source.
 */
export function computeCenteredCropRect(
  srcWidth: number,
  srcHeight: number,
  targetAspect: number = 4.0,
): CropRect {
  if (srcWidth <= 0 || srcHeight <= 0) {
    throw new Error(
      `Source dimensions must be positive: got ${srcWidth}×${srcHeight}.`,
    );
  }

  const srcAspect = srcWidth / srcHeight;
  let cropWidth: number;
  let cropHeight: number;

  if (srcAspect >= targetAspect) {
    // Source is wider than target — crop width, use full height.
    cropHeight = srcHeight;
    cropWidth = Math.round(cropHeight * targetAspect);
  } else {
    // Source is taller than target — crop height, use full width.
    cropWidth = srcWidth;
    cropHeight = Math.round(cropWidth / targetAspect);
  }

  // Clamp to source bounds (handles rounding).
  cropWidth = Math.min(cropWidth, srcWidth);
  cropHeight = Math.min(cropHeight, srcHeight);

  const x = Math.floor((srcWidth - cropWidth) / 2);
  const y = Math.floor((srcHeight - cropHeight) / 2);

  return { x, y, width: cropWidth, height: cropHeight };
}
