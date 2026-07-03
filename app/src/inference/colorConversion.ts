/**
 * Colour-space conversion utilities.
 *
 * Sprint 5E: Camera Frame Capture + Crop + Resize Pipeline.
 *
 * Converts camera frame pixel data from RGBA or RGB to grayscale
 * using the ITU-R BT.601 luminance formula:
 *
 *   Y = 0.299·R + 0.587·G + 0.114·B
 *
 * This matches the Python PIL `Image.convert("L")` behaviour used
 * during model training.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Luminance weight for the red channel (BT.601). */
const LUM_R = 0.299;
/** Luminance weight for the green channel (BT.601). */
const LUM_G = 0.587;
/** Luminance weight for the blue channel (BT.601). */
const LUM_B = 0.114;

// ---------------------------------------------------------------------------
// Conversion functions
// ---------------------------------------------------------------------------

/**
 * Convert an RGBA pixel buffer to grayscale.
 *
 * @param rgba - Flat Uint8Array of RGBA pixels (4 bytes per pixel).
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @returns Uint8Array of grayscale pixel values [0, 255].
 * @throws {Error} if buffer length does not match width × height × 4.
 */
export function rgbaToGrayscale(
  rgba: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const expectedLength = width * height * 4;
  if (rgba.length !== expectedLength) {
    throw new Error(
      `RGBA buffer length ${rgba.length} does not match ` +
      `${width} × ${height} × 4 = ${expectedLength}.`,
    );
  }

  const pixelCount = width * height;
  const gray = new Uint8Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const base = i * 4;
    gray[i] = Math.round(
      LUM_R * rgba[base] +
      LUM_G * rgba[base + 1] +
      LUM_B * rgba[base + 2],
    );
    // Alpha channel (rgba[base + 3]) is ignored.
  }

  return gray;
}

/**
 * Convert an RGB pixel buffer to grayscale.
 *
 * @param rgb - Flat Uint8Array of RGB pixels (3 bytes per pixel).
 * @param width - Image width in pixels.
 * @param height - Image height in pixels.
 * @returns Uint8Array of grayscale pixel values [0, 255].
 * @throws {Error} if buffer length does not match width × height × 3.
 */
export function rgbToGrayscale(
  rgb: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const expectedLength = width * height * 3;
  if (rgb.length !== expectedLength) {
    throw new Error(
      `RGB buffer length ${rgb.length} does not match ` +
      `${width} × ${height} × 3 = ${expectedLength}.`,
    );
  }

  const pixelCount = width * height;
  const gray = new Uint8Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const base = i * 3;
    gray[i] = Math.round(
      LUM_R * rgb[base] +
      LUM_G * rgb[base + 1] +
      LUM_B * rgb[base + 2],
    );
  }

  return gray;
}
