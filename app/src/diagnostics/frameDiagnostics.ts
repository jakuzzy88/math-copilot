/**
 * Frame diagnostics.
 *
 * Tracks per-frame processing metrics for the camera pipeline.
 * Sprint 1: placeholder interface only.
 */

export interface FrameMetrics {
  /** Frame sequence number */
  frameNumber: number;
  /** Time taken to process this frame (ms) */
  processingTimeMs: number;
  /** Whether recognition was attempted on this frame */
  recognitionAttempted: boolean;
  /** Whether a valid equation was detected */
  equationDetected: boolean;
}

/**
 * Create a frame metrics entry.
 */
export function createFrameMetrics(frameNumber: number): FrameMetrics {
  return {
    frameNumber,
    processingTimeMs: 0,
    recognitionAttempted: false,
    equationDetected: false,
  };
}
