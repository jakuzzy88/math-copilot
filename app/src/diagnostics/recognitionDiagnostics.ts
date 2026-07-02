/**
 * Recognition diagnostics.
 *
 * Collects metrics about how well the recognition pipeline is performing.
 * Sprint 1: placeholder interface only – no actual recognition yet.
 */

export interface RecognitionMetrics {
  /** Number of frames processed */
  framesProcessed: number;
  /** Number of successful recognitions */
  successfulRecognitions: number;
  /** Average confidence score (0-1) */
  averageConfidence: number;
  /** Symbols that were rejected as unsupported */
  rejectedSymbols: string[];
}

/**
 * Create a fresh metrics object.
 */
export function createEmptyMetrics(): RecognitionMetrics {
  return {
    framesProcessed: 0,
    successfulRecognitions: 0,
    averageConfidence: 0,
    rejectedSymbols: [],
  };
}
