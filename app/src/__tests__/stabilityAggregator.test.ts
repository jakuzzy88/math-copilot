/**
 * Tests for the Stability Aggregator.
 *
 * Sprint E: Prediction Stability Aggregator.
 *
 * Covers:
 *   1.  No stable result with empty history
 *   2.  One accepted frame is not enough
 *   3.  Same equation 3 times becomes stable
 *   4.  Rejected frames do not immediately clear stable result
 *   5.  Different equations compete correctly
 *   6.  Higher frequency beats higher single confidence
 *   7.  Average confidence threshold enforced
 *   8.  Too many rejected frames clears history
 *   9.  Last stable result persists to prevent flicker
 *   10. Configurable window size and threshold work
 */

import {
  StabilityAggregator,
  DEFAULT_STABILITY_CONFIG,
  type StableRecognitionResult,
  type StabilityConfig,
} from '../pipeline/stabilityAggregator';
import type {
  PipelineResult,
  PipelineAcceptedResult,
  PipelineRejectedResult,
} from '../pipeline/types';


// ── Test helpers ────────────────────────────────────────────────────

/**
 * Create a mock accepted pipeline result.
 */
function makeAccepted(
  equation: string,
  solution: string,
  score: number,
): PipelineAcceptedResult {
  return {
    accepted: true,
    equation,
    solution,
    explanationSteps: [],
    sourceCandidate: { text: equation, confidence: score },
    corrections: [],
    score,
  };
}

/**
 * Create a mock rejected pipeline result.
 */
function makeRejected(): PipelineRejectedResult {
  return {
    accepted: false,
    rejection: {
      code: 'ALL_GRAMMAR_INVALID',
      message: 'Mock rejection.',
      candidateErrors: [],
    },
  };
}


// ── Tests ───────────────────────────────────────────────────────────

describe('StabilityAggregator', () => {
  // ── 1. No stable result with empty history ──────────────────────

  it('should return unstable with no frames', () => {
    const agg = new StabilityAggregator();
    const result = agg.peek();

    expect(result.stable).toBe(false);
    expect(result.equation).toBeNull();
    expect(result.solution).toBeNull();
    expect(result.confidence).toBe(0);
    expect(result.reason).toContain('No frames');
    expect(result.history.totalFrames).toBe(0);
  });


  // ── 2. One accepted frame is not enough ─────────────────────────

  it('should not stabilise with only one accepted frame', () => {
    const agg = new StabilityAggregator();
    const result = agg.addFrame(makeAccepted('2x+3=7', 'x=2', 0.9));

    expect(result.stable).toBe(false);
    expect(result.history.acceptedFrames).toBe(1);
  });


  // ── 3. Same equation 3 times becomes stable ────────────────────

  it('should become stable after 3 agreeing accepted frames', () => {
    const agg = new StabilityAggregator();
    agg.addFrame(makeAccepted('2x+3=7', 'x=2', 0.8));
    agg.addFrame(makeAccepted('2x+3=7', 'x=2', 0.85));
    const result = agg.addFrame(makeAccepted('2x+3=7', 'x=2', 0.9));

    expect(result.stable).toBe(true);
    expect(result.equation).toBe('2x+3=7');
    expect(result.solution).toBe('x=2');
    expect(result.confidence).toBeGreaterThanOrEqual(0.65);
    expect(result.history.acceptedFrames).toBe(3);
    expect(result.history.distinctEquations).toBe(1);
  });


  // ── 4. Rejected frames do not immediately clear stable result ──

  it('should retain stable result when rejected frames arrive', () => {
    const agg = new StabilityAggregator();

    // First: establish stability.
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.85));
    const stable = agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.9));
    expect(stable.stable).toBe(true);

    // Now feed a couple of rejected frames.
    // The 3 accepted frames are still in the window (size=8),
    // so the equation remains genuinely stable.
    const r1 = agg.addFrame(makeRejected());
    expect(r1.stable).toBe(true);
    expect(r1.equation).toBe('x+1=3');

    const r2 = agg.addFrame(makeRejected());
    expect(r2.stable).toBe(true);
    expect(r2.equation).toBe('x+1=3');
  });


  // ── 5. Different equations compete correctly ───────────────────

  it('should pick the equation with highest frequency', () => {
    const agg = new StabilityAggregator();

    // 3x "x+1=3" vs 2x "2x=4"
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.7));
    agg.addFrame(makeAccepted('2x=4', 'x=2', 0.85));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.75));
    agg.addFrame(makeAccepted('2x=4', 'x=2', 0.9));
    const result = agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));

    expect(result.stable).toBe(true);
    expect(result.equation).toBe('x+1=3');
    expect(result.history.distinctEquations).toBe(2);
  });


  // ── 6. Higher frequency beats higher single confidence ─────────

  it('should prefer frequency over single high-confidence frame', () => {
    const agg = new StabilityAggregator();

    // "x+1=3" appears 3 times at moderate confidence.
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.7));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.72));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.68));
    // "5x=10" appears once at very high confidence.
    agg.addFrame(makeAccepted('5x=10', 'x=2', 0.99));

    const result = agg.peek();

    expect(result.stable).toBe(true);
    expect(result.equation).toBe('x+1=3');
  });


  // ── 7. Average confidence threshold enforced ───────────────────

  it('should not stabilise when average confidence is below threshold', () => {
    const agg = new StabilityAggregator({ confidenceThreshold: 0.8 });

    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.5));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.55));
    const result = agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.6));

    // Average confidence: ~0.55, threshold 0.8 → not stable.
    expect(result.stable).toBe(false);
    expect(result.reason).toContain('confidence');
    expect(result.reason).toContain('below threshold');
  });


  // ── 8. Too many rejected frames clears history ─────────────────

  it('should clear history after too many consecutive rejections', () => {
    const agg = new StabilityAggregator({ maxConsecutiveRejections: 3 });

    // Establish stability.
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.85));
    const stable = agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.9));
    expect(stable.stable).toBe(true);

    // Feed 3 consecutive rejections → clears history.
    agg.addFrame(makeRejected());
    agg.addFrame(makeRejected());
    const cleared = agg.addFrame(makeRejected());

    expect(cleared.stable).toBe(false);
    expect(cleared.reason).toContain('too many consecutive');
    expect(cleared.history.totalFrames).toBe(0);
  });

  it('should reset consecutive rejection counter on accepted frame', () => {
    const agg = new StabilityAggregator({ maxConsecutiveRejections: 3 });

    // Two rejections, then an accepted frame → no clear.
    agg.addFrame(makeRejected());
    agg.addFrame(makeRejected());
    const result = agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));

    expect(result.history.totalFrames).toBe(3);
    expect(result.history.consecutiveRejections).toBe(0);
  });


  // ── 9. Last stable result persists to prevent flicker ──────────

  it('should return last stable result when current window is insufficient', () => {
    // Use a small window so old frames are evicted quickly.
    const agg = new StabilityAggregator({ windowSize: 3, minAgreement: 3 });

    // Establish stability with 3 frames (fills window).
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.85));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.9));

    const stable = agg.peek();
    expect(stable.stable).toBe(true);
    expect(stable.equation).toBe('x+1=3');

    // Push 2 different equations → old frames evicted.
    agg.addFrame(makeAccepted('3x=9', 'x=3', 0.95));
    const result = agg.addFrame(makeAccepted('5x=10', 'x=2', 0.9));

    // Window: [x+1=3, 3x=9, 5x=10] — no equation meets minAgreement=3.
    // Flicker prevention: retain the last stable result.
    expect(result.stable).toBe(true);
    expect(result.equation).toBe('x+1=3');
    expect(result.reason).toContain('Retaining');
  });

  it('should update stable result when new equation dominates', () => {
    const agg = new StabilityAggregator({ windowSize: 6 });

    // Old equation.
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));

    const stable1 = agg.peek();
    expect(stable1.stable).toBe(true);
    expect(stable1.equation).toBe('x+1=3');

    // New equation takes over.
    agg.addFrame(makeAccepted('3x=9', 'x=3', 0.85));
    agg.addFrame(makeAccepted('3x=9', 'x=3', 0.9));
    const result = agg.addFrame(makeAccepted('3x=9', 'x=3', 0.88));

    // Window now has 3 old + 3 new = tie on frequency.
    // Tie-break: newest frame wins → 3x=9.
    expect(result.stable).toBe(true);
    expect(result.equation).toBe('3x=9');
    expect(result.solution).toBe('x=3');
  });


  // ── 10. Configurable window size and threshold work ────────────

  it('should respect custom window size', () => {
    const agg = new StabilityAggregator({ windowSize: 3, minAgreement: 2 });

    agg.addFrame(makeAccepted('x=5', 'x=5', 0.9));
    const result = agg.addFrame(makeAccepted('x=5', 'x=5', 0.85));

    // minAgreement = 2, we have 2 → stable.
    expect(result.stable).toBe(true);
    expect(result.equation).toBe('x=5');
  });

  it('should respect custom minAgreement', () => {
    const agg = new StabilityAggregator({ minAgreement: 5 });

    // 4 frames — not enough with minAgreement=5.
    for (let i = 0; i < 4; i++) {
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    }
    const r4 = agg.peek();
    expect(r4.stable).toBe(false);

    // 5th frame → now stable.
    const r5 = agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
    expect(r5.stable).toBe(true);
  });

  it('should enforce default config values', () => {
    expect(DEFAULT_STABILITY_CONFIG.windowSize).toBe(8);
    expect(DEFAULT_STABILITY_CONFIG.minAgreement).toBe(3);
    expect(DEFAULT_STABILITY_CONFIG.confidenceThreshold).toBe(0.65);
    expect(DEFAULT_STABILITY_CONFIG.maxConsecutiveRejections).toBe(5);
  });


  // ── Additional edge cases ──────────────────────────────────────

  describe('reset()', () => {
    it('should clear all state including last stable result', () => {
      const agg = new StabilityAggregator();

      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.85));
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.9));

      const stable = agg.peek();
      expect(stable.stable).toBe(true);

      agg.reset();

      const after = agg.peek();
      expect(after.stable).toBe(false);
      expect(after.equation).toBeNull();
      expect(after.history.totalFrames).toBe(0);
    });
  });

  describe('sliding window eviction', () => {
    it('should evict old frames when window is full', () => {
      const agg = new StabilityAggregator({ windowSize: 4, minAgreement: 3 });

      // 3 frames of "x+1=3" → stable.
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
      expect(agg.peek().stable).toBe(true);

      // 2 more frames of different equation → old ones evicted.
      agg.addFrame(makeAccepted('2x=6', 'x=3', 0.85));
      agg.addFrame(makeAccepted('2x=6', 'x=3', 0.85));

      // Window: [x+1=3, 2x=6, 2x=6] — "x+1=3" has 1, "2x=6" has 2.
      // Neither meets minAgreement=3, but last stable result persists.
      const result = agg.peek();
      expect(result.stable).toBe(true);
      expect(result.equation).toBe('x+1=3');
      expect(result.reason).toContain('Retaining');
    });
  });

  describe('history summary', () => {
    it('should accurately report history metrics', () => {
      const agg = new StabilityAggregator();

      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.8));
      agg.addFrame(makeRejected());
      agg.addFrame(makeAccepted('2x=4', 'x=2', 0.9));
      agg.addFrame(makeAccepted('x+1=3', 'x=2', 0.85));

      const summary = agg.getHistorySummary();
      expect(summary.totalFrames).toBe(4);
      expect(summary.acceptedFrames).toBe(3);
      expect(summary.rejectedFrames).toBe(1);
      expect(summary.distinctEquations).toBe(2);
      expect(summary.consecutiveRejections).toBe(0);
    });
  });
});
