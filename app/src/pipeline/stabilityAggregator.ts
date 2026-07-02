/**
 * Stability Aggregator — prediction stability layer.
 *
 * Sprint E: Prediction Stability Aggregator.
 *
 * Receives accepted OCR pipeline results over multiple frames and
 * only emits a stable equation when enough recent frames agree.
 *
 * Design goals:
 *   - Deterministic: same sequence of inputs always produces same output.
 *   - Flicker-free: last stable result persists until clearly replaced.
 *   - Configurable: window size, agreement threshold, confidence floor.
 *   - Rejection-aware: too many consecutive rejections decay history.
 *
 * This module does NOT duplicate solver logic — it consumes
 * PipelineResult objects from processCandidates().
 */

import type {
  PipelineResult,
  PipelineAcceptedResult,
} from './types';


// ── Configuration ───────────────────────────────────────────────────

/** Configuration for the stability aggregator. */
export interface StabilityConfig {
  /** Number of recent frame results to keep. Default: 8. */
  windowSize: number;
  /** Minimum occurrences of the same equation to be considered stable. Default: 3. */
  minAgreement: number;
  /** Minimum average confidence across agreeing frames. Default: 0.65. */
  confidenceThreshold: number;
  /**
   * Maximum consecutive rejected frames before history is cleared.
   * Default: 5.
   */
  maxConsecutiveRejections: number;
}

/** Default configuration values. */
export const DEFAULT_STABILITY_CONFIG: StabilityConfig = {
  windowSize: 8,
  minAgreement: 3,
  confidenceThreshold: 0.65,
  maxConsecutiveRejections: 5,
};


// ── Output type ─────────────────────────────────────────────────────

/** Summary of recent frame history for diagnostics. */
export interface HistorySummary {
  /** Total frames in the current window. */
  totalFrames: number;
  /** Number of accepted frames in the window. */
  acceptedFrames: number;
  /** Number of rejected frames in the window. */
  rejectedFrames: number;
  /** Distinct equations seen (accepted only). */
  distinctEquations: number;
  /** Current consecutive rejection streak. */
  consecutiveRejections: number;
}

/** Stable recognition result emitted by the aggregator. */
export interface StableRecognitionResult {
  /** Whether the aggregator considers the current equation stable. */
  stable: boolean;
  /** The stable equation string, if stable. */
  equation: string | null;
  /** The solution string (e.g. "x=2"), if stable. */
  solution: string | null;
  /** Average confidence across agreeing frames, or 0. */
  confidence: number;
  /** Human-readable reason for the stability decision. */
  reason: string;
  /** Summary of recent history for diagnostics. */
  history: HistorySummary;
}


// ── Internal frame entry ────────────────────────────────────────────

/** Internal record for a single frame in the sliding window. */
interface FrameEntry {
  /** Monotonic frame index (for ordering / tie-breaking). */
  frameIndex: number;
  /** The pipeline result for this frame. */
  result: PipelineResult;
}


// ── Aggregator class ────────────────────────────────────────────────

/**
 * Stateful stability aggregator.
 *
 * Usage:
 *   const agg = new StabilityAggregator();
 *   const stable = agg.addFrame(pipelineResult);
 *   if (stable.stable) { ... }
 */
export class StabilityAggregator {
  private readonly config: StabilityConfig;
  private readonly window: FrameEntry[] = [];
  private frameCounter = 0;
  private consecutiveRejections = 0;
  private lastStableResult: StableRecognitionResult | null = null;

  constructor(config: Partial<StabilityConfig> = {}) {
    this.config = { ...DEFAULT_STABILITY_CONFIG, ...config };
  }

  // ── Public API ──────────────────────────────────────────────────

  /**
   * Add a new frame result and return the current stability assessment.
   *
   * This is the primary entry point. Call once per frame.
   */
  addFrame(result: PipelineResult): StableRecognitionResult {
    this.frameCounter++;

    // Track consecutive rejections.
    if (result.accepted) {
      this.consecutiveRejections = 0;
    } else {
      this.consecutiveRejections++;
    }

    // Too many consecutive rejections → clear history.
    if (this.consecutiveRejections >= this.config.maxConsecutiveRejections) {
      this.clearHistory();
      return this.buildUnstable(
        'History cleared: too many consecutive rejected frames.',
      );
    }

    // Add to sliding window.
    this.window.push({ frameIndex: this.frameCounter, result });

    // Trim window to configured size.
    while (this.window.length > this.config.windowSize) {
      this.window.shift();
    }

    // Evaluate stability.
    return this.evaluate();
  }

  /**
   * Return the current stability assessment without adding a frame.
   * Useful for polling the current state.
   */
  peek(): StableRecognitionResult {
    if (this.window.length === 0) {
      return this.buildUnstable('No frames received yet.');
    }
    return this.evaluate();
  }

  /** Clear all history and reset state. */
  reset(): void {
    this.clearHistory();
    this.lastStableResult = null;
  }

  /** Get the current history summary. */
  getHistorySummary(): HistorySummary {
    return this.buildHistorySummary();
  }

  // ── Core evaluation ─────────────────────────────────────────────

  private evaluate(): StableRecognitionResult {
    const accepted = this.getAcceptedEntries();

    if (accepted.length === 0) {
      return this.retainOrUnstable(
        'No accepted frames in current window.',
      );
    }

    // Group accepted frames by equation.
    const groups = this.groupByEquation(accepted);

    // Find the best group: highest frequency, then newest strongest frame.
    const bestGroup = this.selectBestGroup(groups);

    if (!bestGroup) {
      return this.retainOrUnstable(
        'No equation group meets agreement threshold.',
      );
    }

    const { equation, entries } = bestGroup;
    const count = entries.length;

    // Check agreement count.
    if (count < this.config.minAgreement) {
      return this.retainOrUnstable(
        `Equation "${equation}" seen ${count} time(s), need ${this.config.minAgreement}.`,
      );
    }

    // Check average confidence.
    const avgConfidence = this.averageConfidence(entries);
    if (avgConfidence < this.config.confidenceThreshold) {
      return this.retainOrUnstable(
        `Average confidence ${avgConfidence.toFixed(3)} below threshold ${this.config.confidenceThreshold}.`,
      );
    }

    // Find the newest entry in the winning group for the solution.
    const newest = entries[entries.length - 1];
    const newestResult = newest.result as PipelineAcceptedResult;

    const stableResult: StableRecognitionResult = {
      stable: true,
      equation,
      solution: newestResult.solution,
      confidence: avgConfidence,
      reason: `Equation "${equation}" stable: ${count}/${this.config.minAgreement} agreement, avg confidence ${avgConfidence.toFixed(3)}.`,
      history: this.buildHistorySummary(),
    };

    this.lastStableResult = stableResult;
    return stableResult;
  }

  // ── Helpers ─────────────────────────────────────────────────────

  /** Get accepted entries from the window, sorted by frameIndex. */
  private getAcceptedEntries(): FrameEntry[] {
    return this.window.filter((e) => e.result.accepted);
  }

  /** Group accepted entries by their equation string. */
  private groupByEquation(
    entries: FrameEntry[],
  ): Map<string, FrameEntry[]> {
    const groups = new Map<string, FrameEntry[]>();
    for (const entry of entries) {
      const result = entry.result as PipelineAcceptedResult;
      const eq = result.equation;
      const group = groups.get(eq);
      if (group) {
        group.push(entry);
      } else {
        groups.set(eq, [entry]);
      }
    }
    return groups;
  }

  /**
   * Select the best equation group.
   *
   * Priority:
   *   1. Highest frequency (count of frames).
   *   2. Tie-break: prefer the group whose newest frame is more recent.
   *
   * Returns null if no group meets the minimum agreement threshold.
   */
  private selectBestGroup(
    groups: Map<string, FrameEntry[]>,
  ): { equation: string; entries: FrameEntry[] } | null {
    let best: { equation: string; entries: FrameEntry[] } | null = null;

    for (const [equation, entries] of groups) {
      if (!best) {
        best = { equation, entries };
        continue;
      }

      // Higher frequency wins.
      if (entries.length > best.entries.length) {
        best = { equation, entries };
        continue;
      }

      // Same frequency → prefer the one with the newest frame.
      if (entries.length === best.entries.length) {
        const newestThis = entries[entries.length - 1].frameIndex;
        const newestBest = best.entries[best.entries.length - 1].frameIndex;
        if (newestThis > newestBest) {
          best = { equation, entries };
        }
      }
    }

    return best;
  }

  /** Compute average confidence of a set of accepted entries. */
  private averageConfidence(entries: FrameEntry[]): number {
    if (entries.length === 0) return 0;
    const sum = entries.reduce((acc, e) => {
      const result = e.result as PipelineAcceptedResult;
      return acc + result.score;
    }, 0);
    return sum / entries.length;
  }

  /**
   * Return the last stable result if available (flicker prevention),
   * otherwise return an unstable result.
   */
  private retainOrUnstable(reason: string): StableRecognitionResult {
    if (this.lastStableResult) {
      return {
        ...this.lastStableResult,
        reason: `Retaining previous stable result. Current: ${reason}`,
        history: this.buildHistorySummary(),
      };
    }
    return this.buildUnstable(reason);
  }

  /** Build an unstable result. */
  private buildUnstable(reason: string): StableRecognitionResult {
    return {
      stable: false,
      equation: null,
      solution: null,
      confidence: 0,
      reason,
      history: this.buildHistorySummary(),
    };
  }

  /** Build a history summary snapshot. */
  private buildHistorySummary(): HistorySummary {
    const accepted = this.window.filter((e) => e.result.accepted);
    const rejected = this.window.filter((e) => !e.result.accepted);

    const equationSet = new Set<string>();
    for (const entry of accepted) {
      const result = entry.result as PipelineAcceptedResult;
      equationSet.add(result.equation);
    }

    return {
      totalFrames: this.window.length,
      acceptedFrames: accepted.length,
      rejectedFrames: rejected.length,
      distinctEquations: equationSet.size,
      consecutiveRejections: this.consecutiveRejections,
    };
  }

  /** Clear the sliding window and reset the consecutive rejection counter. */
  private clearHistory(): void {
    this.window.length = 0;
    this.consecutiveRejections = 0;
  }
}
