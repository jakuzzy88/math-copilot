/**
 * Recognition overlay component tests.
 *
 * Sprint 6A: UI Overlay Integration.
 */

import { computeGuideBoxStyle, renderGuideBoxOverlay } from '../ui/components/GuideBoxOverlay';
import { renderSolutionCard, getConfidenceLevel, getSolutionCardBackground } from '../ui/components/SolutionCard';
import { buildDiagnosticsRows, renderDiagnosticsPanel } from '../ui/components/DiagnosticsPanel';
import { renderRecognitionOverlay } from '../ui/components/RecognitionOverlay';
import type { LiveRecognitionDiagnostics } from '../inference/liveRecognitionController';

function makeDiag(o: Partial<LiveRecognitionDiagnostics> = {}): LiveRecognitionDiagnostics {
  return { framesSeen:0, framesProcessed:0, framesSkippedBusy:0, framesFailedPreprocessing:0, framesFailedRecognition:0, framesRejectedByPipeline:0, stableResultsEmitted:0, lastRawText:null, lastStableEquation:null, averagePreprocessMs:0, averageRecognitionMs:0, averageTotalMs:0, ...o };
}

describe('GuideBoxOverlay', () => {
  it('active style has green border', () => {
    const s = computeGuideBoxStyle({ isActive: true });
    expect(s.borderColor).toBe('#00E676');
    expect(s.opacity).toBe(1);
    expect(s.aspectRatio).toBe(4);
  });
  it('inactive style is dimmed', () => {
    const s = computeGuideBoxStyle({ isActive: false });
    expect(s.opacity).toBe(0.6);
  });
  it('custom border color', () => {
    expect(computeGuideBoxStyle({ isActive: true, borderColor: '#F00' }).borderColor).toBe('#F00');
  });
  it('render returns correct type', () => {
    expect(renderGuideBoxOverlay({ isActive: true }).type).toBe('GuideBoxOverlay');
    expect(renderGuideBoxOverlay({ isActive: true }).accessibilityLabel).toContain('scanning');
    expect(renderGuideBoxOverlay({ isActive: false }).accessibilityLabel).toContain('idle');
  });
});

describe('SolutionCard', () => {
  it('confidence levels', () => {
    expect(getConfidenceLevel(0.9)).toBe('high');
    expect(getConfidenceLevel(0.7)).toBe('medium');
    expect(getConfidenceLevel(0.3)).toBe('low');
    expect(getConfidenceLevel(0)).toBe('none');
  });
  it('background colors', () => {
    expect(getSolutionCardBackground('error', 0)).toContain('244');
    expect(getSolutionCardBackground('stable', 0.9)).toContain('200');
    expect(getSolutionCardBackground('uncertain', 0.5)).toContain('255');
  });
  it('renders stable equation props', () => {
    const d = renderSolutionCard({ mode:'stable', equation:'3x+4=10', solution:'x=2', confidence:0.91, stepTitle:'Sub 4', stepText:'Remove +4', statusMessage:'OK' });
    expect(d.visible).toBe(true);
    expect(d.equationText).toBe('3x+4=10');
    expect(d.solutionText).toBe('x=2');
    expect(d.confidenceLevel).toBe('high');
    expect(d.accessibilityLabel).toContain('3x+4=10');
  });
  it('not visible in idle', () => {
    expect(renderSolutionCard({ mode:'idle', equation:null, solution:null, confidence:0, stepTitle:null, stepText:null, statusMessage:'' }).visible).toBe(false);
  });
});

describe('DiagnosticsPanel', () => {
  it('rows include key fields', () => {
    const rows = buildDiagnosticsRows(makeDiag({ framesProcessed:42, framesSkippedBusy:3, framesRejectedByPipeline:5, lastRawText:'3x+4=10', averageTotalMs:14.8 }));
    const find = (l: string) => rows.find(r => r.label === l);
    expect(find('Frames Processed')?.value).toBe('42');
    expect(find('Frames Skipped (Busy)')?.value).toBe('3');
    expect(find('Frames Skipped (Busy)')?.isWarning).toBe(true);
    expect(find('Avg Total')?.value).toBe('14.8ms');
  });
  it('collapsed panel has empty rows', () => {
    const d = renderDiagnosticsPanel({ isExpanded: false, diagnostics: makeDiag() });
    expect(d.rows.length).toBe(0);
    expect(d.toggleLabel).toBe('Show Diagnostics');
  });
  it('expanded panel has rows', () => {
    const d = renderDiagnosticsPanel({ isExpanded: true, diagnostics: makeDiag() });
    expect(d.rows.length).toBeGreaterThan(0);
    expect(d.toggleLabel).toBe('Hide Diagnostics');
  });
});

describe('RecognitionOverlay', () => {
  it('composes sub-components for stable state', () => {
    const d = renderRecognitionOverlay({
      uiState: { mode:'stable', recognizedEquation:'3x+4=10', solution:'x=2', currentStepTitle:'S', currentStepText:'T', confidence:0.91, statusMessage:'OK', diagnosticsSummary:'', lastError:null },
      diagnostics: makeDiag({ framesProcessed:10 }),
      diagnosticsExpanded: false,
    });
    expect(d.type).toBe('RecognitionOverlay');
    expect(d.solutionCard.equationText).toBe('3x+4=10');
  });
  it('guide box active during scanning', () => {
    const d = renderRecognitionOverlay({
      uiState: { mode:'scanning', recognizedEquation:null, solution:null, currentStepTitle:null, currentStepText:null, confidence:0, statusMessage:'Scanning', diagnosticsSummary:'', lastError:null },
      diagnostics: makeDiag(),
      diagnosticsExpanded: false,
    });
    expect(d.guideBox.style.borderColor).toBe('#00E676');
  });
});
