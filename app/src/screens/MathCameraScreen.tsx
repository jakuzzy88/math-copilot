/**
 * MathCameraScreen — Main camera screen for the Math Copilot.
 *
 * Sprint 6B: React Native Camera Shell.
 *
 * Behaviour:
 *   1. Request camera permission on mount.
 *   2. If permission is denied: show permission explanation.
 *   3. If permission is granted: show camera preview (or placeholder in demo).
 *   4. Overlay the 4:1 equation guide box.
 *   5. Run demo/static recognition mode.
 *   6. Display recognized equation, solution, explanation step, and diagnostics.
 *
 * The screen wraps the existing LiveRecognitionScreen controller
 * (from Sprint 6A) and renders real React Native components from
 * the serializable overlay data.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  Pressable,
  StyleSheet,
  Platform,
  StatusBar,
  SafeAreaView,
  ActivityIndicator,
} from 'react-native';

import {
  LiveRecognitionScreen,
  type LiveRecognitionScreenState,
} from '../ui/LiveRecognitionScreen';
import type { RecognitionUiState, RecognitionMode } from '../ui/recognitionUiState';
import type { LiveRecognitionDiagnostics } from '../inference/liveRecognitionController';
import { buildDiagnosticsRows } from '../ui/components/DiagnosticsPanel';
import { computeGuideBoxStyle } from '../ui/components/GuideBoxOverlay';
import {
  getSolutionCardBackground,
  getConfidenceLevel,
} from '../ui/components/SolutionCard';

// ---------------------------------------------------------------------------
// Camera import (conditional — not available in Jest)
// ---------------------------------------------------------------------------

let Camera: typeof import('react-native-vision-camera').Camera | undefined;
let useCameraPermission: typeof import('react-native-vision-camera').useCameraPermission | undefined;
let useCameraDevice: typeof import('react-native-vision-camera').useCameraDevice | undefined;

try {
  const visionCamera = require('react-native-vision-camera');
  Camera = visionCamera.Camera;
  useCameraPermission = visionCamera.useCameraPermission;
  useCameraDevice = visionCamera.useCameraDevice;
} catch {
  // Not available in Node/Jest — camera features disabled.
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Global demo mode toggle. Set to false for real camera + OCR. */
export const DEMO_MODE = true;

// ---------------------------------------------------------------------------
// Camera permission hook (safe for Jest)
// ---------------------------------------------------------------------------

interface CameraPermissionState {
  hasPermission: boolean;
  requestPermission: () => Promise<void>;
  isLoading: boolean;
}

function useSafeCameraPermission(): CameraPermissionState {
  const [hasPermission, setHasPermission] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Use real hook if available.
  if (useCameraPermission) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const perm = useCameraPermission();
    return {
      hasPermission: perm.hasPermission,
      requestPermission: async () => { await perm.requestPermission(); },
      isLoading: false,
    };
  }

  // Fallback for non-native environments.
  const requestPermission = useCallback(async () => {
    setIsLoading(false);
    setHasPermission(DEMO_MODE); // Auto-grant in demo mode.
  }, []);

  useEffect(() => {
    if (DEMO_MODE) {
      setHasPermission(true);
      setIsLoading(false);
    }
  }, []);

  return { hasPermission, requestPermission, isLoading };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MathCameraScreen(): React.JSX.Element {
  // --- Controller state ---
  const controllerRef = useRef<LiveRecognitionScreen | null>(null);
  const [screenState, setScreenState] = useState<LiveRecognitionScreenState | null>(null);
  const [diagnosticsExpanded, setDiagnosticsExpanded] = useState(false);

  // --- Camera permission ---
  const { hasPermission, requestPermission, isLoading } = useSafeCameraPermission();

  // --- Camera device (safe) ---
  const device = useCameraDevice ? useCameraDevice('back') : undefined;

  // --- Controller lifecycle ---
  useEffect(() => {
    const controller = new LiveRecognitionScreen({ demoMode: DEMO_MODE });
    controllerRef.current = controller;

    const unsubscribe = controller.addListener((state) => {
      setScreenState(state);
    });

    // In demo mode, start processing immediately.
    if (DEMO_MODE || hasPermission) {
      controller.start();
      // Process a few demo frames to populate the UI.
      if (DEMO_MODE) {
        (async () => {
          for (let i = 0; i < 5; i++) {
            await controller.processOneFrame();
          }
        })();
      }
    }

    return () => {
      unsubscribe();
      controller.stop();
      void controller.dispose();
    };
  }, [hasPermission]);

  // --- Derived state ---
  const uiState: RecognitionUiState = screenState?.uiState ?? {
    mode: 'idle',
    recognizedEquation: null,
    solution: null,
    currentStepTitle: null,
    currentStepText: null,
    confidence: 0,
    statusMessage: 'Initializing…',
    diagnosticsSummary: '',
    lastError: null,
  };

  const diagnostics: LiveRecognitionDiagnostics = screenState?.diagnostics ?? {
    framesSeen: 0,
    framesProcessed: 0,
    framesSkippedBusy: 0,
    framesFailedPreprocessing: 0,
    framesFailedRecognition: 0,
    framesRejectedByPipeline: 0,
    stableResultsEmitted: 0,
    lastRawText: null,
    lastStableEquation: null,
    averagePreprocessMs: 0,
    averageRecognitionMs: 0,
    averageTotalMs: 0,
  };

  // --- Permission loading ---
  if (isLoading && !DEMO_MODE) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <View style={styles.centeredContainer}>
          <ActivityIndicator size="large" color="#00E676" />
          <Text style={styles.permissionText}>Checking camera permission…</Text>
        </View>
      </SafeAreaView>
    );
  }

  // --- Permission denied ---
  if (!hasPermission && !DEMO_MODE) {
    return (
      <SafeAreaView style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />
        <View style={styles.centeredContainer}>
          <Text style={styles.permissionTitle}>📷 Camera Access Required</Text>
          <Text style={styles.permissionText}>
            Math Copilot needs camera access to scan handwritten equations.
            Your camera feed is processed entirely on-device — nothing is sent
            to any server.
          </Text>
          <Pressable
            style={styles.permissionButton}
            onPress={requestPermission}
            accessibilityLabel="Grant camera permission"
          >
            <Text style={styles.permissionButtonText}>Grant Camera Access</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  // --- Main camera screen ---
  const guideBoxStyle = computeGuideBoxStyle({
    isActive: uiState.mode === 'scanning' || uiState.mode === 'uncertain',
  });

  const modeColor = getModeIndicatorColor(uiState.mode);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />

      {/* Camera preview or placeholder */}
      <View style={styles.cameraContainer}>
        {renderCameraPreview(device)}

        {/* Demo mode badge */}
        {DEMO_MODE && (
          <View style={styles.demoBadge}>
            <Text style={styles.demoBadgeText}>DEMO MODE</Text>
          </View>
        )}

        {/* Mode indicator dot */}
        <View style={styles.modeIndicator}>
          <View style={[styles.modeIndicatorDot, { backgroundColor: modeColor }]} />
          <Text style={styles.modeIndicatorText}>{uiState.mode.toUpperCase()}</Text>
        </View>

        {/* Guide box overlay */}
        <View style={styles.guideBoxContainer}>
          <View
            style={[
              styles.guideBox,
              {
                borderColor: guideBoxStyle.borderColor,
                borderWidth: guideBoxStyle.borderWidth,
                borderRadius: guideBoxStyle.borderRadius,
                opacity: guideBoxStyle.opacity,
                width: `${guideBoxStyle.widthPercent}%` as unknown as number,
                aspectRatio: guideBoxStyle.aspectRatio,
              },
            ]}
            accessibilityLabel={
              guideBoxStyle.opacity === 1
                ? 'Equation guide box — scanning'
                : 'Equation guide box — idle'
            }
          />
        </View>

        {/* Status text */}
        <View style={styles.statusContainer}>
          <Text style={styles.statusText}>{uiState.statusMessage}</Text>
        </View>
      </View>

      {/* Solution card */}
      <View
        style={[
          styles.solutionCard,
          {
            backgroundColor: getSolutionCardBackground(uiState.mode, uiState.confidence),
          },
        ]}
        accessibilityLabel={
          uiState.recognizedEquation
            ? `Equation: ${uiState.recognizedEquation}. Solution: ${uiState.solution ?? 'pending'}`
            : `Status: ${uiState.statusMessage}`
        }
      >
        {uiState.recognizedEquation ? (
          <>
            <Text style={styles.equationText}>{uiState.recognizedEquation}</Text>
            {uiState.solution && (
              <Text style={styles.solutionText}>→ {uiState.solution}</Text>
            )}
            {uiState.confidence > 0 && (
              <Text style={styles.confidenceText}>
                Confidence: {(uiState.confidence * 100).toFixed(1)}%
                {' · '}
                {getConfidenceLevel(uiState.confidence).toUpperCase()}
              </Text>
            )}
            {uiState.currentStepTitle && (
              <View style={styles.stepContainer}>
                <Text style={styles.stepTitle}>{uiState.currentStepTitle}</Text>
                {uiState.currentStepText && (
                  <Text style={styles.stepText}>{uiState.currentStepText}</Text>
                )}
              </View>
            )}
          </>
        ) : (
          <Text style={styles.placeholderText}>{uiState.statusMessage}</Text>
        )}
      </View>

      {/* Diagnostics toggle */}
      <Pressable
        style={styles.diagnosticsToggle}
        onPress={() => setDiagnosticsExpanded((prev) => !prev)}
        accessibilityLabel={
          diagnosticsExpanded ? 'Hide diagnostics panel' : 'Show diagnostics panel'
        }
      >
        <Text style={styles.diagnosticsToggleText}>
          {diagnosticsExpanded ? '▼ Hide Diagnostics' : '▶ Show Diagnostics'}
        </Text>
      </Pressable>

      {/* Diagnostics panel */}
      {diagnosticsExpanded && (
        <View style={styles.diagnosticsPanel}>
          {buildDiagnosticsRows(diagnostics).map((row, index) => (
            <View key={index} style={styles.diagnosticsRow}>
              <Text
                style={[
                  styles.diagnosticsLabel,
                  row.isWarning && styles.diagnosticsWarning,
                ]}
              >
                {row.label}
              </Text>
              <Text
                style={[
                  styles.diagnosticsValue,
                  row.isWarning && styles.diagnosticsWarning,
                ]}
              >
                {row.value}
              </Text>
            </View>
          ))}
        </View>
      )}
    </SafeAreaView>
  );
}

// ---------------------------------------------------------------------------
// Camera preview helper
// ---------------------------------------------------------------------------

function renderCameraPreview(
  device: ReturnType<NonNullable<typeof useCameraDevice>> | undefined,
): React.JSX.Element {
  // Real camera available.
  if (Camera && device && !DEMO_MODE) {
    const CameraComponent = Camera;
    return (
      <CameraComponent
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
      />
    );
  }

  // Placeholder (demo mode or no camera).
  return (
    <View style={styles.cameraPlaceholder}>
      <Text style={styles.cameraPlaceholderIcon}>📸</Text>
      <Text style={styles.cameraPlaceholderText}>
        {DEMO_MODE
          ? 'Camera preview (demo mode)'
          : 'Camera not available'}
      </Text>
    </View>
  );
}

// ---------------------------------------------------------------------------
// Mode color helper
// ---------------------------------------------------------------------------

function getModeIndicatorColor(mode: RecognitionMode): string {
  switch (mode) {
    case 'stable':
      return '#00E676';
    case 'scanning':
      return '#2196F3';
    case 'uncertain':
      return '#FF9800';
    case 'error':
      return '#F44336';
    case 'idle':
    default:
      return '#9E9E9E';
  }
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centeredContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },

  // --- Permission screen ---
  permissionTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: '#FFFFFF',
    marginBottom: 16,
    textAlign: 'center',
  },
  permissionText: {
    fontSize: 15,
    color: '#CCCCCC',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  permissionButton: {
    backgroundColor: '#00E676',
    paddingHorizontal: 28,
    paddingVertical: 14,
    borderRadius: 12,
  },
  permissionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000000',
  },

  // --- Camera ---
  cameraContainer: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
  },
  cameraPlaceholder: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#1A1A2E',
    justifyContent: 'center',
    alignItems: 'center',
  },
  cameraPlaceholderIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  cameraPlaceholderText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
  },

  // --- Demo badge ---
  demoBadge: {
    position: 'absolute',
    top: Platform.OS === 'android' ? 12 : 8,
    right: 12,
    backgroundColor: 'rgba(255, 152, 0, 0.9)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 6,
    zIndex: 10,
  },
  demoBadgeText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#000',
    letterSpacing: 1,
  },

  // --- Mode indicator ---
  modeIndicator: {
    position: 'absolute',
    top: Platform.OS === 'android' ? 12 : 8,
    left: 12,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    zIndex: 10,
  },
  modeIndicatorDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  modeIndicatorText: {
    fontSize: 11,
    fontWeight: '600',
    color: '#FFF',
    letterSpacing: 0.5,
  },

  // --- Guide box ---
  guideBoxContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 5,
  },
  guideBox: {
    // Dynamic styles are applied inline.
  },

  // --- Status ---
  statusContainer: {
    position: 'absolute',
    bottom: 16,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  statusText: {
    fontSize: 13,
    color: '#FFF',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 16,
    overflow: 'hidden',
  },

  // --- Solution card ---
  solutionCard: {
    marginHorizontal: 12,
    marginTop: -20,
    borderRadius: 16,
    padding: 16,
    zIndex: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  equationText: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFF',
    textAlign: 'center',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  solutionText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FFF',
    textAlign: 'center',
    marginTop: 4,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  confidenceText: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.75)',
    textAlign: 'center',
    marginTop: 6,
  },
  stepContainer: {
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(255, 255, 255, 0.3)',
  },
  stepTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#FFF',
  },
  stepText: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 2,
    lineHeight: 18,
  },
  placeholderText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.6)',
    textAlign: 'center',
    fontStyle: 'italic',
  },

  // --- Diagnostics ---
  diagnosticsToggle: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  diagnosticsToggleText: {
    fontSize: 13,
    color: '#888',
  },
  diagnosticsPanel: {
    marginHorizontal: 12,
    backgroundColor: 'rgba(30, 30, 30, 0.95)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
  },
  diagnosticsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 3,
  },
  diagnosticsLabel: {
    fontSize: 12,
    color: '#AAA',
  },
  diagnosticsValue: {
    fontSize: 12,
    color: '#FFF',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  diagnosticsWarning: {
    color: '#FF9800',
  },
});

export default MathCameraScreen;
