/**
 * CameraPreview — Renders either the real VisionCamera or a placeholder.
 *
 * Sprint 7: Real Camera Preview.
 *
 * This component encapsulates the camera rendering logic:
 *   1. If react-native-vision-camera is available AND a device exists
 *      AND not in demo mode → render the real Camera component.
 *   2. Otherwise → render a styled placeholder.
 *
 * Graceful degradation:
 *   - If VisionCamera is not installed or fails to load: placeholder.
 *   - If no camera device is found: "No camera device" message.
 *   - If in demo mode: placeholder with "demo mode" label.
 *
 * Camera lifecycle (isActive) is controlled externally via the
 * `isActive` prop, which should come from useCameraLifecycle.
 *
 * Pure status logic is in cameraPreviewStatus.ts (testable without React).
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  getCameraPreviewStatus,
  getPlaceholderContent,
} from './cameraPreviewStatus';

// ---------------------------------------------------------------------------
// Conditional VisionCamera import
// ---------------------------------------------------------------------------

let VisionCamera: typeof import('react-native-vision-camera').Camera | undefined;
let useVisionCameraDevice: typeof import('react-native-vision-camera').useCameraDevice | undefined;

/** Whether the VisionCamera native module loaded successfully. */
let isVisionCameraLoaded = false;

/** Error detail if VisionCamera failed to load (for diagnostics). */
let visionCameraLoadError: string | null = null;

try {
  const visionCameraModule = require('react-native-vision-camera');
  VisionCamera = visionCameraModule.Camera;
  useVisionCameraDevice = visionCameraModule.useCameraDevice;
  if (VisionCamera && useVisionCameraDevice) {
    isVisionCameraLoaded = true;
  } else {
    visionCameraLoadError = 'VisionCamera module loaded but Camera or useCameraDevice is undefined.';
  }
} catch (error: unknown) {
  const message = error instanceof Error ? error.message : String(error);
  visionCameraLoadError = message;
  // Log in dev for debugging — this is expected in Jest/Node.
  if (typeof __DEV__ !== 'undefined' && __DEV__) {
    console.warn('[CameraPreview] VisionCamera load failed:', message);
  }
}

/** Diagnostic detail about VisionCamera loading. Exported for UI diagnostics. */
export function getVisionCameraLoadDiagnostic(): string {
  if (isVisionCameraLoaded) return 'VisionCamera: loaded';
  if (visionCameraLoadError) return `VisionCamera: ${visionCameraLoadError}`;
  return 'VisionCamera: not available';
}

// Re-export status types and helpers for convenience.
export { getCameraPreviewStatus, getPlaceholderContent } from './cameraPreviewStatus';
export type { CameraPreviewStatus } from './cameraPreviewStatus';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CameraPreviewProps {
  /** Whether the camera should be actively streaming frames. */
  isActive: boolean;
  /** Whether the app is running in demo mode (no real camera). */
  isDemoMode: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/**
 * CameraPreview renders the real camera feed or an appropriate placeholder.
 *
 * The component handles all failure modes gracefully:
 * - VisionCamera unavailable → shows clear message with diagnostic detail
 * - No camera device found → shows device error
 * - Demo mode → shows demo placeholder
 */
export function CameraPreview({ isActive, isDemoMode }: CameraPreviewProps): React.JSX.Element {
  const status = getCameraPreviewStatus(isDemoMode, isVisionCameraLoaded);

  if (status === 'demo') {
    const content = getPlaceholderContent(status);
    return (
      <CameraPlaceholder
        message={content.message}
        icon={content.icon}
        isError={content.isError}
      />
    );
  }

  if (status === 'unavailable') {
    const content = getPlaceholderContent(status);
    const diagnostic = getVisionCameraLoadDiagnostic();
    return (
      <CameraPlaceholder
        message={content.message}
        icon={content.icon}
        isError={content.isError}
        detail={diagnostic}
      />
    );
  }

  // VisionCamera is available — get the back device.
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const device = useVisionCameraDevice!('back');

  if (!device) {
    const content = getPlaceholderContent('no-device');
    return (
      <CameraPlaceholder
        message={content.message}
        icon={content.icon}
        isError={content.isError}
      />
    );
  }

  // Render real camera.
  const CameraComponent = VisionCamera!;
  return (
    <CameraComponent
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={isActive}
      photo={false}
      video={false}
      audio={false}
    />
  );
}

// ---------------------------------------------------------------------------
// Placeholder component
// ---------------------------------------------------------------------------

interface CameraPlaceholderProps {
  message: string;
  icon: string;
  isError?: boolean;
  /** Optional diagnostic detail line (shown in smaller text below main message). */
  detail?: string;
}

function CameraPlaceholder({ message, icon, isError, detail }: CameraPlaceholderProps): React.JSX.Element {
  return (
    <View style={styles.placeholder} accessibilityLabel={message}>
      <Text style={styles.placeholderIcon}>{icon}</Text>
      <Text style={[styles.placeholderText, isError && styles.placeholderErrorText]}>
        {message}
      </Text>
      {detail && (
        <Text style={styles.placeholderDetail}>{detail}</Text>
      )}
    </View>
  );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
  placeholder: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#1A1A2E',
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  placeholderText: {
    fontSize: 14,
    color: '#666',
    fontStyle: 'italic',
    textAlign: 'center',
    paddingHorizontal: 32,
    lineHeight: 20,
  },
  placeholderErrorText: {
    color: '#FF9800',
  },
  placeholderDetail: {
    fontSize: 11,
    color: '#555',
    textAlign: 'center',
    paddingHorizontal: 32,
    marginTop: 8,
    fontFamily: 'monospace',
  },
});
