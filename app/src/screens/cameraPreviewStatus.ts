/**
 * cameraPreviewStatus — Pure logic for determining camera preview state.
 *
 * Sprint 7: Real Camera Preview.
 *
 * Contains the pure, testable logic for determining what the camera preview
 * should display. Has NO React or React Native dependencies.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Status of the camera preview component.
 *
 * - 'live': Real camera is rendering.
 * - 'no-device': VisionCamera loaded but no camera device found.
 * - 'unavailable': VisionCamera module is not available.
 * - 'demo': Running in demo mode (camera not used).
 */
export type CameraPreviewStatus = 'live' | 'no-device' | 'unavailable' | 'demo';

// ---------------------------------------------------------------------------
// Pure logic
// ---------------------------------------------------------------------------

/**
 * Determine the camera preview status based on current conditions.
 *
 * @param isDemoMode - Whether the app is running in demo mode.
 * @param isVisionCameraAvailable - Whether the VisionCamera module is loaded.
 * @returns The appropriate preview status.
 */
export function getCameraPreviewStatus(
  isDemoMode: boolean,
  isVisionCameraAvailable: boolean = false,
): CameraPreviewStatus {
  if (isDemoMode) return 'demo';
  if (!isVisionCameraAvailable) return 'unavailable';
  // 'live' or 'no-device' is determined at render time (needs the hook).
  return 'live';
}

/**
 * Get the placeholder message for a given camera preview status.
 *
 * @param status - The camera preview status.
 * @returns An object with the message and icon to display.
 */
export function getPlaceholderContent(
  status: CameraPreviewStatus,
): { message: string; icon: string; isError: boolean } {
  switch (status) {
    case 'demo':
      return {
        message: 'Camera preview (demo mode)',
        icon: '📸',
        isError: false,
      };
    case 'unavailable':
      return {
        message: 'Camera not available. VisionCamera module could not be loaded.',
        icon: '⚠️',
        isError: true,
      };
    case 'no-device':
      return {
        message: 'No camera device found on this device.',
        icon: '📷',
        isError: true,
      };
    case 'live':
      return {
        message: '',
        icon: '',
        isError: false,
      };
  }
}
