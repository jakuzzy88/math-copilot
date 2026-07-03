/**
 * React Native CLI configuration.
 *
 * Sprint 7: Real Camera Preview.
 *
 * Configures autolinking behavior for native modules.
 *
 * - VisionCamera: ENABLED — required for real camera preview on Android.
 * - ONNX Runtime: DISABLED — Gradle compatibility issue with RN 0.86.
 *                 Will be re-enabled when the Gradle conflict is resolved.
 */
module.exports = {
  dependencies: {
    'onnxruntime-react-native': {
      platforms: {
        android: null, // Disable: Gradle compatibility issue with RN 0.86.
        ios: null,
      },
    },
    // react-native-vision-camera: autolinked (no override needed).
  },
};
