/**
 * React Native CLI configuration.
 *
 * Sprint 7A: Android Demo Mode Phone Run.
 *
 * Configures autolinking behavior for native modules.
 *
 * For this sprint (demo mode only), native camera and ONNX modules
 * are disabled to allow a clean build without requiring:
 *   - ONNX Runtime native libraries
 *   - VisionCamera native plugin
 *
 * These will be re-enabled when real camera/OCR integration begins.
 */
module.exports = {
  dependencies: {
    'onnxruntime-react-native': {
      platforms: {
        android: null, // Disable: Gradle compatibility issue with RN 0.86.
        ios: null,
      },
    },
    'react-native-vision-camera': {
      platforms: {
        android: null, // Disable: not needed for demo mode.
        ios: null,
      },
    },
  },
};
