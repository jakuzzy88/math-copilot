/**
 * Metro configuration for React Native.
 *
 * Sprint 6B: React Native Camera Shell.
 *
 * Extends the default Metro config to handle .onnx model files
 * as binary assets.
 */
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

const config = {
  resolver: {
    // Add .onnx to asset extensions so Metro can bundle model files.
    assetExts: [...defaultConfig.resolver.assetExts, 'onnx'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
