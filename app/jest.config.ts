import type { Config } from 'jest';

const config: Config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  moduleNameMapper: {
    '^@parser/(.*)$': '<rootDir>/src/parser/$1',
    '^@solver/(.*)$': '<rootDir>/src/solver/$1',
    '^@explanation/(.*)$': '<rootDir>/src/explanation/$1',
    '^@grammar/(.*)$': '<rootDir>/src/grammar/$1',
    '^@diagnostics/(.*)$': '<rootDir>/src/diagnostics/$1',
    '^@pipeline/(.*)$': '<rootDir>/src/pipeline/$1',
  },
  transform: {
    '^.+\\.ts$': ['ts-jest', {
      tsconfig: 'tsconfig.test.json',
    }],
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/__tests__/**',
    '!src/**/index.ts',
  ],
  verbose: true,
};

export default config;
