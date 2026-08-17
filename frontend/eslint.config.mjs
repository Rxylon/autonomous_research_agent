import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Deploy artifacts. `firebase deploy` copies the built bundle here, and
    // linting minified output produced ~15k phantom problems that buried the
    // handful of real ones in src/.
    ".firebase/**",
    "node_modules/**",
  ]),
]);

export default eslintConfig;
