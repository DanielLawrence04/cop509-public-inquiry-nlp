// Copies frontend/src into frontend/dist/src so the in-browser Babel runtime
// in index.html can fetch the .jsx source files from a local Vite build.
// Vite's bundler cannot process the runtime-Babel <script type="text/babel">
// tags, so this shim keeps the existing UMD/Babel setup working without
// rewriting modules.
import { cpSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, "..", "src");
const outDir = resolve(here, "..", "dist", "src");

if (!existsSync(srcDir)) {
  console.error(`[postbuild] missing source dir: ${srcDir}`);
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });
cpSync(srcDir, outDir, { recursive: true });
console.log(`[postbuild] copied ${srcDir} -> ${outDir}`);
