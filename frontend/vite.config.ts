import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // @mediapipe/tasks-vision carga su runtime WASM desde /mediapipe/wasm
  // (public/): no hace falta que esbuild lo pre-empaquete.
  optimizeDeps: {
    exclude: ["@mediapipe/tasks-vision"],
  },
  server: {
    port: 5173,
    watch: {
      ignored: ["**/public/**", "**/node_modules/**"],
    },
  },
});
