import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// Renderer-only build. The runtime host is `python/app.py` via pywebview, so we
// don't need electron-vite's main/preload pipelines. `base: './'` makes asset URLs
// relative — required for pywebview's file:// loading to resolve hashed bundles.
export default defineConfig({
  root: resolve(__dirname, 'src/renderer'),
  base: './',
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(__dirname, 'src/renderer/index.html'),
    },
  },
  plugins: [react()],
  server: { port: 5173 },
});
