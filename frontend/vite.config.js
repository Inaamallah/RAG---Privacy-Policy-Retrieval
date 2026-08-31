import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the page is served by Vite on :5173 and /api is proxied to
// the FastAPI process on :8000, so the browser only ever sees one origin and
// the fetch paths are the same relative ones a production build uses -- where
// FastAPI serves dist/ itself and there is no proxy at all.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
