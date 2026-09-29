import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // Build straight into the Flask app so it can serve the frontend itself
    // (single process/executable, no Node.js needed at runtime).
    outDir: path.resolve(__dirname, '../pyspresso/pyspresso_app/frontend_dist'),
    emptyOutDir: true,
  },
})
