import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base './' so the built index.html works from file:// inside pywebview
export default defineConfig({
  base: './',
  plugins: [react()],
})
