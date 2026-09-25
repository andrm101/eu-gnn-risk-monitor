import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/EU-GNN-Risk-Monitor/',
  test: {
    setupFiles: ['./src/test-setup.ts'],
  },
})
