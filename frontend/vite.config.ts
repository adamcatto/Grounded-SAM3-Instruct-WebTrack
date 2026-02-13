import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Allow large video file responses and range requests
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            // Ensure Accept-Ranges header passes through for video seeking
            if (!proxyRes.headers['accept-ranges']) {
              proxyRes.headers['accept-ranges'] = 'bytes'
            }
          })
        },
      },
    },
  },
})
