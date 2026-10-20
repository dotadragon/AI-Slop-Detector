import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["react", "react-dom", "react-router-dom", "@tanstack/react-query", "lucide-react"],
    exclude: ["react-dropzone"],
  },
  resolve: {
    preserveSymlinks: true
  },
  server: {
    host: "127.0.0.1",
    port: 3000,
    strictPort: true,
    hmr: {
      host: "127.0.0.1",
      port: 3000,
      protocol: "ws"
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        ws: true
      }
    }
  },
  preview: {
    host: "127.0.0.1",
    port: 3000,
    strictPort: true
  },
  build: {
    target: "esnext",
    cssCodeSplit: true,
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom", "react-router-dom"],
          motion: ["framer-motion"],
          icons: ["lucide-react"]
        }
      }
    }
  }
});
