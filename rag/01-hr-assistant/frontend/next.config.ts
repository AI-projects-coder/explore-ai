import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Build a self-contained server (.next/standalone) with only the files it needs: a small Docker image
  // without the full node_modules.
  output: "standalone",
};

export default nextConfig;
