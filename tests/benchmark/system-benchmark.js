class SystemBenchmark {
  config;
  constructor(config) {
    this.config = {
      clearOPFS: false,
      shardLimit: 0,
      debug: false,
      ...config
    };
  }
  async run() {
    const env = await this.collectEnvironment();
    const storage = await this.measureStorage();
    if (this.config.clearOPFS) {
      await this.clearOPFS();
    }
    const download = await this.measureDownload();
    const opfs = await this.measureOPFS(download.totalBytes);
    return {
      schemaVersion: 1,
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      suite: "system",
      runType: this.config.clearOPFS ? "cold" : "warm",
      env,
      storage,
      download,
      opfs
    };
  }
  // ==========================================================================
  // Environment
  // ==========================================================================
  async collectEnvironment() {
    const ua = typeof navigator !== "undefined" ? navigator.userAgent : "unknown";
    let browser = { name: "unknown", version: "unknown" };
    if (ua.includes("Chrome")) {
      const match = ua.match(/Chrome\/(\d+\.\d+\.\d+\.\d+)/);
      browser = { name: "Chrome", version: match?.[1] ?? "unknown" };
    } else if (ua.includes("Safari") && !ua.includes("Chrome")) {
      const match = ua.match(/Version\/(\d+\.\d+)/);
      browser = { name: "Safari", version: match?.[1] ?? "unknown" };
    } else if (ua.includes("Firefox")) {
      const match = ua.match(/Firefox\/(\d+\.\d+)/);
      browser = { name: "Firefox", version: match?.[1] ?? "unknown" };
    }
    let os = { name: "unknown", version: "unknown" };
    if (ua.includes("Mac OS X")) {
      const match = ua.match(/Mac OS X (\d+[._]\d+[._]?\d*)/);
      os = { name: "macOS", version: match?.[1]?.replace(/_/g, ".") ?? "unknown" };
    } else if (ua.includes("Windows")) {
      const match = ua.match(/Windows NT (\d+\.\d+)/);
      os = { name: "Windows", version: match?.[1] ?? "unknown" };
    } else if (ua.includes("Linux")) {
      os = { name: "Linux", version: "unknown" };
    }
    let gpu = { vendor: "unknown", device: "unknown", description: "unknown" };
    let webgpu = { hasF16: false, hasSubgroups: false, hasTimestampQuery: false };
    if (typeof navigator !== "undefined" && "gpu" in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          const info = await adapter.requestAdapterInfo?.() ?? {};
          gpu = {
            vendor: info.vendor ?? "unknown",
            device: info.device ?? "unknown",
            description: info.description ?? adapter.name ?? "unknown"
          };
          webgpu = {
            hasF16: adapter.features.has("shader-f16"),
            hasSubgroups: adapter.features.has("subgroups"),
            hasTimestampQuery: adapter.features.has("timestamp-query")
          };
        }
      } catch (e) {
      }
    }
    return { browser, os, gpu, webgpu };
  }
  // ==========================================================================
  // Storage Metrics
  // ==========================================================================
  async measureStorage() {
    const metrics = {
      mode: "http_only",
      persisted: false,
      quotaBytes: 0,
      usageBytes: 0,
      availableBytes: 0
    };
    if (typeof navigator === "undefined" || !navigator.storage) {
      return metrics;
    }
    try {
      if ("getDirectory" in navigator.storage) {
        metrics.mode = "opfs";
      }
      metrics.persisted = await navigator.storage.persisted?.() ?? false;
      const estimate = await navigator.storage.estimate?.();
      if (estimate) {
        metrics.quotaBytes = estimate.quota ?? 0;
        metrics.usageBytes = estimate.usage ?? 0;
        metrics.availableBytes = metrics.quotaBytes - metrics.usageBytes;
      }
    } catch (e) {
      if (this.config.debug) {
        console.warn("[SystemBenchmark] Storage API error:", e);
      }
    }
    return metrics;
  }
  // ==========================================================================
  // Download Metrics
  // ==========================================================================
  async measureDownload() {
    const metrics = {
      totalBytes: 0,
      shardCount: 0,
      totalTimeMs: 0,
      bytesPerSec: 0,
      shardTimings: [],
      manifestFetchMs: 0
    };
    const manifestStart = performance.now();
    const manifestUrl = this.config.modelPath.endsWith(".json") ? this.config.modelPath : `${this.config.modelPath}/manifest.json`;
    const manifestRes = await fetch(manifestUrl);
    const manifest = await manifestRes.json();
    metrics.manifestFetchMs = performance.now() - manifestStart;
    const shards = [];
    const baseUrl = this.config.modelPath.replace(/\/manifest\.json$/, "");
    if (manifest.shards) {
      for (const shard of manifest.shards) {
        shards.push({
          url: `${baseUrl}/${shard.filename || shard.path}`,
          size: shard.size || shard.byteLength || 0
        });
      }
    }
    const shardsToTest = this.config.shardLimit && this.config.shardLimit > 0 ? shards.slice(0, this.config.shardLimit) : shards;
    const downloadStart = performance.now();
    for (let i = 0; i < shardsToTest.length; i++) {
      const shard = shardsToTest[i];
      const shardStart = performance.now();
      try {
        const res = await fetch(shard.url);
        const buffer = await res.arrayBuffer();
        const shardTime = performance.now() - shardStart;
        const actualSize = buffer.byteLength;
        metrics.shardTimings.push({
          index: i,
          sizeBytes: actualSize,
          fetchMs: shardTime,
          bytesPerSec: shardTime > 0 ? actualSize / shardTime * 1e3 : 0
        });
        metrics.totalBytes += actualSize;
        metrics.shardCount++;
        if (this.config.debug) {
          console.log(`[SystemBenchmark] Shard ${i}: ${actualSize} bytes in ${shardTime.toFixed(0)}ms`);
        }
      } catch (e) {
        if (this.config.debug) {
          console.warn(`[SystemBenchmark] Failed to fetch shard ${i}:`, e);
        }
      }
    }
    metrics.totalTimeMs = performance.now() - downloadStart;
    metrics.bytesPerSec = metrics.totalTimeMs > 0 ? metrics.totalBytes / metrics.totalTimeMs * 1e3 : 0;
    return metrics;
  }
  // ==========================================================================
  // OPFS Metrics
  // ==========================================================================
  async measureOPFS(testDataSize) {
    const metrics = {
      available: false,
      writeTimeMs: 0,
      readTimeMs: 0,
      writeBytesPerSec: 0,
      readBytesPerSec: 0,
      bytesWritten: 0
    };
    if (typeof navigator === "undefined" || !navigator.storage) {
      return metrics;
    }
    try {
      const root = await navigator.storage.getDirectory?.();
      if (!root) {
        return metrics;
      }
      metrics.available = true;
      const testDir = await root.getDirectoryHandle("_benchmark_test", { create: true });
      const chunkSize = Math.min(testDataSize, 10 * 1024 * 1024);
      const testData = new Uint8Array(chunkSize);
      for (let i = 0; i < chunkSize; i++) {
        testData[i] = i % 256;
      }
      const writeStart = performance.now();
      const fileHandle = await testDir.getFileHandle("test.bin", { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(testData);
      await writable.close();
      metrics.writeTimeMs = performance.now() - writeStart;
      metrics.bytesWritten = chunkSize;
      metrics.writeBytesPerSec = metrics.writeTimeMs > 0 ? chunkSize / metrics.writeTimeMs * 1e3 : 0;
      const readStart = performance.now();
      const file = await fileHandle.getFile();
      const readData = await file.arrayBuffer();
      metrics.readTimeMs = performance.now() - readStart;
      metrics.readBytesPerSec = metrics.readTimeMs > 0 ? readData.byteLength / metrics.readTimeMs * 1e3 : 0;
      await testDir.removeEntry("test.bin");
      await root.removeEntry("_benchmark_test");
    } catch (e) {
      if (this.config.debug) {
        console.warn("[SystemBenchmark] OPFS test error:", e);
      }
    }
    return metrics;
  }
  // ==========================================================================
  // OPFS Management
  // ==========================================================================
  async clearOPFS() {
    if (typeof navigator === "undefined" || !navigator.storage) {
      return;
    }
    try {
      const root = await navigator.storage.getDirectory?.();
      if (!root)
        return;
      for await (const [name] of root.entries()) {
        try {
          await root.removeEntry(name, { recursive: true });
          if (this.config.debug) {
            console.log(`[SystemBenchmark] Cleared OPFS entry: ${name}`);
          }
        } catch (e) {
        }
      }
    } catch (e) {
      if (this.config.debug) {
        console.warn("[SystemBenchmark] Failed to clear OPFS:", e);
      }
    }
  }
}
async function runSystemBenchmark(modelPath) {
  const bench = new SystemBenchmark({
    modelPath,
    shardLimit: 3,
    // Only test first 3 shards for speed
    debug: false
  });
  return bench.run();
}
function formatSystemSummary(result) {
  const { storage, download, opfs } = result;
  const formatBytes = (bytes) => {
    if (bytes >= 1024 * 1024 * 1024)
      return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
    if (bytes >= 1024 * 1024)
      return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    if (bytes >= 1024)
      return `${(bytes / 1024).toFixed(2)} KB`;
    return `${bytes} bytes`;
  };
  const formatSpeed = (bytesPerSec) => {
    return `${formatBytes(bytesPerSec)}/s`;
  };
  return [
    `=== System Benchmark ===`,
    `Storage: ${storage.mode} (persisted: ${storage.persisted})`,
    `  Quota: ${formatBytes(storage.quotaBytes)}`,
    `  Used: ${formatBytes(storage.usageBytes)}`,
    `  Available: ${formatBytes(storage.availableBytes)}`,
    ``,
    `Download:`,
    `  Manifest: ${download.manifestFetchMs.toFixed(0)}ms`,
    `  Shards: ${download.shardCount} (${formatBytes(download.totalBytes)})`,
    `  Time: ${download.totalTimeMs.toFixed(0)}ms`,
    `  Speed: ${formatSpeed(download.bytesPerSec)}`,
    ``,
    `OPFS: ${opfs.available ? "available" : "not available"}`,
    opfs.available ? `  Write: ${formatSpeed(opfs.writeBytesPerSec)}` : "",
    opfs.available ? `  Read: ${formatSpeed(opfs.readBytesPerSec)}` : ""
  ].filter(Boolean).join("\n");
}
export {
  SystemBenchmark,
  formatSystemSummary,
  runSystemBenchmark
};
