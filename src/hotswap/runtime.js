let activeManifest = null;

export function getHotSwapManifest() {
  return activeManifest;
}

export function setHotSwapManifest(manifest) {
  activeManifest = manifest;
}
