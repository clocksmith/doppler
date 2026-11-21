export const DEFAULT_HF_CDN_BASE_URL = 'https://huggingface.co';

export function isPinnedHostedSource(source) {
  return typeof source?.manifestUrl === 'string'
    && URL.canParse(source.manifestUrl)
    && new URL(source.manifestUrl).protocol === 'https:'
    && /^[0-9a-f]{64}$/.test(source.expectedManifestHash)
    && typeof source.weights?.repoId === 'string' && source.weights.repoId.trim().length > 0
    && typeof source.weights.path === 'string' && source.weights.path.trim().length > 0
    && /^[0-9a-f]{40}$/.test(source.weights.revision);
}

export function buildHfResolveBaseUrl(hfConfig, options = {}) {
  const repoId = typeof hfConfig?.repoId === 'string' ? hfConfig.repoId.trim() : '';
  const repoPath = typeof hfConfig?.path === 'string' ? hfConfig.path.trim().replace(/^\/+/, '') : '';
  if (!repoId || !repoPath) {
    throw new Error('Hosted Hugging Face source requires repoId and path.');
  }

  const revision = typeof hfConfig?.revision === 'string' && hfConfig.revision.trim().length > 0
    ? hfConfig.revision.trim()
    : 'main';
  const cdnBasePath = typeof options?.cdnBasePath === 'string' && options.cdnBasePath.trim().length > 0
    ? options.cdnBasePath.trim()
    : DEFAULT_HF_CDN_BASE_URL;
  return `${cdnBasePath.replace(/\/$/, '')}/${repoId}/resolve/${revision}/${repoPath}`;
}
