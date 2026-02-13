#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models_dir="$(cd "${script_dir}/../models" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
firebase_config="${repo_root}/firebase.json"

if [[ ! -f "${models_dir}/catalog.json" ]]; then
  echo "[models-validate] Missing required file: models/catalog.json" >&2
  exit 1
fi

disallowed=()
while IFS= read -r file_path; do
  rel_path="${file_path#${models_dir}/}"
  case "${rel_path}" in
    catalog.json|README.md|curated/*|local/*)
      ;;
    *)
      disallowed+=("${rel_path}")
      ;;
  esac
done < <(find "${models_dir}" -type f | sort)

if [[ "${#disallowed[@]}" -gt 0 ]]; then
  echo "[models-validate] Only the following model paths are allowed under models/:" >&2
  echo "  - models/catalog.json" >&2
  echo "  - models/README.md" >&2
  echo "  - models/curated/** (hosted)" >&2
  echo "  - models/local/** (local only; not deployed)" >&2
  echo "[models-validate] Disallowed files found:" >&2
  printf '  - %s\n' "${disallowed[@]}" >&2
  exit 1
fi

if [[ -f "${firebase_config}" ]] && ! grep -Fq '"models/local/**"' "${firebase_config}"; then
  echo "[models-validate] firebase.json must ignore models/local/** for the doppler host target." >&2
  exit 1
fi

echo "[models-validate] Model layout OK (curated hosted, local excluded)."
