#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIFORGE_DIR="${PROJECT_DIR}/.miniforge"
INSTALLER="/private/tmp/Miniforge3-MacOSX-arm64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"

if [ ! -x "${MINIFORGE_DIR}/bin/conda" ]; then
  echo "Downloading Miniforge..."
  curl -L -o "${INSTALLER}" "${MINIFORGE_URL}"

  echo "Installing Miniforge to ${MINIFORGE_DIR}..."
  bash "${INSTALLER}" -b -p "${MINIFORGE_DIR}"
fi

source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "tdc-admet"; then
  echo "Updating existing tdc-admet environment..."
  conda env update -n tdc-admet -f "${PROJECT_DIR}/environment.yml" --prune
else
  echo "Creating tdc-admet environment..."
  conda env create -f "${PROJECT_DIR}/environment.yml"
fi

conda activate tdc-admet

if [ "${INSTALL_CHEMPROP:-0}" = "1" ]; then
  echo "Installing optional Chemprop dependency..."
  python -m pip install -r "${PROJECT_DIR}/requirements-optional-chemprop.txt"
fi

echo "Verifying packages..."
python - <<'PY'
import sys
print(sys.version)
for name in ["numpy", "pandas", "rdkit", "torch", "sklearn"]:
    mod = __import__(name)
    print(name, "OK", getattr(mod, "__version__", ""))
PY

echo "Environment ready."
