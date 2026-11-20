#!/usr/bin/env bash
# Install MetaMetro tag v0.20.1, the pin this repository is tested against.
# The package is not on conda. An editable install keeps VERSION beside the
# sources, which is how MetaMetro reports its version.
# Keep the tag in this file only; other docs should point here instead of
# repeating it.
set -euo pipefail

PIN=v0.20.1
ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/MetaMetro-${PIN}"

if [[ ! -f "${ROOT}/VERSION" ]]; then
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"
  git -C "${ROOT}" init
  git -C "${ROOT}" remote add origin https://github.com/dsmutin/MetaMetro.git
  git -C "${ROOT}" fetch --depth 1 origin "refs/tags/${PIN}:refs/tags/${PIN}"
  git -C "${ROOT}" checkout --detach "${PIN}"
fi

python -m pip install -e "${ROOT}" --no-deps
python -c 'import metametro; print("metametro", metametro.__version__)'
git -C "${ROOT}" rev-parse --short HEAD
