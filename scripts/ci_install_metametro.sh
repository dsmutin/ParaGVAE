#!/usr/bin/env bash
# Install the MetaMetro commit this repository is tested against.
# The package is not on conda. An editable install keeps VERSION beside the
# sources, which is how MetaMetro reports its version.
set -euo pipefail

PIN=d3cc7ac5df2cc1bd8141459cd8d34e6495fbb2bd
ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/MetaMetro-${PIN}"

if [[ ! -f "${ROOT}/VERSION" ]]; then
  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"
  git -C "${ROOT}" init
  git -C "${ROOT}" remote add origin https://github.com/dsmutin/MetaMetro.git
  git -C "${ROOT}" fetch --depth 1 origin "${PIN}"
  git -C "${ROOT}" checkout --detach FETCH_HEAD
fi

python -m pip install -e "${ROOT}" --no-deps
python -c 'import metametro; print("metametro", metametro.__version__)'
