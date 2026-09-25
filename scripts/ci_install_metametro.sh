#!/usr/bin/env bash
# Install the current MetaMetro default branch into the active environment.
# The package is not on conda. An editable install keeps VERSION beside the
# sources, which is how MetaMetro reports its version.
set -euo pipefail

ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/MetaMetro-current"
rm -rf "${ROOT}"
git clone --depth 1 https://github.com/dsmutin/MetaMetro.git "${ROOT}"
python -m pip install -e "${ROOT}" --no-deps
python -c 'import metametro; print("metametro", metametro.__version__)'
git -C "${ROOT}" rev-parse --short HEAD
