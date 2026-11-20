#!/usr/bin/env bash
# Install the MetaMetro tag in PIN, the pin this repository is tested against.
# The package is not on conda. An editable install keeps VERSION beside the
# sources, which is how MetaMetro reports its version.
# PIN is the single source of truth. Prose docs point here instead of
# repeating it; tests/test_pin_consistency.py checks the machine-readable
# copies (pyproject.toml, environment.yml, REQUIRED_METAMETRO) against it.
set -euo pipefail

PIN=v0.22.0
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
