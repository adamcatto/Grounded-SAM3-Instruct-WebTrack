#!/usr/bin/env bash
set -euo pipefail
NODE22="${NVM_NODE:-$HOME/.nvm/versions/node/v22.18.0/bin}"
export PATH="$NODE22:$PATH"
cd "$(dirname "$0")"
exec "$NODE22/npx" playwright test "$@"
