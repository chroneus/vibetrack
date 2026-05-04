#!/usr/bin/env bash
# Refresh vendored JS dependencies in this directory.
# Pinned versions match README.md.

set -euo pipefail

THREE_VERSION="0.128.0"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$DEST"
curl -sSfL -o "$DEST/three.min.js" \
  "https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/build/three.min.js"
curl -sSfL -o "$DEST/OrbitControls.js" \
  "https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/examples/js/controls/OrbitControls.js"

echo "Refreshed vendored JS at $DEST:"
ls -lh "$DEST"
