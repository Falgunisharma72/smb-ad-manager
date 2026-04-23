#!/usr/bin/env bash
# Push latest commit to both GitHub and HuggingFace Space in one command.
# Prerequisite: `hf` remote configured (see README Step 7b).
#
# Usage:  ./scripts/deploy.sh [commit-message]
#         or just: ./scripts/deploy.sh    (uses last commit)

set -euo pipefail

cd "$(dirname "$0")/.."

MSG="${1:-}"

if [ -n "$MSG" ]; then
    git add -A
    git -c user.name="Falguni Sharma" -c user.email="fsharma0207@gmail.com" \
        commit -m "$MSG" || echo "[deploy] nothing to commit"
fi

echo "[deploy] pushing to GitHub (origin)..."
git push origin main

echo "[deploy] pushing to HuggingFace Space (hf)..."
git push hf main

echo ""
echo "[deploy] done."
echo "  GitHub:  https://github.com/Falgunisharma72/smb-ad-manager"
echo "  Space:   https://huggingface.co/spaces/Falgunisharma/smb-ad-manager"
echo "  Live:    https://falgunisharma-smb-ad-manager.hf.space"
echo ""
echo "[deploy] Space rebuild takes ~60-90s. Check:"
echo "         curl https://falgunisharma-smb-ad-manager.hf.space/healthz"
