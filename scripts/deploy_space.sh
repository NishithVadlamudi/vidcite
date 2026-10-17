#!/usr/bin/env bash
# Push the serving code to a Hugging Face Docker Space.
#   scripts/deploy_space.sh https://huggingface.co/spaces/<user>/vidcite
# Builds the Space's tree from the committed HEAD (never from uncommitted files or .env),
# adds the Space README (YAML header), and pushes. git asks for your Hugging Face
# username and an access token with write permission.
set -euo pipefail
SPACE_URL="${1:?usage: scripts/deploy_space.sh <space git url>}"
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain)" ]; then
  echo "Commit your changes first: the Space is built from HEAD." >&2
  exit 1
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git archive HEAD Dockerfile .dockerignore requirements-serve.txt LICENSE api ingest retrieval | tar -x -C "$WORK"
cp deploy/hf-space/README.md "$WORK/README.md"
cd "$WORK"
git init -q -b main
git add -A
git commit -q -m "Deploy vidcite $(git -C "$OLDPWD" rev-parse --short HEAD)"
git push --force "$SPACE_URL" main
echo "Pushed. Build logs: ${SPACE_URL} (Logs tab)."
