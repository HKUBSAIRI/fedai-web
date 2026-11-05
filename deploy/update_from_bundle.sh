#!/usr/bin/env bash
# Update the site on the Aliyun ECS server from a git bundle (the server has no GitHub access).
# Run on the machine that has the repo and the deploy key:  bash deploy/update_from_bundle.sh
set -euo pipefail
# Server login (user@address) comes from FEDAI_HOST_SSH or the untracked file deploy/.host (kept out of the public repository).
HOST=${FEDAI_HOST_SSH:-$(cat "$(dirname "$0")/.host" 2>/dev/null || true)}
[ -n "$HOST" ] || { echo "Set FEDAI_HOST_SSH=user@address or put it in deploy/.host"; exit 1; }
KEY=${FEDAI_SSH_KEY:-$HOME/.ssh/fedai_ecs}
cd "$(dirname "$0")/.."
git bundle create /tmp/fedai-web-deploy.bundle main >/dev/null 2>&1
scp -q -i "$KEY" /tmp/fedai-web-deploy.bundle "$HOST:/tmp/fedai-web.bundle"
ssh -i "$KEY" -o BatchMode=yes "$HOST" 'set -e
  sudo chmod 644 /tmp/fedai-web.bundle
  cd /opt/fedai-web
  sudo -u fedai git fetch -q /tmp/fedai-web.bundle main
  # fast-forward; if the history was rewritten (e.g. squashed before publishing), reset, but only when the server copy has no local edits
  sudo -u fedai git merge -q --ff-only FETCH_HEAD 2>/dev/null || { [ -z "$(sudo -u fedai git status --porcelain --untracked-files=no)" ] && sudo -u fedai git reset -q --hard FETCH_HEAD; }
  sudo -u fedai env HOME=/var/lib/fedai npm ci --no-audit --no-fund --registry=https://registry.npmmirror.com >/dev/null
  sudo -u fedai env HOME=/var/lib/fedai npm run build >/dev/null
  sudo systemctl restart fedai-web
  for i in $(seq 1 40); do curl -fs 127.0.0.1:8787/api/status >/dev/null 2>&1 && break; sleep 1; done
  echo "deployed $(sudo -u fedai git log --oneline -1)"; systemctl is-active fedai-web
  # cleanup: uploaded bundle, npm download cache, unreachable git objects
  sudo rm -f /tmp/fedai-web.bundle
  sudo -u fedai env HOME=/var/lib/fedai npm cache clean --force >/dev/null 2>&1 || true
  sudo -u fedai git gc -q --prune=now
  echo "disk: $(du -sh /opt/fedai-web | cut -f1) app, $(df -h / | awk "NR==2{print \$5}") of disk used"'
rm -f /tmp/fedai-web-deploy.bundle
