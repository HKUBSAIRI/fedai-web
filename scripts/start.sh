#!/usr/bin/env bash
# Build the UI and start the read-only server on 127.0.0.1 (background). Logs: .run/server.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .run
if [[ -f .run/server.pid ]] && kill -0 "$(cat .run/server.pid)" 2>/dev/null; then
  echo "already running (pid $(cat .run/server.pid))"; exit 0
fi
npm run -s build >/dev/null
setsid nohup node_modules/.bin/tsx server/src/index.ts > .run/server.log 2>&1 < /dev/null &
echo $! > .run/server.pid
for _ in $(seq 1 60); do
  kill -0 "$(cat .run/server.pid)" 2>/dev/null || { echo "server exited; see .run/server.log" >&2; tail -5 .run/server.log >&2; exit 1; }
  if curl -fs "http://127.0.0.1:${FEDAI_PORT:-8787}/api/status" >/dev/null 2>&1; then
    grep -m1 'mode=' .run/server.log || true
    exit 0
  fi
  sleep 1
done
echo "server did not become ready; see .run/server.log" >&2; exit 1
