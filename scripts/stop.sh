#!/usr/bin/env bash
# Stop the server started by scripts/start.sh (also stops the private demo PostgreSQL).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .run/server.pid ]]; then
  pid=$(cat .run/server.pid)
  # start.sh uses setsid, so pid == process-group id: stop tsx -> node -> demo postgres together
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
  kill -0 -- "-$pid" 2>/dev/null && { echo "still running; sending KILL"; kill -KILL -- "-$pid" 2>/dev/null || true; }
  rm -f .run/server.pid
  echo "stopped"
else
  echo "not running"
fi
