#!/usr/bin/env bash
# start_scanner.sh — Launch the signal scanner daemon in a tmux session.
#
# Usage:
#   ./start_scanner.sh            # start (or re-attach if already running)
#   ./start_scanner.sh stop       # stop the daemon
#   ./start_scanner.sh status     # show running state
#   ./start_scanner.sh logs       # tail the log file

set -euo pipefail

SESSION="stock-scanner"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$WORKDIR/logs/scanner.log"
PYTHON="${PYTHON:-python3}"

mkdir -p "$WORKDIR/logs"

case "${1:-start}" in

  start)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Scanner already running in tmux session '$SESSION'."
      echo "  Re-attach : tmux attach -t $SESSION"
      echo "  Stop      : $0 stop"
      exit 0
    fi
    echo "Starting scanner daemon…"
    tmux new-session -d -s "$SESSION" \
      "cd '$WORKDIR' && $PYTHON scanner_daemon.py 2>&1 | tee -a '$LOGFILE'"
    sleep 1
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "✓ Scanner started (session: $SESSION)"
      echo "  Attach  : tmux attach -t $SESSION"
      echo "  Logs    : tail -f $LOGFILE"
    else
      echo "✗ Failed to start. Check: $LOGFILE"
    fi
    ;;

  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux send-keys -t "$SESSION" C-c
      sleep 2
      tmux kill-session -t "$SESSION" 2>/dev/null || true
      echo "✓ Scanner stopped."
    else
      echo "Scanner is not running."
    fi
    ;;

  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "✓ Running (tmux session: $SESSION)"
      echo "  Attach : tmux attach -t $SESSION"
    else
      echo "✗ Not running"
    fi
    ;;

  logs)
    if [ -f "$LOGFILE" ]; then
      tail -f "$LOGFILE"
    else
      echo "No log file yet: $LOGFILE"
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
