#!/bin/sh
set -e

# Fallback to 8000 if PORT isn't provided by the platform for some reason.
PORT="${PORT:-8000}"

echo "Starting uvicorn on port: ${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}"
