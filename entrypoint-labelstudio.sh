#!/bin/bash
set -e

echo "🚀 Starting Label Studio entrypoint..."

# Set logging levels before starting
export DJANGO_LOG_LEVEL=WARNING
export LOG_LEVEL=WARNING

# Start Label Studio with aggressive filtering
label-studio start --init \
  --username admin@example.com \
  --password pass123 \
  --no-browser \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level WARNING 2>&1 | \
  grep -vE "(faker\.|Looking for locale|localized|Specified locale|django\.server::|django\.request::|Invalid page|Not Found: /api/projects|Traceback|File \"|^\s+|raise |EmptyPage|During handling|paginate_queryset|Redis not connected|deprecated method)" &

# Wait for Label Studio to be ready
echo "⏳ Waiting for Label Studio to start..."
until curl -s http://127.0.0.1:8080/api/health > /dev/null 2>&1; do
    sleep 2
done
echo "✅ Label Studio is up and running!"

echo "📍 Access Label Studio at: http://localhost:8026"
echo "🔐 Login with email: admin@example.com, password: pass123"

# Keep Label Studio running
wait -n