#!/bin/sh
set -eu

API_BASE_URL_VALUE="${API_BASE_URL:-}"
printf 'window.__API_BASE_URL__ = "%s";\n' "$API_BASE_URL_VALUE" > /usr/share/nginx/html/config.js

exec nginx -g 'daemon off;'