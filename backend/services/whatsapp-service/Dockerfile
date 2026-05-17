FROM node:18-alpine

RUN apk add --no-cache \
    chromium \
    nss \
    freetype \
    harfbuzz \
    ca-certificates \
    ttf-freefont \
    bash \
    curl

ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser
ENV WWEBJS_DATA_PATH=/data/.wwebjs_auth

WORKDIR /app
COPY package*.json ./
RUN npm install --omit=dev
COPY src/ ./src/

# Create a persistent data dir for whatsapp-web.js and set permissions
RUN mkdir -p /data/.wwebjs_auth && chown -R node:node /data /app

# Ensure common chromium binary names exist (some alpine variants use /usr/bin/chromium)
RUN if [ -f /usr/bin/chromium ] && [ ! -f /usr/bin/chromium-browser ]; then ln -s /usr/bin/chromium /usr/bin/chromium-browser || true; fi
RUN if [ -f /usr/bin/chromium-browser ] && [ ! -f /usr/bin/chromium ]; then ln -s /usr/bin/chromium-browser /usr/bin/chromium || true; fi

USER node

EXPOSE 3004
CMD ["node", "src/index.js"]

