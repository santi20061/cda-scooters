'use strict';

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const fs = require('fs');
const path = require('path');

let clientReady = false;

const DATA_PATH = process.env.WWEBJS_DATA_PATH || '/data/.wwebjs_auth';

function sleep(ms) {
  return new Promise((res) => setTimeout(res, ms));
}

function ensureDataPath() {
  try {
    fs.mkdirSync(DATA_PATH, { recursive: true });
    try { fs.chmodSync(DATA_PATH, 0o700); } catch (e) { /* ignore */ }
  } catch (err) {
    console.warn('No se pudo crear DATA_PATH:', DATA_PATH, err && err.message);
  }
}

function cleanLockFiles() {
  try {
    const candidates = ['SingletonLock', 'SingletonSocket', 'LOCK', 'lockfile', 'Local State'];
    for (const name of candidates) {
      const p = path.join(DATA_PATH, name);
      if (fs.existsSync(p)) {
        try { fs.rmSync(p, { force: true }); console.log('Removed lock file:', p); } catch (e) { /* ignore */ }
      }
    }
  } catch (e) { /* ignore */ }
}

function resetAuth() {
  try {
    if (fs.existsSync(DATA_PATH)) {
      fs.rmSync(DATA_PATH, { recursive: true, force: true });
      console.log('Auth data reset successfully:', DATA_PATH);
    }
  } catch (e) {
    console.warn('Failed to reset auth data:', e && e.message);
  }
}

ensureDataPath();

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: DATA_PATH, clientId: 'main-client' }),
  puppeteer: {
    headless: 'new',
    executablePath: (function findChromium() {
      const candidates = [process.env.PUPPETEER_EXECUTABLE_PATH, '/usr/bin/chromium-browser', '/usr/bin/chromium', '/usr/bin/google-chrome-stable', '/usr/bin/google-chrome'].filter(Boolean);
      for (const c of candidates) {
        try { if (fs.existsSync(c)) return c; } catch (e) { /* ignore */ }
      }
      return process.env.PUPPETEER_EXECUTABLE_PATH || '/usr/bin/chromium-browser';
    })(),
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--no-first-run',
      '--no-zygote',
      '--remote-debugging-port=9222',
      `--user-data-dir=${DATA_PATH}`,
    ],
  },
  webVersionCache: { type: 'none' },
});

// Event handlers
client.on('qr', (qr) => {
  console.log('\n=======================================================');
  console.log('📱 WHATSAPP: Escanea el QR con la app de WhatsApp');
  console.log('=======================================================\n');
  qrcode.generate(qr, { small: true });
  console.log('\n=======================================================\n');
});

client.on('loading_screen', (percent, message) => {
  console.log(`[whatsapp] Cargando... ${percent}% — ${message}`);
});

client.on('authenticated', () => {
  console.log('[whatsapp] ✅ Autenticado — sesión guardada en disco');
});

client.on('ready', () => {
  clientReady = true;
  console.log('[whatsapp] ✅ Cliente listo — mensajes habilitados');
});

client.on('message', (msg) => {
  console.log(`[whatsapp] 📨 Mensaje recibido de ${msg.from}: ${msg.body.substring(0, 60)}`);
});

client.on('message_create', (msg) => {
  if (msg.fromMe) {
    console.log(`[whatsapp] 📤 Mensaje enviado a ${msg.to}`);
  }
});

client.on('auth_failure', (msg) => {
  console.error('[whatsapp] ❌ auth_failure:', msg);
  clientReady = false;
  try { resetAuth(); } catch (e) { console.error('[whatsapp] resetAuth error:', e && e.message); }
  setTimeout(() => initializeClient().catch(console.error), 3000);
});

client.on('disconnected', (reason) => {
  clientReady = false;
  console.warn('[whatsapp] ⚠️  Desconectado:', reason, '— reintentando en 5s');
  setTimeout(() => initializeClient().catch(console.error), 5000);
});

// Initialization with retries
async function initializeClient() {
  const maxAttempts = 6;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      console.log(`Inicializando cliente WhatsApp — intento ${attempt}/${maxAttempts}...`);
      // staggered wait to reduce race conditions
      await sleep(1000 * attempt);
      ensureDataPath();
      cleanLockFiles();
      await client.initialize();
      console.log('Inicialización completada');
      return;
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      console.error(`Error inicializando cliente (intento ${attempt}):`, msg);
      if (msg.toLowerCase().includes('requesting main frame') || msg.toLowerCase().includes('frame')) {
        console.warn('Detected FrameManager issue — cleaning auth data and retrying');
        try { resetAuth(); } catch (e) { console.error('Error resetting auth during retry:', e && e.message); }
      }
      await sleep(2000 * attempt);
    }
  }
  throw new Error('No se pudo inicializar el cliente WhatsApp tras varios intentos');
}

// Public sendMessage that waits for readiness
async function sendMessage(phone, message) {
  if (!clientReady) throw new Error('El cliente WhatsApp aún no está listo.');
  const cleanPhone = String(phone).replace(/\D/g, '');
  const chatId = `${cleanPhone}@c.us`;
  console.log('[whatsapp-client] sending to chatId:', chatId);
  return client.sendMessage(chatId, message);
}

function isReady() { return clientReady; }

// Start
initializeClient().catch((e) => console.error('Inicialización fallida:', e && e.message));

module.exports = { client, sendMessage, isReady, resetAuth, initializeClient };
