'use strict';

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

let clientReady = false;

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth' }),
  puppeteer: {
    headless: true,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
    ],
  },
});

client.on('qr', (qr) => {
  console.log('\n📱 Escanea este QR con la app de WhatsApp:\n');
  qrcode.generate(qr, { small: true });
});

client.on('authenticated', () => {
  console.log('✅ WhatsApp autenticado — sesión guardada');
});

client.on('ready', () => {
  clientReady = true;
  console.log('✅ Cliente WhatsApp listo para enviar mensajes');
});

client.on('disconnected', (reason) => {
  clientReady = false;
  console.warn('⚠️  WhatsApp desconectado:', reason);
  setTimeout(() => client.initialize().catch(console.error), 5000);
});

client.initialize().catch(console.error);

function normalizarNumero(phone) {
  const digits = phone.replace(/\D/g, '');
  if (digits.length === 10) return `57${digits}@c.us`;
  return `${digits}@c.us`;
}

async function sendMessage(phone, message) {
  if (!clientReady) {
    throw new Error('El cliente WhatsApp aún no está listo. Escanea el QR primero.');
  }
  const chatId = normalizarNumero(phone);
  await client.sendMessage(chatId, message);
}

function isReady() {
  return clientReady;
}

module.exports = { client, sendMessage, isReady };
