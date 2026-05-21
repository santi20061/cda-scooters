'use strict';

const axios = require('axios');
const config = require('../config/config');

function normalizarTelefono(telefono) {
  const numero = String(telefono || '').replace(/[^0-9]/g, '');
  if (!numero) return '';
  return numero.startsWith('57') ? numero : `57${numero}`;
}

function isHabilitado() {
  return Boolean(
    config.WHATSAPP_HABILITADO &&
    config.WHATSAPP_INSTANCE_ID &&
    config.WHATSAPP_TOKEN
  );
}

async function enviar(telefono, mensaje) {
  if (!isHabilitado()) {
    console.warn('[whatsapp-service] WhatsApp deshabilitado o sin credenciales');
    return false;
  }

  const numero = normalizarTelefono(telefono);
  if (!numero || !mensaje) return false;

  try {
    const body = new URLSearchParams({
      token: config.WHATSAPP_TOKEN,
      to: numero,
      body: mensaje,
    }).toString();

    const response = await axios.post(
      `https://api.ultramsg.com/${config.WHATSAPP_INSTANCE_ID}/messages/chat`,
      body,
      {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        timeout: config.WHATSAPP_TIMEOUT_MS,
      }
    );

    const responseBody = typeof response.data === 'string'
      ? response.data
      : JSON.stringify(response.data);

    return response.status === 200 && responseBody.includes('"sent":"true"');
  } catch (error) {
    const status = error.response?.status;
    const data = error.response?.data;
    console.error('[whatsapp-service] Error enviando WhatsApp:', {
      message: error.message,
      status,
      data,
    });
    return false;
  }
}

module.exports = { enviar, isHabilitado, normalizarTelefono };