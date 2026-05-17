'use strict';

const { sendMessage, isReady } = require('../services/whatsappClient');

async function send(req, res, next) {
  try {
    const { phone, message } = req.body;
    console.log(`[whatsapp] POST /send → phone=${phone} msg="${String(message).substring(0, 60)}"`);

    if (!phone || !message) {
      console.warn('[whatsapp] Petición rechazada: faltan phone o message');
      return res.status(400).json({ ok: false, error: 'phone y message son requeridos' });
    }

    if (!isReady()) {
      console.warn('[whatsapp] Cliente no listo — devolviendo 503');
      return res.status(503).json({
        ok: false,
        error: 'WhatsApp aún no está listo. Escanee el QR en los logs del servicio.',
      });
    }

    await sendMessage(phone, message);
    console.log(`[whatsapp] ✅ Mensaje enviado a ${phone}`);
    res.json({ ok: true, message: 'Mensaje enviado correctamente' });
  } catch (err) {
    console.error('[whatsapp] ❌ Error al enviar:', err && err.message);
    next(err);
  }
}

function status(_req, res) {
  const ready = isReady();
  console.log(`[whatsapp] GET /status → ready=${ready}`);
  res.json({ ok: true, ready });
}

module.exports = { send, status };
