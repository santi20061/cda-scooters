'use strict';

const { sendMessage, isReady } = require('../services/whatsappClient');

async function send(req, res, next) {
  try {
    const { phone, message } = req.body;
    if (!phone || !message) {
      return res.status(400).json({ error: 'phone y message son requeridos' });
    }
    await sendMessage(phone, message);
    res.json({ ok: true, message: 'Mensaje enviado correctamente' });
  } catch (err) {
    next(err);
  }
}

function status(_req, res) {
  res.json({ ok: true, ready: isReady() });
}

module.exports = { send, status };
