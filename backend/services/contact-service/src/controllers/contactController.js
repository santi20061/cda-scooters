'use strict';

const { sendContactEmail } = require('../services/emailService');
const { sendWhatsApp }     = require('../services/whatsappService');

async function createContact(req, res, next) {
  try {
    const { nombre, correo, telefono, mensaje } = req.body;

    await sendContactEmail({ nombre, correo, telefono, mensaje });

    // WhatsApp es no-crítico: si falla no bloquea la respuesta
    sendWhatsApp(telefono,
      `Hola ${nombre}, hemos recibido tu mensaje correctamente. Te contactaremos pronto. 🚗`
    ).catch((e) => console.warn('[contact] WhatsApp falló:', e.message));

    res.status(201).json({ ok: true, mensaje: '¡Mensaje recibido! Te contactaremos pronto ✅' });
  } catch (err) {
    next(err);
  }
}

module.exports = { createContact };
