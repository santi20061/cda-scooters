'use strict';

const { sendContactEmail } = require('../services/emailService');

async function createContact(req, res, next) {
  try {
    const { nombre, correo, telefono, mensaje } = req.body;
    console.log(`[contact] nuevo mensaje de ${nombre} <${correo}> tel:${telefono}`);

    // Email en segundo plano — no bloquea la respuesta al usuario
    sendContactEmail({ nombre, correo, telefono, mensaje })
      .then(() => console.log('[contact] email enviado correctamente'))
      .catch((e) => console.warn('[contact] email falló (no crítico):', e.message));

    console.log('[contact] respuesta 201 enviada al gateway');
    res.status(201).json({ ok: true, mensaje: '¡Mensaje recibido! Te contactaremos pronto ✅' });
  } catch (err) {
    console.error('[contact] error inesperado:', err.message);
    next(err);
  }
}

module.exports = { createContact };
