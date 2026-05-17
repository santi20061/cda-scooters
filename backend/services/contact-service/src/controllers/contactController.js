'use strict';

const { sendContactEmail } = require('../services/emailService');
const { sendWhatsApp }     = require('../services/whatsappService');

const ADMIN_PHONE = process.env.ADMIN_PHONE || '3202294468';

async function createContact(req, res, next) {
  try {
    const { nombre, correo, telefono, mensaje } = req.body;
    console.log(`[contact] nuevo mensaje de ${nombre} <${correo}> tel:${telefono}`);

    // Email en segundo plano — no bloquea la respuesta al usuario
    sendContactEmail({ nombre, correo, telefono, mensaje })
      .then(() => console.log('[contact] email enviado correctamente'))
      .catch((e) => console.warn('[contact] email falló (no crítico):', e.message));

    // WhatsApp al cliente
    sendWhatsApp(
      telefono,
      `Hola ${nombre}, hemos recibido tu mensaje correctamente. Te contactaremos pronto. 🚗`
    ).then(() => console.log('[contact] WA cliente enviado'))
     .catch((e) => console.warn('[contact] WA cliente falló:', e.message));

    // WhatsApp al admin
    sendWhatsApp(
      ADMIN_PHONE,
      `📩 Nuevo contacto:\nNombre: ${nombre}\nTeléfono: ${telefono}\nCorreo: ${correo}\nMensaje: ${mensaje}`
    ).then(() => console.log('[contact] WA admin enviado'))
     .catch((e) => console.warn('[contact] WA admin falló:', e.message));

    console.log('[contact] respuesta 201 enviada al gateway');
    res.status(201).json({ ok: true, mensaje: '¡Mensaje recibido! Te contactaremos pronto ✅' });
  } catch (err) {
    console.error('[contact] error inesperado:', err.message);
    next(err);
  }
}

module.exports = { createContact };
