'use strict';

const { sendContactEmail } = require('../services/emailService');
const { sendWhatsApp }     = require('../services/whatsappService');

const ADMIN_PHONE = process.env.ADMIN_PHONE || '3202294468';

async function createContact(req, res, next) {
  try {
    const { nombre, correo, telefono, mensaje } = req.body;

    // Email al admin
    await sendContactEmail({ nombre, correo, telefono, mensaje });

    // WhatsApp al cliente
    sendWhatsApp(telefono,
      `Hola ${nombre}, hemos recibido tu mensaje correctamente. Te contactaremos pronto. 🚗`
    ).catch((e) => console.warn('[contact] WA cliente falló:', e.message));

    // WhatsApp al admin
    sendWhatsApp(ADMIN_PHONE,
      `📩 Nuevo contacto:\nNombre: ${nombre}\nTeléfono: ${telefono}\nCorreo: ${correo}\nMensaje: ${mensaje}`
    ).catch((e) => console.warn('[contact] WA admin falló:', e.message));

    res.status(201).json({ ok: true, mensaje: '¡Mensaje recibido! Te contactaremos pronto ✅' });
  } catch (err) {
    next(err);
  }
}

module.exports = { createContact };
