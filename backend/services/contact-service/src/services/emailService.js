'use strict';

const nodemailer = require('nodemailer');
const config     = require('../config/config');

const transporter = nodemailer.createTransport({
  host:   config.EMAIL_HOST,
  port:   Number(config.EMAIL_PORT),
  secure: false,
  auth: {
    user: config.EMAIL_USER,
    pass: config.EMAIL_PASS,
  },
});

async function sendContactEmail({ nombre, correo, telefono, mensaje }) {
  await transporter.sendMail({
    from:    `"CDA Scooters SAS" <${config.EMAIL_USER}>`,
    to:      config.EMAIL_DEST,
    subject: `Nuevo contacto de ${nombre}`,
    html: `
      <h2 style="color:#1a1a2e">Nuevo mensaje de contacto</h2>
      <table>
        <tr><td><strong>Nombre:</strong></td><td>${nombre}</td></tr>
        <tr><td><strong>Correo:</strong></td><td>${correo}</td></tr>
        <tr><td><strong>Teléfono:</strong></td><td>${telefono}</td></tr>
      </table>
      <p><strong>Mensaje:</strong></p>
      <p>${mensaje}</p>
    `,
  });
}

module.exports = { sendContactEmail };
