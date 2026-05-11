'use strict';

const nodemailer = require('nodemailer');
const config     = require('../config/config');

const transporter = nodemailer.createTransport({
  host:   config.EMAIL_HOST,
  port:   Number(config.EMAIL_PORT),
  secure: false,
  auth: { user: config.EMAIL_USER, pass: config.EMAIL_PASS },
});

async function sendAppointmentEmail({ nombre, correo, telefono, fecha, hora, servicio }) {
  await transporter.sendMail({
    from:    `"CDA Scooters SAS" <${config.EMAIL_USER}>`,
    to:      config.EMAIL_DEST,
    subject: `📅 Nueva cita — ${nombre} (${fecha} ${hora})`,
    html: `
      <h2 style="color:#1a1a2e">Nueva cita agendada</h2>
      <table cellpadding="6">
        <tr><td><strong>Nombre:</strong></td><td>${nombre}</td></tr>
        <tr><td><strong>Correo:</strong></td><td>${correo}</td></tr>
        <tr><td><strong>Teléfono:</strong></td><td>${telefono}</td></tr>
        <tr><td><strong>Fecha:</strong></td><td>${fecha}</td></tr>
        <tr><td><strong>Hora:</strong></td><td>${hora}</td></tr>
        <tr><td><strong>Servicio:</strong></td><td>${servicio}</td></tr>
      </table>
    `,
  });
}

module.exports = { sendAppointmentEmail };
