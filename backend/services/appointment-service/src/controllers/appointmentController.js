'use strict';

const Appointment      = require('../models/Appointment');
const { sendWhatsApp } = require('../services/whatsappService');

async function createAppointment(req, res, next) {
  try {
    const { nombre, correo, telefono, fecha, hora, servicio } = req.body;

    if (!nombre || !correo || !telefono || !fecha || !hora || !servicio) {
      return res.status(400).json({
        ok: false,
        error: 'Campos requeridos: nombre, correo, telefono, fecha, hora, servicio',
      });
    }

    const appointment = await Appointment.create({
      nombre, correo, telefono, fecha, hora, servicio,
    });

    sendWhatsApp(
      telefono,
      `Hola ${nombre}, tu cita fue agendada para el ${fecha} a las ${hora}. ¡Te esperamos en CDA Scooters! 🏍️`
    ).catch((e) => console.warn('[appointment] WhatsApp falló:', e.message));

    res.status(201).json({ ok: true, appointment });
  } catch (err) {
    next(err);
  }
}

async function getAppointments(_req, res, next) {
  try {
    const appointments = await Appointment.find().sort({ createdAt: -1 });
    res.json({ ok: true, total: appointments.length, appointments });
  } catch (err) {
    next(err);
  }
}

async function cancelAppointment(req, res, next) {
  try {
    const appointment = await Appointment.findOneAndUpdate(
      { id: req.params.id },
      { estado: 'cancelada' },
      { new: true }
    );
    if (!appointment) {
      return res.status(404).json({ ok: false, error: 'Cita no encontrada' });
    }
    res.json({ ok: true, appointment });
  } catch (err) {
    next(err);
  }
}

module.exports = { createAppointment, getAppointments, cancelAppointment };
