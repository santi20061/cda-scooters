'use strict';

const { randomUUID } = require('crypto');
const Appointment = require('../models/Appointment');
const { enviar } = require('../services/whatsappService');
const config = require('../config/config');

const DESTINO_WHATSAPP = config.WHATSAPP_DESTINO || '3202294468';
const NOTA_CHAT = '\n\nRecuerde: los cambios o cancelaciones solo son posibles desde el chat de la página.';

function construirMensaje(appointment) {
  return [
    '📅 Nueva cita agendada',
    '',
    `Nombre: ${appointment.nombre}`,
    `Teléfono: ${appointment.telefono}`,
    `Fecha: ${appointment.fecha}`,
    `Hora: ${appointment.hora}`,
    `Servicio: ${appointment.servicio}`,
    appointment.placa ? `Placa: ${appointment.placa}` : null,
    appointment.tipoVehiculo ? `Tipo de vehículo: ${appointment.tipoVehiculo}` : null,
    appointment.observaciones ? `Observaciones: ${appointment.observaciones}` : null,
    '',
  ].filter(Boolean).join('\n');
}

async function createAppointment(req, res, next) {
  try {
    const {
      nombre,
      telefono,
      fecha,
      hora,
      servicio,
      placa = '',
      tipoVehiculo = '',
      observaciones = '',
    } = req.body;

    const appointment = await Appointment.create({
      id: randomUUID(),
      nombre,
      telefono,
      fecha,
      hora,
      servicio,
      placa,
      tipoVehiculo,
      observaciones,
      estado: 'pendiente',
    });

    const mensaje = construirMensaje(appointment);
    // Enviar notificación a la empresa (sin recordatorio, la cita queda confirmada al agendar)
    const whatsappEnviado = await enviar(DESTINO_WHATSAPP, mensaje);

    // Enviar confirmación al cliente con el ID de turno y guía para control de turnos
    const mensajeCliente = [
      `Hola ${appointment.nombre || ''},`,
      '',
      'Tu agendamiento fue un éxito. Aquí están los detalles:',
      `Fecha: ${appointment.fecha}`,
      `Hora: ${appointment.hora}`,
      `Servicio: ${appointment.servicio}`,
      '',
      `ID de tu turno: ${appointment.id}`,
      '',
      'Si deseas cancelar o reprogramar, puedes hacerlo desde el chat indicando el comando:',
      `"cancelar ${appointment.id}" o "reprogramar ${appointment.id} fecha: YYYY-MM-DD hora: HH:MM"`,
      '',
      'Gracias por preferirnos.'
    ].join('\n');

    const whatsappEnviadoAlCliente = await enviar(appointment.telefono, mensajeCliente + NOTA_CHAT);

    res.status(201).json({
      ok: true,
      appointment,
      whatsappEnviadoEmpresa: whatsappEnviado,
      whatsappEnviadoCliente: whatsappEnviadoAlCliente,
      mensaje: whatsappEnviado && whatsappEnviadoAlCliente
        ? 'Cita registrada y notificada por WhatsApp (empresa y cliente)'
        : whatsappEnviado
          ? 'Cita registrada; notificada a la empresa, pero no al cliente'
          : whatsappEnviadoAlCliente
            ? 'Cita registrada; notificada al cliente, pero no a la empresa'
            : 'Cita registrada, pero no se pudieron enviar notificaciones por WhatsApp',
    });
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

async function updateAppointmentState(req, res, next) {
  try {
    const { estado } = req.body;
    const estadosValidos = ['pendiente', 'confirmada', 'cancelada', 'reprogramada'];

    if (!estadosValidos.includes(estado)) {
      return res.status(400).json({ ok: false, error: 'Estado inválido' });
    }

    // Si solicitan cancelar, notificar a la empresa y eliminar la cita
    if (estado === 'cancelada') {
      const appointment = await Appointment.findById(req.params.id);
      if (!appointment) return res.status(404).json({ ok: false, error: 'Cita no encontrada' });

      // Notificar a la empresa
      const mensajeEmpresa = [
        '⚠️ Cita cancelada',
        '',
        `Nombre: ${appointment.nombre}`,
        `Teléfono: ${appointment.telefono}`,
        `Fecha: ${appointment.fecha}`,
        `Hora: ${appointment.hora}`,
        `Servicio: ${appointment.servicio}`,
        '',
        `ID: ${appointment.id}`,
      ].join('\n');

      const sentEmpresa = await enviar(DESTINO_WHATSAPP, mensajeEmpresa + NOTA_CHAT);

      // Notificar al cliente antes de eliminar
      const mensajeClienteCancel = [
        `Hola ${appointment.nombre || ''},`,
        '',
        'Tu cita ha sido cancelada.',
        `Fecha: ${appointment.fecha}`,
        `Hora: ${appointment.hora}`,
        '',
        `ID: ${appointment.id}`,
        '',
        'Si deseas reagendar, usa el chat de la página con: "reprogramar <ID> fecha: YYYY-MM-DD hora: HH:MM"',
      ].join('\n');

      const sentCliente = await enviar(appointment.telefono, mensajeClienteCancel + NOTA_CHAT);

      // Eliminar la cita de la base de datos
      await Appointment.findByIdAndDelete(req.params.id);

      return res.json({ ok: true, deleted: true, whatsappEmpresa: sentEmpresa, whatsappCliente: sentCliente });
    }

    // Otros estados: actualizar campo estado
    const appointment = await Appointment.findByIdAndUpdate(
      req.params.id,
      { estado },
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

async function updateAppointment(req, res, next) {
  try {
    const allowed = ['fecha', 'hora', 'estado', 'nombre', 'telefono', 'servicio', 'placa', 'tipoVehiculo', 'observaciones'];
    const updates = {};
    allowed.forEach((k) => {
      if (req.body[k] !== undefined) updates[k] = req.body[k];
    });

    if (!Object.keys(updates).length) {
      return res.status(400).json({ ok: false, error: 'No hay campos para actualizar' });
    }

    const appointment = await Appointment.findByIdAndUpdate(req.params.id, updates, { new: true });
    if (!appointment) {
      return res.status(404).json({ ok: false, error: 'Cita no encontrada' });
    }

    // Notificar a la empresa
    const mensajeEmpresa = [
      '📅 Cita actualizada',
      '',
      `Nombre: ${appointment.nombre}`,
      `Teléfono: ${appointment.telefono}`,
      `Fecha: ${appointment.fecha}`,
      `Hora: ${appointment.hora}`,
      `Servicio: ${appointment.servicio}`,
      '',
    ].join('\n');
    const sentEmpresa = await enviar(DESTINO_WHATSAPP, mensajeEmpresa + NOTA_CHAT);

    // Notificar al cliente
    const mensajeCliente = [
      `Hola ${appointment.nombre || ''},`,
      '',
      'Tu cita ha sido reprogramada.',
      `Nueva fecha: ${appointment.fecha}`,
      `Nueva hora: ${appointment.hora}`,
      '',
      `ID de tu turno: ${appointment.id}`,
      '',
      'Si deseas cancelar o reprogramar, usa el chat: "cancelar <ID>" o "reprogramar <ID> fecha: YYYY-MM-DD hora: HH:MM"',
    ].join('\n');
    const sentCliente = await enviar(appointment.telefono, mensajeCliente + NOTA_CHAT);

    res.json({ ok: true, appointment, whatsappEmpresa: sentEmpresa, whatsappCliente: sentCliente });
  } catch (err) {
    next(err);
  }
}

module.exports = {
  createAppointment,
  getAppointments,
  updateAppointmentState,
  updateAppointment,
};
