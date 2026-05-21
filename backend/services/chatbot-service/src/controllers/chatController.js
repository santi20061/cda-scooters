'use strict';

const { findResponse } = require('../services/chatbotService');
const {
  crearCitaDesdeTexto,
  // consultarCitasRecientes,
  findAppointmentByUUID,
  updateAppointmentEstadoByUUID,
  updateAppointmentFechaHoraByUUID,
} = require('../services/appointmentClient');

function esConsultaCitas(texto) {
  return /\b(consultar|ver|mostrar|mis)\s+(citas?|agendas?|turnos?)\b/i.test(texto)
    || /\b(citas?|agendas?|turnos?)\s+(recientes|registradas|pendientes)\b/i.test(texto);
}

function esIntentoAgendar(texto) {
  return /\b(agendar|agenda|reservar|programar|crear)\b/i.test(texto)
    && /\b(nombre|telefono|fecha|hora|servicio)\b/i.test(texto);
}

async function chat(req, res) {
  const { message } = req.body;
  if (!message?.trim()) {
    return res.status(400).json({ error: 'message es requerido' });
  }

  const texto = message.trim();

  if (esIntentoAgendar(texto)) {
    try {
      const resultado = await crearCitaDesdeTexto(texto);
      if (!resultado.ok) {
        return res.json({
          response: resultado.error,
          action: 'appointment.create',
        });
      }

      const cita = resultado.data?.appointment;
      return res.json({
        response: cita
          ? `Listo, agendé tu cita para ${cita.fecha} a las ${cita.hora}.`
          : 'Listo, tu cita fue registrada.',
        action: 'appointment.create',
        appointment: cita || null,
      });
    } catch (error) {
      return res.status(502).json({
        error: 'No pude agendar la cita en este momento.',
        details: error.response?.data || error.message,
      });
    }
  }

  // Control de turnos: confirmar / cancelar por id
  // Reprogramar cita: "reprogramar <id> fecha: YYYY-MM-DD hora: HH:MM"
  if (/\b(reprogramar|reagendar|reprogramo|reagendo)\b/i.test(texto)) {
    const uuidMatch = texto.match(/[0-9a-fA-F\-]{36,36}/);
    if (!uuidMatch) {
      return res.json({ response: 'Incluye el ID (UUID) de la cita para reprogramar. Ej: "reprogramar <ID> fecha: 2026-05-25 hora: 11:00"', action: 'appointment.reprogram' });
    }
    const uuid = uuidMatch[0];
    const dateMatch = texto.match(/\b\d{4}-\d{2}-\d{2}\b/);
    const timeMatch = texto.match(/\b\d{1,2}:\d{2}\b/);
    if (!dateMatch || !timeMatch) {
      return res.json({ response: 'Por favor indica la nueva fecha y hora en formato YYYY-MM-DD y HH:MM. Ej: "reprogramar <ID> fecha: 2026-05-25 hora: 11:00"', action: 'appointment.reprogram' });
    }
    const fecha = dateMatch[0];
    const hora = timeMatch[0];
    try {
      const result = await updateAppointmentFechaHoraByUUID(uuid, fecha, hora);
      const cita = result?.appointment || result?.data?.appointment || null;
      return res.json({ response: cita ? `Listo, reprogramé la cita para ${cita.fecha} a las ${cita.hora}.` : 'Cita reprogramada.', action: 'appointment.reprogram', appointment: cita, result });
    } catch (err) {
      return res.status(502).json({ error: 'No pude reprogramar la cita.', details: err.message });
    }
  }
  if (/\b(confirmar|confirmo|confirmar cita)\b/i.test(texto) || /\b(cancelar|cancelo|cancelar cita)\b/i.test(texto)) {
    const accion = /\b(cancelar|cancelo|cancelar cita)\b/i.test(texto) ? 'cancelar' : 'confirmar';
    // Buscar UUID dentro del texto
    const uuidMatch = texto.match(/[0-9a-fA-F\-]{36,36}/);
    if (!uuidMatch) {
      return res.json({ response: 'Por favor incluye el id de la cita (UUID) para confirmar o cancelar. Ej: "confirmar 9779b041-c04c-491e-a8c5-855209257ebf"', action: 'appointment.control' });
    }
    const uuid = uuidMatch[0];
    try {
      const nuevoEstado = accion === 'cancelar' ? 'cancelada' : 'confirmada';
      const result = await updateAppointmentEstadoByUUID(uuid, nuevoEstado);
      return res.json({ response: `Cita ${accion}da correctamente.`, action: 'appointment.control', result });
    } catch (err) {
      return res.status(502).json({ error: 'No pude actualizar la cita.', details: err.message });
    }
  }

  const response = findResponse(message);
  res.json({ response, action: 'faq' });
}

module.exports = { chat };
