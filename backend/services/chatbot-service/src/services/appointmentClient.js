'use strict';

const axios = require('axios');
const { APPOINTMENT_SERVICE_URL } = require('../config/config');

function normalizarTexto(texto) {
  return String(texto || '').trim().replace(/\s+/g, ' ');
}

function escaparRegex(texto) {
  return String(texto).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function extraerValorPorEtiquetas(texto, etiquetas) {
  const fuente = String(texto || '');
  const coincidencias = [];

  etiquetas.forEach((etiqueta) => {
    const regex = new RegExp(`\\b${escaparRegex(etiqueta)}\\s*[:=]`, 'ig');
    let match;
    while ((match = regex.exec(fuente)) !== null) {
      coincidencias.push({ etiqueta, inicio: match.index, finEtiqueta: regex.lastIndex });
    }
  });

  if (!coincidencias.length) return '';

  coincidencias.sort((a, b) => a.inicio - b.inicio);

  const valores = coincidencias.map((actual, index) => {
    const siguiente = coincidencias[index + 1];
    const valor = fuente.slice(actual.finEtiqueta, siguiente ? siguiente.inicio : fuente.length);
    return {
      etiqueta: actual.etiqueta,
      valor: normalizarTexto(valor.replace(/^[\s,:-]+/, '')),
    };
  });

  return valores;
}

function parsearCampos(texto) {
  const valores = extraerValorPorEtiquetas(texto, [
    'nombre',
    'telefono',
    'fecha',
    'hora',
    'servicio',
    'placa',
    'tipoVehiculo',
    'tipo de vehiculo',
    'observaciones',
    'nota',
  ]);

  const obtener = (...etiquetas) => {
    for (const etiqueta of etiquetas) {
      const encontrado = valores.find(item => item.etiqueta.toLowerCase() === etiqueta.toLowerCase());
      if (encontrado?.valor) return encontrado.valor;
    }
    return '';
  };

  return {
    nombre: obtener('nombre'),
    telefono: obtener('telefono'),
    fecha: obtener('fecha'),
    hora: obtener('hora'),
    servicio: obtener('servicio') || 'Revisión técnico-mecánica',
    placa: obtener('placa'),
    tipoVehiculo: obtener('tipoVehiculo') || obtener('tipo de vehiculo') || '',
    observaciones: obtener('observaciones') || obtener('nota') || '',
  };
}

function tieneCamposMinimos(datos) {
  return Boolean(datos.nombre && datos.telefono && datos.fecha && datos.hora && datos.servicio);
}

function formatearCita(cita, indice) {
  return [
    `${indice}. ${cita.nombre} - ${cita.fecha} ${cita.hora}`,
    `   Tel: ${cita.telefono}`,
    cita.placa ? `   Placa: ${cita.placa}` : null,
    cita.estado ? `   Estado: ${cita.estado}` : null,
  ].filter(Boolean).join('\n');
}

async function crearCitaDesdeTexto(texto) {
  const datos = parsearCampos(texto);
  if (!tieneCamposMinimos(datos)) {
    return {
      ok: false,
      error: 'Faltan datos. Usa: nombre, telefono, fecha, hora y servicio. También puedes incluir placa, tipoVehiculo y observaciones.',
    };
  }

  const response = await axios.post(`${APPOINTMENT_SERVICE_URL}/appointments`, datos, {
    headers: { 'Content-Type': 'application/json' },
    timeout: 15000,
  });

  return {
    ok: true,
    data: response.data,
  };
}

async function consultarCitasRecientes(limite = 3) {
  const response = await axios.get(`${APPOINTMENT_SERVICE_URL}/appointments`, {
    timeout: 15000,
  });

  const appointments = Array.isArray(response.data?.appointments) ? response.data.appointments : [];
  return appointments.slice(0, limite).map(formatearCita);
}

async function findAppointmentByUUID(uuid) {
  const response = await axios.get(`${APPOINTMENT_SERVICE_URL}/appointments`, { timeout: 15000 });
  const appointments = Array.isArray(response.data?.appointments) ? response.data.appointments : [];
  return appointments.find(a => a.id === uuid || String(a.id) === String(uuid));
}

async function updateAppointmentEstadoByUUID(uuid, estado) {
  const appt = await findAppointmentByUUID(uuid);
  if (!appt) throw new Error('Cita no encontrada con ese id');
  const id = appt._id || appt.id;
  // appointment-service expects Mongo _id for the PATCH route
  const url = `${APPOINTMENT_SERVICE_URL}/appointments/${id}/estado`;
  const response = await axios.patch(url, { estado }, { timeout: 15000 });
  return response.data;
}

async function updateAppointmentFechaHoraByUUID(uuid, fecha, hora) {
  const appt = await findAppointmentByUUID(uuid);
  if (!appt) throw new Error('Cita no encontrada con ese id');
  const id = appt._id || appt.id;
  const url = `${APPOINTMENT_SERVICE_URL}/appointments/${id}`;
  const body = {};
  if (fecha) body.fecha = fecha;
  if (hora) body.hora = hora;
  // marcar como reprogramada
  body.estado = 'reprogramada';
  const response = await axios.patch(url, body, { timeout: 15000 });
  return response.data;
}

module.exports = {
  crearCitaDesdeTexto,
  consultarCitasRecientes,
  parsearCampos,
  findAppointmentByUUID,
  updateAppointmentEstadoByUUID,
  updateAppointmentFechaHoraByUUID,
};