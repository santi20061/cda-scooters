'use strict';

function validateAppointment(req, res, next) {
  const { nombre, telefono, fecha, hora, servicio } = req.body;
  const errors = [];

  if (!nombre?.trim()) errors.push('nombre es requerido');
  if (!telefono?.trim()) errors.push('telefono es requerido');
  if (!fecha?.trim()) errors.push('fecha es requerida');
  if (!hora?.trim()) errors.push('hora es requerida');
  if (!servicio?.trim()) errors.push('servicio es requerido');

  if (errors.length) return res.status(400).json({ ok: false, errors });
  next();
}

module.exports = { validateAppointment };