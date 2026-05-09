'use strict';

function validateContact(req, res, next) {
  const { nombre, correo, telefono, mensaje } = req.body;
  const errors = [];

  if (!nombre?.trim())                              errors.push('nombre es requerido');
  if (!correo || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(correo)) errors.push('correo inválido');
  if (!telefono?.trim())                            errors.push('telefono es requerido');
  if (!mensaje?.trim())                             errors.push('mensaje es requerido');

  if (errors.length) return res.status(400).json({ ok: false, errors });
  next();
}

module.exports = { validateContact };
