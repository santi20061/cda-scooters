'use strict';

const { Schema, model } = require('mongoose');

const appointmentSchema = new Schema(
  {
    id: { type: String, required: true, unique: true, trim: true },
    nombre: { type: String, required: true, trim: true },
    telefono: { type: String, required: true, trim: true },
    fecha: { type: String, required: true, trim: true },
    hora: { type: String, required: true, trim: true },
    servicio: { type: String, required: true, trim: true },
    placa: { type: String, trim: true, default: '' },
    tipoVehiculo: { type: String, trim: true, default: '' },
    observaciones: { type: String, trim: true, default: '' },
    estado: {
      type: String,
      enum: ['pendiente', 'confirmada', 'cancelada', 'reprogramada'],
      default: 'pendiente',
    },
  },
  { timestamps: true }
);

module.exports = model('Appointment', appointmentSchema, 'nose');