'use strict';

const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const AppointmentSchema = new mongoose.Schema(
  {
    id:       { type: String, default: uuidv4, unique: true },
    nombre:   { type: String, required: true, trim: true },
    correo:   { type: String, required: true, trim: true },
    telefono: { type: String, required: true, trim: true },
    fecha:    { type: String, required: true },
    hora:     { type: String, required: true },
    servicio: {
      type: String,
      required: true,
      enum: ['motocicleta', 'liviano', 'motocarro'],
    },
    estado: {
      type: String,
      default: 'pendiente',
      enum: ['pendiente', 'confirmada', 'cancelada'],
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model('Appointment', AppointmentSchema);
