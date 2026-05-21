'use strict';

const { Router } = require('express');
const {
  createAppointment,
  getAppointments,
  updateAppointmentState,
  updateAppointment,
} = require('../controllers/appointmentController');
const { validateAppointment } = require('../middleware/validator');

const router = Router();

router.get('/appointments', getAppointments);
router.post('/appointments', validateAppointment, createAppointment);
router.patch('/appointments/:id', updateAppointment);
router.patch('/appointments/:id/estado', updateAppointmentState);

module.exports = router;