'use strict';

const { Router } = require('express');
const {
  createAppointment,
  getAppointments,
  cancelAppointment,
} = require('../controllers/appointmentController');

const router = Router();
router.post('/appointments',       createAppointment);
router.get('/appointments',        getAppointments);
router.delete('/appointments/:id', cancelAppointment);

module.exports = router;
