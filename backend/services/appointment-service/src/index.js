'use strict';

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { PORT } = require('./config/config');
const { connectDatabase } = require('./config/database');
const appointmentRoutes = require('./routes/appointmentRoutes');
const errorHandler = require('./middleware/errorHandler');

const app = express();

app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'appointment-service' });
});

app.use('/', appointmentRoutes);
app.use(errorHandler);

async function start() {
  await connectDatabase();
  app.listen(PORT, () => {
    console.log(`📅 Appointment Service corriendo en puerto ${PORT}`);
  });
}

start().catch((error) => {
  console.error('[appointment-service] No se pudo iniciar:', error.message);
  process.exit(1);
});