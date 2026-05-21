'use strict';

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
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
  const server = app.listen(PORT, () => {
    console.log(`📅 Appointment Service corriendo en puerto ${PORT}`);
  });

  function shutdown(signal) {
    console.error(`[appointment-service] Recibido ${signal}, cerrando...`);
    server.close(() => {
      mongoose.disconnect().then(() => {
        console.error('[appointment-service] MongoDB desconectado');
        process.exit(0);
      }).catch(() => process.exit(1));
    });

    // Forzar salida si no termina en 10s
    setTimeout(() => {
      console.error('[appointment-service] Timeout al cerrar, forzando salida');
      process.exit(1);
    }, 10000).unref();
  }

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('uncaughtException', (err) => {
    console.error('[appointment-service] uncaughtException', err && err.stack ? err.stack : err);
    process.exit(1);
  });
  process.on('unhandledRejection', (reason) => {
    console.error('[appointment-service] unhandledRejection', reason);
    process.exit(1);
  });
}

start().catch((error) => {
  console.error('[appointment-service] No se pudo iniciar:', error.message);
  process.exit(1);
});