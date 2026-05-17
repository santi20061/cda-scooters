'use strict';

require('dotenv').config();
const express = require('express');
const { PORT } = require('./config/config');
const whatsappRoutes = require('./routes/whatsappRoutes');
const errorHandler  = require('./middleware/errorHandler');

// Inicializar cliente WhatsApp al arrancar el servicio
require('./services/whatsappClient');

const app = express();
app.use((req, _res, next) => {
  console.log(`[whatsapp-service] ${req.method} ${req.originalUrl}`);
  next();
});

app.use(express.json({ limit: '1mb' }));

app.use((err, _req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({ error: 'JSON invalido' });
  }
  next(err);
});

app.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'whatsapp-service' })
);
app.use('/', whatsappRoutes);
app.use(errorHandler);

app.listen(PORT, () =>
  console.log(`📱 WhatsApp Service corriendo en puerto ${PORT}`)
);
