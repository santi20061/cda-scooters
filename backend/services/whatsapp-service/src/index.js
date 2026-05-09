'use strict';

require('dotenv').config();
const express = require('express');
const { PORT } = require('./config/config');
const whatsappRoutes = require('./routes/whatsappRoutes');
const errorHandler  = require('./middleware/errorHandler');

// Inicializar cliente WhatsApp al arrancar el servicio
require('./services/whatsappClient');

const app = express();
app.use(express.json());

app.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'whatsapp-service' })
);
app.use('/', whatsappRoutes);
app.use(errorHandler);

app.listen(PORT, () =>
  console.log(`📱 WhatsApp Service corriendo en puerto ${PORT}`)
);
