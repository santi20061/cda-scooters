'use strict';

require('dotenv').config();
const express       = require('express');
const cors          = require('cors');
const { PORT }      = require('./config/config');
const contactRoutes = require('./routes/contactRoutes');
const errorHandler  = require('./middleware/errorHandler');

const app = express();
app.use(cors());
app.use(express.json());

app.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'contact-service' })
);
app.use('/', contactRoutes);
app.use(errorHandler);

app.listen(PORT, () =>
  console.log(`📧 Contact Service corriendo en puerto ${PORT}`)
);
