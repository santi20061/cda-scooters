'use strict';

require('dotenv').config();

module.exports = {
  PORT: process.env.PORT || 3002,
  MONGO_URI: process.env.MONGO_URI_ATLAS || process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/cda_appointments',
  MONGO_DB_NAME: process.env.MONGO_DB_NAME || 'cda_appointments',
  WHATSAPP_HABILITADO: process.env.WHATSAPP_HABILITADO !== 'false',
  WHATSAPP_INSTANCE_ID: process.env.WHATSAPP_INSTANCE_ID || '',
  WHATSAPP_TOKEN: process.env.WHATSAPP_TOKEN || '',
  WHATSAPP_DESTINO: process.env.WHATSAPP_DESTINO || '3202294468',
  WHATSAPP_TIMEOUT_MS: Number(process.env.WHATSAPP_TIMEOUT_MS || 15000),
};
