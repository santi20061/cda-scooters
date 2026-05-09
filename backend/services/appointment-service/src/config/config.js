'use strict';
require('dotenv').config();

module.exports = {
  PORT:                 process.env.PORT                 || 3002,
  MONGODB_URI:          process.env.MONGODB_URI          || 'mongodb://localhost:27017/cda-appointments',
  WHATSAPP_SERVICE_URL: process.env.WHATSAPP_SERVICE_URL || 'http://localhost:3004',
};
