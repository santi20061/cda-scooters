'use strict';
require('dotenv').config();

module.exports = {
  PORT:                  process.env.PORT                  || 3001,
  EMAIL_HOST:            process.env.EMAIL_HOST            || 'smtp.gmail.com',
  EMAIL_PORT:            process.env.EMAIL_PORT            || 587,
  EMAIL_USER:            process.env.EMAIL_USER,
  EMAIL_PASS:            process.env.EMAIL_PASS,
  EMAIL_DEST:            process.env.EMAIL_DEST            || process.env.EMAIL_USER,
  WHATSAPP_SERVICE_URL:  process.env.WHATSAPP_SERVICE_URL  || 'http://localhost:3004',
};
