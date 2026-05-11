'use strict';
require('dotenv').config();

module.exports = {
  PORT:                 process.env.PORT                 || 3002,
  MONGODB_URI:          process.env.MONGODB_URI          || 'mongodb://localhost:27017/CDA_db',
  WHATSAPP_SERVICE_URL: process.env.WHATSAPP_SERVICE_URL || 'http://localhost:3004',
  ADMIN_PHONE:          process.env.ADMIN_PHONE          || '3202294468',
  EMAIL_HOST:           process.env.EMAIL_HOST           || 'smtp.gmail.com',
  EMAIL_PORT:           process.env.EMAIL_PORT           || 587,
  EMAIL_USER:           process.env.EMAIL_USER,
  EMAIL_PASS:           process.env.EMAIL_PASS,
  EMAIL_DEST:           process.env.EMAIL_DEST           || 'santiagomartinezaaz009@gmail.com',
};
