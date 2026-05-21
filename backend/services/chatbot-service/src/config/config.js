'use strict';
require('dotenv').config();

module.exports = {
  PORT: process.env.PORT || 3003,
  APPOINTMENT_SERVICE_URL: process.env.APPOINTMENT_SERVICE_URL || 'http://localhost:3002',
};
