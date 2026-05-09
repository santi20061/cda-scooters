'use strict';

const axios  = require('axios');
const config = require('../config/config');

async function sendWhatsApp(phone, message) {
  await axios.post(`${config.WHATSAPP_SERVICE_URL}/send`, { phone, message });
}

module.exports = { sendWhatsApp };
