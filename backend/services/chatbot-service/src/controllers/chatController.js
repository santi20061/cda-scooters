'use strict';

const { findResponse } = require('../services/chatbotService');

async function chat(req, res) {
  const { message } = req.body;
  if (!message?.trim()) {
    return res.status(400).json({ error: 'message es requerido' });
  }
  const response = findResponse(message);
  res.json({ response });
}

module.exports = { chat };
