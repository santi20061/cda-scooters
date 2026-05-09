'use strict';

const { intents, default: defaultResponse } = require('../data/intents.json');

function findResponse(message) {
  const lower = message.toLowerCase().trim();
  for (const intent of intents) {
    for (const pattern of intent.patterns) {
      if (lower.includes(pattern)) return intent.response;
    }
  }
  return defaultResponse;
}

module.exports = { findResponse };
