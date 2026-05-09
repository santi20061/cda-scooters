'use strict';
require('dotenv').config();

module.exports = {
  PORT:            process.env.PORT || 3003,
  OPENAI_API_KEY:  process.env.OPENAI_API_KEY || null,
};
