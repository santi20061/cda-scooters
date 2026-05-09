'use strict';

const mongoose   = require('mongoose');
const { MONGODB_URI } = require('./config');

async function connectDB() {
  await mongoose.connect(MONGODB_URI);
  console.log('✅ MongoDB conectado:', MONGODB_URI);
}

module.exports = { connectDB };
