'use strict';

const mongoose = require('mongoose');
const { MONGO_URI, MONGO_DB_NAME } = require('./config');

async function connectDatabase() {
  mongoose.set('strictQuery', true);
  await mongoose.connect(MONGO_URI, { dbName: MONGO_DB_NAME });
  console.log('[appointment-service] MongoDB conectado');
}

module.exports = { connectDatabase };