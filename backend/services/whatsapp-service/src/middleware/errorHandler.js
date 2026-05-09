'use strict';

function errorHandler(err, _req, res, _next) {
  console.error('[whatsapp-service]', err.message);
  res.status(500).json({ error: err.message || 'Error interno del servidor' });
}

module.exports = errorHandler;
