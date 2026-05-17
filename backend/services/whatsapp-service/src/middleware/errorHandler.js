'use strict';

function errorHandler(err, _req, res, _next) {
  console.error('[whatsapp-service:error]', {
    message: err && err.message,
    stack: err && err.stack,
  });
  res.status(err && err.statusCode ? err.statusCode : 500).json({
    error: (err && err.message) || 'Error interno del servidor',
  });
}

module.exports = errorHandler;
