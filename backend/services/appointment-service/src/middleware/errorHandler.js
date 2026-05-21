'use strict';

function errorHandler(err, _req, res, _next) {
  console.error('[appointment-service:error]', err.message);
  res.status(500).json({ ok: false, error: 'Error interno del servidor' });
}

module.exports = errorHandler;