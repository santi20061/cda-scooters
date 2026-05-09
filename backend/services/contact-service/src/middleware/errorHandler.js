'use strict';

function errorHandler(err, _req, res, _next) {
  console.error('[contact-service]', err.message);
  res.status(500).json({ ok: false, error: err.message || 'Error interno del servidor' });
}

module.exports = errorHandler;
