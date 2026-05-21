'use strict';

require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const morgan  = require('morgan');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app  = express();
const PORT = process.env.PORT || 3000;

const CONTACT_SERVICE_URL     = process.env.CONTACT_SERVICE_URL     || 'http://localhost:3001';
const APPOINTMENT_SERVICE_URL = process.env.APPOINTMENT_SERVICE_URL || 'http://localhost:3002';
const CHATBOT_SERVICE_URL     = process.env.CHATBOT_SERVICE_URL     || 'http://localhost:3003';

app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));
app.use(morgan('dev'));

app.use(express.json({ limit: '1mb' }));

// Capturar errores de parseo JSON antes de llegar al proxy
app.use((err, _req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    console.warn('[gateway] JSON malformado recibido');
    return res.status(400).json({ ok: false, error: 'JSON inválido en el cuerpo de la petición' });
  }
  next(err);
});

app.use((req, _res, next) => {
  console.log(`[gateway] ${req.method} ${req.originalUrl}`);
  next();
});

function createServiceProxy(basePath, target, pathRewrite, serviceName) {
  return app.use(
    basePath,
    createProxyMiddleware({
      target,
      changeOrigin: true,
      pathRewrite,
      logLevel: 'debug',
      onProxyReq: (proxyReq, req) => {
        // express.json() consume el stream; debemos re-escribir el body al proxy
        if (req.body && Object.keys(req.body).length > 0) {
          const bodyData = JSON.stringify(req.body);
          proxyReq.setHeader('Content-Type', 'application/json');
          proxyReq.setHeader('Content-Length', Buffer.byteLength(bodyData));
          proxyReq.write(bodyData);
          proxyReq.end();
        }
      },
      onProxyRes: (proxyRes, req) => {
        console.log(`[gateway:proxy] ${req.method} ${req.originalUrl} → ${target} — HTTP ${proxyRes.statusCode}`);
      },
      onError: (err, _req, res) => {
        console.error(`[gateway:proxy:error] ${serviceName}: ${err.message}`);
        if (!res.headersSent) {
          res.status(502).json({
            ok: false,
            error: `${serviceName} no disponible`,
            details: err.message,
          });
        }
      },
    })
  );
}

// Health check propio del gateway
app.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'api-gateway' })
);

createServiceProxy('/api/contact', CONTACT_SERVICE_URL, { '^/api/contact': '/contact' }, 'contact-service');
createServiceProxy('/api/appointments', APPOINTMENT_SERVICE_URL, { '^/api/appointments': '/appointments' }, 'appointment-service');
createServiceProxy('/api/chatbot', CHATBOT_SERVICE_URL, { '^/api/chatbot': '' }, 'chatbot-service');

app.use((req, res) => {
  res.status(404).json({ error: 'Ruta no encontrada', path: req.originalUrl });
});

app.listen(PORT, () => {
  console.log(`\n🚀 API Gateway corriendo en puerto ${PORT}`);
  console.log(`   /api/contact      → ${CONTACT_SERVICE_URL}`);
  console.log(`   /api/appointments → ${APPOINTMENT_SERVICE_URL}`);
  console.log(`   /api/chatbot      → ${CHATBOT_SERVICE_URL}`);
});
