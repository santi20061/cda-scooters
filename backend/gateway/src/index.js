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

app.use(cors());
app.use(morgan('dev'));

// Health check propio del gateway
app.get('/health', (_req, res) =>
  res.json({ status: 'ok', service: 'api-gateway' })
);

// ── Proxy: /api/contact → contact-service ──────────────────────────────────
app.use(
  '/api/contact',
  createProxyMiddleware({
    target:      CONTACT_SERVICE_URL,
    changeOrigin: true,
    pathRewrite:  { '^/api/contact': '/contact' },
    on: {
      error: (_err, _req, res) =>
        res.status(502).json({ error: 'contact-service no disponible' }),
    },
  })
);

// ── Proxy: /api/appointments → appointment-service ─────────────────────────
app.use(
  '/api/appointments',
  createProxyMiddleware({
    target:      APPOINTMENT_SERVICE_URL,
    changeOrigin: true,
    pathRewrite:  { '^/api/appointments': '/appointments' },
    on: {
      error: (_err, _req, res) =>
        res.status(502).json({ error: 'appointment-service no disponible' }),
    },
  })
);

// ── Proxy: /api/chatbot → chatbot-service ──────────────────────────────────
app.use(
  '/api/chatbot',
  createProxyMiddleware({
    target:      CHATBOT_SERVICE_URL,
    changeOrigin: true,
    pathRewrite:  { '^/api/chatbot': '' },
    on: {
      error: (_err, _req, res) =>
        res.status(502).json({ error: 'chatbot-service no disponible' }),
    },
  })
);

app.listen(PORT, () => {
  console.log(`\n🚀 API Gateway corriendo en puerto ${PORT}`);
  console.log(`   /api/contact      → ${CONTACT_SERVICE_URL}`);
  console.log(`   /api/appointments → ${APPOINTMENT_SERVICE_URL}`);
  console.log(`   /api/chatbot      → ${CHATBOT_SERVICE_URL}\n`);
});
