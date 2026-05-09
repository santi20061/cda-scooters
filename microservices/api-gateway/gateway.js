/**
 * API Gateway — CDA Scooters Asociados SAS
 * Enrutador central que conecta todos los microservicios
 * 
 * PUERTOS:
 *  - Gateway: 3000
 *  - Contact Service: 3001
 *  - Appointment Service: 3002
 *  - Chatbot Service: 3003
 */

"use strict";

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const proxy = require("express-http-proxy");
const { v4: uuidv4 } = require("uuid");

const app = express();
const PORT = process.env.GATEWAY_PORT || 3000;

// ═══════════════════════════════════════════════════════════════════════════
//  MIDDLEWARE GLOBAL
// ═══════════════════════════════════════════════════════════════════════════

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Logger middleware
app.use((req, res, next) => {
  const requestId = uuidv4();
  console.log(`[${new Date().toISOString()}] ${requestId} → ${req.method} ${req.path}`);
  res.set("X-Request-ID", requestId);
  next();
});

// ═══════════════════════════════════════════════════════════════════════════
//  HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════════════

app.get("/health", (req, res) => {
  res.json({ status: "ok", gateway: "CDA API Gateway" });
});

// ═══════════════════════════════════════════════════════════════════════════
//  RUTAS DEL API GATEWAY
// ═══════════════════════════════════════════════════════════════════════════

// Servicios disponibles
const services = {
  contact: process.env.CONTACT_SERVICE_URL || "http://localhost:3001",
  appointment: process.env.APPOINTMENT_SERVICE_URL || "http://localhost:3002",
  chatbot: process.env.CHATBOT_SERVICE_URL || "http://localhost:3003",
};

console.log("📍 Servicios configurados:");
console.log(`   📧 Contact Service: ${services.contact}`);
console.log(`   📅 Appointment Service: ${services.appointment}`);
console.log(`   🤖 Chatbot Service: ${services.chatbot}`);

// ─────────────────────────────────────────────────────────────────────────────
// CONTACT SERVICE (Puerto 3001)
// ─────────────────────────────────────────────────────────────────────────────

app.post("/api/contact/send", proxy(services.contact, {
  proxyReqPathResolver: () => "/contact/send",
}));

app.get("/api/contact/messages", proxy(services.contact, {
  proxyReqPathResolver: () => "/contact/messages",
}));

// ─────────────────────────────────────────────────────────────────────────────
// APPOINTMENT SERVICE (Puerto 3002)
// ─────────────────────────────────────────────────────────────────────────────

app.post("/api/appointments/book", proxy(services.appointment, {
  proxyReqPathResolver: () => "/appointments/book",
}));

app.get("/api/appointments", proxy(services.appointment, {
  proxyReqPathResolver: () => "/appointments",
}));

app.put("/api/appointments/:id", proxy(services.appointment, {
  proxyReqPathResolver: (req) => `/appointments/${req.params.id}`,
}));

app.delete("/api/appointments/:id", proxy(services.appointment, {
  proxyReqPathResolver: (req) => `/appointments/${req.params.id}`,
}));

// ─────────────────────────────────────────────────────────────────────────────
// CHATBOT SERVICE (Puerto 3003)
// ─────────────────────────────────────────────────────────────────────────────

app.post("/api/chat/message", proxy(services.chatbot, {
  proxyReqPathResolver: () => "/chat/message",
}));

app.get("/api/chat/history", proxy(services.chatbot, {
  proxyReqPathResolver: () => "/chat/history",
}));

// ─────────────────────────────────────────────────────────────────────────────
// RUTAS DE INFORMACIÓN
// ─────────────────────────────────────────────────────────────────────────────

app.get("/api/status", (req, res) => {
  res.json({
    gateway: "CDA API Gateway",
    version: "1.0.0",
    timestamp: new Date().toISOString(),
    services: {
      contact: { url: services.contact, status: "active" },
      appointment: { url: services.appointment, status: "active" },
      chatbot: { url: services.chatbot, status: "active" },
    },
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// ERROR HANDLING
// ─────────────────────────────────────────────────────────────────────────────

app.use((err, req, res, next) => {
  console.error("❌ Error:", err.message);
  res.status(err.status || 500).json({
    error: err.message,
    requestId: res.get("X-Request-ID"),
  });
});

app.use((req, res) => {
  res.status(404).json({
    error: "Ruta no encontrada",
    path: req.path,
  });
});

// ═══════════════════════════════════════════════════════════════════════════
//  INICIAR GATEWAY
// ═══════════════════════════════════════════════════════════════════════════

app.listen(PORT, () => {
  console.log(`\n✅ API Gateway corriendo en http://localhost:${PORT}`);
  console.log(`📊 Ver status: GET /api/status`);
  console.log(`❤️  Health check: GET /health\n`);
});

module.exports = app;
