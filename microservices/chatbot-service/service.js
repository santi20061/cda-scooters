/**
 * Chatbot Service — CDA Scooters Asociados SAS
 * 
 * Servicio de chatbot impulsado por OpenAI
 * Especializado en CDA Scooters
 */

"use strict";

require("dotenv").config();
const express = require("express");
const OpenAI = require("openai");
const { v4: uuidv4 } = require("uuid");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.CHATBOT_SERVICE_PORT || 3003;

// ═══════════════════════════════════════════════════════════════════════════
//  MIDDLEWARE
// ═══════════════════════════════════════════════════════════════════════════

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ═══════════════════════════════════════════════════════════════════════════
//  CONFIGURACIÓN DE OPENAI
// ═══════════════════════════════════════════════════════════════════════════

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const MODEL = process.env.OPENAI_MODEL || "gpt-3.5-turbo";

// Prompt del sistema para el chatbot
const SYSTEM_PROMPT = `Eres un asistente amable y profesional de CDA Scooters Asociados SAS, una empresa especializada en scooters eléctricos en Colombia.

INFORMACIÓN SOBRE CDA SCOOTERS:
- Empresa: CDA Scooters Asociados SAS
- Ubicación: Colombia
- Especialidad: Scooters eléctricos, alquiler, reparación y mantenimiento
- Servicios:
  * Alquiler de scooters
  * Reparación y mantenimiento
  * Venta de accesorios
  * Capacitación en seguridad vial

TUS RESPONSABILIDADES:
1. Responder preguntas sobre servicios y productos
2. Ayudar a agendar citas
3. Proporcionar información sobre seguridad vial
4. Ofrecer recomendaciones personalizadas
5. Ser empático y resolver problemas del cliente

TONO:
- Amable y profesional
- Usar emojis ocasionalmente
- Responder en español
- Ser conciso pero informativo

Si el usuario quiere agendar una cita, menciona que puede hacerlo a través del formulario de citas.
Si quiere contactar, menciona que puede usar el formulario de contacto.`;

// Base de datos de conversaciones (archivo JSON)
const conversationsFile = path.join(__dirname, "conversaciones.json");

function loadConversations() {
  if (fs.existsSync(conversationsFile)) {
    return JSON.parse(fs.readFileSync(conversationsFile, "utf-8"));
  }
  return {};
}

function saveConversations(conversations) {
  fs.writeFileSync(conversationsFile, JSON.stringify(conversations, null, 2));
}

// ═══════════════════════════════════════════════════════════════════════════
//  RUTAS
// ═══════════════════════════════════════════════════════════════════════════

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "Chatbot Service" });
});

// ─────────────────────────────────────────────────────────────────────────────
// ENVIAR MENSAJE AL CHATBOT
// ─────────────────────────────────────────────────────────────────────────────

app.post("/chat/message", async (req, res) => {
  try {
    const { sessionId = uuidv4(), mensaje, userId = "anonymous" } = req.body;

    if (!mensaje) {
      return res.status(400).json({ error: "El campo 'mensaje' es requerido" });
    }

    // Cargar o crear conversación
    const conversations = loadConversations();
    if (!conversations[sessionId]) {
      conversations[sessionId] = {
        id: sessionId,
        userId,
        createdAt: new Date().toISOString(),
        messages: [],
      };
    }

    const conversation = conversations[sessionId];

    // Agregar mensaje del usuario
    conversation.messages.push({
      role: "user",
      content: mensaje,
      timestamp: new Date().toISOString(),
    });

    // Preparar historial para OpenAI
    const messages = conversation.messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));

    console.log(`🤖 Procesando mensaje de ${sessionId}...`);

    // Llamar a OpenAI
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        ...messages,
      ],
      temperature: 0.7,
      max_tokens: 500,
    });

    const assistantMessage = response.choices[0].message.content;

    // Guardar respuesta
    conversation.messages.push({
      role: "assistant",
      content: assistantMessage,
      timestamp: new Date().toISOString(),
    });

    conversation.updatedAt = new Date().toISOString();
    saveConversations(conversations);

    console.log("✅ Respuesta generada");

    res.json({
      success: true,
      sessionId,
      userMessage: mensaje,
      assistantMessage,
      conversation: {
        id: sessionId,
        messageCount: conversation.messages.length,
      },
    });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({
      error: error.message || "Error al procesar el mensaje",
    });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// OBTENER HISTORIAL DE CHAT
// ─────────────────────────────────────────────────────────────────────────────

app.get("/chat/history", (req, res) => {
  try {
    const { sessionId } = req.query;

    if (!sessionId) {
      return res.status(400).json({
        error: "El parámetro 'sessionId' es requerido",
      });
    }

    const conversations = loadConversations();
    const conversation = conversations[sessionId];

    if (!conversation) {
      return res.status(404).json({ error: "Sesión no encontrada" });
    }

    res.json({
      sessionId,
      createdAt: conversation.createdAt,
      updatedAt: conversation.updatedAt,
      messageCount: conversation.messages.length,
      messages: conversation.messages,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// OBTENER TODAS LAS CONVERSACIONES (Admin)
// ─────────────────────────────────────────────────────────────────────────────

app.get("/chat/conversations", (req, res) => {
  try {
    const conversations = loadConversations();

    const summary = Object.values(conversations).map((conv) => ({
      id: conv.id,
      userId: conv.userId,
      createdAt: conv.createdAt,
      updatedAt: conv.updatedAt,
      messageCount: conv.messages.length,
    }));

    res.json({
      total: summary.length,
      conversations: summary,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// LIMPIAR CONVERSACIÓN
// ─────────────────────────────────────────────────────────────────────────────

app.delete("/chat/history/:sessionId", (req, res) => {
  try {
    const conversations = loadConversations();
    const { sessionId } = req.params;

    if (conversations[sessionId]) {
      delete conversations[sessionId];
      saveConversations(conversations);
      res.json({ success: true, message: "Conversación eliminada" });
    } else {
      res.status(404).json({ error: "Sesión no encontrada" });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// ERROR HANDLING
// ─────────────────────────────────────────────────────────────────────────────

app.use((err, req, res, next) => {
  console.error("❌ Error:", err.message);
  res.status(err.status || 500).json({ error: err.message });
});

// ═══════════════════════════════════════════════════════════════════════════
//  INICIAR SERVICIO
// ═══════════════════════════════════════════════════════════════════════════

app.listen(PORT, () => {
  console.log(`✅ Chatbot Service corriendo en puerto ${PORT}`);
  console.log(`📝 Modelo: ${MODEL}`);
});

module.exports = app;
