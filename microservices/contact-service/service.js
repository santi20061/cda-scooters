/**
 * Contact Service — CDA Scooters Asociados SAS
 * 
 * Envía mensajes de contacto a través de:
 *  ✉️  Email (nodemailer)
 *  💬 WhatsApp (Twilio)
 */

"use strict";

require("dotenv").config();
const express = require("express");
const nodemailer = require("nodemailer");
const twilio = require("twilio");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.CONTACT_SERVICE_PORT || 3001;

// ═══════════════════════════════════════════════════════════════════════════
//  MIDDLEWARE
// ═══════════════════════════════════════════════════════════════════════════

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ═══════════════════════════════════════════════════════════════════════════
//  CONFIGURACIÓN DE SERVICIOS
// ═══════════════════════════════════════════════════════════════════════════

// Email (Nodemailer)
const emailConfig = {
  host: process.env.SMTP_HOST || "smtp.gmail.com",
  port: process.env.SMTP_PORT || 587,
  secure: process.env.SMTP_SECURE === "true",
  auth: {
    user: process.env.SMTP_USER || "tu-email@gmail.com",
    pass: process.env.SMTP_PASS || "tu-contraseña",
  },
};

const transporter = nodemailer.createTransport(emailConfig);

// WhatsApp (Twilio)
const TWILIO_PHONE = process.env.TWILIO_PHONE || "+1234567890";

function createTwilioClient() {
  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;

  if (!accountSid || !authToken || !accountSid.startsWith("AC")) {
    console.warn("⚠️ Twilio no configurado; WhatsApp quedará deshabilitado");
    return null;
  }

  return twilio(accountSid, authToken);
}

const twilioClient = createTwilioClient();

// Almacenamiento de mensajes (en un proyecto real, usar base de datos)
const messagesFile = path.join(__dirname, "mensajes.json");

function loadMessages() {
  if (fs.existsSync(messagesFile)) {
    return JSON.parse(fs.readFileSync(messagesFile, "utf-8"));
  }
  return [];
}

function saveMessages(messages) {
  fs.writeFileSync(messagesFile, JSON.stringify(messages, null, 2));
}

// ═══════════════════════════════════════════════════════════════════════════
//  RUTAS
// ═══════════════════════════════════════════════════════════════════════════

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "Contact Service" });
});

// ─────────────────────────────────────────────────────────────────────────────
// ENVIAR CONTACTO (Email + WhatsApp)
// ─────────────────────────────────────────────────────────────────────────────

app.post("/contact/send", async (req, res) => {
  try {
    const { nombre, email, telefono, asunto, mensaje, whatsapp = true } = req.body;

    // Validación
    if (!nombre || !email || !asunto || !mensaje) {
      return res.status(400).json({
        error: "Faltan campos requeridos: nombre, email, asunto, mensaje",
      });
    }

    const contactData = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      nombre,
      email,
      telefono,
      asunto,
      mensaje,
      status: "pendiente",
    };

    // Guardar en archivo
    const messages = loadMessages();
    messages.push(contactData);
    saveMessages(messages);

    // Enviar Email
    console.log(`📧 Enviando email a ${email}...`);
    try {
      await transporter.sendMail({
        from: process.env.SMTP_FROM || '"CDA Scooters" <noreply@cdascooters.com>',
        to: email,
        subject: `Confirmación: ${asunto}`,
        html: `
          <h2>¡Hola ${nombre}!</h2>
          <p>Recibimos tu mensaje:</p>
          <hr>
          <p><strong>Asunto:</strong> ${asunto}</p>
          <p><strong>Mensaje:</strong> ${mensaje}</p>
          <hr>
          <p>Nos pondremos en contacto pronto.</p>
          <p>CDA Scooters Asociados SAS</p>
        `,
      });

      await transporter.sendMail({
        from: process.env.SMTP_FROM || '"CDA Scooters" <noreply@cdascooters.com>',
        to: process.env.ADMIN_EMAIL || "admin@cdascooters.com",
        subject: `[CONTACTO] ${asunto}`,
        html: `
          <h3>Nuevo mensaje de contacto</h3>
          <p><strong>De:</strong> ${nombre} (${email})</p>
          <p><strong>Teléfono:</strong> ${telefono || "No proporcionado"}</p>
          <p><strong>Asunto:</strong> ${asunto}</p>
          <p><strong>Mensaje:</strong></p>
          <pre>${mensaje}</pre>
        `,
      });

      console.log("✅ Email enviado exitosamente");
    } catch (emailError) {
      console.error("❌ Error al enviar email:", emailError.message);
    }

    // Enviar WhatsApp (opcional)
    if (whatsapp && telefono && twilioClient) {
      console.log(`💬 Enviando WhatsApp a ${telefono}...`);
      try {
        await twilioClient.messages.create({
          body: `¡Hola ${nombre}! Recibimos tu mensaje sobre "${asunto}". Nos pondremos en contacto pronto.\n\nCDA Scooters`,
          from: `whatsapp:${TWILIO_PHONE}`,
          to: `whatsapp:${telefono}`,
        });
        console.log("✅ WhatsApp enviado exitosamente");
        contactData.status = "enviado";
      } catch (whatsappError) {
        console.error("❌ Error al enviar WhatsApp:", whatsappError.message);
      }
    } else if (whatsapp && telefono) {
      console.warn("⚠️ WhatsApp solicitado, pero Twilio no está configurado");
    }

    // Actualizar estado
    const messages2 = loadMessages();
    const lastMessage = messages2[messages2.length - 1];
    lastMessage.status = "enviado";
    saveMessages(messages2);

    res.json({
      success: true,
      message: "Contacto recibido",
      data: contactData,
    });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// OBTENER MENSAJES
// ─────────────────────────────────────────────────────────────────────────────

app.get("/contact/messages", (req, res) => {
  try {
    const messages = loadMessages();
    res.json({ count: messages.length, messages });
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
  console.log(`✅ Contact Service corriendo en puerto ${PORT}`);
});

module.exports = app;
