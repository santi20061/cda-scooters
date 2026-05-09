/**
 * Appointment Service — CDA Scooters Asociados SAS
 * 
 * Gestiona citas y envía notificaciones por WhatsApp
 */

"use strict";

require("dotenv").config();
const express = require("express");
const twilio = require("twilio");
const { v4: uuidv4 } = require("uuid");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.APPOINTMENT_SERVICE_PORT || 3002;

// ═══════════════════════════════════════════════════════════════════════════
//  MIDDLEWARE
// ═══════════════════════════════════════════════════════════════════════════

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ═══════════════════════════════════════════════════════════════════════════
//  CONFIGURACIÓN
// ═══════════════════════════════════════════════════════════════════════════

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

// Base de datos de citas (archivo JSON)
const appointmentsFile = path.join(__dirname, "citas.json");

function loadAppointments() {
  if (fs.existsSync(appointmentsFile)) {
    return JSON.parse(fs.readFileSync(appointmentsFile, "utf-8"));
  }
  return [];
}

function saveAppointments(appointments) {
  fs.writeFileSync(appointmentsFile, JSON.stringify(appointments, null, 2));
}

// ═══════════════════════════════════════════════════════════════════════════
//  RUTAS
// ═══════════════════════════════════════════════════════════════════════════

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "Appointment Service" });
});

// ─────────────────────────────────────────────────────────────────────────────
// AGENDAR CITA
// ─────────────────────────────────────────────────────────────────────────────

app.post("/appointments/book", async (req, res) => {
  try {
    const { nombre, telefono, email, fecha, hora, servicio, notas } = req.body;

    // Validación
    if (!nombre || !telefono || !fecha || !hora) {
      return res.status(400).json({
        error: "Faltan campos requeridos: nombre, telefono, fecha, hora",
      });
    }

    // Crear cita
    const appointment = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      nombre,
      telefono,
      email,
      fecha,
      hora,
      servicio: servicio || "Servicio General",
      notas: notas || "",
      status: "confirmada",
      reminderSent: false,
    };

    // Guardar
    const appointments = loadAppointments();
    appointments.push(appointment);
    saveAppointments(appointments);

    // Enviar confirmación por WhatsApp
    if (twilioClient) {
      console.log(`💬 Enviando confirmación de cita a ${telefono}...`);
      try {
        await twilioClient.messages.create({
          body: `¡Hola ${nombre}! 📅\n\nTu cita ha sido confirmada:\n📍 Fecha: ${fecha}\n⏰ Hora: ${hora}\n🔧 Servicio: ${servicio || "General"}\n\nCDA Scooters Asociados SAS`,
          from: `whatsapp:${TWILIO_PHONE}`,
          to: `whatsapp:${telefono}`,
        });
        console.log("✅ WhatsApp de confirmación enviado");
        appointment.reminderSent = true;
      } catch (error) {
        console.error("❌ Error al enviar WhatsApp:", error.message);
      }
    } else {
      console.warn("⚠️ No se envió confirmación por WhatsApp porque Twilio no está configurado");
    }

    // Actualizar
    const appointments2 = loadAppointments();    appointments2[appointments2.length - 1] = appointment;
    saveAppointments(appointments2);

    res.json({
      success: true,
      message: "Cita agendada exitosamente",
      appointment,
    });
  } catch (error) {
    console.error("❌ Error:", error);
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// OBTENER TODAS LAS CITAS
// ─────────────────────────────────────────────────────────────────────────────

app.get("/appointments", (req, res) => {
  try {
    const appointments = loadAppointments();
    res.json({ count: appointments.length, appointments });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// OBTENER CITA POR ID
// ─────────────────────────────────────────────────────────────────────────────

app.get("/appointments/:id", (req, res) => {
  try {
    const appointments = loadAppointments();
    const appointment = appointments.find((a) => a.id === req.params.id);

    if (!appointment) {
      return res.status(404).json({ error: "Cita no encontrada" });
    }

    res.json(appointment);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// ACTUALIZAR CITA
// ─────────────────────────────────────────────────────────────────────────────

app.put("/appointments/:id", async (req, res) => {
  try {
    const { fecha, hora, status, notas } = req.body;
    const appointments = loadAppointments();
    const index = appointments.findIndex((a) => a.id === req.params.id);

    if (index === -1) {
      return res.status(404).json({ error: "Cita no encontrada" });
    }

    const appointment = appointments[index];

    if (fecha) appointment.fecha = fecha;
    if (hora) appointment.hora = hora;
    if (status) appointment.status = status;
    if (notas) appointment.notas = notas;

    saveAppointments(appointments);

    // Enviar notificación de cambio
    if ((fecha || hora) && appointment.telefono && twilioClient) {
      console.log(`💬 Enviando notificación de cambio a ${appointment.telefono}...`);
      try {
        await twilioClient.messages.create({
          body: `¡Hola ${appointment.nombre}! 📅\n\nTu cita ha sido actualizada:\n📍 Fecha: ${fecha || appointment.fecha}\n⏰ Hora: ${hora || appointment.hora}\n\nCDA Scooters Asociados SAS`,
          from: `whatsapp:${TWILIO_PHONE}`,
          to: `whatsapp:${appointment.telefono}`,
        });
        console.log("✅ Notificación de cambio enviada");
      } catch (error) {
        console.error("❌ Error al enviar WhatsApp:", error.message);
      }
    } else if (fecha || hora) {
      console.warn("⚠️ No se envió notificación de cambio porque Twilio no está configurado");
    }

    res.json({
      success: true,
      message: "Cita actualizada",
      appointment,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// CANCELAR CITA
// ─────────────────────────────────────────────────────────────────────────────

app.delete("/appointments/:id", async (req, res) => {
  try {
    const appointments = loadAppointments();
    const index = appointments.findIndex((a) => a.id === req.params.id);

    if (index === -1) {
      return res.status(404).json({ error: "Cita no encontrada" });
    }

    const appointment = appointments[index];

    // Enviar notificación de cancelación
    if (appointment.telefono && twilioClient) {
      console.log(`💬 Enviando notificación de cancelación a ${appointment.telefono}...`);
      try {
        await twilioClient.messages.create({
          body: `¡Hola ${appointment.nombre}! 📅\n\nTu cita del ${appointment.fecha} a las ${appointment.hora} ha sido cancelada.\n\nSi deseas agendar una nueva, contáctanos.\n\nCDA Scooters Asociados SAS`,
          from: `whatsapp:${TWILIO_PHONE}`,
          to: `whatsapp:${appointment.telefono}`,
        });
        console.log("✅ Notificación de cancelación enviada");
      } catch (error) {
        console.error("❌ Error al enviar WhatsApp:", error.message);
      }
    } else if (appointment.telefono) {
      console.warn("⚠️ No se envió notificación de cancelación porque Twilio no está configurado");
    }

    appointments.splice(index, 1);
    saveAppointments(appointments);

    res.json({ success: true, message: "Cita cancelada" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// ENVIAR RECORDATORIO
// ─────────────────────────────────────────────────────────────────────────────

app.post("/appointments/:id/remind", async (req, res) => {
  try {
    const appointments = loadAppointments();
    const appointment = appointments.find((a) => a.id === req.params.id);

    if (!appointment) {
      return res.status(404).json({ error: "Cita no encontrada" });
    }

    if (appointment.telefono && twilioClient) {
      console.log(`💬 Enviando recordatorio a ${appointment.telefono}...`);
      try {
        await twilioClient.messages.create({
          body: `¡Hola ${appointment.nombre}! 📅 RECORDATORIO\n\nTienes una cita el ${appointment.fecha} a las ${appointment.hora}.\n\nCDA Scooters Asociados SAS`,
          from: `whatsapp:${TWILIO_PHONE}`,
          to: `whatsapp:${appointment.telefono}`,
        });
        console.log("✅ Recordatorio enviado");
      } catch (error) {
        console.error("❌ Error al enviar WhatsApp:", error.message);
      }
    } else if (appointment.telefono) {
      console.warn("⚠️ No se envió recordatorio porque Twilio no está configurado");
    }

    res.json({ success: true, message: "Recordatorio enviado" });
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
  console.log(`✅ Appointment Service corriendo en puerto ${PORT}`);
});

module.exports = app;
