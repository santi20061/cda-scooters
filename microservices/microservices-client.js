/**
 * microservices-client.js
 * 
 * Cliente para integrar los microservicios de CDA Scooters
 * con el frontend
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════════════
//  CONFIGURACIÓN
// ═══════════════════════════════════════════════════════════════════════════

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:3000";

const endpoints = {
  gateway: {
    status: "/api/status",
    health: "/health",
  },
  contact: {
    send: "/api/contact/send",
    messages: "/api/contact/messages",
  },
  appointments: {
    book: "/api/appointments/book",
    list: "/api/appointments",
    get: (id) => `/api/appointments/${id}`,
    update: (id) => `/api/appointments/${id}`,
    delete: (id) => `/api/appointments/${id}`,
    remind: (id) => `/api/appointments/${id}/remind`,
  },
  chatbot: {
    message: "/api/chat/message",
    history: "/api/chat/history",
  },
};

// ═══════════════════════════════════════════════════════════════════════════
//  CONTACT SERVICE
// ═══════════════════════════════════════════════════════════════════════════

class ContactService {
  /**
   * Enviar mensaje de contacto por Email + WhatsApp
   * @param {Object} contactData - Datos del contacto
   * @returns {Promise}
   */
  static async sendContact(contactData) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoints.contact.send}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          nombre: contactData.nombre,
          email: contactData.email,
          telefono: contactData.telefono || "",
          asunto: contactData.asunto,
          mensaje: contactData.mensaje,
          whatsapp: contactData.whatsapp !== false,
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error enviando contacto:", error);
      throw error;
    }
  }

  /**
   * Obtener todos los mensajes de contacto
   * @returns {Promise}
   */
  static async getMessages() {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.contact.messages}`
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error obteniendo mensajes:", error);
      throw error;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  APPOINTMENT SERVICE
// ═══════════════════════════════════════════════════════════════════════════

class AppointmentService {
  /**
   * Agendar una cita
   * @param {Object} appointmentData - Datos de la cita
   * @returns {Promise}
   */
  static async bookAppointment(appointmentData) {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.appointments.book}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            nombre: appointmentData.nombre,
            telefono: appointmentData.telefono,
            email: appointmentData.email || "",
            fecha: appointmentData.fecha,
            hora: appointmentData.hora,
            servicio: appointmentData.servicio || "Servicio General",
            notas: appointmentData.notas || "",
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error agendando cita:", error);
      throw error;
    }
  }

  /**
   * Obtener todas las citas
   * @returns {Promise}
   */
  static async getAppointments() {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoints.appointments.list}`);

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error obteniendo citas:", error);
      throw error;
    }
  }

  /**
   * Obtener una cita específica
   * @param {string} id - ID de la cita
   * @returns {Promise}
   */
  static async getAppointment(id) {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.appointments.get(id)}`
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error obteniendo cita:", error);
      throw error;
    }
  }

  /**
   * Actualizar una cita
   * @param {string} id - ID de la cita
   * @param {Object} updateData - Datos a actualizar
   * @returns {Promise}
   */
  static async updateAppointment(id, updateData) {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.appointments.update(id)}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updateData),
        }
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error actualizando cita:", error);
      throw error;
    }
  }

  /**
   * Cancelar una cita
   * @param {string} id - ID de la cita
   * @returns {Promise}
   */
  static async cancelAppointment(id) {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.appointments.delete(id)}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error cancelando cita:", error);
      throw error;
    }
  }

  /**
   * Enviar recordatorio de cita
   * @param {string} id - ID de la cita
   * @returns {Promise}
   */
  static async sendReminder(id) {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.appointments.remind(id)}`,
        {
          method: "POST",
        }
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error enviando recordatorio:", error);
      throw error;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  CHATBOT SERVICE
// ═══════════════════════════════════════════════════════════════════════════

class ChatbotService {
  /**
   * Enviar mensaje al chatbot
   * @param {string} mensaje - Mensaje del usuario
   * @param {string} sessionId - ID de la sesión (opcional)
   * @returns {Promise}
   */
  static async sendMessage(mensaje, sessionId = null) {
    try {
      // Generar ID de sesión si no existe
      const session = sessionId || this._generateSessionId();

      const response = await fetch(
        `${API_BASE_URL}${endpoints.chatbot.message}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            sessionId: session,
            mensaje,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error enviando mensaje:", error);
      throw error;
    }
  }

  /**
   * Obtener historial de conversación
   * @param {string} sessionId - ID de la sesión
   * @returns {Promise}
   */
  static async getHistory(sessionId) {
    try {
      const response = await fetch(
        `${API_BASE_URL}${endpoints.chatbot.history}?sessionId=${sessionId}`
      );

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error obteniendo historial:", error);
      throw error;
    }
  }

  /**
   * Generar ID de sesión único
   * @private
   * @returns {string}
   */
  static _generateSessionId() {
    return `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  GATEWAY SERVICE (Status y Health)
// ═══════════════════════════════════════════════════════════════════════════

class GatewayService {
  /**
   * Obtener estado del sistema
   * @returns {Promise}
   */
  static async getStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoints.gateway.status}`);

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error obteniendo estado:", error);
      throw error;
    }
  }

  /**
   * Health check del gateway
   * @returns {Promise}
   */
  static async health() {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoints.gateway.health}`);

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error("❌ Error en health check:", error);
      throw error;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  EXPORTAR SERVICIOS
// ═══════════════════════════════════════════════════════════════════════════

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    ContactService,
    AppointmentService,
    ChatbotService,
    GatewayService,
  };
}

// Para uso directo en el navegador
window.CDAMicroservices = {
  ContactService,
  AppointmentService,
  ChatbotService,
  GatewayService,
};
