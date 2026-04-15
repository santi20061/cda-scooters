/**
 * ============================================================
 *  CDA Scooters SAS — server.js (Backend Refactorizado)
 *  Patrón aplicado: FACADE (Estructural)
 * ============================================================
 *
 *  ¿Qué es el patrón Facade?
 *  --------------------------
 *  El patrón Facade proporciona una interfaz simplificada
 *  para un subsistema complejo. En lugar de que el cliente
 *  (frontend o rutas Express) llame directamente a OpenAI,
 *  al sistema de archivos y a la validación por separado,
 *  existe UNA sola clase (CDAFacade) que lo coordina todo.
 *
 *  Estructura:
 *    ┌─────────────┐     llama a     ┌──────────────────┐
 *    │  Rutas      │ ─────────────▶  │   CDAFacade      │
 *    │  Express    │                 │  (interfaz única) │
 *    └─────────────┘                 └──────────────────┘
 *                                           │
 *                        ┌─────────────────┬┴────────────────────┐
 *                        ▼                 ▼                      ▼
 *                   OpenAIService    ValidadorMensajes    RepositorioContactos
 *                   (subsistema 1)   (subsistema 2)       (subsistema 3)
 */

require("dotenv").config();
const express = require("express");
const cors    = require("cors");
const fs      = require("fs");
const OpenAI  = require("openai");

const app    = express();
app.use(cors());
app.use(express.json());

// ─────────────────────────────────────────────────────────────
//  SUBSISTEMA 1: OpenAIService
//  Encapsula toda la lógica de comunicación con la API de OpenAI.
//  El resto del sistema NO sabe cómo funciona OpenAI por dentro.
// ─────────────────────────────────────────────────────────────
class OpenAIService {
  constructor(apiKey) {
    this.client = new OpenAI({ apiKey });

    // Contexto del CDA: define la "personalidad" del asistente
    this.systemPrompt = `
Eres un asistente virtual de CDA Scooters SAS, un Centro de Diagnóstico 
Automotriz legalmente habilitado por el Ministerio de Transporte de Colombia,
ubicado en Acacías, Meta.

Tu función es ayudar a los usuarios con:
- Revisión técnico-mecánica y de emisiones
- Requisitos y documentos necesarios (SOAT, licencia, tarjeta de propiedad)
- Precios de revisión (Moto: $180.000 | Carro: $280.000 | Motocarro: $220.000)
- Horarios (Lun–Vie 7AM–5PM · Sáb 7AM–1PM · Dom cerrado)
- Agendamiento de citas
- Fallas comunes que hacen reprobar la revisión

Responde de forma clara, profesional y amigable. Si la pregunta no tiene 
relación con el CDA o vehículos, redirige amablemente la conversación.

Teléfono: +57 311 800 6270
Email: cda.scootersacacias@gmail.com
    `;
  }

  /**
   * Envía un mensaje a OpenAI y retorna la respuesta.
   * @param {string} userMessage - El mensaje del usuario
   * @returns {Promise<string>} - La respuesta del modelo
   */
  async enviarMensaje(userMessage) {
    const completion = await this.client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: this.systemPrompt },
        { role: "user",   content: userMessage }
      ]
    });
    return completion.choices[0].message.content;
  }
}

// ─────────────────────────────────────────────────────────────
//  SUBSISTEMA 2: ValidadorMensajes
//  Contiene todas las reglas de validación de datos entrantes.
//  Si mañana cambias las reglas, solo tocas esta clase.
// ─────────────────────────────────────────────────────────────
class ValidadorMensajes {
  /**
   * Valida que un mensaje del chatbot no esté vacío.
   * @param {string} mensaje
   * @returns {{ valido: boolean, error?: string }}
   */
  validarChat(mensaje) {
    if (!mensaje || typeof mensaje !== "string" || mensaje.trim() === "") {
      return { valido: false, error: "El mensaje no puede estar vacío." };
    }
    if (mensaje.length > 1000) {
      return { valido: false, error: "El mensaje es demasiado largo (máx 1000 caracteres)." };
    }
    return { valido: true };
  }

  /**
   * Valida los campos del formulario de contacto.
   * @param {{ nombre, correo, telefono, vehiculo, mensaje }} datos
   * @returns {{ valido: boolean, error?: string }}
   */
  validarContacto({ nombre, correo, telefono, vehiculo, mensaje }) {
    if (!nombre || nombre.trim() === "") {
      return { valido: false, error: "El nombre es obligatorio." };
    }
    if (!correo || !correo.includes("@")) {
      return { valido: false, error: "El correo electrónico no es válido." };
    }
    return { valido: true };
  }
}

// ─────────────────────────────────────────────────────────────
//  SUBSISTEMA 3: RepositorioContactos
//  Maneja toda la persistencia de datos (leer/escribir JSON).
//  Si en el futuro cambias a una base de datos, solo tocas aquí.
// ─────────────────────────────────────────────────────────────
class RepositorioContactos {
  constructor(rutaArchivo = "mensajes.json") {
    this.rutaArchivo = rutaArchivo;
  }

  /**
   * Lee todos los contactos guardados.
   * @returns {Array} - Lista de contactos
   */
  leerContactos() {
    try {
      if (fs.existsSync(this.rutaArchivo)) {
        return JSON.parse(fs.readFileSync(this.rutaArchivo, "utf-8"));
      }
    } catch (error) {
      console.error("Error leyendo contactos:", error);
    }
    return [];
  }

  /**
   * Guarda un nuevo contacto en el archivo JSON.
   * @param {object} contacto - Datos del contacto
   * @returns {boolean} - true si se guardó correctamente
   */
  guardarContacto(contacto) {
    try {
      const datos = this.leerContactos();
      datos.push({ ...contacto, fecha: new Date() });
      fs.writeFileSync(this.rutaArchivo, JSON.stringify(datos, null, 2));
      return true;
    } catch (error) {
      console.error("Error guardando contacto:", error);
      return false;
    }
  }
}

// ─────────────────────────────────────────────────────────────
//  ★ FACHADA PRINCIPAL: CDAFacade
//  ─────────────────────────────────────────────────────────────
//  Esta es la interfaz unificada que expone solo dos métodos
//  simples al mundo exterior:
//    → procesarChat(mensaje)
//    → procesarContacto(datos)
//
//  Las rutas de Express SOLO hablan con esta clase.
//  No saben nada de OpenAI, de validaciones ni de archivos.
// ─────────────────────────────────────────────────────────────
class CDAFacade {
  constructor() {
    // Instancia los tres subsistemas internamente
    this.ia          = new OpenAIService(process.env.OPENAI_API_KEY);
    this.validador   = new ValidadorMensajes();
    this.repositorio = new RepositorioContactos();
  }

  /**
   * Punto de entrada unificado para el chatbot.
   * Internamente valida, llama a la IA y maneja errores.
   *
   * @param {string} mensaje - Mensaje enviado por el usuario
   * @returns {Promise<{ ok: boolean, reply?: string, error?: string }>}
   */
  async procesarChat(mensaje) {
    // Paso 1: Validar
    const validacion = this.validador.validarChat(mensaje);
    if (!validacion.valido) {
      return { ok: false, error: validacion.error };
    }

    // Paso 2: Llamar a la IA
    try {
      const reply = await this.ia.enviarMensaje(mensaje);
      return { ok: true, reply };
    } catch (error) {
      console.error("Error en IA:", error);
      return {
        ok: false,
        error: "El asistente no está disponible. Contáctanos al WhatsApp."
      };
    }
  }

  /**
   * Punto de entrada unificado para el formulario de contacto.
   * Internamente valida y guarda en el repositorio.
   *
   * @param {object} datos - Datos del formulario
   * @returns {{ ok: boolean, mensaje: string }}
   */
  procesarContacto(datos) {
    // Paso 1: Validar
    const validacion = this.validador.validarContacto(datos);
    if (!validacion.valido) {
      return { ok: false, mensaje: validacion.error };
    }

    // Paso 2: Guardar
    const guardado = this.repositorio.guardarContacto(datos);
    if (guardado) {
      return { ok: true, mensaje: "¡Mensaje recibido! Te contactaremos pronto ✅" };
    } else {
      return { ok: false, mensaje: "Error al guardar. Intenta nuevamente ❌" };
    }
  }
}

// ─────────────────────────────────────────────────────────────
//  INICIALIZACIÓN
//  Se crea UNA sola instancia del facade (Singleton implícito)
// ─────────────────────────────────────────────────────────────
const cdaFacade = new CDAFacade();

// ─────────────────────────────────────────────────────────────
//  RUTAS EXPRESS
//  Observa lo simple que son: solo llaman al facade.
//  No contienen lógica de negocio.
// ─────────────────────────────────────────────────────────────

/** Ruta del chatbot — delega todo al facade */
app.post("/chat", async (req, res) => {
  const resultado = await cdaFacade.procesarChat(req.body.message);

  if (resultado.ok) {
    res.json({ reply: resultado.reply });
  } else {
    res.status(400).json({ reply: resultado.error });
  }
});

/** Ruta del formulario — delega todo al facade */
app.post("/contacto", (req, res) => {
  const resultado = cdaFacade.procesarContacto(req.body);
  res.json({ mensaje: resultado.mensaje });
});

// ─────────────────────────────────────────────────────────────
app.listen(3000, () => {
  console.log("🚀 Servidor CDA corriendo en http://localhost:3000");
  console.log("📐 Patrón FACADE activo — CDAFacade centraliza todo");
});