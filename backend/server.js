/**
 * server.js — CDA Scooters Asociados SAS
 * ─────────────────────────────────────────────────────────────────────────────
 * PATRONES APLICADOS EN ESTE ARCHIVO:
 *
 *  1. SINGLETON  (Creacional)
 *     → AppServer: garantiza una única instancia del servidor Express.
 *     → Por qué: evita crear múltiples instancias accidentales en imports o
 *       en futuros tests; centraliza la configuración del servidor.
 *
 *  2. STRATEGY  (Comportamiento)
 *     → IAStrategy / OpenAIStrategy / FallbackStrategy: permiten intercambiar
 *       el motor de IA en tiempo de ejecución sin tocar las rutas.
 *     → Por qué: si mañana se migra a Claude, Gemini o un modelo local, solo
 *       se crea una nueva clase que implemente `responder(mensaje)`.
 *
 *  3. CHAIN OF RESPONSIBILITY  (Comportamiento)
 *     → ValidacionHandler y sus subclases: cada eslabón de la cadena decide
 *       si puede manejar la validación o pasa al siguiente.
 *     → Por qué: el código original tenía validaciones mezcladas en un solo
 *       método. Ahora cada regla es una unidad independiente, fácil de
 *       agregar, quitar u ordenar.
 *
 *  4. REPOSITORY  (Estructural / Arquitectónico)
 *     → IContactoRepository / JsonContactoRepository / FirebaseContactoRepository
 *       (stub): abstrae el mecanismo de persistencia.
 *     → Por qué: cuando se migre de JSON a Firebase o Postgres, solo se cambia
 *       la implementación concreta; las rutas y el facade no cambian.
 *
 *  5. FACADE  (Estructural)  ← ya existía, se refina
 *     → CDAFacade: sigue siendo la única interfaz que conocen las rutas.
 *       Ahora delega en Strategy y Chain en lugar de hacerlo todo inline.
 *
 *  6. OBSERVER  (Comportamiento) — nuevo
 *     → EventBus / LoggerObserver / StatsObserver: publican eventos del sistema
 *       (chat procesado, contacto guardado, error) de forma desacoplada.
 *     → Por qué: sin Observer, los logs y métricas estaban dispersos con
 *       console.log. Ahora se pueden agregar / quitar listeners sin tocar
 *       la lógica de negocio.
 * ─────────────────────────────────────────────────────────────────────────────
 */

"use strict";

require("dotenv").config();
const express = require("express");
const cors    = require("cors");
const fs      = require("fs");
const path    = require("path");
const OpenAI  = require("openai");

// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: OBSERVER — EventBus
//  Permite que distintas partes del sistema reaccionen a eventos sin
//  acoplarse entre sí. Logger y Stats son "suscriptores" independientes.
// ═══════════════════════════════════════════════════════════════════════════

class EventBus {
  constructor() {
    /** @type {Map<string, Function[]>} */
    this._listeners = new Map();
  }

  /** Suscribe un callback a un evento */
  on(evento, callback) {
    if (!this._listeners.has(evento)) this._listeners.set(evento, []);
    this._listeners.get(evento).push(callback);
    return this; // fluent
  }

  /** Emite un evento a todos sus suscriptores */
  emit(evento, datos) {
    (this._listeners.get(evento) || []).forEach(cb => {
      try { cb(datos); } catch (e) { console.error(`[EventBus] Error en listener '${evento}':`, e); }
    });
  }
}

// Instancia global del bus (shared state intencional)
const eventBus = new EventBus();

// ── Observadores concretos ──────────────────────────────────────────────────

/** LoggerObserver: imprime eventos con timestamp */
const LoggerObserver = {
  registrar(bus) {
    bus
      .on("chat:procesado",    ({ input, duracionMs }) =>
        console.log(`[LOG] chat procesado (${duracionMs}ms): "${input.slice(0, 60)}…"`))
      .on("contacto:guardado", ({ nombre }) =>
        console.log(`[LOG] contacto guardado: ${nombre}`))
      .on("error:ia",          ({ mensaje }) =>
        console.error(`[LOG][ERROR] fallo IA: ${mensaje}`))
      .on("error:repositorio", ({ mensaje }) =>
        console.error(`[LOG][ERROR] fallo repo: ${mensaje}`));
  }
};

/** StatsObserver: acumula métricas en memoria (podría volcarlas a BD) */
class StatsObserver {
  constructor() {
    this.totalChats     = 0;
    this.totalContactos = 0;
    this.totalErrores   = 0;
    this.tiempoTotalMs  = 0;
  }
  registrar(bus) {
    bus
      .on("chat:procesado",    ({ duracionMs }) => {
        this.totalChats++;
        this.tiempoTotalMs += duracionMs;
      })
      .on("contacto:guardado", () => { this.totalContactos++; })
      .on("error:ia",          () => { this.totalErrores++; })
      .on("error:repositorio", () => { this.totalErrores++; });
    return this;
  }
  resumen() {
    const avg = this.totalChats ? Math.round(this.tiempoTotalMs / this.totalChats) : 0;
    return { totalChats: this.totalChats, totalContactos: this.totalContactos,
             totalErrores: this.totalErrores, tiempoPromedioMs: avg };
  }
}

const stats = new StatsObserver().registrar(eventBus);
LoggerObserver.registrar(eventBus);


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: CHAIN OF RESPONSIBILITY — Validaciones
//  Cada Handler es un nodo de la cadena. Llama a this._next si pasa,
//  o retorna un error si no. Se componen sin condicionales anidados.
// ═══════════════════════════════════════════════════════════════════════════

class ValidacionHandler {
  constructor() { this._next = null; }

  /** Encadena el siguiente validador (fluent) */
  setSiguiente(handler) {
    this._next = handler;
    return handler; // permite cadenas: a.setSiguiente(b).setSiguiente(c)
  }

  /** Subclases deben sobreescribir este método */
  validar(datos) {
    return this._next ? this._next.validar(datos) : { valido: true };
  }
}

// ── Eslabones concretos ─────────────────────────────────────────────────────

class ValidarNoVacio extends ValidacionHandler {
  validar(datos) {
    const { mensaje } = datos;
    if (!mensaje || typeof mensaje !== "string" || !mensaje.trim())
      return { valido: false, error: "El mensaje no puede estar vacío." };
    return super.validar(datos);
  }
}

class ValidarLongitud extends ValidacionHandler {
  constructor(maxChars = 1000) { super(); this.maxChars = maxChars; }
  validar(datos) {
    if (datos.mensaje.length > this.maxChars)
      return { valido: false, error: `Mensaje demasiado largo (máx ${this.maxChars} caracteres).` };
    return super.validar(datos);
  }
}

class ValidarCorreo extends ValidacionHandler {
  validar(datos) {
    const { correo } = datos;
    if (!correo || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(correo))
      return { valido: false, error: "Correo electrónico inválido." };
    return super.validar(datos);
  }
}

class ValidarNombreObligatorio extends ValidacionHandler {
  validar(datos) {
    if (!datos.nombre || !datos.nombre.trim())
      return { valido: false, error: "El nombre es obligatorio." };
    return super.validar(datos);
  }
}

// ── Fábricas de cadenas ─────────────────────────────────────────────────────

/** Construye la cadena de validación para el chat */
function crearCadenaChat() {
  const vacio    = new ValidarNoVacio();
  const longitud = new ValidarLongitud(1000);
  vacio.setSiguiente(longitud);
  return vacio; // entrada a la cadena
}

/** Construye la cadena de validación para el formulario de contacto */
function crearCadenaContacto() {
  const nombre = new ValidarNombreObligatorio();
  const correo = new ValidarCorreo();
  nombre.setSiguiente(correo);
  return nombre;
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: STRATEGY — Motor de IA
//  IAStrategy define la interfaz. Las implementaciones concretas son
//  intercambiables: OpenAIStrategy, FallbackStrategy.
//  El CDAFacade recibe la estrategia por constructor (inyección).
// ═══════════════════════════════════════════════════════════════════════════

/** Interfaz (contrato) de cualquier motor de IA */
class IAStrategy {
  /** @returns {Promise<string>} */
  async responder(_mensaje) {
    throw new Error("responder() debe implementarse en la subclase.");
  }
}

/** Estrategia concreta: OpenAI GPT-4o-mini */
class OpenAIStrategy extends IAStrategy {
  constructor(apiKey) {
    super();
    this.client = new OpenAI({ apiKey });
    this.systemPrompt = `
Eres un asistente virtual de CDA Scooters SAS, Centro de Diagnóstico Automotor
habilitado por el Ministerio de Transporte de Colombia, ubicado en Acacías, Meta.
Registro ONAC: 17-OIN-054, Clase D.

Ayuda con:
- Revisión técnico-mecánica y de emisiones contaminantes
- Documentos requeridos: SOAT vigente, matrícula, cédula del propietario
- Precios: Motocicleta $218.000 | Liviano/Motocarro $318.000
- Horarios: Lun–Vie 7AM–6PM · Sáb 7AM–4PM · Festivos 8AM–12M
- Agendamiento de citas
- Fallas comunes que causan no aprobación

Responde de forma clara, profesional y amigable en español.
Si la consulta no tiene relación con el CDA o vehículos, redirige amablemente.

Contacto: +57 311 800 6270 | cda.scootersacacias@gmail.com`.trim();
  }

  async responder(mensaje) {
    const completion = await this.client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: this.systemPrompt },
        { role: "user",   content: mensaje }
      ]
    });
    return completion.choices[0].message.content;
  }
}

/**
 * Estrategia de respaldo: respuestas predefinidas sin IA externa.
 * Útil cuando la API de OpenAI no está disponible o la clave no está
 * configurada (desarrollo local, tests, etc.).
 */
class FallbackStrategy extends IAStrategy {
  constructor() {
    super();
    this._respuestas = [
      { patron: /precio|tarifa|valor|costo/i,
        resp: "La revisión para motocicletas tiene un costo de $218.000 y para vehículos livianos $318.000. ¿Desea agendar una cita?" },
      { patron: /hora|horario|atiend/i,
        resp: "Atendemos Lun–Vie 7AM–6PM, Sábados 7AM–4PM y Festivos 8AM–12M." },
      { patron: /document|requisi|necesit/i,
        resp: "Necesita traer: SOAT vigente, matrícula del vehículo y cédula del propietario." },
      { patron: /cit|agend/i,
        resp: "Para agendar su cita use el formulario de la página o escríbanos por correo." },
    ];
    this._default = "En este momento no puedo procesar su consulta automáticamente. Contáctenos por el formulario o por correo.";
  }

  async responder(mensaje) {
    const encontrada = this._respuestas.find(r => r.patron.test(mensaje));
    return encontrada ? encontrada.resp : this._default;
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: REPOSITORY — Persistencia de contactos
//  La interfaz IContactoRepository define el contrato.
//  JsonContactoRepository es la implementación actual (archivo JSON).
//  FirebaseContactoRepository es un stub para la migración futura.
// ═══════════════════════════════════════════════════════════════════════════

/** Interfaz del repositorio */
class IContactoRepository {
  async guardar(_contacto)    { throw new Error("guardar() no implementado."); }
  async obtenerTodos()        { throw new Error("obtenerTodos() no implementado."); }
}

/** Implementación concreta: persiste en un archivo JSON local */
class JsonContactoRepository extends IContactoRepository {
  constructor(rutaArchivo = "mensajes.json") {
    super();
    this.rutaArchivo = rutaArchivo;
  }

  async guardar(contacto) {
    const datos = await this.obtenerTodos();
    datos.push({ ...contacto, fecha: new Date().toISOString() });
    fs.writeFileSync(this.rutaArchivo, JSON.stringify(datos, null, 2), "utf-8");
    return true;
  }

  async obtenerTodos() {
    try {
      if (fs.existsSync(this.rutaArchivo))
        return JSON.parse(fs.readFileSync(this.rutaArchivo, "utf-8"));
    } catch { /* archivo corrupto o inexistente */ }
    return [];
  }
}

/**
 * Stub para Firebase / Firestore.
 * Cuando se active, solo se pasa esta clase al Facade en lugar de la JSON.
 * El resto del código no necesita cambios.
 */
class FirebaseContactoRepository extends IContactoRepository {
  constructor(db, addDoc, collection) {
    super();
    this.db = db; this.addDoc = addDoc; this.collection = collection;
  }
  async guardar(contacto) {
    await this.addDoc(this.collection(this.db, "contactos"), {
      ...contacto, fecha: new Date()
    });
    return true;
  }
  async obtenerTodos() { return []; } // implementar con getDocs si se necesita
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: FACADE — CDAFacade  (refinado)
//  Sigue siendo la única clase que conocen las rutas Express.
//  Ahora orquesta Strategy + Chain + Repository + EventBus.
// ═══════════════════════════════════════════════════════════════════════════

class CDAFacade {
  /**
   * @param {IAStrategy}            estrategiaIA
   * @param {IContactoRepository}   repositorio
   * @param {EventBus}              bus
   */
  constructor(estrategiaIA, repositorio, bus) {
    this._ia          = estrategiaIA;
    this._repositorio = repositorio;
    this._bus         = bus;
    // Cadenas de validación construidas una sola vez
    this._cadenaChat     = crearCadenaChat();
    this._cadenaContacto = crearCadenaContacto();
  }

  /**
   * Procesa un mensaje del chatbot.
   * Chain valida → Strategy responde → Observer notifica.
   */
  async procesarChat(mensaje) {
    const validacion = this._cadenaChat.validar({ mensaje });
    if (!validacion.valido) return { ok: false, error: validacion.error };

    const inicio = Date.now();
    try {
      const reply      = await this._ia.responder(mensaje);
      const duracionMs = Date.now() - inicio;
      this._bus.emit("chat:procesado", { input: mensaje, duracionMs });
      return { ok: true, reply };
    } catch (err) {
      this._bus.emit("error:ia", { mensaje: err.message });
      return { ok: false, error: "El asistente no está disponible en este momento. Contáctanos por el formulario o por correo." };
    }
  }

  /**
   * Procesa el formulario de contacto.
   * Chain valida → Repository guarda → Observer notifica.
   */
  async procesarContacto(datos) {
    const validacion = this._cadenaContacto.validar(datos);
    if (!validacion.valido) return { ok: false, mensaje: validacion.error };

    try {
      await this._repositorio.guardar(datos);
      this._bus.emit("contacto:guardado", { nombre: datos.nombre });
      return { ok: true, mensaje: "¡Mensaje recibido! Te contactaremos pronto ✅" };
    } catch (err) {
      this._bus.emit("error:repositorio", { mensaje: err.message });
      return { ok: false, mensaje: "Error al guardar. Por favor intenta nuevamente ❌" };
    }
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: SINGLETON — AppServer
//  Garantiza una única instancia del servidor Express en todo el proceso.
//  getInstance() retorna siempre la misma referencia.
// ═══════════════════════════════════════════════════════════════════════════

class AppServer {
  /** @type {AppServer} */
  static _instancia = null;

  constructor() {
    if (AppServer._instancia) return AppServer._instancia;

    this.app = express();
    this._configurarMiddleware();
    this._configurarRutas();

    AppServer._instancia = this;
  }

  static getInstance() {
    if (!AppServer._instancia) new AppServer();
    return AppServer._instancia;
  }

  _configurarMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.static(path.join(__dirname, "../frontend")));
  }

  _configurarRutas() {
    // ── Selección de estrategia según entorno ──────────────────────────────
    // STRATEGY: si hay clave de OpenAI se usa la estrategia real;
    // si no, la de respaldo. Fácil de extender con más estrategias.
    const estrategiaIA = process.env.OPENAI_API_KEY
      ? new OpenAIStrategy(process.env.OPENAI_API_KEY)
      : new FallbackStrategy();

    const repositorio = new JsonContactoRepository();

    // FACADE: única interfaz que conocen las rutas
    const facade = new CDAFacade(estrategiaIA, repositorio, eventBus);

    // ── Rutas ──────────────────────────────────────────────────────────────

    /** Chat — delega todo al facade */
    this.app.post("/chat", async (req, res) => {
      const resultado = await facade.procesarChat(req.body.message);
      if (resultado.ok) return res.json({ reply: resultado.reply });
      res.status(400).json({ reply: resultado.error });
    });

    /** Contacto — delega todo al facade */
    this.app.post("/contacto", async (req, res) => {
      const resultado = await facade.procesarContacto(req.body);
      res.status(resultado.ok ? 200 : 400).json({ mensaje: resultado.mensaje });
    });

    /** Stats — expone métricas del Observer (útil para monitoreo) */
    this.app.get("/stats", (_req, res) => {
      res.json(stats.resumen());
    });
  }

  /** Inicia el servidor en el puerto indicado */
  listen(puerto = process.env.PORT || 3000) {
    this.app.listen(puerto, () =>
      console.log(`🚀 CDA Scooters SAS — servidor en puerto ${puerto}`)
    );
  }
}


// ── Punto de entrada ────────────────────────────────────────────────────────
AppServer.getInstance().listen();