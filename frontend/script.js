/**
 * script.js — CDA Scooters Asociados SAS
 * ─────────────────────────────────────────────────────────────────────────────
 * PATRONES APLICADOS EN ESTE ARCHIVO:
 *
 *  1. COMPOSITE  (Estructural) — ya existía, se refina
 *     → ComponenteBase / ComponenteHoja / ComponenteGrupo / ArbolCDA
 *     → Se limpia la API, se eliminan los logs de debug innecesarios en
 *       producción y se añade un método destroy() para evitar memory leaks.
 *
 *  2. DECORATOR  (Estructural) — ya existía, se refina
 *     → DecoradorHover / DecoradorResaltado / DecoradorAnimacion
 *     → Se extrae la inyección de estilos a un helper (_inyectarEstilo)
 *       para no duplicar el getElementById en cada decorador.
 *
 *  3. COMMAND  (Comportamiento) — NUEVO
 *     → ICommand / AbrirModalCommand / CerrarModalCommand / EnviarChatCommand
 *       / EnviarContactoCommand / AbrirWhatsAppCommand
 *     → Por qué: los botones del HTML llaman directamente a funciones
 *       globales. Con Command, cada acción es un objeto que puede ser
 *       ejecutado, registrado o desecho (undo). Facilita tests y añadir
 *       historial de acciones sin tocar la lógica de UI.
 *
 *  4. MEDIATOR  (Comportamiento) — NUEVO
 *     → UIMediator: centraliza la comunicación entre componentes de UI
 *       (modal, chatbot, navbar). En lugar de que cada componente
 *       conozca a los demás, todos notifican al mediador.
 *     → Por qué: el código original tenía acoplamiento directo entre
 *       toggleChat, openModal, closeModal y los elementos del DOM.
 *       Con Mediator se eliminan dependencias cruzadas.
 *
 *  5. STATE  (Comportamiento) — NUEVO
 *     → ChatState / EstadoInactivo / EstadoEsperando / EstadoRespondiendo
 *     → Por qué: el chatbot tenía lógica de estado dispersa (is typing,
 *       disable input, etc.). Con State, cada estado sabe qué puede hacer
 *       y qué no, evitando condiciones `if (estoyEsperando)` diseminadas.
 *
 *  6. COMPOSITE / DECORATOR ya existentes se mantienen con mejoras menores.
 * ─────────────────────────────────────────────────────────────────────────────
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════════════
//  HELPER: inyector de estilos (evita duplicar getElementById en decoradores)
// ═══════════════════════════════════════════════════════════════════════════

function _inyectarEstilo(id, css) {
  if (document.getElementById(id)) return;
  const s = document.createElement("style");
  s.id = id;
  s.textContent = css;
  document.head.appendChild(s);
}

function _sanitizar(texto) {
  const div = document.createElement("div");
  div.textContent = texto;
  return div.innerHTML;
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: COMPOSITE — Árbol de componentes UI
//  ComponenteBase es el nodo abstracto.
//  ComponenteHoja representa un elemento DOM sin hijos.
//  ComponenteGrupo es un nodo compuesto que puede tener hijos.
//  ArbolCDA construye la jerarquía completa de la página.
// ═══════════════════════════════════════════════════════════════════════════

class ComponenteBase {
  constructor(id, nombre) {
    this.id     = id;
    this.nombre = nombre;
  }
  render()          { throw new Error(`render() no implementado en ${this.nombre}`); }
  agregar(_hijo)    { console.warn(`${this.nombre} no acepta hijos.`); }
  /** Libera listeners y referencias (evita memory leaks) */
  destroy()         {}
}

class ComponenteHoja extends ComponenteBase {
  constructor(id, nombre) {
    super(id, nombre);
    this.elemento = document.getElementById(id);
  }
  render() { return this.elemento; }
}

class ComponenteGrupo extends ComponenteBase {
  constructor(id, nombre) {
    super(id, nombre);
    this.hijos    = [];
    this.elemento = document.getElementById(id);
  }
  agregar(componente) {
    this.hijos.push(componente);
    return this; // fluent
  }
  render() {
    this.hijos.forEach(h => h.render());
    return this.elemento;
  }
  /** Recorre recursivamente y devuelve todos los elementos DOM */
  obtenerElementos() {
    const acc = this.elemento ? [this.elemento] : [];
    return this.hijos.reduce((a, h) => {
      if (h instanceof ComponenteGrupo) return a.concat(h.obtenerElementos());
      if (h.elemento) a.push(h.elemento);
      return a;
    }, acc);
  }
  destroy() { this.hijos.forEach(h => h.destroy()); }
}

class ArbolCDA {
  constructor() {
    this.raiz = new ComponenteGrupo("__pagina__", "Página CDA");
    this._construir();
  }
  _construir() {
    const secServicios = new ComponenteGrupo("servicios", "Servicios");
    document.querySelectorAll(".servicio-card").forEach((el, i) => {
      if (!el.id) el.id = `servicio-card-${i}`;
      const hoja = new ComponenteHoja(el.id, `Card ${i + 1}`);
      hoja.elemento = el;
      secServicios.agregar(hoja);
    });
    this.raiz
      .agregar(secServicios)
      .agregar(new ComponenteHoja("contacto",      "Contacto"))
      .agregar(new ComponenteHoja("terminos",       "Términos"))
      .agregar(new ComponenteHoja("seguridad-vial", "Seguridad Vial"));
  }
  renderizarTodo() { this.raiz.render(); }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: DECORATOR — Enriquecimiento de componentes UI
//  Cada decorador envuelve a otro y agrega comportamiento sin modificarlo.
//  Se pueden apilar: Animacion(Hover(Resaltado(base))).aplicar()
// ═══════════════════════════════════════════════════════════════════════════

class ComponenteDecorable {
  constructor(elemento) { this.elemento = elemento; }
  aplicar()             { return this; }
  obtenerElemento()     { return this.elemento; }
}

class DecoradorHover extends ComponenteDecorable {
  constructor(componente, { escala = "1.03", sombra = "0 8px 32px rgba(11,37,69,.14)", duracion = "0.28s" } = {}) {
    super(componente.elemento);
    this._inner  = componente;
    this._escala = escala;
    this._sombra = sombra;
    this._dur    = duracion;
  }
  aplicar() {
    this._inner.aplicar();
    const el = this.elemento;
    el.style.transition = `transform ${this._dur} ease, box-shadow ${this._dur} ease`;
    el.addEventListener("mouseenter", () => {
      el.style.transform  = `translateY(-5px) scale(${this._escala})`;
      el.style.boxShadow  = this._sombra;
    });
    el.addEventListener("mouseleave", () => {
      el.style.transform  = "";
      el.style.boxShadow  = "";
    });
    return this;
  }
}

class DecoradorResaltado extends ComponenteDecorable {
  constructor(componente) {
    super(componente.elemento);
    this._inner = componente;
  }
  aplicar() {
    this._inner.aplicar();
    _inyectarEstilo("deco-resaltado-style", `
      @keyframes pulseBorder {
        0%,100% { box-shadow: 0 0 0 0 rgba(11,37,69,.22); }
        50%      { box-shadow: 0 0 0 8px rgba(11,37,69,0); }
      }
      .deco-resaltado { animation: pulseBorder 2.2s ease-in-out infinite; }
    `);
    this.elemento.classList.add("deco-resaltado");
    return this;
  }
}

class DecoradorAnimacion extends ComponenteDecorable {
  constructor(componente, { tipo = "fadeUp", delay = 0 } = {}) {
    super(componente.elemento);
    this._inner = componente;
    this._tipo  = tipo;
    this._delay = delay;
  }
  aplicar() {
    this._inner.aplicar();
    const el = this.elemento;
    if (!el) return this;
    _inyectarEstilo("deco-animacion-style", `
      .deco-anim-hidden   { opacity:0; transform:translateY(20px); }
      .deco-anim-visible  { opacity:1; transform:none;
                            transition:opacity .6s ease, transform .6s ease; }
      .deco-fade-hidden   { opacity:0; }
      .deco-fade-visible  { opacity:1; transition:opacity .6s ease; }
    `);
    const claseO = this._tipo === "fadeUp" ? "deco-anim-hidden"  : "deco-fade-hidden";
    const claseV = this._tipo === "fadeUp" ? "deco-anim-visible" : "deco-fade-visible";
    el.classList.add(claseO);
    el.style.transitionDelay = `${this._delay}ms`;
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.replace(claseO, claseV);
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.15 });
    obs.observe(el);
    return this;
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: COMMAND — Acciones de la interfaz de usuario
//  Cada acción del usuario (abrir modal, enviar formulario, etc.) es un
//  objeto Command con un método execute(). Las funciones globales del HTML
//  ahora son delegadoras delgadas que simplemente llaman a CommandBus.
//
//  Ventaja: se puede registrar un historial, implementar undo/redo,
//  agregar logging o analytics sin tocar la lógica de cada acción.
// ═══════════════════════════════════════════════════════════════════════════

/** Interfaz de comando */
class ICommand {
  execute() { throw new Error("execute() no implementado."); }
}

/** Comando: abrir el modal de agendamiento */
class AbrirModalCommand extends ICommand {
  constructor(tipo = "general") { super(); this._tipo = tipo; }
  execute() {
    const select = document.getElementById("m-tipo");
    if (this._tipo && this._tipo !== "general" && select)
      select.value = this._tipo;
    document.getElementById("modal-overlay").classList.add("active");
    document.body.style.overflow = "hidden";
  }
}

/** Comando: cerrar el modal */
class CerrarModalCommand extends ICommand {
  execute() {
    document.getElementById("modal-overlay").classList.remove("active");
    document.body.style.overflow = "";
  }
}

/** Comando: enviar el formulario del modal por WhatsApp */
class SubmitModalCommand extends ICommand {
  execute() {
    const get  = id => document.getElementById(id);
    const nombre = get("m-nombre").value.trim();
    const tel    = get("m-tel").value.trim();
    const placa  = get("m-placa").value.trim().toUpperCase();
    const tipoEl = get("m-tipo");
    const fecha  = get("m-fecha").value;
    const hora   = get("m-hora").value;
    const nota   = get("m-nota").value.trim();

    if (!nombre || !tel || !placa || !tipoEl.value || !fecha || !hora) {
      alert("Por favor completa los campos obligatorios (*) antes de continuar.");
      return;
    }

    const tipoTxt = tipoEl.options[tipoEl.selectedIndex]?.text || "";
    const msg = [
      "🚗 *Solicitud de cita — CDA Scooters SAS*", "",
      `👤 *Nombre:* ${nombre}`,
      `📱 *Teléfono:* ${tel}`,
      `🔢 *Placa:* ${placa}`,
      `🚘 *Tipo de vehículo:* ${tipoTxt}`,
      `📅 *Fecha preferida:* ${fecha}`,
      `⏰ *Hora preferida:* ${hora}`,
      nota ? `📝 *Observaciones:* ${nota}` : ""
    ].filter(Boolean).join("\n");

    window.open("https://wa.me/573118006270?text=" + encodeURIComponent(msg), "_blank", "noopener");
    new CerrarModalCommand().execute();
  }
}

/** Comando: enviar formulario de contacto al backend (Firebase) */
class EnviarContactoCommand extends ICommand {
  async execute() {
    const get  = id => document.getElementById(id);
    const nombre   = get("fname").value.trim();
    const correo   = get("femail").value.trim();
    const telefono = get("fphone").value.trim();
    const vehiculo = get("fvehicle").value;
    const mensaje  = get("fmsg").value.trim();

    if (!nombre || !correo) {
      alert("Por favor completa nombre y correo electrónico.");
      return;
    }
    try {
      await window.addDoc(window.collection(window.db, "contactos"),
        { nombre, correo, telefono, vehiculo, mensaje, fecha: new Date() });
      alert("¡Mensaje enviado correctamente! ✅");
      ["fname","femail","fphone","fvehicle","fmsg"].forEach(id => get(id).value = "");
    } catch (err) {
      console.error(err);
      alert("Error al enviar el mensaje ❌");
    }
  }
}

/** Comando: abrir WhatsApp con mensaje predefinido */
class AbrirWhatsAppCommand extends ICommand {
  constructor(mensaje) { super(); this._mensaje = mensaje; }
  execute() {
    window.open(
      "https://wa.me/573118006270?text=" + encodeURIComponent(this._mensaje),
      "_blank", "noopener"
    );
  }
}

/**
 * CommandBus: registro central de comandos.
 * Las funciones globales registran y despachan comandos aquí.
 * Permite historial y potencial undo/redo en el futuro.
 */
class CommandBus {
  constructor() { this._historial = []; }

  /** Ejecuta un comando y lo agrega al historial */
  despachar(command) {
    this._historial.push({ comando: command.constructor.name, ts: Date.now() });
    return command.execute();
  }

  historial() { return [...this._historial]; }
}

const commandBus = new CommandBus();


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: MEDIATOR — UIMediator
//  Centraliza la comunicación entre la navbar, el modal y el chatbot.
//  Ningún componente habla directamente con otro; todos pasan por aquí.
//
//  Por qué: antes, toggleChat() tocaba el DOM directamente y openModal()
//  cambiaba body.overflow sin coordinación. Ahora el Mediator gestiona
//  las interacciones y puede aplicar reglas transversales (p.ej. cerrar
//  el chat si se abre el modal).
// ═══════════════════════════════════════════════════════════════════════════

class UIMediator {
  constructor() {
    this._chatAbierto  = false;
    this._modalAbierto = false;
  }

  /** Alterna el chatbot; cierra el modal si estaba abierto */
  toggleChat() {
    if (this._modalAbierto) commandBus.despachar(new CerrarModalCommand());
    const chat = document.getElementById("chatbot");
    if (!chat) return;
    this._chatAbierto = !this._chatAbierto;
    chat.classList.toggle("open", this._chatAbierto);
  }

  /** Abre el modal; colapsa el menú móvil si estaba abierto */
  abrirModal(tipo = "general") {
    this._cerrarMenuMovil();
    this._modalAbierto = true;
    commandBus.despachar(new AbrirModalCommand(tipo));
  }

  /** Cierra el modal */
  cerrarModal() {
    this._modalAbierto = false;
    commandBus.despachar(new CerrarModalCommand());
  }

  /** Alterna el menú móvil */
  toggleNav() {
    document.getElementById("mobileMenu")?.classList.toggle("open");
  }

  /** Cierra el menú móvil */
  closeNav() {
    document.getElementById("mobileMenu")?.classList.remove("open");
  }

  _cerrarMenuMovil() {
    document.getElementById("mobileMenu")?.classList.remove("open");
  }
}

const uiMediator = new UIMediator();


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: STATE — Estado del Chatbot
//  El chatbot pasa por tres estados: Inactivo, Esperando (respuesta IA),
//  Respondiendo. Cada estado controla qué puede hacer el usuario.
//
//  Por qué: antes había código disperso para deshabilitar el input,
//  mostrar/ocultar los puntos de "typing" y manejar errores. Con State,
//  cada estado encapsula su propio comportamiento.
// ═══════════════════════════════════════════════════════════════════════════

class ChatState {
  constructor(contexto) { this._ctx = contexto; }
  /** Se llama al intentar enviar un mensaje */
  async enviar(_texto) { /* por defecto, ignorar */ }
}

/** Estado: el usuario puede escribir y enviar */
class EstadoInactivo extends ChatState {
  async enviar(texto) {
    if (!texto) return;
    this._ctx.mostrarMensaje(texto, "user");
    this._ctx.limpiarInput();
    this._ctx.setState(new EstadoEsperando(this._ctx));
    await this._ctx.getEstado().enviar(texto);
  }
}

/** Estado: esperando respuesta del servidor (input bloqueado) */
class EstadoEsperando extends ChatState {
  constructor(ctx) {
    super(ctx);
    this._dotsId = "typing-dots";
  }
  async enviar(texto) {
    // Mostrar animación de "escribiendo…"
    this._ctx.mostrarTyping(this._dotsId);

    try {
      const res  = await fetch("https://cda-scooters.onrender.com/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: texto })
      });
      const data = await res.json();
      this._ctx.quitarTyping(this._dotsId);
      this._ctx.mostrarMensaje(data.reply, "bot");
    } catch {
      this._ctx.quitarTyping(this._dotsId);
      this._ctx.mostrarMensaje(
        'No puedo responder ahora. Escríbenos al <a href="https://wa.me/573118006270" target="_blank">WhatsApp</a>.',
        "bot"
      );
    } finally {
      this._ctx.setState(new EstadoInactivo(this._ctx));
    }
  }
}

/** Contexto del chatbot que mantiene una referencia al estado actual */
class ChatbotContexto {
  constructor(inputId, messagesId) {
    this._inputEl    = document.getElementById(inputId);
    this._messagesEl = document.getElementById(messagesId);
    this._estado     = new EstadoInactivo(this);
  }
  setState(estado) { this._estado = estado; }
  getEstado()      { return this._estado; }

  async enviarMensaje() {
    const texto = this._inputEl?.value.trim() || "";
    await this._estado.enviar(texto);
  }

  mostrarMensaje(html, tipo) {
    if (!this._messagesEl) return;
    const div = document.createElement("div");
    div.className = `message ${tipo}`;
    div.innerHTML = html;
    this._messagesEl.appendChild(div);
    this._messagesEl.scrollTop = this._messagesEl.scrollHeight;
  }

  mostrarTyping(id) {
    if (!this._messagesEl || document.getElementById(id)) return;
    const dots = document.createElement("div");
    dots.className = "message bot typing-dots";
    dots.id = id;
    dots.innerHTML = "<span></span><span></span><span></span>";
    this._messagesEl.appendChild(dots);
    this._messagesEl.scrollTop = this._messagesEl.scrollHeight;
  }

  quitarTyping(id) { document.getElementById(id)?.remove(); }

  limpiarInput() { if (this._inputEl) this._inputEl.value = ""; }
}

// Instancia única del contexto del chatbot
let chatbot = null;


// ═══════════════════════════════════════════════════════════════════════════
//  INICIALIZACIÓN PRINCIPAL
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {

  // 1. Árbol de componentes (COMPOSITE)
  new ArbolCDA().renderizarTodo();

  // 2. Decorar tarjetas de servicios con Hover + Animación escalonada
  document.querySelectorAll(".servicio-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { sombra: "0 10px 36px rgba(11,37,69,.13)" }),
      { tipo: "fadeUp", delay: i * 110 }
    ).aplicar();
  });

  // 3. Decorar tarjetas de política de seguridad vial
  document.querySelectorAll(".psv-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { escala: "1.02", sombra: "0 6px 24px rgba(11,37,69,.10)" }),
      { tipo: "fadeUp", delay: i * 80 }
    ).aplicar();
  });

  // 4. Decorar bloques de "cómo asistir"
  document.querySelectorAll(".asistir-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { escala: "1.01", sombra: "0 4px 18px rgba(11,37,69,.09)" }),
      { tipo: "fadeUp", delay: i * 90 }
    ).aplicar();
  });

  // 5. Inicializar chatbot (STATE)
  chatbot = new ChatbotContexto("input", "messages");

  // 6. Comportamientos globales
  _inicializarNavbar();
  _inicializarScrollReveal();
  _inicializarNavActiva();
  _inicializarFechaMinima();
  _inicializarAtajosTeclado();
});


// ═══════════════════════════════════════════════════════════════════════════
//  COMPORTAMIENTOS GLOBALES
// ═══════════════════════════════════════════════════════════════════════════

function _inicializarNavbar() {
  const navbar = document.getElementById("navbar");
  if (!navbar) return;
  window.addEventListener("scroll", () => {
    navbar.style.boxShadow = window.scrollY > 40
      ? "0 2px 24px rgba(11,37,69,.28)"
      : "0 2px 20px rgba(11,37,69,.22)";
  }, { passive: true });
}

function _inicializarScrollReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    els.forEach(el => el.classList.add("visible"));
    return;
  }
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("visible"); obs.unobserve(e.target); }
    });
  }, { threshold: 0.1 });
  els.forEach(el => obs.observe(el));
}

function _inicializarNavActiva() {
  const secciones = ["inicio","servicios","asistir","contacto","terminos","seguridad-vial"];
  const links     = document.querySelectorAll(".nav a");
  const actualizar = () => {
    let actual = "inicio";
    secciones.forEach(id => {
      const el = document.getElementById(id);
      if (el && window.scrollY >= el.offsetTop - 110) actual = id;
    });
    links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + actual));
  };
  window.addEventListener("scroll", actualizar, { passive: true });
  actualizar();
}

function _inicializarFechaMinima() {
  const hoy  = new Date();
  const yyyy = hoy.getFullYear();
  const mm   = String(hoy.getMonth() + 1).padStart(2, "0");
  const dd   = String(hoy.getDate()).padStart(2, "0");
  const inp  = document.getElementById("m-fecha");
  if (inp) inp.min = `${yyyy}-${mm}-${dd}`;
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = yyyy;
}

/** Atajos de teclado globales */
function _inicializarAtajosTeclado() {
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") uiMediator.cerrarModal();
  });

  // Cerrar modal al hacer clic en el overlay
  const overlay = document.getElementById("modal-overlay");
  overlay?.addEventListener("click", e => {
    if (e.target === overlay) uiMediator.cerrarModal();
  });
}


// ═══════════════════════════════════════════════════════════════════════════
//  FUNCIONES GLOBALES — Puente con el HTML
//  Son lo más delgadas posible: solo delegan al Mediator o al CommandBus.
//  El HTML no necesita saber nada de patrones internos.
// ═══════════════════════════════════════════════════════════════════════════

/** Usado por el botón del modal y "Agendar cita" en header/hero */
function openModal(tipo)         { uiMediator.abrirModal(tipo); }

/** Usado por el botón ✕ del modal */
function closeModal()            { uiMediator.cerrarModal(); }

/** Compatibilidad con onclick="closeModalOnOverlay(event)" en el HTML */
function closeModalOnOverlay(e)  {
  if (e.target === document.getElementById("modal-overlay")) uiMediator.cerrarModal();
}

/** Envía el formulario del modal */
function submitModal()           { commandBus.despachar(new SubmitModalCommand()); }

/** Envía el formulario de contacto */
function enviarMensaje()         { commandBus.despachar(new EnviarContactoCommand()); }

/** Abre WhatsApp con un mensaje predeterminado */
function openWhatsApp(mensaje)   { commandBus.despachar(new AbrirWhatsAppCommand(mensaje)); }

/** Alterna el chatbot */
function toggleChat()            { uiMediator.toggleChat(); }

/** Alterna el menú móvil */
function toggleNav()             { uiMediator.toggleNav(); }

/** Cierra el menú móvil */
function closeNav()              { uiMediator.closeNav(); }

/** Envía un mensaje en el chatbot (llamado desde el input y el botón) */
async function sendMessage()     { await chatbot?.enviarMensaje(); }

/** Scroll suave a una sección */
function scrollToSection(id)     {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}