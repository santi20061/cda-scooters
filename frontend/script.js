/**
 * script.js — CDA Scooters Asociados SAS
 * ─────────────────────────────────────────────────────────────────────────────
 * PATRONES APLICADOS:
 *
 *  1. COMPOSITE  — ComponenteBase / ComponenteHoja / ComponenteGrupo / ArbolCDA
 *  2. DECORATOR  — DecoradorHover / DecoradorResaltado / DecoradorAnimacion
 *  3. COMMAND    — AbrirModalCommand / CerrarModalCommand / SubmitModalCommand
 *                   EnviarContactoCommand / AbrirWhatsAppCommand / CommandBus
 *  4. MEDIATOR   — UIMediator: centraliza navbar, modal y chatbot
 *  5. STATE      — ChatState / EstadoInactivo / EstadoEsperando / ChatbotContexto
 *  6. ROUTER     — ViewRouter: carga vistas con fetch(), historial y cache
 * ─────────────────────────────────────────────────────────────────────────────
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════════════
//  HELPERS
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

function _obtenerApiBase() {
  return (window.location.protocol === "file:" || ((window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") && window.location.port !== "3000"))
    ? "http://localhost:3000"
    : "";
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: COMPOSITE
// ═══════════════════════════════════════════════════════════════════════════

class ComponenteBase {
  constructor(id, nombre) {
    this.id     = id;
    this.nombre = nombre;
  }
  render()       { throw new Error(`render() no implementado en ${this.nombre}`); }
  agregar(_hijo) { console.warn(`${this.nombre} no acepta hijos.`); }
  destroy()      {}
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
  agregar(componente) { this.hijos.push(componente); return this; }
  render() {
    this.hijos.forEach(h => h.render());
    return this.elemento;
  }
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
//  PATRÓN: DECORATOR
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
      el.style.transform = `translateY(-5px) scale(${this._escala})`;
      el.style.boxShadow = this._sombra;
    });
    el.addEventListener("mouseleave", () => {
      el.style.transform = "";
      el.style.boxShadow = "";
    });
    return this;
  }
}

class DecoradorResaltado extends ComponenteDecorable {
  constructor(componente) { super(componente.elemento); this._inner = componente; }
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
      .deco-anim-hidden  { opacity:0; transform:translateY(20px); }
      .deco-anim-visible { opacity:1; transform:none; transition:opacity .6s ease, transform .6s ease; }
      .deco-fade-hidden  { opacity:0; }
      .deco-fade-visible { opacity:1; transition:opacity .6s ease; }
    `);
    const claseO = this._tipo === "fadeUp" ? "deco-anim-hidden"  : "deco-fade-hidden";
    const claseV = this._tipo === "fadeUp" ? "deco-anim-visible" : "deco-fade-visible";
    el.classList.add(claseO);
    el.style.transitionDelay = `${this._delay}ms`;
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.replace(claseO, claseV); obs.unobserve(e.target); }
      });
    }, { threshold: 0.15 });
    obs.observe(el);
    return this;
  }
}


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: COMMAND
// ═══════════════════════════════════════════════════════════════════════════

class ICommand { execute() { throw new Error("execute() no implementado."); } }

class AbrirModalCommand extends ICommand {
  constructor(tipo = "general") { super(); this._tipo = tipo; }
  execute() {
    const select = document.getElementById("m-tipo");
    if (this._tipo && this._tipo !== "general" && select) select.value = this._tipo;
    document.getElementById("modal-overlay").classList.add("active");
    document.body.style.overflow = "hidden";
  }
}

class CerrarModalCommand extends ICommand {
  execute() {
    document.getElementById("modal-overlay").classList.remove("active");
    document.body.style.overflow = "";
  }
}

class SubmitModalCommand extends ICommand {
  async execute() {
    const get    = id => document.getElementById(id);
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
    const apiBase = _obtenerApiBase();
    const btn = document.getElementById("appointment-submit-btn");
    const btnOriginal = btn ? btn.innerHTML : "";

    if (btn) {
      btn.disabled = true;
      btn.textContent = "Enviando…";
    }

    try {
      const res = await fetch(`${apiBase}/api/appointments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nombre,
          telefono: tel,
          placa,
          tipoVehiculo: tipoTxt,
          fecha,
          hora,
          servicio: "Revisión técnico-mecánica",
          observaciones: nota,
        }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const errorMsg = Array.isArray(data?.errors)
          ? data.errors.join(", ")
          : (data?.error || data?.mensaje || "No se pudo registrar la cita");
        throw new Error(errorMsg);
      }

      alert(data?.mensaje || "Solicitud registrada y enviada por WhatsApp.");
      ["m-nombre", "m-tel", "m-placa", "m-tipo", "m-fecha", "m-hora", "m-nota"].forEach(id => {
        const el = get(id);
        if (el) el.value = "";
      });
      new CerrarModalCommand().execute();
    } catch (error) {
      console.error("[appointment]", error);
      alert(`Error: ${error.message || "No se pudo enviar la solicitud"}`);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = btnOriginal;
      }
    }
  }
}

class EnviarContactoCommand extends ICommand {
  async execute() {
    const get      = id => document.getElementById(id);
    const nombre   = get("fname")?.value.trim()   || "";
    const correo   = get("femail")?.value.trim()   || "";
    const telefono = get("fphone")?.value.trim()   || "";
    const vehiculo = get("fvehicle")?.value        || "";
    const mensaje  = get("fmsg")?.value.trim()     || "";

    if (!nombre || !correo || !telefono || !mensaje) {
      _mostrarFeedbackContacto("error", "Por favor completa nombre, correo, teléfono y mensaje.");
      return;
    }

    const vehiculoTexto = get("fvehicle")?.selectedOptions?.[0]?.text || "";
    const mensajeCompleto = [
      `Nombre: ${nombre}`,
      `Correo: ${correo}`,
      `Teléfono: ${telefono}`,
      vehiculoTexto ? `Tipo de vehículo: ${vehiculoTexto}` : "",
      `Mensaje: ${mensaje}`,
    ].filter(Boolean).join("\n");

    const btn        = document.querySelector("#contacto form button[type='submit']");
    const btnOrigHTML = btn ? btn.innerHTML : "";
    if (btn) { btn.disabled = true; btn.textContent = "Enviando…"; }
    _mostrarFeedbackContacto("loading", "Enviando su mensaje, por favor espere…");

    // Usar URL absoluta cuando se abre desde file:// o desde un servidor de desarrollo distinto al gateway
    const apiBase = _obtenerApiBase();

    try {
      const res = await fetch(`${apiBase}/api/contact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nombre, correo, telefono, mensaje: mensajeCompleto, vehiculo }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const errMsg = Array.isArray(data?.errors)
          ? data.errors.join(", ")
          : (data?.error || data?.mensaje || "No se pudo enviar el mensaje");
        throw new Error(errMsg);
      }

      _mostrarFeedbackContacto("success", data?.mensaje || "¡Mensaje enviado correctamente! Le responderemos pronto.");
      ["fname","femail","fphone","fvehicle","fmsg"].forEach(id => { const el = get(id); if (el) el.value = ""; });
    } catch (err) {
      console.error("[contacto]", err);
      _mostrarFeedbackContacto("error", `Error: ${err.message || "No se pudo enviar. Intente de nuevo o escríbanos por WhatsApp."}`);
    } finally {
      if (btn) { btn.disabled = false; btn.innerHTML = btnOrigHTML; }
    }
  }
}

function _mostrarFeedbackContacto(tipo, texto) {
  const el = document.getElementById("contact-feedback");
  if (!el) return;
  el.className = `contact-feedback contact-feedback--${tipo}`;
  el.textContent = texto;
  el.style.display = "block";
  if (tipo === "success") setTimeout(() => { if (el) el.style.display = "none"; }, 7000);
}

class AbrirWhatsAppCommand extends ICommand {
  constructor(mensaje) { super(); this._mensaje = mensaje; }
  execute() {
    window.open("https://wa.me/573118006270?text=" + encodeURIComponent(this._mensaje), "_blank", "noopener");
  }
}

class CommandBus {
  constructor() { this._historial = []; }
  despachar(command) {
    this._historial.push({ comando: command.constructor.name, ts: Date.now() });
    return command.execute();
  }
  historial() { return [...this._historial]; }
}

const commandBus = new CommandBus();


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: MEDIATOR
// ═══════════════════════════════════════════════════════════════════════════

class UIMediator {
  constructor() {
    this._chatAbierto  = false;
    this._modalAbierto = false;
  }

  toggleChat() {
    if (this._modalAbierto) commandBus.despachar(new CerrarModalCommand());
    const chat = document.getElementById("chatbot");
    if (!chat) return;
    this._chatAbierto = !this._chatAbierto;
    chat.classList.toggle("open", this._chatAbierto);
  }

  abrirModal(tipo = "general") {
    this._cerrarMenuMovil();
    this._modalAbierto = true;
    commandBus.despachar(new AbrirModalCommand(tipo));
  }

  cerrarModal() {
    this._modalAbierto = false;
    commandBus.despachar(new CerrarModalCommand());
  }

  toggleNav() { document.getElementById("mobileMenu")?.classList.toggle("open"); }

  closeNav()  { document.getElementById("mobileMenu")?.classList.remove("open"); }

  _cerrarMenuMovil() { document.getElementById("mobileMenu")?.classList.remove("open"); }
}

const uiMediator = new UIMediator();


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: STATE — Chatbot
// ═══════════════════════════════════════════════════════════════════════════

class ChatState {
  constructor(contexto) { this._ctx = contexto; }
  async enviar(_texto) {}
}

class EstadoInactivo extends ChatState {
  async enviar(texto) {
    if (!texto) return;
    this._ctx.mostrarMensaje(texto, "user");
    this._ctx.limpiarInput();
    this._ctx.setState(new EstadoEsperando(this._ctx));
    await this._ctx.getEstado().enviar(texto);
  }
}

class EstadoEsperando extends ChatState {
  constructor(ctx) { super(ctx); this._dotsId = "typing-dots"; }
  async enviar(texto) {
    this._ctx.mostrarTyping(this._dotsId);
    try {
      const apiBase = _obtenerApiBase();
      const res  = await fetch(`${apiBase}/api/chatbot/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: texto })
      });
      const data = await res.json();
      this._ctx.quitarTyping(this._dotsId);
      this._ctx.mostrarMensaje(data.response || data.reply || "No pude obtener una respuesta.", "bot");
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
  limpiarInput()   { if (this._inputEl) this._inputEl.value = ""; }
}

let chatbot = null;


// ═══════════════════════════════════════════════════════════════════════════
//  PATRÓN: ROUTER — ViewRouter
//  Carga vistas HTML desde /views/ mediante fetch(), gestiona historial
//  con history.pushState y aplica una función de inicialización tras cada
//  carga para activar decoradores, scroll reveal y comportamientos de vista.
// ═══════════════════════════════════════════════════════════════════════════

class ViewRouter {
  constructor(containerId) {
    this._container = document.getElementById(containerId);
    this._current   = null;
    this._binding();
  }

  _binding() {
    window.addEventListener("popstate", e => {
      const view = e.state?.view || "inicio";
      this._load(view, false);
    });
  }

  /** Pública: carga una vista. Ignorada si es la misma que la actual. */
  async cargar(nombre, pushState = true) {
    if (nombre === this._current) {
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    await this._load(nombre, pushState);
  }

  async _load(nombre, pushState) {
    const container = this._container;
    if (!container) return;

    try {
      // Fade-out
      container.style.opacity = "0";
      await _esperar(120);

      // Obtener HTML fresco para evitar que el navegador o el router
      // sirvan una versión vieja de la vista.
      const res = await fetch(`views/${nombre}.html?ts=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Vista no encontrada: ${nombre}`);
      const html = await res.text();

      // Cargar CSS específico de la vista (esperar carga antes de mostrar)
      await this._cargarCSS(nombre);

      // Inyectar
      container.innerHTML = html;
      this._current = nombre;

      // Scroll al inicio instantáneo
      window.scrollTo({ top: 0, behavior: "instant" });

      // Fade-in
      container.style.opacity = "1";

      // Historial del navegador
      if (pushState) {
        history.pushState({ view: nombre }, "", `#${nombre}`);
      }

      // Actualizar estado visual del nav
      _actualizarNavActivo(nombre);

      // Cerrar menú móvil si estaba abierto
      uiMediator.closeNav();

      // Esperar un frame para que el navegador pinte el HTML inyectado
      // antes de activar el IntersectionObserver y los decoradores
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          _inicializarVista(nombre);
        });
      });

    } catch (err) {
      container.style.opacity = "1";
      console.error("[ViewRouter]", err);
      container.innerHTML = `
        <div style="padding:80px 28px; text-align:center; color:var(--gris-medio);">
          <p style="font-size:1.1rem;">No se pudo cargar la sección. Intente de nuevo.</p>
          <button class="btn btn-azul" style="margin-top:20px;" onclick="showPanel('inicio')">Volver al inicio</button>
        </div>`;
    }
  }

  /** Carga dinámicamente el CSS de la vista y elimina el anterior */
  _cargarCSS(nombre) {
    // Eliminar CSS de vista anterior
    const prev = document.getElementById("view-css-dinamico");
    if (prev) prev.remove();

    // Lista de vistas que tienen CSS propio
    const vistasConCSS = ["inicio","nosotros","servicios","proceso","contacto","terminos","seguridad-vial"];
    if (!vistasConCSS.includes(nombre)) return Promise.resolve();

    return new Promise((resolve) => {
      const link = document.createElement("link");
      link.rel   = "stylesheet";
      link.href  = `css/${nombre}.css`;
      link.id    = "view-css-dinamico";
      link.onload  = resolve;
      link.onerror = resolve; // continuar aunque falle
      document.head.appendChild(link);
    });
  }
}

function _esperar(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

let router = null;


// ═══════════════════════════════════════════════════════════════════════════
//  POST-CARGA: inicialización de cada vista
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Se ejecuta después de inyectar cada vista.
 * Reactiva scroll reveal, decoradores y cualquier lógica específica.
 */
function _inicializarVista(nombre) {
  // Reconstruir árbol de componentes (asigna IDs a .servicio-card, etc.)
  new ArbolCDA().renderizarTodo();

  // Si la vista de contacto llegó recortada por caché o por una versión vieja,
  // completamos el contenido crítico para que el usuario vea mapa y horarios.
  _asegurarVistaCompleta(nombre);

  // Scroll reveal sobre los nuevos elementos .reveal
  _inicializarScrollReveal();

  // Decoradores sobre tarjetas de servicios
  document.querySelectorAll(".servicio-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { sombra: "0 10px 36px rgba(11,37,69,.13)" }),
      { tipo: "fadeUp", delay: i * 110 }
    ).aplicar();
  });

  // Decoradores sobre tarjetas de seguridad vial
  document.querySelectorAll(".psv-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { escala: "1.02", sombra: "0 6px 24px rgba(11,37,69,.10)" }),
      { tipo: "fadeUp", delay: i * 80 }
    ).aplicar();
  });

  // Decoradores sobre tarjetas asistir (si las hubiera en vistas)
  document.querySelectorAll(".asistir-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { escala: "1.01", sombra: "0 4px 18px rgba(11,37,69,.09)" }),
      { tipo: "fadeUp", delay: i * 90 }
    ).aplicar();
  });

  // Garantiza que el contenido de la vista quede visible aunque el observer
  // no alcance algunos nodos al cargar por router o por scroll inicial.
  _mostrarContenidoVisible();
}

function _asegurarVistaCompleta(nombre) {
  if (nombre === "nosotros") {
    // Inyectar todas las 6 tarjetas de certificación SIEMPRE
    // para garantizar que nunca se vean vacías
    const certsGrid = document.querySelector("#nosotros .nosotros-certs-grid");
    
    if (certsGrid) {
      // Reemplazar contenido del grid con las 6 tarjetas completas
      certsGrid.innerHTML = `
        <div class="cert-card">
          <div class="cert-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div class="cert-title">ONAC 17-OIN-054</div>
          <p class="cert-desc">Acreditación oficial del Organismo Nacional de Acreditación de Colombia</p>
        </div>
        
        <div class="cert-card">
          <div class="cert-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 11l3 3L22 4"/>
              <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
          <div class="cert-title">Clase D</div>
          <p class="cert-desc">Habilitados para revisar motocicletas, motocarros y vehículos livianos</p>
        </div>
        
        <div class="cert-card">
          <div class="cert-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
          </div>
          <div class="cert-title">Decreto 948/1995</div>
          <p class="cert-desc">Cumplimiento normativo en medición de emisiones contaminantes</p>
        </div>
        
        <div class="cert-card">
          <div class="cert-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
          </div>
          <div class="cert-title">Registro RUNT</div>
          <p class="cert-desc">Todos los resultados se registran en la base de datos nacional de tránsito</p>
        </div>
        
        <div class="cert-card">
          <div class="cert-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <div class="cert-title">Horario continuo</div>
          <p class="cert-desc">Atención lunes a viernes 7am–6pm, sábados 7am–4pm, festivos 8am–12m</p>
        </div>
        
        <div class="cert-card">
          <div class="cert-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
              <circle cx="12" cy="10" r="3"/>
            </svg>
          </div>
          <div class="cert-title">Ubicación central</div>
          <p class="cert-desc">Cra. 21 #15-24, Barrio Cooperativo, Acacías, Meta</p>
        </div>
      `;
    }
    return;
  }

  if (nombre === "servicios") {
    const layout = document.querySelector("#servicios .servicios-layout");
    if (!layout || document.querySelector("#servicios .servicios-video-card")) return;

    const serviciosLista = layout.querySelector(".servicios-lista");
    if (!serviciosLista) return;

    // Insertar panel liviano después del article de vehículos livianos
    const livianoCard = document.querySelector("#servicios .servicio-card:nth-of-type(2)");
    if (livianoCard && !document.querySelector(".servicio-panel-liviano")) {
      livianoCard.insertAdjacentHTML("afterend", `
        <div class="servicio-panel-liviano reveal" style="grid-column:1/-1; margin-top:8px;">
          <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;">
            <div style="background:var(--azul-xlight); border-radius:var(--radio-lg); padding:22px;">
              <div class="label-seccion">Duración</div>
              <div style="font-size:1.3rem; font-weight:700; color:var(--azul); margin:8px 0;">45–60 minutos</div>
              <p style="font-size:.85rem; color:var(--gris-texto); line-height:1.6;">
                Incluye toda la inspección técnica, prueba de emisiones y emisión del certificado.
              </p>
            </div>
            <div style="background:var(--azul-xlight); border-radius:var(--radio-lg); padding:22px;">
              <div class="label-seccion">Tipos incluidos</div>
              <div style="font-size:.95rem; color:var(--azul); margin:8px 0; font-weight:600;">
                Autos, SUV, Busetas, Motocarros
              </div>
              <p style="font-size:.85rem; color:var(--gris-texto); line-height:1.6;">
                Cualquier vehículo clasificado como liviano según el RUNT.
              </p>
            </div>
            <div style="background:var(--azul-xlight); border-radius:var(--radio-lg); padding:22px;">
              <div class="label-seccion">Documentos requeridos</div>
              <ul style="font-size:.85rem; color:var(--gris-texto); line-height:1.6; margin:8px 0; padding-left:18px;">
                <li>Matrícula original</li>
                <li>SOAT vigente</li>
                <li>Cédula del propietario</li>
              </ul>
            </div>
          </div>
        </div>
      `);
    }

    serviciosLista.insertAdjacentHTML("afterend", `
      <aside class="servicios-lateral">
        <div class="servicios-video-card reveal">
          <div class="label-seccion">Video informativo</div>
          <h3>Conozca nuestro proceso</h3>
          <p>
            Revise cómo atendemos a cada usuario, cómo se realiza la inspección y por qué
            el proceso de revisión es claro y ordenado.
          </p>
          <div class="video-wrapper">
            <iframe
              src="https://www.youtube.com/embed/n00Cn333etE"
              title="Revisión técnico-mecánica CDA Scooters"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowfullscreen>
            </iframe>
          </div>
        </div>

        <div class="servicios-info-card reveal">
          <div class="label-seccion">¿Qué incluye?</div>
          <h3>Revisión con cobertura completa</h3>
          <ul>
            <li>Diagnóstico de frenos, luces y dirección.</li>
            <li>Verificación de llantas, suspensión y carrocería.</li>
            <li>Prueba de emisiones contaminantes.</li>
            <li>Emisión del resultado y certificado según corresponda.</li>
          </ul>
        </div>
      </aside>
    `);

    layout.insertAdjacentHTML("afterend", `
      <div class="servicio-cta reveal">
        <div>
          <div class="label-seccion">Atención personalizada</div>
          <h3>Información completa para su vehículo</h3>
          <p>
            Si necesita saber qué revisamos, cuánto tarda el proceso o cuáles son los documentos,
            aquí tiene el resumen completo de servicios, requisitos y medios de pago.
          </p>
        </div>
        <div class="servicio-cta-actions">
          <button class="btn btn-amarillo" onclick="openModal('general')">Agendar cita</button>
          <a class="btn btn-outline-azul"
             href="https://wa.me/573118006270?text=Hola%2C%20quiero%20información%20sobre%20la%20revisión%20técnico-mecánica"
             target="_blank" rel="noopener">WhatsApp</a>
        </div>
      </div>

      <div class="servicios-proceso-grid reveal">
        <div class="servicios-proceso-card">
          <div class="label-seccion">Proceso</div>
          <h3>Así atendemos su revisión</h3>
          <p>
            El proceso está diseñado para ser rápido, transparente y claro. Desde el registro
            de documentos hasta la entrega del resultado, cada paso está guiado por personal técnico.
          </p>
        </div>
        <div class="servicios-proceso-card">
          <div class="label-seccion">Resultados</div>
          <h3>¿Qué obtiene al finalizar?</h3>
          <p>
            Un certificado válido si aprueba, o un diagnóstico detallado para corregir fallas y
            volver a presentar el vehículo dentro del plazo legal.
          </p>
        </div>
        <div class="servicios-proceso-card">
          <div class="label-seccion">Pago</div>
          <h3>Medios de pago disponibles</h3>
          <p>
            Efectivo, tarjeta débito/crédito y financiación con Addi o Credi S-Mart.
          </p>
        </div>
      </div>
    `);
    return;
  }

  if (nombre !== "contacto") return;

  const contacto = document.getElementById("contacto");
  if (!contacto) return;
  if (contacto.querySelector(".mapa-wrapper") && contacto.querySelector(".horarios-tab-wrap")) return;

  contacto.insertAdjacentHTML("beforeend", `
    <div class="ubicacion-grid reveal" style="margin-top:48px;">
      <div class="contact-info-list">
        <div class="contact-info-item">
          <div class="ci-icon-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
              <circle cx="12" cy="10" r="3"/>
            </svg>
          </div>
          <div>
            <div class="ci-label">Ubicación exacta</div>
            <div class="ci-value">Cra. 21 #15-24, Barrio Cooperativo<br>Acacías, Meta, Colombia</div>
          </div>
        </div>
        <div class="contact-info-item">
          <div class="ci-icon-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/>
            </svg>
          </div>
          <div>
            <div class="ci-label">Teléfono</div>
            <div class="ci-value"><a href="tel:+573118006270">+57 311 800 6270</a></div>
          </div>
        </div>
        <div class="contact-info-item">
          <div class="ci-icon-box">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
          </div>
          <div>
            <div class="ci-label">Horario</div>
            <div class="ci-value">
              Lun–Vie 7AM–6PM<br>
              Sáb 7AM–4PM<br>
              Fest 8AM–12M
            </div>
          </div>
        </div>
        <div style="padding-top:20px;">
          <a class="btn btn-amarillo btn-block"
             href="https://www.google.com/maps/search/?api=1&query=3.988342985633988,-73.76894482430293"
             target="_blank" rel="noopener">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
              <circle cx="12" cy="10" r="3"/>
            </svg>
            Abrir en Google Maps
          </a>
        </div>
      </div>

      <div class="mapa-wrapper">
        <iframe
          src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3980.156228067716!2d-73.76885262430281!3d3.9882680856342625!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x8e3e3950821d8495%3A0xecc19d81e5ba85a7!2sCDA%20Scooters!5e0!3m2!1ses-419!2sco!4v1778336301638!5m2!1ses-419!2sco"
          width="600"
          height="450"
          style="border:0;"
          allowfullscreen=""
          loading="lazy"
          title="Ubicación CDA Scooters — Acacías, Meta">
        </iframe>
        <div style="text-align:right; padding:4px 0;">
          <a href="https://www.google.com/maps/search/?api=1&query=3.988342985633988,-73.76894482430293"
             target="_blank" rel="noopener"
             style="font-size:.72rem; color:var(--gris-medio);">
            Ver en Google Maps ↗
          </a>
        </div>
      </div>
    </div>

    <div class="horarios-tab-wrap reveal" style="margin-top:48px;">
      <div class="horarios-table-card">
        <h2 class="horarios-tab-titulo">Horarios de atención</h2>
        <table class="horarios-table">
          <thead>
            <tr>
              <th>Día</th>
              <th>Apertura</th>
              <th>Cierre</th>
              <th>Duración revisión</th>
            </tr>
          </thead>
          <tbody>
            <tr class="tr-highlight">
              <td><strong>Lunes a viernes</strong></td>
              <td>7:00 a.m.</td>
              <td>6:00 p.m.</td>
              <td>45–60 min</td>
            </tr>
            <tr>
              <td><strong>Sábados</strong></td>
              <td>7:00 a.m.</td>
              <td>4:00 p.m.</td>
              <td>45–60 min</td>
            </tr>
            <tr>
              <td><strong>Festivos</strong></td>
              <td>8:00 a.m.</td>
              <td>12:00 m.</td>
              <td>45–60 min</td>
            </tr>
            <tr>
              <td><strong>Domingos</strong></td>
              <td colspan="3" style="color:var(--gris-medio);">Cerrado</td>
            </tr>
          </tbody>
        </table>
        <div class="horarios-nota">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          Se recomienda llegar <strong>15 minutos antes</strong> de su turno con todos los
          documentos listos. La última revisión se admite 45 minutos antes del cierre.
        </div>
      </div>

      <div class="horarios-cta">
        <div class="label-seccion">Agenda tu cita</div>
        <h3>¿Listo para su revisión?</h3>
        <p>
          Reserva con anticipación y evita las filas. Le atendemos con puntualidad
          y garantizamos un proceso ágil y sin contratiempos.
        </p>
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:20px;">
          <button class="btn btn-amarillo" onclick="openModal('general')">Agendar cita</button>
          <a class="btn btn-outline-azul"
             href="https://wa.me/573118006270?text=Hola%2C%20quiero%20agendar%20una%20cita%20para%20revisión%20técnico-mecánica"
             target="_blank" rel="noopener">WhatsApp</a>
        </div>

        <div style="margin-top:28px; padding-top:20px; border-top:1px solid rgba(255,255,255,.15);">
          <p style="font-size:.8rem; color:rgba(255,255,255,.5); margin-bottom:10px;">
            También puede llamarnos directamente:
          </p>
          <a href="tel:+573118006270"
             style="font-size:1.1rem; font-weight:700; color:var(--amarillo); display:flex; align-items:center; gap:8px;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
              <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"/>
            </svg>
            +57 311 800 6270
          </a>
        </div>
      </div>
    </div>
  `);
}

function _mostrarContenidoVisible() {
  document.querySelectorAll(".reveal, .deco-anim-hidden, .deco-fade-hidden").forEach(el => {
    el.classList.add("visible", "deco-anim-visible");
    el.classList.remove("deco-anim-hidden", "deco-fade-hidden");
    el.style.opacity = "1";
    el.style.transform = "none";
  });
}

/** Marca el enlace de nav correspondiente a la vista activa */
function _actualizarNavActivo(nombre) {
  document.querySelectorAll(".nav a, .mobile-menu a").forEach(a => {
    const href = (a.getAttribute("href") || "").replace("#", "");
    a.classList.toggle("active", href === nombre);
  });
}


// ═══════════════════════════════════════════════════════════════════════════
//  COMPORTAMIENTOS GLOBALES (shell)
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
  const els = document.querySelectorAll(".reveal:not(.visible)");
  if (!("IntersectionObserver" in window)) {
    els.forEach(el => el.classList.add("visible"));
    return;
  }
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("visible"); obs.unobserve(e.target); }
    });
  }, { threshold: 0.05, rootMargin: "0px 0px 60px 0px" });

  els.forEach(el => {
    // Elementos ya en el viewport se hacen visibles de inmediato (evita parpadeo)
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight + 80 && rect.bottom >= 0) {
      el.classList.add("visible");
    } else {
      obs.observe(el);
    }
  });
}

function _inicializarFechaMinima() {
  const hoy    = new Date();
  const yyyy   = hoy.getFullYear();
  const mm     = String(hoy.getMonth() + 1).padStart(2, "0");
  const dd     = String(hoy.getDate()).padStart(2, "0");
  const inp    = document.getElementById("m-fecha");
  if (inp) inp.min = `${yyyy}-${mm}-${dd}`;
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = yyyy;
}

function _inicializarAtajosTeclado() {
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
      uiMediator.cerrarModal();
      closeAsistirModal();
    }
  });
  const overlay = document.getElementById("modal-overlay");
  overlay?.addEventListener("click", e => {
    if (e.target === overlay) uiMediator.cerrarModal();
  });
}


// ═══════════════════════════════════════════════════════════════════════════
//  INICIALIZACIÓN PRINCIPAL
// ═══════════════════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  // Chatbot (elementos en el shell)
  chatbot = new ChatbotContexto("input", "messages");

  // Comportamientos del shell (navbar, modal, atajos)
  _inicializarNavbar();
  _inicializarFechaMinima();
  _inicializarAtajosTeclado();

  // Inicializar router
  router = new ViewRouter("view-container");

  // Cargar vista inicial desde el hash de la URL o 'inicio' por defecto
  const vistasValidas = ["inicio","nosotros","servicios","proceso","contacto","terminos","seguridad-vial"];
  const hashInicial   = window.location.hash.replace("#", "");
  const vistaInicial  = vistasValidas.includes(hashInicial) ? hashInicial : "inicio";

  router.cargar(vistaInicial, !hashInicial);
});


// ═══════════════════════════════════════════════════════════════════════════
//  FUNCIONES GLOBALES — Puente con el HTML
// ═══════════════════════════════════════════════════════════════════════════

function openModal(tipo)          { uiMediator.abrirModal(tipo); }
function closeModal()             { uiMediator.cerrarModal(); }
function closeModalOnOverlay(e)   {
  if (e.target === document.getElementById("modal-overlay")) uiMediator.cerrarModal();
}
function submitModal()            { commandBus.despachar(new SubmitModalCommand()); }
function enviarMensaje()          { commandBus.despachar(new EnviarContactoCommand()); }
function openWhatsApp(mensaje)    { commandBus.despachar(new AbrirWhatsAppCommand(mensaje)); }
function toggleChat()             { uiMediator.toggleChat(); }
function toggleNav()              { uiMediator.toggleNav(); }
function closeNav()               { uiMediator.closeNav(); }
async function sendMessage()      { await chatbot?.enviarMensaje(); }

/**
 * showPanel — adaptada al router.
 * El HTML la llama desde onclick en nav, footer y botones de vista.
 */
function showPanel(name) {
  router?.cargar(name);
}


// ═══════════════════════════════════════════════════════════════════════════
//  ACORDEÓN — Términos y Seguridad Vial
// ═══════════════════════════════════════════════════════════════════════════

function toggleAcc(btn) {
  const item      = btn.closest(".acc-item");
  const accordion = btn.closest(".accordion");
  const isOpen    = item.classList.contains("open");

  if (accordion) {
    accordion.querySelectorAll(".acc-item").forEach(i => i.classList.remove("open"));
  }
  if (!isOpen) item.classList.add("open");
}


// ═══════════════════════════════════════════════════════════════════════════
//  TABS — Contacto
// ═══════════════════════════════════════════════════════════════════════════

function switchTab(id, btn) {
  document.querySelectorAll("#contacto .tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll("#contacto .tab-btn").forEach(b => b.classList.remove("active"));
  const panel = document.getElementById("tab-" + id);
  if (panel) panel.classList.add("active");
  if (btn)   btn.classList.add("active");
}


// ═══════════════════════════════════════════════════════════════════════════
//  MODAL ASISTIR
// ═══════════════════════════════════════════════════════════════════════════

function openAsistirModal() {
  const overlay = document.getElementById("modal-asistir");
  if (!overlay) return;
  overlay.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeAsistirModal() {
  const overlay = document.getElementById("modal-asistir");
  if (!overlay) return;
  overlay.classList.remove("active");
  document.body.style.overflow = "";
}

function closeAsistirOnOverlay(e) {
  if (e.target === document.getElementById("modal-asistir")) closeAsistirModal();
}