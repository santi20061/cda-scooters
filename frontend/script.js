"use strict";

// =============================================================
//  ★ PATRÓN: COMPOSITE
// =============================================================

class ComponenteBase {
  constructor(id, nombre) {
    this.id     = id;
    this.nombre = nombre;
  }
  render() { throw new Error(`render() no implementado en ${this.nombre}`); }
  agregar(componente) { console.warn(`${this.nombre} no es un grupo.`); }
}

class ComponenteHoja extends ComponenteBase {
  constructor(id, nombre) {
    super(id, nombre);
    this.elemento = document.getElementById(id);
  }
  render() {
    if (this.elemento) console.log(`🌿 Hoja: [${this.nombre}] #${this.id}`);
    return this.elemento;
  }
}

class ComponenteGrupo extends ComponenteBase {
  constructor(id, nombre) {
    super(id, nombre);
    this.hijos    = [];
    this.elemento = document.getElementById(id);
  }
  agregar(componente) {
    this.hijos.push(componente);
    return this;
  }
  render() {
    console.log(`📦 Grupo: [${this.nombre}] con ${this.hijos.length} hijo(s)`);
    this.hijos.forEach(h => h.render());
    return this.elemento;
  }
  obtenerElementos() {
    let elementos = [];
    if (this.elemento) elementos.push(this.elemento);
    this.hijos.forEach(h => {
      if (h instanceof ComponenteGrupo) elementos = elementos.concat(h.obtenerElementos());
      else if (h.elemento) elementos.push(h.elemento);
    });
    return elementos;
  }
}

class ArbolCDA {
  constructor() {
    this.raiz = new ComponenteGrupo("__pagina__", "Página CDA Scooters SAS");
    this._construir();
  }
  _construir() {
    const seccionServicios = new ComponenteGrupo("servicios", "Sección Servicios");
    document.querySelectorAll(".service-card").forEach((el, i) => {
      if (!el.id) el.id = `service-card-${i}`;
      const hoja = new ComponenteHoja(el.id, `Card Servicio ${i + 1}`);
      hoja.elemento = el;
      seccionServicios.agregar(hoja);
    });

    const seccionContacto  = new ComponenteHoja("contacto",      "Sección Contacto");
    const seccionTerminos  = new ComponenteHoja("terminos",       "Sección Términos");
    const seccionSegVial   = new ComponenteHoja("seguridad-vial", "Sección Seg. Vial");

    this.raiz
      .agregar(seccionServicios)
      .agregar(seccionContacto)
      .agregar(seccionTerminos)
      .agregar(seccionSegVial);
  }
  renderizarTodo() {
    console.group("🌳 Árbol de Componentes CDA — COMPOSITE");
    this.raiz.render();
    console.groupEnd();
  }
}


// =============================================================
//  ★ PATRÓN: DECORATOR
// =============================================================

class ComponenteDecorable {
  constructor(elemento) { this.elemento = elemento; }
  aplicar() { return this; }
  obtenerElemento() { return this.elemento; }
}

class DecoradorHover extends ComponenteDecorable {
  constructor(componente, opciones = {}) {
    super(componente.elemento);
    this.componente = componente;
    this.opciones   = {
      escala:   opciones.escala   || "1.03",
      sombra:   opciones.sombra   || "0 8px 32px rgba(11,37,69,0.14)",
      duracion: opciones.duracion || "0.28s"
    };
  }
  aplicar() {
    this.componente.aplicar();
    const el  = this.elemento;
    const ops = this.opciones;
    el.style.transition = `transform ${ops.duracion} ease, box-shadow ${ops.duracion} ease`;
    el.addEventListener("mouseenter", () => {
      el.style.transform = `translateY(-5px) scale(${ops.escala})`;
      el.style.boxShadow = ops.sombra;
    });
    el.addEventListener("mouseleave", () => {
      el.style.transform = "";
      el.style.boxShadow = "";
    });
    return this;
  }
}

class DecoradorResaltado extends ComponenteDecorable {
  constructor(componente, etiqueta = "⭐ Destacado") {
    super(componente.elemento);
    this.componente = componente;
    this.etiqueta   = etiqueta;
  }
  aplicar() {
    this.componente.aplicar();
    const el      = this.elemento;
    const styleId = "deco-resaltado-style";
    if (!document.getElementById(styleId)) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = `
        @keyframes pulseBorder {
          0%, 100% { box-shadow: 0 0 0 0 rgba(11,37,69,0.22); }
          50%       { box-shadow: 0 0 0 8px rgba(11,37,69,0); }
        }
        .deco-resaltado { animation: pulseBorder 2.2s ease-in-out infinite; }
      `;
      document.head.appendChild(style);
    }
    el.classList.add("deco-resaltado");
    return this;
  }
}

class DecoradorAnimacion extends ComponenteDecorable {
  constructor(componente, opciones = {}) {
    super(componente.elemento);
    this.componente = componente;
    this.tipo  = opciones.tipo  || "fadeUp";
    this.delay = opciones.delay || 0;
  }
  aplicar() {
    this.componente.aplicar();
    const el = this.elemento;
    if (!el) return this;
    if (!document.getElementById("deco-animacion-style")) {
      const style = document.createElement("style");
      style.id = "deco-animacion-style";
      style.textContent = `
        .deco-anim-hidden  { opacity: 0; transform: translateY(20px); }
        .deco-anim-visible { opacity: 1; transform: none;
                             transition: opacity 0.6s ease, transform 0.6s ease; }
        .deco-fadeIn-hidden  { opacity: 0; }
        .deco-fadeIn-visible { opacity: 1; transition: opacity 0.6s ease; }
      `;
      document.head.appendChild(style);
    }
    const claseO = this.tipo === "fadeUp" ? "deco-anim-hidden"    : "deco-fadeIn-hidden";
    const claseV = this.tipo === "fadeUp" ? "deco-anim-visible"   : "deco-fadeIn-visible";
    el.classList.add(claseO);
    el.style.transitionDelay = `${this.delay}ms`;
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.remove(claseO);
          e.target.classList.add(claseV);
          obs.unobserve(e.target);
        }
      });
    }, { threshold: 0.15 });
    obs.observe(el);
    return this;
  }
}


// =============================================================
//  INICIALIZACIÓN PRINCIPAL
// =============================================================

document.addEventListener("DOMContentLoaded", () => {

  // 1. Árbol de componentes (COMPOSITE)
  const arbol = new ArbolCDA();
  arbol.renderizarTodo();

  // 2. Decorar cards de servicios con Hover + Animación escalonada
  document.querySelectorAll(".service-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { sombra: "0 10px 36px rgba(11,37,69,0.13)" }),
      { tipo: "fadeUp", delay: i * 110 }
    ).aplicar();
  });

  // 3. Decorar cards de política de seguridad vial
  document.querySelectorAll(".psv-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { escala: "1.02", sombra: "0 6px 24px rgba(11,37,69,0.1)" }),
      { tipo: "fadeUp", delay: i * 80 }
    ).aplicar();
  });

  // 4. Decorar cards de "cómo asistir"
  document.querySelectorAll(".asistir-card").forEach((el, i) => {
    const base = new ComponenteDecorable(el);
    new DecoradorAnimacion(
      new DecoradorHover(base, { escala: "1.01", sombra: "0 4px 18px rgba(11,37,69,0.09)" }),
      { tipo: "fadeUp", delay: i * 90 }
    ).aplicar();
  });

  // 5. Comportamientos globales
  _inicializarNavbar();
  _inicializarScrollReveal();
  _inicializarNavActiva();
  _inicializarFechaMinima();

  console.log("✅ CDA Scooters SAS — Patrones Composite y Decorator activos");
});


// =============================================================
//  NAVBAR
// =============================================================

function _inicializarNavbar() {
  const navbar = document.getElementById("navbar");
  window.addEventListener("scroll", () => {
    navbar.style.boxShadow = window.scrollY > 40
      ? "0 2px 24px rgba(11,37,69,.28)"
      : "0 2px 20px rgba(11,37,69,.22)";
  }, { passive: true });
}

function toggleNav() {
  document.getElementById("mobileMenu").classList.toggle("open");
}
function closeNav() {
  document.getElementById("mobileMenu").classList.remove("open");
}


// =============================================================
//  SCROLL REVEAL (para elementos con clase .reveal)
// =============================================================

function _inicializarScrollReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    els.forEach(el => el.classList.add("visible"));
    return;
  }
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });
  els.forEach(el => obs.observe(el));
}


// =============================================================
//  NAV ACTIVA AL HACER SCROLL
// =============================================================

function _inicializarNavActiva() {
  const sections = ["inicio","servicios","asistir","contacto","terminos","seguridad-vial"];
  const links    = document.querySelectorAll(".nav a");
  function actualizar() {
    let actual = "inicio";
    sections.forEach(id => {
      const el = document.getElementById(id);
      if (el && window.scrollY >= el.offsetTop - 110) actual = id;
    });
    links.forEach(a => {
      a.classList.toggle("active", a.getAttribute("href") === "#" + actual);
    });
  }
  window.addEventListener("scroll", actualizar, { passive: true });
  actualizar();
}


// =============================================================
//  FECHA MÍNIMA EN EL MODAL (hoy)
// =============================================================

function _inicializarFechaMinima() {
  const hoy  = new Date();
  const yyyy = hoy.getFullYear();
  const mm   = String(hoy.getMonth() + 1).padStart(2, "0");
  const dd   = String(hoy.getDate()).padStart(2, "0");
  const inp  = document.getElementById("m-fecha");
  if (inp) inp.min = `${yyyy}-${mm}-${dd}`;

  // Año dinámico en el footer
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = yyyy;
}


// =============================================================
//  MODAL DE AGENDAMIENTO
// =============================================================

function openModal(tipo) {
  const select = document.getElementById("m-tipo");
  if (tipo && tipo !== "general" && select) select.value = tipo;
  document.getElementById("modal-overlay").classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  document.getElementById("modal-overlay").classList.remove("active");
  document.body.style.overflow = "";
}

function closeModalOnOverlay(e) {
  if (e.target === document.getElementById("modal-overlay")) closeModal();
}

document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeModal();
});

/** Envía el formulario del modal por WhatsApp */
function submitModal() {
  const nombre = document.getElementById("m-nombre").value.trim();
  const tel    = document.getElementById("m-tel").value.trim();
  const placa  = document.getElementById("m-placa").value.trim().toUpperCase();
  const tipo   = document.getElementById("m-tipo");
  const tipoTxt = tipo.options[tipo.selectedIndex]?.text || "";
  const fecha  = document.getElementById("m-fecha").value;
  const hora   = document.getElementById("m-hora").value;
  const nota   = document.getElementById("m-nota").value.trim();

  if (!nombre || !tel || !placa || !tipo.value || !fecha || !hora) {
    alert("Por favor completa los campos obligatorios (*) antes de continuar.");
    return;
  }

  const msg = [
    "🚗 *Solicitud de cita — CDA Scooters SAS*",
    "",
    `👤 *Nombre:* ${nombre}`,
    `📱 *Teléfono:* ${tel}`,
    `🔢 *Placa:* ${placa}`,
    `🚘 *Tipo de vehículo:* ${tipoTxt}`,
    `📅 *Fecha preferida:* ${fecha}`,
    `⏰ *Hora preferida:* ${hora}`,
    nota ? `📝 *Observaciones:* ${nota}` : "",
  ].filter(Boolean).join("\n");

  window.open("https://wa.me/573118006270?text=" + encodeURIComponent(msg), "_blank", "noopener");
  closeModal();
}


// =============================================================
//  FORMULARIO DE CONTACTO → Backend
// =============================================================

async function enviarMensaje() {
  const nombre   = document.getElementById("fname").value.trim();
  const correo   = document.getElementById("femail").value.trim();
  const telefono = document.getElementById("fphone").value.trim();
  const vehiculo = document.getElementById("fvehicle").value;
  const mensaje  = document.getElementById("fmsg").value.trim();

  if (!nombre || !correo) {
    alert("Por favor completa nombre y correo electrónico.");
    return;
  }

  try {
    const res  = await fetch("http://localhost:3000/contacto", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ nombre, correo, telefono, vehiculo, mensaje })
    });
    const data = await res.json();
    alert(data.mensaje || "¡Mensaje enviado! Te contactaremos pronto.");
  } catch {
    alert("Error al conectar con el servidor. Puedes contactarnos directamente al +57 311 800 6270.");
  }
}


// =============================================================
//  ABRIR WHATSAPP CON MENSAJE PREDEFINIDO
// =============================================================

function openWhatsApp(mensaje) {
  window.open(
    "https://wa.me/573118006270?text=" + encodeURIComponent(mensaje),
    "_blank",
    "noopener"
  );
}


// =============================================================
//  CHATBOT (se mantiene para compatibilidad con server.js)
// =============================================================

function toggleChat() {
  const chat = document.getElementById("chatbot");
  if (chat) chat.classList.toggle("open");
}

async function sendMessage() {
  const inputEl    = document.getElementById("input");
  const messagesEl = document.getElementById("messages");
  if (!inputEl || !messagesEl) return;
  const texto = inputEl.value.trim();
  if (!texto) return;

  messagesEl.innerHTML += `<div class="message user">${_sanitizar(texto)}</div>`;
  inputEl.value = "";
  messagesEl.scrollTop = messagesEl.scrollHeight;

  const dots = document.createElement("div");
  dots.className = "message bot typing-dots";
  dots.innerHTML = "<span></span><span></span><span></span>";
  dots.id = "typing-dots";
  messagesEl.appendChild(dots);
  messagesEl.scrollTop = messagesEl.scrollHeight;

try {
  const res = await fetch("https://cda-scooters.onrender.com/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message: texto })
  });

  const data = await res.json();

  document.getElementById("typing-dots")?.remove();

  messagesEl.innerHTML += `<div class="message bot">${data.reply}</div>`;

} catch {
  document.getElementById("typing-dots")?.remove();

  messagesEl.innerHTML += `
    <div class="message bot">
      No puedo responder ahora. Escríbenos al
      <a href="https://wa.me/573118006270" target="_blank">WhatsApp</a>.
    </div>`;
}

messagesEl.scrollTop = messagesEl.scrollHeight;
}


// =============================================================
//  UTILIDADES
// =============================================================

function _sanitizar(texto) {
  const div = document.createElement("div");
  div.textContent = texto;
  return div.innerHTML;
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: "smooth" });
}