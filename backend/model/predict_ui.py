"""
=============================================================
  Sistema Predictivo CDA — Centro de Diagnóstico Automotriz
  Aplicación avanzada de predicción interactiva
=============================================================
Archivo  : predict_ui.py
Modelo   : modelo_cda.pkl  (misma carpeta)
Librerías: tkinter, ttk, joblib, pandas, matplotlib, numpy
=============================================================
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "modelo_cda.pkl"
TRAIN_SCRIPT = "train.py"

C = {
    "bg":       "#0F1923",
    "panel":    "#162030",
    "card":     "#1C2B3A",
    "accent":   "#00C2FF",
    "accent2":  "#0077FF",
    "success":  "#00E5A0",
    "danger":   "#FF4D6A",
    "warn":     "#FFB020",
    "text":     "#E8F0F8",
    "muted":    "#5C7A96",
    "border":   "#1E3448",
    "plot_bg":  "#111C27",
    "grid":     "#1A2E40",
}

FONT_TITLE  = ("Courier New", 13, "bold")
FONT_HEADER = ("Courier New", 10, "bold")
FONT_BODY   = ("Courier New",  9)
FONT_MONO   = ("Courier New",  9)
FONT_METRIC = ("Courier New", 20, "bold")
FONT_LABEL  = ("Courier New",  8)
FONT_RESULT = ("Courier New", 17, "bold")


# ─────────────────────────────────────────────────────────────
# LÓGICA DE MODELO
# ─────────────────────────────────────────────────────────────

class ModelManager:
    """Carga y mantiene el modelo. Expone predict y predict_proba."""

    def __init__(self):
        self.model    = None
        self.features = []   # orden de features según el modelo

    def cargar(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró '{MODEL_PATH}'.\n"
                "Ejecuta primero train.py para generar el modelo.")
        self.model = joblib.load(MODEL_PATH)
        # Orden de features guardado en el modelo (sklearn ≥1.0)
        if hasattr(self.model, "feature_names_in_"):
            self.features = list(self.model.feature_names_in_)
        else:
            # Fallback: orden esperado por el script de entrenamiento
            self.features = ["kilometraje", "anio", "tiempo_desde_revision"]
        return self.model

    def predecir(self, anio: float, kilometraje: float, meses: float):
        """
        Construye el DataFrame con el orden EXACTO de features del modelo
        y retorna (clase, prob_falla).
        """
        if self.model is None:
            self.cargar()

        # Mapeo flexible de valores a nombres de features conocidos
        aliases = {
            "anio":                  anio,
            "año":                   anio,
            "year":                  anio,
            "kilometraje":           kilometraje,
            "km":                    kilometraje,
            "mileage":               kilometraje,
            "tiempo_desde_revision": meses,
            "meses":                 meses,
            "tiempo":                meses,
            "months":                meses,
        }
        row = {}
        for feat in self.features:
            key = feat.strip().lower()
            if key in aliases:
                row[feat] = aliases[key]
            else:
                row[feat] = 0.0   # valor neutro para features desconocidas

        X = pd.DataFrame([row])
        clase = int(self.model.predict(X)[0])
        prob  = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            clases = list(self.model.classes_)
            idx = clases.index(1) if 1 in clases else -1
            prob = float(proba[idx]) * 100 if idx >= 0 else 50.0
        return clase, prob

    def importancias(self):
        """Devuelve dict {feature: importancia} si el modelo está cargado."""
        if self.model is None:
            return {}
        return dict(zip(self.features, self.model.feature_importances_))


# ─────────────────────────────────────────────────────────────
# HELPERS DE UI
# ─────────────────────────────────────────────────────────────

def apply_style(root):
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=C["bg"], foreground=C["text"], font=FONT_BODY)
    s.configure("TFrame",    background=C["bg"])
    s.configure("TNotebook", background=C["bg"], tabmargins=[0, 0, 0, 0])
    s.configure("TNotebook.Tab",
                background=C["panel"], foreground=C["muted"],
                font=FONT_HEADER, padding=[20, 8])
    s.map("TNotebook.Tab",
          background=[("selected", C["card"])],
          foreground=[("selected", C["accent"])])
    s.configure("TScale",  background=C["card"], troughcolor=C["bg"],
                sliderlength=18)
    s.configure("TScrollbar", background=C["panel"],
                troughcolor=C["bg"], arrowcolor=C["muted"])


def make_btn(parent, text, cmd, color=None, width=16):
    color = color or C["accent2"]
    f = tk.Frame(parent, bg=color, cursor="hand2")
    lbl = tk.Label(f, text=text, font=FONT_HEADER, bg=color, fg="white",
                   padx=14, pady=8, width=width)
    lbl.pack()
    lbl.bind("<Button-1>", lambda e: cmd())
    f.bind("<Button-1>",   lambda e: cmd())

    def hover_on(e):  f.config(bg=C["accent"]);  lbl.config(bg=C["accent"])
    def hover_off(e): f.config(bg=color);         lbl.config(bg=color)
    for w in (f, lbl):
        w.bind("<Enter>", hover_on)
        w.bind("<Leave>", hover_off)
    return f


def make_card(parent, title="", padx=16, pady=12):
    outer = tk.Frame(parent, bg=C["border"])
    inner = tk.Frame(outer, bg=C["card"])
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    if title:
        tk.Label(inner, text=title, font=FONT_HEADER,
                 bg=C["card"], fg=C["accent"]).pack(anchor="w", padx=padx, pady=(pady, 4))
        tk.Frame(inner, height=1, bg=C["border"]).pack(fill="x", padx=padx)
    return outer, inner


def campo_entry(parent, label: str, placeholder: str):
    """Entry con placeholder estilizado. Retorna (frame, StringVar)."""
    var = tk.StringVar()
    f = tk.Frame(parent, bg=C["card"])
    tk.Label(f, text=label, font=FONT_LABEL, bg=C["card"], fg=C["muted"]).pack(anchor="w")
    border = tk.Frame(f, bg=C["muted"])
    border.pack(fill="x", pady=(2, 0))
    e = tk.Entry(border, textvariable=var, font=FONT_BODY,
                 bg=C["panel"], fg=C["muted"], relief="flat",
                 insertbackground=C["accent"], bd=8)
    e.pack(fill="x", ipady=6)
    e.insert(0, placeholder)

    def fi(_):
        if e.get() == placeholder:
            e.delete(0, tk.END); e.config(fg=C["text"])
        border.config(bg=C["accent"])
    def fo(_):
        if not e.get().strip():
            e.insert(0, placeholder); e.config(fg=C["muted"])
        border.config(bg=C["muted"])
    e.bind("<FocusIn>", fi); e.bind("<FocusOut>", fo)
    return f, var, placeholder


# ─────────────────────────────────────────────────────────────
# WIDGET: RESULTADO (compartido entre tabs)
# ─────────────────────────────────────────────────────────────

class ResultWidget(tk.Frame):
    """Muestra emoji, texto de riesgo, probabilidad y barra."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["card"], **kwargs)
        self.lbl_emoji = tk.Label(self, text="—", font=("Courier New", 36),
                                  bg=C["card"], fg=C["muted"])
        self.lbl_emoji.pack(pady=(18, 4))

        self.lbl_risk = tk.Label(self, text="SIN PREDICCIÓN",
                                 font=FONT_RESULT, bg=C["card"], fg=C["muted"])
        self.lbl_risk.pack()

        self.lbl_prob = tk.Label(self, text="Probabilidad de falla: —",
                                 font=FONT_BODY, bg=C["card"], fg=C["muted"])
        self.lbl_prob.pack(pady=(8, 4))

        bar_bg = tk.Frame(self, bg=C["bg"], height=10)
        bar_bg.pack(fill="x", padx=40, pady=(0, 4))
        self._bar_bg = bar_bg
        self._bar    = tk.Frame(bar_bg, bg=C["muted"], height=10)
        self._bar.place(relwidth=0, relheight=1)

        self.lbl_tip = tk.Label(self, text="", font=FONT_LABEL,
                                bg=C["card"], fg=C["muted"], justify="center")
        self.lbl_tip.pack(pady=(0, 16))

    def update(self, clase: int, prob: float | None):
        if clase == 1:
            emoji, riesgo, color = "⚠", "ALTO RIESGO DE FALLA", C["danger"]
            tip = "Revisión inmediata recomendada\nantes de presentar al CDA."
        else:
            emoji, riesgo, color = "✓", "BAJO RIESGO", C["success"]
            tip = "Buenas condiciones para\nla revisión técnico-mecánica."

        self.lbl_emoji.config(text=emoji, fg=color)
        self.lbl_risk.config(text=riesgo, fg=color)
        self.lbl_tip.config(text=tip)

        if prob is not None:
            self.lbl_prob.config(
                text=f"Probabilidad de falla:  {prob:.1f} %", fg=C["text"])
            pct = max(0.03, prob / 100)
            self._bar.config(bg=color)
            self._bar.place(relwidth=pct, relheight=1)
        else:
            self.lbl_prob.config(text="Probabilidad: no disponible", fg=C["muted"])
            self._bar.place(relwidth=0, relheight=1)


# ─────────────────────────────────────────────────────────────
# PESTAÑA 1: PREDICCIÓN
# ─────────────────────────────────────────────────────────────

class TabPrediccion(tk.Frame):
    def __init__(self, parent, manager: ModelManager, status_cb):
        super().__init__(parent, bg=C["bg"])
        self.manager   = manager
        self.set_status = status_cb
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # — Formulario —
        card_o, card_i = make_card(self, "■ DATOS DEL VEHÍCULO")
        card_o.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        f1, self.var_anio, self.ph_anio   = campo_entry(card_i, "AÑO DEL VEHÍCULO", "Ej: 2018")
        f2, self.var_km,   self.ph_km     = campo_entry(card_i, "KILOMETRAJE",       "Ej: 85000")
        f3, self.var_mes,  self.ph_mes    = campo_entry(card_i, "MESES DESDE REVISIÓN", "Ej: 14")
        for f in (f1, f2, f3):
            f.pack(fill="x", padx=18, pady=10)

        tk.Frame(card_i, height=8, bg=C["card"]).pack()
        make_btn(card_i, "  Predecir", self._predecir).pack(padx=18, pady=(0, 18), fill="x")

        # Botón reentrenar
        make_btn(card_i, "  Reentrenar Modelo", self._reentrenar,
                 color=C["warn"], width=20).pack(padx=18, pady=(0, 18), fill="x")

        # — Resultado —
        card_o2, card_i2 = make_card(self, "■ RESULTADO")
        card_o2.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self.result_widget = ResultWidget(card_i2)
        self.result_widget.pack(fill="both", expand=True)

    def _leer(self, var, ph, nombre) -> float:
        v = var.get().strip()
        if not v or v == ph:
            raise ValueError(f"El campo '{nombre}' está vacío.")
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"'{nombre}' debe ser un número. Recibido: '{v}'")

    def _predecir(self):
        try:
            anio = self._leer(self.var_anio, self.ph_anio, "Año")
            km   = self._leer(self.var_km,   self.ph_km,   "Kilometraje")
            mes  = self._leer(self.var_mes,  self.ph_mes,  "Meses")
        except ValueError as e:
            messagebox.showerror("Datos inválidos", str(e)); return
        try:
            clase, prob = self.manager.predecir(anio, km, mes)
            self.result_widget.update(clase, prob)
            self.set_status(
                f"Predicción: {'FALLA' if clase else 'APRUEBA'}  |  "
                f"Prob. falla: {prob:.1f}%  |  "
                f"Año: {int(anio)}  Km: {km:,.0f}  Meses: {mes:.0f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reentrenar(self):
        if not os.path.exists(TRAIN_SCRIPT):
            messagebox.showerror("Error", f"No se encontró '{TRAIN_SCRIPT}'.")
            return
        self.set_status("Lanzando entrenamiento…")
        try:
            subprocess.Popen([sys.executable, TRAIN_SCRIPT])
            messagebox.showinfo("Reentrenamiento",
                                f"Se abrió '{TRAIN_SCRIPT}'.\n"
                                "Cuando termine, recarga el modelo cerrando y reabriendo esta app.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ─────────────────────────────────────────────────────────────
# PESTAÑA 2: SIMULACIÓN
# ─────────────────────────────────────────────────────────────

class TabSimulacion(tk.Frame):
    def __init__(self, parent, manager: ModelManager, status_cb):
        super().__init__(parent, bg=C["bg"])
        self.manager    = manager
        self.set_status = status_cb
        self._build()
        self._actualizar()   # estado inicial

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # — Sliders —
        card_o, card_i = make_card(self, "■ SIMULADOR DE PARÁMETROS")
        card_o.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        self.var_anio = tk.IntVar(value=2015)
        self.var_km   = tk.IntVar(value=60000)
        self.var_mes  = tk.IntVar(value=12)

        sliders = [
            ("📅  AÑO",             self.var_anio, 1990, 2024, 1),
            ("🛣️  KILOMETRAJE",     self.var_km,   0, 300000, 5000),
            ("🕐  MESES REVISIÓN",  self.var_mes,  0, 60, 1),
        ]

        self._val_labels = {}
        for label, var, lo, hi, res in sliders:
            f = tk.Frame(card_i, bg=C["card"])
            f.pack(fill="x", padx=18, pady=12)

            top = tk.Frame(f, bg=C["card"])
            top.pack(fill="x")
            tk.Label(top, text=label, font=FONT_LABEL,
                     bg=C["card"], fg=C["muted"]).pack(side="left")
            val_lbl = tk.Label(top, text=str(var.get()), font=FONT_HEADER,
                               bg=C["card"], fg=C["accent"])
            val_lbl.pack(side="right")
            self._val_labels[label] = val_lbl

            s = ttk.Scale(f, from_=lo, to=hi, variable=var,
                          orient="horizontal",
                          command=lambda v, vr=var, lbl=val_lbl, r=res:
                              self._on_slide(vr, lbl, r))
            s.pack(fill="x", pady=(4, 0))

        # Resultado en tiempo real
        card_o2, card_i2 = make_card(card_i, "■ PREDICCIÓN EN TIEMPO REAL")
        card_o2.pack(fill="x", padx=18, pady=(8, 18))
        self.result_sim = ResultWidget(card_i2)
        self.result_sim.pack(fill="both", expand=True)

        # — Gráfico dinámico —
        card_o3, card_i3 = make_card(self, "■ GRÁFICO DINÁMICO DE PROBABILIDAD")
        card_o3.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self._build_plot(card_i3)

    def _build_plot(self, parent):
        self.fig_sim = plt.Figure(figsize=(5.5, 5), facecolor=C["plot_bg"])
        self.fig_sim.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        self.ax_sim = self.fig_sim.add_subplot(111)
        self.ax_sim.set_facecolor(C["plot_bg"])
        for sp in self.ax_sim.spines.values():
            sp.set_edgecolor(C["border"])
        self.canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=parent)
        self.canvas_sim.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _on_slide(self, var, lbl, res):
        v = round(var.get() / res) * res
        var.set(v)
        fmt = f"{v:,.0f}" if res >= 1000 else str(v)
        lbl.config(text=fmt)
        self._actualizar()

    def _actualizar(self):
        anio = self.var_anio.get()
        km   = self.var_km.get()
        mes  = self.var_mes.get()
        try:
            clase, prob = self.manager.predecir(float(anio), float(km), float(mes))
            self.result_sim.update(clase, prob)
            self._plot_gauge(prob or 50.0, clase)
        except Exception:
            pass   # modelo no cargado aún, silencioso

    def _plot_gauge(self, prob: float, clase: int):
        ax = self.ax_sim
        ax.clear()
        ax.set_facecolor(C["plot_bg"])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Fondo barra
        ax.barh([0.5], [100], height=0.22, color=C["bg"], align="center",
                zorder=1)
        # Relleno
        color = C["danger"] if clase == 1 else C["success"]
        ax.barh([0.5], [prob], height=0.22, color=color, align="center",
                zorder=2)

        # Líneas de referencia
        for x, lbl in [(25, "25%"), (50, "50%"), (75, "75%")]:
            ax.axvline(x, color=C["grid"], linewidth=0.8, linestyle="--", zorder=3)
            ax.text(x, 0.3, lbl, ha="center", va="top",
                    color=C["muted"], fontsize=8, fontfamily="Courier New")

        # Valor principal
        ax.text(prob, 0.5, f" {prob:.1f}%",
                ha="left" if prob < 80 else "right",
                va="center", color="white",
                fontsize=14, fontweight="bold", fontfamily="Courier New", zorder=4)
        ax.set_title(f"Probabilidad de falla — {'ALTO RIESGO' if clase else 'BAJO RIESGO'}",
                     color=color, fontsize=9, pad=10, fontfamily="Courier New")

        # Curva de probabilidad vs kilometraje
        ax2 = self.fig_sim.add_axes([0.1, 0.08, 0.85, 0.38],
                                    facecolor=C["plot_bg"])
        km_range = np.linspace(0, 300000, 60)
        probs = []
        anio = float(self.var_anio.get())
        mes  = float(self.var_mes.get())
        for k in km_range:
            try:
                _, p = self.manager.predecir(anio, k, mes)
                probs.append(p if p is not None else 50)
            except Exception:
                probs.append(50)
        cmap_line = LinearSegmentedColormap.from_list("g2r",
                        [C["success"], C["warn"], C["danger"]])
        for i in range(len(km_range) - 1):
            c = cmap_line(probs[i] / 100)
            ax2.plot(km_range[i:i+2], probs[i:i+2], color=c, linewidth=2)
        ax2.axvline(self.var_km.get(), color=C["accent"], linewidth=1.5,
                    linestyle="--", label=f"Km actual: {self.var_km.get():,}")
        ax2.set_xlabel("Kilometraje", color=C["muted"], fontsize=7,
                       fontfamily="Courier New")
        ax2.set_ylabel("Prob. falla %", color=C["muted"], fontsize=7,
                       fontfamily="Courier New")
        ax2.tick_params(colors=C["muted"], labelsize=6)
        ax2.set_facecolor(C["plot_bg"])
        for sp in ax2.spines.values():
            sp.set_edgecolor(C["border"])
        ax2.legend(fontsize=7, facecolor=C["panel"],
                   labelcolor=C["text"], edgecolor=C["border"])
        ax2.set_ylim(0, 100)

        self.canvas_sim.draw()
        # Limpiar el eje extra para próxima actualización
        self.fig_sim.delaxes(ax2)


# ─────────────────────────────────────────────────────────────
# PESTAÑA 3: ANÁLISIS
# ─────────────────────────────────────────────────────────────

class TabAnalisis(tk.Frame):
    def __init__(self, parent, manager: ModelManager, status_cb):
        super().__init__(parent, bg=C["bg"])
        self.manager    = manager
        self.set_status = status_cb
        self._build()
        self.actualizar()

    def _build(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # — Gráfico de importancias —
        card_o, card_i = make_card(self, "■ IMPORTANCIA DE VARIABLES")
        card_o.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        self._build_plot_imp(card_i)

        # — Panel de análisis textual —
        card_o2, card_i2 = make_card(self, "■ ANÁLISIS Y RECOMENDACIONES")
        card_o2.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self._build_analysis(card_i2)

    def _build_plot_imp(self, parent):
        self.fig_imp = plt.Figure(figsize=(5.5, 4.8), facecolor=C["plot_bg"])
        self.fig_imp.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.12)
        self.ax_imp = self.fig_imp.add_subplot(111)
        self.ax_imp.set_facecolor(C["plot_bg"])
        for sp in self.ax_imp.spines.values():
            sp.set_edgecolor(C["border"])
        self.canvas_imp = FigureCanvasTkAgg(self.fig_imp, master=parent)
        self.canvas_imp.get_tk_widget().pack(fill="both", expand=True,
                                             padx=8, pady=8)

    def _build_analysis(self, parent):
        self.lbl_principal = tk.Label(
            parent, text="Cargando análisis…", font=FONT_HEADER,
            bg=C["card"], fg=C["accent"], wraplength=280, justify="left")
        self.lbl_principal.pack(anchor="w", padx=18, pady=(14, 8))

        tk.Frame(parent, height=1, bg=C["border"]).pack(fill="x", padx=18)

        self.lbl_detalle = tk.Label(
            parent, text="", font=FONT_BODY,
            bg=C["card"], fg=C["text"], wraplength=280, justify="left")
        self.lbl_detalle.pack(anchor="w", padx=18, pady=(10, 8))

        tk.Frame(parent, height=1, bg=C["border"]).pack(fill="x", padx=18)

        tk.Label(parent, text="■ RANKING DE FACTORES", font=FONT_LABEL,
                 bg=C["card"], fg=C["muted"]).pack(anchor="w", padx=18, pady=(12, 4))
        self.rank_frame = tk.Frame(parent, bg=C["card"])
        self.rank_frame.pack(fill="x", padx=18, pady=(0, 12))

        make_btn(parent, "↺ Recargar análisis", self.actualizar,
                 color=C["accent2"]).pack(padx=18, pady=(8, 18), fill="x")

    def actualizar(self):
        imps = self.manager.importancias()
        if not imps:
            self.lbl_principal.config(text="Modelo no cargado aún.")
            return

        # Ordenar
        sorted_imps = sorted(imps.items(), key=lambda x: x[1], reverse=True)
        top_feat, top_val = sorted_imps[0]

        # Texto de análisis
        consejos = {
            "kilometraje":           "El desgaste acumulado por distancia es\nel predictor más determinante.",
            "km":                    "El desgaste acumulado por distancia es\nel predictor más determinante.",
            "anio":                  "La antigüedad del vehículo influye\ndirectamente en su estado mecánico.",
            "año":                   "La antigüedad del vehículo influye\ndirectamente en su estado mecánico.",
            "tiempo_desde_revision": "El tiempo sin mantenimiento aumenta\nla probabilidad de falla.",
            "meses":                 "El tiempo sin mantenimiento aumenta\nla probabilidad de falla.",
        }
        consejo = consejos.get(top_feat.lower(),
                               f"'{top_feat}' es el factor más relevante\nen la predicción del modelo.")
        self.lbl_principal.config(
            text=f"Factor principal de riesgo:\n{top_feat.upper()}\n({top_val*100:.1f}% de peso)")
        self.lbl_detalle.config(text=consejo)

        # Ranking
        for w in self.rank_frame.winfo_children():
            w.destroy()
        for i, (feat, val) in enumerate(sorted_imps):
            row = tk.Frame(self.rank_frame, bg=C["card"])
            row.pack(fill="x", pady=3)
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f" {i+1}."
            tk.Label(row, text=f"{medal}  {feat}", font=FONT_BODY,
                     bg=C["card"], fg=C["text"]).pack(side="left")
            tk.Label(row, text=f"{val*100:.1f}%", font=FONT_HEADER,
                     bg=C["card"], fg=C["accent"]).pack(side="right")
            bar_bg = tk.Frame(self.rank_frame, bg=C["bg"], height=4)
            bar_bg.pack(fill="x", pady=(0, 2))
            pct = max(0.04, val)
            tk.Frame(bar_bg, bg=C["accent2"], height=4).place(
                relwidth=pct, relheight=1)

        # Gráfico
        self._plot_importancias(sorted_imps)

    def _plot_importancias(self, sorted_imps):
        ax = self.ax_imp
        ax.clear()
        ax.set_facecolor(C["plot_bg"])
        items_asc = list(reversed(sorted_imps))
        labels = [f[0] for f in items_asc]
        vals   = [f[1] for f in items_asc]
        max_val = max(vals) if vals else 1

        palette = [C["accent"] if v == max_val else C["accent2"] for v in vals]
        bars = ax.barh(labels, vals, color=palette, height=0.55)

        for bar, val in zip(bars, vals):
            ax.text(val + 0.004, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center",
                    color=C["text"], fontsize=8, fontfamily="Courier New")

        ax.set_title("Peso de cada variable en la predicción",
                     color=C["text"], fontsize=8, pad=8, fontfamily="Courier New")
        ax.tick_params(colors=C["muted"], labelsize=8)
        ax.xaxis.set_visible(False)
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
        ax.set_facecolor(C["plot_bg"])
        self.canvas_imp.draw()


# ─────────────────────────────────────────────────────────────
# VENTANA PRINCIPAL
# ─────────────────────────────────────────────────────────────

class PredictApp:
    def __init__(self, root: tk.Tk):
        self.root    = root
        self.manager = ModelManager()
        self._cargar_modelo_inicial()
        self._setup_window()
        self._build_ui()

    def _cargar_modelo_inicial(self):
        try:
            self.manager.cargar()
        except FileNotFoundError:
            pass   # Se mostrará error contextual al intentar predecir

    def _setup_window(self):
        self.root.title("CDA · Sistema Predictivo Avanzado")
        self.root.configure(bg=C["bg"])
        self.root.minsize(900, 620)
        w, h = 1050, 720
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        apply_style(self.root)

    def _build_ui(self):
        # Cabecera
        hdr = tk.Frame(self.root, bg=C["panel"], height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  CDA COLOMBIA",
                 font=("Courier New", 15, "bold"),
                 bg=C["panel"], fg=C["accent"]).place(x=24, rely=0.5, anchor="w")
        tk.Label(hdr, text="SISTEMA PREDICTIVO — REVISIÓN TÉCNICO-MECÁNICA",
                 font=FONT_BODY, bg=C["panel"], fg=C["muted"]).place(x=210, rely=0.5, anchor="w")

        model_info = f"Modelo: {MODEL_PATH}  ({'✓ cargado' if self.manager.model else '✗ no encontrado'})"
        tk.Label(hdr, text=model_info, font=FONT_LABEL,
                 bg=C["panel"],
                 fg=C["success"] if self.manager.model else C["danger"]).place(
            relx=1, x=-18, rely=0.5, anchor="e")

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.status_var = tk.StringVar(value="Listo.")

        tab1 = TabPrediccion(nb, self.manager, self._set_status)
        tab2 = TabSimulacion(nb, self.manager, self._set_status)
        tab3 = TabAnalisis(nb, self.manager, self._set_status)

        nb.add(tab1, text="  Predicción  ")
        nb.add(tab2, text="  Simulación  ")
        nb.add(tab3, text="  Análisis    ")

        # Barra de estado
        sb = tk.Frame(self.root, bg=C["panel"], height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self.status_var, font=FONT_LABEL,
                 bg=C["panel"], fg=C["muted"], anchor="w").pack(fill="x", padx=12)

    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")


# ─────────────────────────────────────────────────────────────
# ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    PredictApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()