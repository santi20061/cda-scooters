"""
=============================================================
  Sistema de Entrenamiento — CDA Colombia
  Interfaz visual para entrenar modelo predictivo
=============================================================
Archivo  : train.py
Dataset  : dataset_cda_colombia.csv  (misma carpeta)
Salida   : modelo_cda.pkl
=============================================================
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, precision_score,
                              recall_score, f1_score)

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
DATASET_DEFAULT = "dataset_cda_colombia.csv"
MODEL_OUTPUT    = "modelo_cda.pkl"
TARGET_SYNONYMS = ["fallo","falla","resultado","estado","aprobado",
                   "reprobado","pass","fail","label","target","clase"]
ID_LIKE         = ["id","codigo","placa","matricula","nro","num","numero"]
TEST_SIZE       = 0.20
RANDOM_STATE    = 42

# Paleta unificada con predict_ui
C = {
    "bg":        "#0F1923",   # fondo oscuro profundo
    "panel":     "#162030",   # panel secundario
    "card":      "#1C2B3A",   # tarjeta
    "accent":    "#00C2FF",   # cian eléctrico
    "accent2":   "#0077FF",   # azul medio
    "success":   "#00E5A0",   # verde menta
    "danger":    "#FF4D6A",   # rojo coral
    "warn":      "#FFB020",   # ámbar
    "text":      "#E8F0F8",   # texto principal
    "muted":     "#5C7A96",   # texto secundario
    "border":    "#1E3448",   # borde sutil
    "plot_bg":   "#111C27",   # fondo gráficas
    "grid":      "#1A2E40",   # líneas de cuadrícula
}

FONT_TITLE  = ("Courier New", 13, "bold")
FONT_HEADER = ("Courier New", 10, "bold")
FONT_BODY   = ("Courier New",  9)
FONT_MONO   = ("Courier New",  9)
FONT_METRIC = ("Courier New", 22, "bold")
FONT_LABEL  = ("Courier New",  8)


# ─────────────────────────────────────────────────────────────
# LÓGICA DE ML
# ─────────────────────────────────────────────────────────────

class MLEngine:
    """Encapsula toda la lógica de carga, limpieza y entrenamiento."""

    def __init__(self):
        self.df       = None
        self.features = []
        self.target   = None
        self.model    = None
        self.X_test   = None
        self.y_test   = None
        self.y_pred   = None
        self.metrics  = {}

    def cargar_dataset(self, path: str) -> dict:
        df = pd.read_csv(path)
        self.df = df
        return {"shape": df.shape, "columns": list(df.columns),
                "head": df.head(8), "nulls": df.isnull().sum().sum()}

    def _encontrar_target(self):
        for col in self.df.columns:
            if col.strip().lower() in TARGET_SYNONYMS:
                return col
        return None

    def entrenar(self) -> dict:
        df = self.df.copy()

        # Target
        target_col = self._encontrar_target()
        if target_col is None:
            raise ValueError("No se encontró columna objetivo (fallo/falla/resultado…).\n"
                             f"Columnas disponibles: {list(df.columns)}")
        self.target = target_col

        # Limpiar nulos
        df.dropna(inplace=True)
        if df.shape[0] < 20:
            raise ValueError("Muy pocas filas válidas tras eliminar nulos (<20).")

        # Features numéricas
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if target_col in numeric:
            numeric.remove(target_col)
        features = [c for c in numeric
                    if not any(c.strip().lower().startswith(p) for p in ID_LIKE)]
        if not features:
            raise ValueError("No se encontraron columnas numéricas utilizables.")
        self.features = features

        X = df[features]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

        model = RandomForestClassifier(
            n_estimators=200, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        self.model  = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

        avg = "weighted"
        self.metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
            "recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
            "f1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
            "cm":        confusion_matrix(y_test, y_pred),
            "classes":   sorted(y.unique()),
            "dist":      y.value_counts().sort_index(),
            "importances": dict(zip(features, model.feature_importances_)),
            "n_train":   len(X_train),
            "n_test":    len(X_test),
            "features":  features,
        }
        return self.metrics

    def guardar_modelo(self):
        joblib.dump(self.model, MODEL_OUTPUT)


# ─────────────────────────────────────────────────────────────
# COMPONENTES UI
# ─────────────────────────────────────────────────────────────

def apply_dark_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=C["bg"], foreground=C["text"],
                    font=FONT_BODY, borderwidth=0)
    style.configure("TFrame", background=C["bg"])
    style.configure("Card.TFrame", background=C["card"])
    style.configure("TLabel", background=C["bg"], foreground=C["text"])
    style.configure("Card.TLabel", background=C["card"], foreground=C["text"])
    style.configure("Accent.TLabel", background=C["card"], foreground=C["accent"])
    style.configure("Muted.TLabel", background=C["card"], foreground=C["muted"])
    style.configure("TScrollbar", background=C["panel"], troughcolor=C["bg"],
                    arrowcolor=C["muted"])


def make_btn(parent, text, cmd, color=None, width=18):
    color = color or C["accent2"]
    f = tk.Frame(parent, bg=color, cursor="hand2")
    lbl = tk.Label(f, text=text, font=FONT_HEADER,
                   bg=color, fg="white", padx=14, pady=8, width=width)
    lbl.pack()
    lbl.bind("<Button-1>", lambda e: cmd())
    f.bind("<Button-1>", lambda e: cmd())

    def on_enter(e):  f.config(bg=C["accent"]);  lbl.config(bg=C["accent"])
    def on_leave(e):  f.config(bg=color);         lbl.config(bg=color)
    f.bind("<Enter>", on_enter);  lbl.bind("<Enter>", on_enter)
    f.bind("<Leave>", on_leave);  lbl.bind("<Leave>", on_leave)
    return f


def make_card(parent, title="", padx=16, pady=12):
    outer = tk.Frame(parent, bg=C["border"], bd=0)
    inner = tk.Frame(outer, bg=C["card"], bd=0)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    if title:
        tk.Label(inner, text=title, font=FONT_HEADER,
                 bg=C["card"], fg=C["accent"]).pack(anchor="w", padx=padx, pady=(pady, 4))
        tk.Frame(inner, height=1, bg=C["border"]).pack(fill="x", padx=padx)
    return outer, inner


# ─────────────────────────────────────────────────────────────
# VENTANA PRINCIPAL
# ─────────────────────────────────────────────────────────────

class TrainApp:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.engine = MLEngine()
        self.dataset_path = None
        self._setup_window()
        self._build_ui()

    def _setup_window(self):
        self.root.title("CDA · Sistema de Entrenamiento")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1100, 720)
        w, h = 1200, 800
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        apply_dark_style(self.root)

    # ── Layout principal ───────────────────────────────────────
    def _build_ui(self):
        # Cabecera
        hdr = tk.Frame(self.root, bg=C["panel"], height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  CDA COLOMBIA",
                 font=("Courier New", 15, "bold"),
                 bg=C["panel"], fg=C["accent"]).place(x=24, rely=0.5, anchor="w")
        tk.Label(hdr, text="MÓDULO DE ENTRENAMIENTO — RandomForestClassifier",
                 font=FONT_BODY, bg=C["panel"], fg=C["muted"]).place(x=210, rely=0.5, anchor="w")

        # Cuerpo: izquierda (controles) + derecha (gráficas)
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=12)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

        # Barra de estado
        self.status_var = tk.StringVar(value="Listo. Carga un dataset para comenzar.")
        sb = tk.Frame(self.root, bg=C["panel"], height=26)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self.status_var, font=FONT_LABEL,
                 bg=C["panel"], fg=C["muted"], anchor="w").pack(fill="x", padx=12)

    # ── Panel izquierdo (controles + métricas) ─────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C["bg"], width=290)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.pack_propagate(False)

        # — Botones de control —
        card_o, card_i = make_card(left, "■ CONTROL")
        card_o.pack(fill="x", pady=(0, 10))
        make_btn(card_i, "  Cargar Dataset", self._cargar_dataset).pack(padx=16, pady=(10, 6), fill="x")
        make_btn(card_i, "  Entrenar Modelo", self._entrenar, color=C["accent2"]).pack(padx=16, pady=(0, 10), fill="x")

        # — Info dataset —
        card_o2, card_i2 = make_card(left, "■ DATASET")
        card_o2.pack(fill="x", pady=(0, 10))
        self.lbl_archivo = tk.Label(card_i2, text="Sin dataset cargado",
                                    font=FONT_BODY, bg=C["card"], fg=C["muted"],
                                    wraplength=240, justify="left")
        self.lbl_archivo.pack(anchor="w", padx=16, pady=(8, 4))
        self.lbl_shape   = tk.Label(card_i2, text="", font=FONT_BODY,
                                    bg=C["card"], fg=C["text"])
        self.lbl_shape.pack(anchor="w", padx=16, pady=(0, 10))

        # — Métricas —
        card_o3, card_i3 = make_card(left, "■ MÉTRICAS")
        card_o3.pack(fill="x", pady=(0, 10))
        self.metric_widgets = {}
        for key, label in [("accuracy","ACCURACY"),("precision","PRECISION"),
                            ("recall","RECALL"),("f1","F1-SCORE")]:
            row = tk.Frame(card_i3, bg=C["card"])
            row.pack(fill="x", padx=14, pady=4)
            tk.Label(row, text=label, font=FONT_LABEL,
                     bg=C["card"], fg=C["muted"]).pack(anchor="w")
            bar_bg = tk.Frame(row, bg=C["bg"], height=6)
            bar_bg.pack(fill="x", pady=(2, 0))
            bar_fill = tk.Frame(bar_bg, bg=C["accent"], height=6, width=0)
            bar_fill.place(x=0, y=0, relheight=1)
            val_lbl = tk.Label(row, text="—", font=("Courier New", 10, "bold"),
                               bg=C["card"], fg=C["accent"])
            val_lbl.pack(anchor="e")
            self.metric_widgets[key] = (val_lbl, bar_fill, bar_bg)

        # — Features usadas —
        card_o4, card_i4 = make_card(left, "■ VARIABLES")
        card_o4.pack(fill="x", pady=(0, 0))
        self.lbl_features = tk.Label(card_i4, text="—", font=FONT_MONO,
                                     bg=C["card"], fg=C["text"],
                                     justify="left", wraplength=240)
        self.lbl_features.pack(anchor="w", padx=14, pady=(6, 12))

    # ── Panel derecho (vista previa + gráficas) ────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Vista previa del dataset (tabla)
        card_o, card_i = make_card(right, "■ VISTA PREVIA DEL DATASET")
        card_o.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._build_table(card_i)

        # Gráficas
        card_o2, card_i2 = make_card(right, "■ VISUALIZACIONES")
        card_o2.grid(row=1, column=0, sticky="nsew")
        self._build_plots(card_i2)

    def _build_table(self, parent):
        frame = tk.Frame(parent, bg=C["card"])
        frame.pack(fill="x", padx=14, pady=(8, 12))

        # Canvas scrollable horizontal
        self.table_canvas = tk.Canvas(frame, bg=C["card"], height=120,
                                      highlightthickness=0)
        hscroll = ttk.Scrollbar(frame, orient="horizontal",
                                command=self.table_canvas.xview)
        self.table_canvas.configure(xscrollcommand=hscroll.set)
        self.table_canvas.pack(fill="x")
        hscroll.pack(fill="x")
        self.table_inner = tk.Frame(self.table_canvas, bg=C["card"])
        self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")
        self.table_inner.bind("<Configure>",
            lambda e: self.table_canvas.configure(
                scrollregion=self.table_canvas.bbox("all")))

    def _build_plots(self, parent):
        self.fig = plt.Figure(figsize=(10, 4.2), facecolor=C["plot_bg"])
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.88,
                                  bottom=0.14, wspace=0.35)
        gs = gridspec.GridSpec(1, 3, figure=self.fig)
        self.ax_cm   = self.fig.add_subplot(gs[0])
        self.ax_imp  = self.fig.add_subplot(gs[1])
        self.ax_dist = self.fig.add_subplot(gs[2])
        for ax in [self.ax_cm, self.ax_imp, self.ax_dist]:
            ax.set_facecolor(C["plot_bg"])
            ax.tick_params(colors=C["muted"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True,
                                              padx=14, pady=(4, 12))
        self._placeholder_plots()

    def _placeholder_plots(self):
        for ax, title in [(self.ax_cm,   "Matriz de Confusión"),
                          (self.ax_imp,  "Importancia de Variables"),
                          (self.ax_dist, "Distribución de Clases")]:
            ax.clear()
            ax.set_facecolor(C["plot_bg"])
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    color=C["muted"], fontsize=9, transform=ax.transAxes)
            ax.set_title(title, color=C["muted"], fontsize=8, pad=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])
        self.canvas_plot.draw()

    # ── Acciones ───────────────────────────────────────────────
    def _cargar_dataset(self):
        path = filedialog.askopenfilename(
            title="Seleccionar dataset",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")],
            initialfile=DATASET_DEFAULT)
        if not path:
            return
        try:
            info = self.engine.cargar_dataset(path)
            self.dataset_path = path
            nombre = os.path.basename(path)
            self.lbl_archivo.config(text=f"📄 {nombre}", fg=C["success"])
            self.lbl_shape.config(
                text=f"{info['shape'][0]:,} filas · {info['shape'][1]} columnas  |  "
                     f"{info['nulls']:,} nulos")
            self._render_table(info["head"])
            self._set_status(f"Dataset cargado: {nombre}  ({info['shape'][0]:,} filas)")
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))

    def _entrenar(self):
        if self.engine.df is None:
            messagebox.showwarning("Sin datos", "Primero carga un dataset.")
            return
        self._set_status("Entrenando modelo…  ⏳")
        self.root.update()
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            m = self.engine.entrenar()
            self.engine.guardar_modelo()
            self.root.after(0, self._update_ui, m)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error de entrenamiento", str(e)))
            self.root.after(0, self._set_status, "Error durante el entrenamiento.")

    def _update_ui(self, m: dict):
        # Métricas con barras animadas
        colors = {"accuracy": C["accent"], "precision": C["success"],
                  "recall": C["warn"], "f1": C["accent2"]}
        for key, (lbl, bar, bg) in self.metric_widgets.items():
            val = m[key]
            lbl.config(text=f"{val:.4f}", fg=colors[key])
            bg.update_idletasks()
            w = int(bg.winfo_width() * val)
            bar.config(bg=colors[key], width=max(w, 1))
            bar.place(x=0, y=0, relheight=1)

        # Features
        self.lbl_features.config(
            text="\n".join(f"  · {f}" for f in m["features"]))

        # Gráficas
        self._plot_cm(m)
        self._plot_importance(m)
        self._plot_dist(m)
        self.canvas_plot.draw()

        self._set_status(
            f"✓ Modelo entrenado  |  Accuracy: {m['accuracy']:.4f}  |  "
            f"Train: {m['n_train']} muestras  |  Test: {m['n_test']} muestras  |  "
            f"Guardado: {MODEL_OUTPUT}")

    # ── Gráficas ───────────────────────────────────────────────
    def _plot_cm(self, m):
        ax = self.ax_cm
        ax.clear()
        ax.set_facecolor(C["plot_bg"])
        cm = m["cm"]
        cmap = LinearSegmentedColormap.from_list("cda", [C["plot_bg"], C["accent2"]])
        im = ax.imshow(cm, cmap=cmap)
        classes = m["classes"]
        ticks = range(len(classes))
        ax.set_xticks(ticks); ax.set_xticklabels(
            [f"Pred {c}" for c in classes], color=C["muted"], fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels(
            [f"Real {c}" for c in classes], color=C["muted"], fontsize=7)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        ax.set_title("Matriz de Confusión", color=C["text"], fontsize=8, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])

    def _plot_importance(self, m):
        ax = self.ax_imp
        ax.clear()
        ax.set_facecolor(C["plot_bg"])
        items = sorted(m["importances"].items(), key=lambda x: x[1])
        labels, vals = zip(*items)
        colors_bar = [C["accent"] if v == max(vals) else C["accent2"] for v in vals]
        bars = ax.barh(labels, vals, color=colors_bar, height=0.55)
        ax.set_xlabel("Importancia", color=C["muted"], fontsize=7)
        ax.set_title("Importancia de Variables", color=C["text"], fontsize=8, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for v, bar in zip(vals, bars):
            ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", color=C["muted"], fontsize=7)
        ax.set_facecolor(C["plot_bg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.xaxis.label.set_color(C["muted"])

    def _plot_dist(self, m):
        ax = self.ax_dist
        ax.clear()
        ax.set_facecolor(C["plot_bg"])
        dist = m["dist"]
        bar_colors = [C["success"], C["danger"]][:len(dist)]
        bars = ax.bar([str(k) for k in dist.index], dist.values,
                      color=bar_colors, width=0.5)
        labels_map = {0: "Aprueba", 1: "Falla"}
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels([labels_map.get(k, str(k)) for k in dist.index],
                           color=C["muted"], fontsize=8)
        ax.set_title("Distribución de Clases", color=C["text"], fontsize=8, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for bar, val in zip(bars, dist.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:,}", ha="center", color=C["muted"], fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])

    # ── Tabla de vista previa ──────────────────────────────────
    def _render_table(self, df_head: pd.DataFrame):
        for w in self.table_inner.winfo_children():
            w.destroy()
        cols = list(df_head.columns)
        # Headers
        for j, col in enumerate(cols):
            tk.Label(self.table_inner, text=col, font=FONT_HEADER,
                     bg=C["panel"], fg=C["accent"], padx=10, pady=4,
                     relief="flat", width=12, anchor="w").grid(
                row=0, column=j, sticky="ew", padx=1, pady=1)
        # Filas
        for i, row_data in enumerate(df_head.itertuples(index=False)):
            bg = C["card"] if i % 2 == 0 else C["panel"]
            for j, val in enumerate(row_data):
                tk.Label(self.table_inner, text=str(val)[:14], font=FONT_MONO,
                         bg=bg, fg=C["text"], padx=10, pady=3,
                         width=12, anchor="w").grid(
                    row=i+1, column=j, sticky="ew", padx=1, pady=0)

    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")


# ─────────────────────────────────────────────────────────────
# ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    TrainApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()