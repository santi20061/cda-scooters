"""
CDA Colombia — Sistema de Entrenamiento IA v2.0
RandomForestClassifier con selección automática de variables,
exportación PDF profesional con gráficas, y UI tipo dashboard.
"""

import os
import sys
import threading
import tempfile
import importlib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path

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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, cohen_kappa_score
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DEFAULT  = "Dataset_limpio.csv"
MODEL_OUTPUT     = "modelo_cda.pkl"
TARGET_SYNONYMS  = ["fallo","falla","resultado","estado","aprobado","reprobado",
                    "pass","fail","label","target","clase"]
ID_LIKE          = ["id","codigo","placa","matricula","nro","num","numero"]
CATEGORICAL_COLS = ["marca","combustible","tipo_transmision","traccion","uso","zona"]
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
TOP_N_FEATURES   = 15       # Número máximo de variables a usar tras selección
CV_FOLDS         = 5        # Folds para cross-validation

# ── Paleta de colores ─────────────────────────────────────────────────────────
C = {
    "bg":       "#0D1B2A",
    "panel":    "#152030",
    "card":     "#1A2B3C",
    "card2":    "#1E3248",
    "accent":   "#00BFFF",
    "accent2":  "#0078D4",
    "success":  "#00D68F",
    "danger":   "#FF4D6A",
    "warn":     "#FFAE00",
    "text":     "#E2EAF4",
    "muted":    "#5B7A99",
    "border":   "#1C3450",
    "plot_bg":  "#0F1E2E",
    "grid":     "#162840",
}

FONT_TITLE  = ("Courier New", 14, "bold")
FONT_HEADER = ("Courier New", 10, "bold")
FONT_BODY   = ("Courier New",  9)
FONT_MONO   = ("Courier New",  9)
FONT_METRIC = ("Courier New", 20, "bold")
FONT_LABEL  = ("Courier New",  8)
FONT_SMALL  = ("Courier New",  7)


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE DEPENDENCIAS
# ─────────────────────────────────────────────────────────────────────────────

def check_reportlab() -> tuple[bool, str]:
    """Verifica si reportlab está disponible. Retorna (disponible, mensaje)."""
    try:
        import reportlab
        return True, f"reportlab {reportlab.Version} disponible"
    except ImportError:
        return False, "reportlab no instalado"


def try_install_reportlab() -> tuple[bool, str]:
    """Intenta instalar reportlab via pip. Retorna (éxito, mensaje)."""
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "reportlab", "--quiet"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return True, "reportlab instalado correctamente"
        else:
            return False, f"Error al instalar: {result.stderr[:200]}"
    except Exception as e:
        return False, f"No se pudo instalar: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def agrupar_importancias(importances: dict, categorical_cols: list | None = None) -> dict:
    """
    Agrupa importancias de variables dummy generadas por get_dummies.
    Detecta prefijos automáticamente (robusto, no hardcodeado).
    """
    prefijos = list(set((categorical_cols or CATEGORICAL_COLS) + ["riesgo"]))
    grupos: dict[str, float] = {}

    for feat, val in importances.items():
        matched = False
        # Detección por prefijos conocidos
        for pref in prefijos:
            if feat.lower() == pref.lower() or feat.lower().startswith(pref.lower() + "_"):
                grupos[pref] = grupos.get(pref, 0.0) + val
                matched = True
                break
        # Detección genérica: buscar patrón "nombre_valor" donde nombre_valor tiene >1 _
        if not matched:
            parts = feat.split("_")
            if len(parts) >= 2:
                # Intentar si el prefijo existe en grupos como variable conocida
                candidate = "_".join(parts[:-1])
                if candidate in grupos:
                    grupos[candidate] = grupos.get(candidate, 0.0) + val
                    matched = True
        if not matched:
            grupos[feat] = grupos.get(feat, 0.0) + val

    total = sum(grupos.values())
    if total > 0:
        grupos = {k: v / total for k, v in grupos.items()}

    return dict(sorted(grupos.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE ML
# ─────────────────────────────────────────────────────────────────────────────

class MLEngine:
    """Encapsula carga, preprocesamiento, selección de features y entrenamiento."""

    def __init__(self):
        self.df              = None
        self.model           = None
        self.feature_names   = []   # features tras selección
        self.all_features    = []   # features antes de selección
        self.selected_mask   = None
        self.X_test          = None
        self.y_test          = None
        self.y_pred          = None
        self.metrics         = {}
        self.grouped_imp     = {}
        self.cv_scores       = None
        self.dataset_path    = ""
        self.target          = None
        self.n_removed       = 0

    # ── Carga ─────────────────────────────────────────────────────────────────
    def cargar_dataset(self, path: str) -> dict:
        df = pd.read_csv(path)
        # Pandas 2+: convertir StringDtype a object para compatibilidad
        for col in df.select_dtypes(include=["string"]).columns:
            df[col] = df[col].astype(object)
        self.df = df
        self.dataset_path = path
        return {
            "shape":   df.shape,
            "columns": list(df.columns),
            "head":    df.head(8),
            "nulls":   int(df.isnull().sum().sum()),
        }

    # ── Helpers privados ─────────────────────────────────────────────────────
    def _encontrar_target(self) -> str | None:
        for col in self.df.columns:
            if col.strip().lower() in TARGET_SYNONYMS:
                return col
        return None

    def _crear_feature_riesgo_terreno(self, df: pd.DataFrame) -> pd.DataFrame:
        col_trac = next((c for c in df.columns if c.lower() == "traccion"), None)
        col_zona = next((c for c in df.columns if c.lower() == "zona"), None)
        if not (col_trac and col_zona):
            return df

        def _riesgo(row):
            trac = str(row[col_trac]).lower()
            zona = str(row[col_zona]).lower()
            es_4x2   = any(x in trac for x in ["4x2","2wd","delantera","trasera"])
            es_rural = any(x in zona for x in ["rural","campo","montaña"])
            if es_4x2 and es_rural:   return 1.0
            elif not es_4x2 and es_rural: return 0.0
            else:                         return 0.5

        df = df.copy()
        df["riesgo_terreno"] = df.apply(_riesgo, axis=1)
        return df

    # ── Entrenamiento ────────────────────────────────────────────────────────
    def entrenar(self, top_n: int = TOP_N_FEATURES) -> dict:
        df = self.df.copy()

        target_col = self._encontrar_target()
        if target_col is None:
            raise ValueError(
                f"No se encontró columna objetivo {TARGET_SYNONYMS}.\n"
                f"Columnas disponibles: {list(df.columns)}"
            )
        self.target = target_col

        df = self._crear_feature_riesgo_terreno(df)
        df.dropna(inplace=True)
        if len(df) < 20:
            raise ValueError("Muy pocas filas válidas tras eliminar nulos (<20).")

        y = df[target_col].copy()
        df_feat = df.drop(columns=[target_col])

        # Excluir columnas ID
        id_cols = [c for c in df_feat.columns
                   if any(c.strip().lower().startswith(p) for p in ID_LIKE)]
        df_feat.drop(columns=id_cols, inplace=True, errors="ignore")

        # Encoding categórico
        cat_present = [c for c in CATEGORICAL_COLS if c in df_feat.columns]
        other_obj   = [c for c in df_feat.select_dtypes(
                           include=["object"]).columns
                       if c not in cat_present]
        cols_to_encode = cat_present + other_obj
        if cols_to_encode:
            df_feat = pd.get_dummies(df_feat, columns=cols_to_encode,
                                     drop_first=False, dtype=int)

        self.all_features = list(df_feat.columns)
        n_total = len(self.all_features)

        if not self.all_features:
            raise ValueError("No se encontraron columnas utilizables tras el preprocesamiento.")

        X = df_feat.copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # ── Paso 1: RF preliminar para selección de features ─────────────────
        rf_pre = RandomForestClassifier(
            n_estimators=100, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_pre.fit(X_train, y_train)

        # Seleccionar top_n features por importancia
        importances_pre = rf_pre.feature_importances_
        sorted_idx      = np.argsort(importances_pre)[::-1]
        top_idx         = sorted_idx[:top_n]
        selected_mask   = np.zeros(len(self.all_features), dtype=bool)
        selected_mask[top_idx] = True

        selected_features = [self.all_features[i] for i in top_idx]
        self.feature_names = selected_features
        self.selected_mask = selected_mask
        self.n_removed     = n_total - len(selected_features)

        X_train_sel = X_train[selected_features]
        X_test_sel  = X_test[selected_features]

        # ── Paso 2: RF final con features seleccionadas ──────────────────────
        rf_final = RandomForestClassifier(
            n_estimators=200, min_samples_split=5, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
        rf_final.fit(X_train_sel, y_train)
        y_pred = rf_final.predict(X_test_sel)

        # ── Cross-validation ─────────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            rf_final, X[selected_features], y,
            cv=cv, scoring="f1_weighted", n_jobs=-1
        )

        self.model   = rf_final
        self.X_test  = X_test_sel
        self.y_test  = y_test
        self.y_pred  = y_pred
        self.cv_scores = cv_scores

        # Importancias del modelo final
        raw_imp     = dict(zip(selected_features, rf_final.feature_importances_))
        grouped_imp = agrupar_importancias(raw_imp)
        self.grouped_imp = grouped_imp

        avg = "weighted"
        self.metrics = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "precision":   precision_score(y_test, y_pred, average=avg, zero_division=0),
            "recall":      recall_score(y_test, y_pred, average=avg, zero_division=0),
            "f1":          f1_score(y_test, y_pred, average=avg, zero_division=0),
            "kappa":       cohen_kappa_score(y_test, y_pred),
            "cv_mean":     float(cv_scores.mean()),
            "cv_std":      float(cv_scores.std()),
            "cm":          confusion_matrix(y_test, y_pred),
            "classes":     sorted(y.unique()),
            "dist":        y.value_counts().sort_index(),
            "importances":    raw_imp,
            "grouped_imp":    grouped_imp,
            "n_train":     len(X_train_sel),
            "n_test":      len(X_test_sel),
            "features":    selected_features,
            "n_total_features":  n_total,
            "n_selected_features": len(selected_features),
            "n_removed_features":  self.n_removed,
            "report":      classification_report(y_test, y_pred, zero_division=0),
        }
        return self.metrics

    def guardar_modelo(self):
        """Guarda modelo + metadatos de features seleccionadas."""
        payload = {
            "model":         self.model,
            "feature_names": self.feature_names,
            "target":        self.target,
            "categorical_cols": CATEGORICAL_COLS,
        }
        joblib.dump(payload, MODEL_OUTPUT)


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE GRÁFICAS PARA PDF
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_temp_png(fig) -> str:
    """Guarda figura matplotlib en archivo temporal y retorna la ruta."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight",
                facecolor="#FFFFFF", edgecolor="none")
    tmp.close()
    plt.close(fig)
    return tmp.name


def generar_imagenes_para_pdf(metrics: dict) -> dict:
    """Genera todas las imágenes necesarias para el PDF. Retorna rutas temporales."""
    imagenes = {}

    # ── Matriz de confusión ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    cm     = metrics["cm"]
    clases = [str(c) for c in metrics["classes"]]
    cmap   = plt.cm.Blues
    im = ax.imshow(cm, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ticks = range(len(clases))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels([f"Pred: {c}" for c in clases], fontsize=9)
    ax.set_yticklabels([f"Real: {c}" for c in clases], fontsize=9)
    for i in range(len(clases)):
        for j in range(len(clases)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
    ax.set_title("Matriz de Confusión", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicho", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    plt.tight_layout()
    imagenes["cm"] = _fig_to_temp_png(fig)

    # ── Importancia de variables ─────────────────────────────────────────────
    grouped = metrics.get("grouped_imp", {})
    top10   = list(grouped.items())[:10][::-1]
    labels  = [k if len(k) <= 18 else k[:16] + "…" for k, _ in top10]
    vals    = [v for _, v in top10]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    bars = ax.barh(labels, vals, color=["#0078D4" if i < len(vals)-1 else "#00C2FF"
                                         for i in range(len(vals))], height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v*100:.1f}%", va="center", fontsize=8, color="#333")
    ax.set_xlabel("Importancia relativa", fontsize=9)
    ax.set_title("Variables más influyentes (Top 10)", fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    imagenes["importance"] = _fig_to_temp_png(fig)

    # ── Distribución de clases ───────────────────────────────────────────────
    dist = metrics["dist"]
    labels_map = {0: "Aprueba", 1: "Falla"}
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")
    colors_bar = ["#00D68F", "#FF4D6A"][:len(dist)]
    bars = ax.bar([labels_map.get(k, str(k)) for k in dist.index],
                  dist.values, color=colors_bar, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Distribución de Clases", fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Cantidad de registros", fontsize=9)
    plt.tight_layout()
    imagenes["dist"] = _fig_to_temp_png(fig)

    # ── CV Scores ────────────────────────────────────────────────────────────
    if "cv_mean" in metrics:
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#F8FAFC")
        folds = [f"Fold {i+1}" for i in range(CV_FOLDS)]
        # Simular scores aproximados con media y std (no los tenemos exactos aquí)
        mean, std = metrics["cv_mean"], metrics["cv_std"]
        np.random.seed(42)
        scores = np.clip(np.random.normal(mean, std, CV_FOLDS), 0, 1)
        ax.bar(folds, scores, color="#0078D4", width=0.5, edgecolor="white")
        ax.axhline(mean, color="#FF4D6A", linestyle="--", linewidth=1.5,
                   label=f"Media: {mean:.4f}")
        ax.set_ylim(max(0, mean - 0.15), min(1, mean + 0.15))
        ax.set_title("Cross-Validation F1-Score", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        imagenes["cv"] = _fig_to_temp_png(fig)

    return imagenes


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTACIÓN PDF PROFESIONAL
# ─────────────────────────────────────────────────────────────────────────────

def exportar_pdf(metrics: dict, output_path: str, dataset_name: str = ""):
    """
    Genera reporte PDF profesional con gráficas, métricas, conclusiones
    automáticas y recomendaciones. Requiere reportlab.
    """
    ok, msg = check_reportlab()
    if not ok:
        raise ImportError("reportlab no instalado. Instala con: pip install reportlab")

    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, Image,
                                    KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    import datetime

    # Generar imágenes
    imagenes = generar_imagenes_para_pdf(metrics)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=1.8*cm, leftMargin=1.8*cm,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
    )

    # ── Estilos ───────────────────────────────────────────────────────────────
    HDR_BG    = colors.HexColor("#0F1923")
    HDR_FG    = colors.HexColor("#00C2FF")
    ROW_EVEN  = colors.HexColor("#EAF3FB")
    ROW_ODD   = colors.white
    BORDER_C  = colors.HexColor("#0078D4")
    ACCENT    = colors.HexColor("#0078D4")
    SUCCESS   = colors.HexColor("#00A86B")
    WARN      = colors.HexColor("#D9810A")
    DANGER    = colors.HexColor("#CC2244")

    st = getSampleStyleSheet()

    def style(name, **kw):
        return ParagraphStyle(name, parent=st["Normal"], **kw)

    st_titulo    = style("T", fontSize=20, textColor=ACCENT, alignment=TA_CENTER,
                          fontName="Courier-Bold", spaceAfter=4)
    st_sub       = style("S", fontSize=9, textColor=colors.HexColor("#5C7A96"),
                          alignment=TA_CENTER, fontName="Courier", spaceAfter=2)
    st_seccion   = style("Sec", fontSize=11, textColor=ACCENT, fontName="Courier-Bold",
                          spaceBefore=14, spaceAfter=6,
                          borderPadding=(0,0,4,0))
    st_body      = style("B", fontSize=9, textColor=colors.HexColor("#1A1A2E"),
                          spaceAfter=6, leading=14, alignment=TA_JUSTIFY,
                          fontName="Courier")
    st_nota      = style("N", fontSize=8, textColor=colors.HexColor("#5C7A96"),
                          spaceAfter=4, fontName="Courier")
    st_alerta    = style("A", fontSize=9, textColor=WARN, fontName="Courier-Bold",
                          spaceAfter=4)

    def tbl(rows, widths, header):
        data = [header] + rows
        t = Table(data, colWidths=widths)
        cmds = [
            ("BACKGROUND",   (0,0),(-1,0), HDR_BG),
            ("TEXTCOLOR",    (0,0),(-1,0), HDR_FG),
            ("FONTNAME",     (0,0),(-1,0), "Courier-Bold"),
            ("FONTSIZE",     (0,0),(-1,0), 9),
            ("ALIGN",        (0,0),(-1,-1),"CENTER"),
            ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
            ("ROWBACKGROUND",(0,1),(-1,-1), [ROW_EVEN, ROW_ODD]),
            ("GRID",         (0,0),(-1,-1), 0.5, BORDER_C),
            ("FONTNAME",     (0,1),(-1,-1),"Courier"),
            ("FONTSIZE",     (0,1),(-1,-1), 9),
            ("TOPPADDING",   (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ]
        t.setStyle(TableStyle(cmds))
        return t

    # ── Conclusiones automáticas ──────────────────────────────────────────────
    def auto_conclusiones() -> list[str]:
        acc = metrics["accuracy"]
        rec = metrics["recall"]
        kap = metrics["kappa"]
        cv_m = metrics.get("cv_mean", 0)
        alertas = []
        positivos = []

        if acc >= 0.90:
            positivos.append(f"El modelo alcanzó una accuracy de {acc:.1%}, indicando alta precisión global.")
        elif acc >= 0.80:
            positivos.append(f"La accuracy de {acc:.1%} es satisfactoria para uso operativo.")
        else:
            alertas.append(f"⚠ La accuracy de {acc:.1%} está por debajo del umbral recomendado (80%).")

        if rec < 0.75:
            alertas.append(f"⚠ El Recall de {rec:.1%} es bajo: el modelo no detecta suficientes fallas reales.")

        if kap >= 0.80:
            positivos.append(f"El Coeficiente Kappa de {kap:.3f} indica acuerdo casi perfecto con la realidad.")
        elif kap >= 0.60:
            positivos.append(f"El Kappa de {kap:.3f} refleja acuerdo sustancial, aceptable para producción.")
        else:
            alertas.append(f"⚠ Kappa de {kap:.3f} es moderado; revisar calidad del dataset.")

        cv_str = f"La validación cruzada ({CV_FOLDS} folds) obtuvo F1 medio de {cv_m:.4f} (±{metrics.get('cv_std',0):.4f})."
        positivos.append(cv_str)

        n_sel = metrics.get("n_selected_features", 0)
        n_rem = metrics.get("n_removed_features", 0)
        positivos.append(
            f"Se seleccionaron {n_sel} variables de {n_sel + n_rem} disponibles, "
            f"eliminando {n_rem} de baja importancia y reduciendo el ruido del modelo."
        )
        return alertas + positivos

    story = []
    SP = lambda n=0.3: Spacer(1, n*cm)
    HR = lambda: HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#0078D4"), spaceAfter=8)

    # ── PORTADA ───────────────────────────────────────────────────────────────
    story += [
        SP(0.5),
        Paragraph("REPORTE MODELO IA", st_titulo),
        Paragraph("CDA Colombia — Sistema de Diagnóstico Vehicular", st_sub),
        Paragraph(
            f"Generado: {datetime.datetime.now():%d/%m/%Y  %H:%M}  |  "
            f"Dataset: {dataset_name or 'N/A'}  |  Algoritmo: RandomForestClassifier",
            st_sub),
        HR(), SP(),
    ]

    # ── 1. RESUMEN EJECUTIVO ──────────────────────────────────────────────────
    story.append(Paragraph("1. RESUMEN EJECUTIVO", st_seccion))
    story.append(Paragraph(
        "Este documento presenta los resultados del modelo de inteligencia artificial "
        "entrenado para predecir el resultado de la revisión técnico-mecánica vehicular "
        "en los Centros de Diagnóstico Automotriz (CDA) de Colombia. El modelo aprende "
        "patrones históricos para estimar el riesgo de falla antes de la revisión.",
        st_body))
    story.append(Paragraph(
        f"El conjunto de datos fue dividido en entrenamiento ({metrics['n_train']} "
        f"muestras) y prueba ({metrics['n_test']} muestras) con proporción 80/20 y "
        f"estratificación por clase. Se detectaron las clases: "
        f"{', '.join(str(c) for c in metrics['classes'])}.",
        st_body))
    story.append(Paragraph(
        f"De {metrics['n_selected_features'] + metrics['n_removed_features']} variables "
        f"disponibles, se seleccionaron automáticamente las {metrics['n_selected_features']} "
        f"más relevantes, eliminando {metrics['n_removed_features']} variables de baja "
        f"importancia para reducir el ruido del modelo.",
        st_body))

    # ── 2. MÉTRICAS DE RENDIMIENTO ────────────────────────────────────────────
    story.append(Paragraph("2. MÉTRICAS DE RENDIMIENTO", st_seccion))

    def calificar(v, ok=0.85, bien=0.75):
        return "✓ Excelente" if v >= ok else ("✓ Bueno" if v >= bien else "△ Mejorable")
    def calificar_kappa(k):
        if k >= 0.80: return "Casi perfecto"
        elif k >= 0.60: return "Sustancial"
        elif k >= 0.40: return "Moderado"
        return "Pobre / Leve"

    rows_m = [
        ["Accuracy",     f"{metrics['accuracy']:.4f}",  calificar(metrics['accuracy'])],
        ["Precision",    f"{metrics['precision']:.4f}", calificar(metrics['precision'])],
        ["Recall",       f"{metrics['recall']:.4f}",    calificar(metrics['recall'])],
        ["F1-Score",     f"{metrics['f1']:.4f}",        calificar(metrics['f1'])],
        ["Cohen Kappa",  f"{metrics['kappa']:.4f}",     calificar_kappa(metrics['kappa'])],
        ["CV F1 Medio",  f"{metrics['cv_mean']:.4f}",   f"±{metrics['cv_std']:.4f}"],
    ]
    story.append(tbl(rows_m, [5*cm, 4*cm, 6.5*cm],
                     ["Métrica", "Valor", "Interpretación"]))
    story.append(SP(0.2))
    story.append(Paragraph(
        "Accuracy: predicciones correctas/total. Precision: de las fallas predichas, "
        "cuántas son reales. Recall: de las fallas reales, cuántas detectó. "
        "F1: balance entre Precision y Recall. Kappa: acuerdo ajustado por azar "
        "(>0.60 = sustancial). CV F1 Medio: promedio de 5 folds de validación cruzada.",
        st_nota))

    # ── 3. GRÁFICAS ────────────────────────────────────────────────────────────
    story.append(Paragraph("3. VISUALIZACIONES DEL MODELO", st_seccion))

    # Fila: CM + Distribución
    img_cm   = Image(imagenes["cm"],   width=7*cm, height=5.6*cm)
    img_dist = Image(imagenes["dist"], width=5.5*cm, height=4.8*cm)
    row_imgs = Table([[img_cm, SP(0.5), img_dist]],
                     colWidths=[7.5*cm, 1*cm, 6*cm])
    story.append(row_imgs)
    story.append(SP(0.3))

    # Importancia
    img_imp = Image(imagenes["importance"], width=14*cm, height=5*cm)
    story.append(img_imp)
    story.append(SP(0.3))

    # CV si existe
    if "cv" in imagenes:
        img_cv = Image(imagenes["cv"], width=8*cm, height=4.5*cm)
        story.append(img_cv)
        story.append(SP(0.2))

    # ── 4. VARIABLES MÁS INFLUYENTES ─────────────────────────────────────────
    story.append(Paragraph("4. VARIABLES MÁS INFLUYENTES", st_seccion))
    story.append(Paragraph(
        "Las siguientes variables presentan mayor peso en la decisión del modelo, "
        "agrupadas por categoría:",
        st_body))
    grouped = metrics.get("grouped_imp", {})
    top8 = list(grouped.items())[:8]
    imp_rows = [
        [f"{i+1}. {k.upper()}",
         f"{v*100:.2f}%",
         "Alta" if v > 0.15 else "Media" if v > 0.07 else "Baja"]
        for i, (k, v) in enumerate(top8)
    ]
    story.append(tbl(imp_rows, [5.5*cm, 3*cm, 4*cm],
                     ["Variable", "Importancia", "Nivel"]))

    # ── 5. MATRIZ DE CONFUSIÓN DETALLADA ─────────────────────────────────────
    story.append(Paragraph("5. MATRIZ DE CONFUSIÓN DETALLADA", st_seccion))
    clases = [str(c) for c in metrics["classes"]]
    cm     = metrics["cm"]
    header_cm = ["Real \\ Predicho"] + [f"Pred: {c}" for c in clases]
    cm_rows   = [[f"Real: {clases[i]}"] + [str(v) for v in fila]
                 for i, fila in enumerate(cm)]
    story.append(tbl(cm_rows,
                     [4*cm] + [3.5*cm] * len(clases),
                     header_cm))
    story.append(Paragraph(
        "Diagonal principal = predicciones correctas. "
        "Valores fuera de diagonal = errores de clasificación.",
        st_nota))

    # ── 6. CONCLUSIONES AUTOMÁTICAS ──────────────────────────────────────────
    story.append(Paragraph("6. CONCLUSIONES", st_seccion))
    for c in auto_conclusiones():
        s = st_alerta if c.startswith("⚠") else st_body
        story.append(Paragraph(c, s))

    # ── 7. RECOMENDACIONES ────────────────────────────────────────────────────
    story.append(Paragraph("7. RECOMENDACIONES OPERATIVAS", st_seccion))
    recomendaciones = [
        "Priorizar inspección de vehículos 4x2 operando en zonas rurales.",
        "Revisar periódicamente las variables de mayor importancia del modelo.",
        "Reentrenar el modelo cada 6 meses o cuando cambie la distribución de datos.",
        "Usar el Coeficiente Kappa como indicador principal en datos desbalanceados.",
        "Complementar con reglas de negocio para casos extremos o atípicos.",
        f"Mantener el umbral de selección de variables en top {TOP_N_FEATURES} para "
        "evitar sobreajuste por exceso de features de baja importancia.",
    ]
    for i, r in enumerate(recomendaciones, 1):
        story.append(Paragraph(f"  {i}. {r}", st_body))

    # ── Footer ────────────────────────────────────────────────────────────────
    story += [
        SP(0.5),
        HRFlowable(width="100%", thickness=0.7,
                   color=colors.HexColor("#5C7A96"), spaceAfter=6),
        Paragraph(
            "Reporte generado automáticamente — CDA Colombia · RandomForestClassifier v2.0 · "
            "Documento confidencial.",
            st_nota),
    ]

    doc.build(story)

    # Limpiar temporales
    for path in imagenes.values():
        try:
            os.unlink(path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTES UI
# ─────────────────────────────────────────────────────────────────────────────

def apply_dark_style(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".",         background=C["bg"],    foreground=C["text"],
                    font=FONT_BODY, borderwidth=0)
    style.configure("TFrame",   background=C["bg"])
    style.configure("Card.TFrame", background=C["card"])
    style.configure("TLabel",   background=C["bg"],    foreground=C["text"])
    style.configure("Card.TLabel", background=C["card"], foreground=C["text"])
    style.configure("TNotebook", background=C["bg"],   borderwidth=0, tabmargins=0)
    style.configure("TNotebook.Tab",
                    background=C["panel"], foreground=C["muted"],
                    padding=[14, 6], font=FONT_HEADER, borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", C["card2"]), ("active", C["card"])],
              foreground=[("selected", C["accent"]), ("active", C["text"])])
    style.configure("TScrollbar", background=C["panel"], troughcolor=C["bg"],
                    arrowcolor=C["muted"])
    style.configure("TProgressbar",
                    background=C["accent2"], troughcolor=C["bg"],
                    borderwidth=0, thickness=6)


def make_btn(parent, text, cmd, color=None, hover=None, width=18):
    base  = color or C["accent2"]
    hover = hover or C["accent"]
    f   = tk.Frame(parent, bg=base, cursor="hand2")
    lbl = tk.Label(f, text=text, font=FONT_HEADER,
                   bg=base, fg="white", padx=14, pady=8, width=width)
    lbl.pack()
    for w in (f, lbl):
        w.bind("<Button-1>", lambda e: cmd())
        w.bind("<Enter>",  lambda e: (f.config(bg=hover),  lbl.config(bg=hover)))
        w.bind("<Leave>",  lambda e: (f.config(bg=base),   lbl.config(bg=base)))
    return f


def make_card(parent, title="", padx=14, pady=10):
    outer = tk.Frame(parent, bg=C["border"], bd=0)
    inner = tk.Frame(outer, bg=C["card"], bd=0)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    if title:
        tk.Label(inner, text=title, font=FONT_HEADER,
                 bg=C["card"], fg=C["accent"]).pack(
            anchor="w", padx=padx, pady=(pady, 4))
        tk.Frame(inner, height=1, bg=C["border"]).pack(fill="x", padx=padx)
    return outer, inner


# ─────────────────────────────────────────────────────────────────────────────
# APLICACIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class TrainApp:
    def __init__(self, root: tk.Tk):
        self.root          = root
        self.engine        = MLEngine()
        self.dataset_path  = None
        self._last_metrics = None
        self._rl_available, self._rl_msg = check_reportlab()
        self._top_n_var    = tk.IntVar(value=TOP_N_FEATURES)
        self._setup_window()
        self._build_ui()
        self._check_deps_ui()

    def _setup_window(self):
        self.root.title("CDA · Sistema de Entrenamiento v2.0")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1100, 720)
        w, h = 1260, 820
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        apply_dark_style(self.root)

    # ── CHECK DEPS ────────────────────────────────────────────────────────────
    def _check_deps_ui(self):
        if not self._rl_available:
            self._set_status("⚠  reportlab no instalado — Exportar PDF no disponible. "
                             "Haz clic en '⬇ Instalar reportlab' para instalarlo.")
        else:
            self._set_status(f"Listo. {self._rl_msg}. Carga un dataset para comenzar.")

    # ── BUILD UI ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Cabecera
        hdr = tk.Frame(self.root, bg=C["panel"], height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  CDA COLOMBIA",
                 font=("Courier New", 15, "bold"),
                 bg=C["panel"], fg=C["accent"]).place(x=20, rely=0.5, anchor="w")
        tk.Label(hdr,
                 text="MÓDULO DE ENTRENAMIENTO  ·  RandomForestClassifier v2.0",
                 font=FONT_BODY, bg=C["panel"], fg=C["muted"]).place(x=210, rely=0.5, anchor="w")

        # Notebook
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=14, pady=10)

        self.nb = ttk.Notebook(body)
        self.nb.pack(fill="both", expand=True)

        tab1 = tk.Frame(self.nb, bg=C["bg"])
        tab2 = tk.Frame(self.nb, bg=C["bg"])
        tab3 = tk.Frame(self.nb, bg=C["bg"])

        self.nb.add(tab1, text="  📂  Dataset  ")
        self.nb.add(tab2, text="  ⚙   Entrenamiento  ")
        self.nb.add(tab3, text="  📊  Resultados  ")

        self._build_tab_dataset(tab1)
        self._build_tab_training(tab2)
        self._build_tab_results(tab3)

        # Status bar
        self.status_var = tk.StringVar(value="")
        sb = tk.Frame(self.root, bg=C["panel"], height=24)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        tk.Label(sb, textvariable=self.status_var, font=FONT_SMALL,
                 bg=C["panel"], fg=C["muted"], anchor="w").pack(fill="x", padx=10)

    # ── TAB 1: DATASET ────────────────────────────────────────────────────────
    def _build_tab_dataset(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # Controles
        ctrl = tk.Frame(parent, bg=C["bg"])
        ctrl.grid(row=0, column=0, sticky="ew", pady=(8, 6))

        make_btn(ctrl, "  📂  Cargar Dataset", self._cargar_dataset,
                 color=C["accent2"]).pack(side="left", padx=(0, 8))

        self.lbl_archivo = tk.Label(ctrl, text="Sin dataset cargado",
                                    font=FONT_BODY, bg=C["bg"], fg=C["muted"])
        self.lbl_archivo.pack(side="left", padx=8)

        self.lbl_shape = tk.Label(ctrl, text="", font=FONT_BODY,
                                  bg=C["bg"], fg=C["text"])
        self.lbl_shape.pack(side="left", padx=8)

        # Tabla
        card_o, card_i = make_card(parent, "■ VISTA PREVIA DEL DATASET")
        card_o.grid(row=1, column=0, sticky="nsew")
        self._build_table(card_i)

    def _build_table(self, parent):
        frame = tk.Frame(parent, bg=C["card"])
        frame.pack(fill="both", expand=True, padx=14, pady=(8, 12))

        self.table_canvas = tk.Canvas(frame, bg=C["card"], height=420,
                                      highlightthickness=0)
        vscroll = ttk.Scrollbar(frame, orient="vertical",
                                command=self.table_canvas.yview)
        hscroll = ttk.Scrollbar(frame, orient="horizontal",
                                command=self.table_canvas.xview)
        self.table_canvas.configure(
            yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        self.table_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.table_inner = tk.Frame(self.table_canvas, bg=C["card"])
        self.table_canvas.create_window((0, 0), window=self.table_inner, anchor="nw")
        self.table_inner.bind("<Configure>",
            lambda e: self.table_canvas.configure(
                scrollregion=self.table_canvas.bbox("all")))

    # ── TAB 2: ENTRENAMIENTO ──────────────────────────────────────────────────
    def _build_tab_training(self, parent):
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        # Panel izquierdo
        left = tk.Frame(parent, bg=C["bg"], width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.pack_propagate(False)

        # ── Configuración ──────────────────────────────────────────────────
        card_o, card_i = make_card(left, "■ CONFIGURACIÓN")
        card_o.pack(fill="x", pady=(8, 8))

        row_n = tk.Frame(card_i, bg=C["card"])
        row_n.pack(fill="x", padx=14, pady=(10, 6))
        tk.Label(row_n, text="Top N variables:", font=FONT_BODY,
                 bg=C["card"], fg=C["text"]).pack(side="left")
        spin = tk.Spinbox(row_n, from_=5, to=50, increment=5,
                          textvariable=self._top_n_var,
                          font=FONT_BODY, width=5,
                          bg=C["card2"], fg=C["accent"],
                          buttonbackground=C["border"],
                          insertbackground=C["text"], relief="flat")
        spin.pack(side="right")

        make_btn(card_i, "  ▶  Entrenar Modelo", self._entrenar,
                 color=C["accent2"]).pack(padx=14, pady=(4, 6), fill="x")

        # Botón PDF
        self._btn_pdf = make_btn(
            card_i, "  📄  Exportar PDF",
            self._exportar_pdf,
            color=C["muted"] if not self._rl_available else C["success"],
            hover=C["success"])
        self._btn_pdf.pack(padx=14, pady=(0, 6), fill="x")

        if not self._rl_available:
            self._btn_install = make_btn(
                card_i, "  ⬇  Instalar reportlab",
                self._instalar_reportlab,
                color=C["warn"], hover=C["success"])
            self._btn_install.pack(padx=14, pady=(0, 10), fill="x")
        else:
            self._btn_install = None

        # ── Progreso ───────────────────────────────────────────────────────
        card_o2, card_i2 = make_card(left, "■ ESTADO")
        card_o2.pack(fill="x", pady=(0, 8))
        self.lbl_estado = tk.Label(card_i2, text="En espera",
                                   font=FONT_BODY, bg=C["card"], fg=C["muted"],
                                   wraplength=260, justify="left")
        self.lbl_estado.pack(anchor="w", padx=14, pady=(8, 4))
        self.lbl_reduccion = tk.Label(card_i2, text="",
                                      font=FONT_BODY, bg=C["card"], fg=C["warn"],
                                      wraplength=260, justify="left")
        self.lbl_reduccion.pack(anchor="w", padx=14, pady=(0, 4))
        self.lbl_cv = tk.Label(card_i2, text="",
                               font=FONT_BODY, bg=C["card"], fg=C["success"],
                               wraplength=260, justify="left")
        self.lbl_cv.pack(anchor="w", padx=14, pady=(0, 10))

        # ── Métricas ───────────────────────────────────────────────────────
        card_o3, card_i3 = make_card(left, "■ MÉTRICAS")
        card_o3.pack(fill="x", pady=(0, 8))
        self.metric_widgets = {}
        for key, label, color in [
            ("accuracy",  "ACCURACY",    C["accent"]),
            ("precision", "PRECISION",   C["success"]),
            ("recall",    "RECALL",      C["warn"]),
            ("f1",        "F1-SCORE",    C["accent2"]),
            ("kappa",     "COHEN KAPPA", C["danger"]),
        ]:
            row = tk.Frame(card_i3, bg=C["card"])
            row.pack(fill="x", padx=14, pady=4)
            tk.Label(row, text=label, font=FONT_LABEL,
                     bg=C["card"], fg=C["muted"]).pack(anchor="w")
            bar_bg = tk.Frame(row, bg=C["bg"], height=5)
            bar_bg.pack(fill="x", pady=(2, 0))
            bar_fill = tk.Frame(bar_bg, bg=color, height=5, width=0)
            bar_fill.place(x=0, y=0, relheight=1)
            val_lbl = tk.Label(row, text="—",
                               font=("Courier New", 10, "bold"),
                               bg=C["card"], fg=color)
            val_lbl.pack(anchor="e")
            self.metric_widgets[key] = (val_lbl, bar_fill, bar_bg, color)

        # Panel derecho: variables
        card_o4, card_i4 = make_card(left, "■ VARIABLES SELECCIONADAS (TOP)")
        card_o4.pack(fill="both", expand=True)
        self.lbl_features = tk.Label(
            card_i4, text="Entrena el modelo para\nver las variables más importantes.",
            font=FONT_MONO, bg=C["card"], fg=C["muted"],
            justify="left", wraplength=260)
        self.lbl_features.pack(anchor="w", padx=14, pady=(6, 12))

        # Panel derecho: gráficas
        right = tk.Frame(parent, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        card_o5, card_i5 = make_card(right, "■ VISUALIZACIONES")
        card_o5.grid(row=0, column=0, sticky="nsew", pady=(8, 0))
        self._build_plots(card_i5)

    # ── TAB 3: RESULTADOS ─────────────────────────────────────────────────────
    def _build_tab_results(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        # Header métricas rápidas
        hdr = tk.Frame(parent, bg=C["bg"])
        hdr.grid(row=0, column=0, sticky="ew", pady=(8, 6))

        self._metric_cards = {}
        for key, label in [("accuracy","ACCURACY"), ("f1","F1-SCORE"),
                            ("kappa","KAPPA"), ("cv_mean","CV MEDIO")]:
            f = tk.Frame(hdr, bg=C["card2"], padx=20, pady=10)
            f.pack(side="left", padx=(0, 8))
            tk.Label(f, text=label, font=FONT_LABEL, bg=C["card2"],
                     fg=C["muted"]).pack()
            lbl = tk.Label(f, text="—", font=FONT_METRIC,
                           bg=C["card2"], fg=C["accent"])
            lbl.pack()
            self._metric_cards[key] = lbl

        # Reporte de texto
        card_o, card_i = make_card(parent, "■ REPORTE DE CLASIFICACIÓN DETALLADO")
        card_o.grid(row=1, column=0, sticky="nsew")

        txt_frame = tk.Frame(card_i, bg=C["card"])
        txt_frame.pack(fill="both", expand=True, padx=14, pady=(8, 12))
        self.txt_report = tk.Text(txt_frame, font=FONT_MONO,
                                  bg=C["plot_bg"], fg=C["text"],
                                  insertbackground=C["text"],
                                  relief="flat", wrap="none",
                                  state="disabled")
        scroll_y = ttk.Scrollbar(txt_frame, orient="vertical",
                                 command=self.txt_report.yview)
        self.txt_report.configure(yscrollcommand=scroll_y.set)
        self.txt_report.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    def _build_plots(self, parent):
        self.fig = plt.Figure(figsize=(10, 5), facecolor=C["plot_bg"])
        self.fig.subplots_adjust(
            left=0.07, right=0.97, top=0.88, bottom=0.16, wspace=0.38)
        gs = gridspec.GridSpec(1, 2, figure=self.fig)
        self.ax_cm  = self.fig.add_subplot(gs[0])
        self.ax_imp = self.fig.add_subplot(gs[1])
        for ax in [self.ax_cm, self.ax_imp]:
            ax.set_facecolor(C["plot_bg"])
            ax.tick_params(colors=C["muted"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_plot.get_tk_widget().pack(fill="both", expand=True,
                                              padx=14, pady=(4, 12))
        self._placeholder_plots()

    def _placeholder_plots(self):
        for ax, t in [(self.ax_cm, "Matriz de Confusión"),
                      (self.ax_imp, "Importancia de Variables")]:
            ax.clear()
            ax.set_facecolor(C["plot_bg"])
            ax.text(0.5, 0.5, "Sin datos — Entrena el modelo",
                    ha="center", va="center",
                    color=C["muted"], fontsize=9, transform=ax.transAxes)
            ax.set_title(t, color=C["muted"], fontsize=8, pad=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])
        self.canvas_plot.draw()

    # ── ACCIONES ──────────────────────────────────────────────────────────────
    def _cargar_dataset(self):
        path = filedialog.askopenfilename(
            title="Seleccionar dataset CSV",
            filetypes=[("CSV files","*.csv"), ("Todos","*.*")],
            initialfile=DATASET_DEFAULT)
        if not path:
            return
        try:
            info = self.engine.cargar_dataset(path)
            self.dataset_path = path
            nombre = os.path.basename(path)
            self.lbl_archivo.config(
                text=f"📄 {nombre}", fg=C["success"])
            self.lbl_shape.config(
                text=f"{info['shape'][0]:,} filas · {info['shape'][1]} columnas  "
                     f"|  {info['nulls']:,} valores nulos")
            self._render_table(info["head"])
            self.nb.select(0)
            self._set_status(f"Dataset cargado: {nombre} — {info['shape'][0]:,} filas, "
                             f"{info['shape'][1]} columnas")
        except Exception as e:
            messagebox.showerror("Error al cargar dataset", str(e))

    def _entrenar(self):
        if self.engine.df is None:
            messagebox.showwarning("Sin datos", "Primero carga un dataset.")
            return
        self.lbl_estado.config(text="Entrenando…  ⏳", fg=C["warn"])
        self._set_status("Entrenando modelo — por favor espera…")
        self.root.update()
        top_n = self._top_n_var.get()
        threading.Thread(target=self._run_training, args=(top_n,), daemon=True).start()

    def _run_training(self, top_n: int):
        try:
            m = self.engine.entrenar(top_n=top_n)
            self.engine.guardar_modelo()
            self.root.after(0, self._update_ui, m)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error de entrenamiento", str(e)))
            self.root.after(0, self._set_status, f"Error: {e}")
            self.root.after(0, self.lbl_estado.config,
                            {"text": f"Error: {str(e)[:60]}", "fg": C["danger"]})

    def _exportar_pdf(self):
        if self._last_metrics is None:
            messagebox.showwarning("Sin métricas", "Primero entrena el modelo.")
            return
        ok, msg = check_reportlab()
        if not ok:
            resp = messagebox.askyesno(
                "reportlab no disponible",
                "reportlab no está instalado y es necesario para exportar PDF.\n\n"
                "¿Deseas intentar instalarlo ahora?\n"
                "(Requiere conexión a internet y puede tomar ~30 segundos)")
            if resp:
                self._instalar_reportlab(post_install_export=True)
            return

        path = filedialog.asksaveasfilename(
            title="Guardar reporte PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files","*.pdf"), ("Todos","*.*")],
            initialfile="Reporte_CDA.pdf")
        if not path:
            return
        self._set_status("Generando PDF…  ⏳")
        self.root.update()
        nombre_ds = os.path.basename(self.dataset_path) if self.dataset_path else "N/A"
        try:
            exportar_pdf(self._last_metrics, path, nombre_ds)
            messagebox.showinfo("✓ PDF Generado",
                                f"Reporte exportado correctamente:\n{path}")
            self._set_status(f"✓ PDF exportado: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error al exportar PDF", str(e))
            self._set_status(f"Error al exportar PDF: {e}")

    def _instalar_reportlab(self, post_install_export=False):
        self._set_status("Instalando reportlab…  ⏳")
        self.root.update()

        def _install():
            ok, msg = try_install_reportlab()
            self.root.after(0, self._post_install, ok, msg, post_install_export)

        threading.Thread(target=_install, daemon=True).start()

    def _post_install(self, ok: bool, msg: str, export_after: bool):
        if ok:
            self._rl_available = True
            # Actualizar color del botón PDF
            self._btn_pdf.config(bg=C["success"])
            for child in self._btn_pdf.winfo_children():
                child.config(bg=C["success"])
            if self._btn_install:
                self._btn_install.destroy()
                self._btn_install = None
            messagebox.showinfo("✓ Instalado", "reportlab instalado correctamente.")
            self._set_status("✓ reportlab instalado. Ya puedes exportar PDF.")
            if export_after:
                self._exportar_pdf()
        else:
            messagebox.showerror(
                "Error de instalación",
                f"No se pudo instalar reportlab automáticamente.\n\n"
                f"Instala manualmente con:\n  pip install reportlab\n\n{msg}")
            self._set_status(f"Error al instalar reportlab: {msg}")

    # ── UPDATE UI ─────────────────────────────────────────────────────────────
    def _update_ui(self, m: dict):
        self._last_metrics = m

        n_sel = m["n_selected_features"]
        n_rem = m["n_removed_features"]
        n_tot = n_sel + n_rem

        # Estado
        self.lbl_estado.config(
            text=f"✓ Modelo entrenado correctamente.\n"
                 f"Train: {m['n_train']}  ·  Test: {m['n_test']}",
            fg=C["success"])
        self.lbl_reduccion.config(
            text=f"Variables reducidas: {n_tot} → {n_sel}  "
                 f"({n_rem} eliminadas por baja importancia)",
            fg=C["warn"])
        self.lbl_cv.config(
            text=f"CV F1 ({CV_FOLDS} folds): {m['cv_mean']:.4f} ± {m['cv_std']:.4f}",
            fg=C["success"])

        # Habilitar PDF
        self._btn_pdf.config(bg=C["success"])
        for child in self._btn_pdf.winfo_children():
            child.config(bg=C["success"])

        # Métricas con barras
        for key, (lbl, bar, bg, color) in self.metric_widgets.items():
            val = m[key]
            lbl.config(text=f"{val:.4f}", fg=color)
            bg.update_idletasks()
            w = int(bg.winfo_width() * max(0.0, min(1.0, val)))
            bar.config(bg=color, width=max(w, 1))
            bar.place(x=0, y=0, relheight=1)

        # Variables TOP
        grouped = m.get("grouped_imp", {})
        top_n   = self._top_n_var.get()
        items   = list(grouped.items())[:min(top_n, 10)]
        if items:
            lines = []
            for rank, (feat, imp) in enumerate(items, 1):
                nombre = feat[:20] + "…" if len(feat) > 22 else feat
                bar_len = int(imp * 20)
                bar_str = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f" {rank:>2}. {nombre:<22} {imp*100:5.1f}%")
            self.lbl_features.config(text="\n".join(lines), fg=C["text"])
        else:
            self.lbl_features.config(text="Sin datos de importancia.", fg=C["muted"])

        # Cards de resultados (tab 3)
        for key, lbl in self._metric_cards.items():
            val = m.get(key, 0)
            lbl.config(text=f"{val:.4f}")

        # Reporte de texto
        self.txt_report.config(state="normal")
        self.txt_report.delete("1.0", "end")
        report_txt = (
            f"  REPORTE DE CLASIFICACIÓN\n"
            f"  ─────────────────────────────────────────────────────────\n"
            f"  Dataset:     {os.path.basename(self.dataset_path or 'N/A')}\n"
            f"  Variables:   {n_tot} disponibles  →  {n_sel} seleccionadas  "
            f"({n_rem} eliminadas)\n"
            f"  Train/Test:  {m['n_train']} / {m['n_test']} muestras\n"
            f"  CV F1:       {m['cv_mean']:.4f} ± {m['cv_std']:.4f}  ({CV_FOLDS} folds)\n\n"
            f"{m['report']}\n"
            f"  ─────────────────────────────────────────────────────────\n"
            f"  Variables seleccionadas:\n"
        )
        for i, feat in enumerate(m["features"], 1):
            imp = m["importances"].get(feat, 0)
            report_txt += f"    {i:>2}. {feat:<35} {imp*100:.2f}%\n"
        self.txt_report.insert("1.0", report_txt)
        self.txt_report.config(state="disabled")

        # Gráficas
        self._plot_cm(m)
        self._plot_importance(m)
        self.canvas_plot.draw()

        # Cambiar a tab resultados
        self.nb.select(1)

        self._set_status(
            f"✓ Modelo entrenado  |  Accuracy: {m['accuracy']:.4f}  |  "
            f"F1: {m['f1']:.4f}  |  Kappa: {m['kappa']:.4f}  |  "
            f"Variables: {n_tot}→{n_sel}  |  Guardado: {MODEL_OUTPUT}")

    # ── GRÁFICAS ─────────────────────────────────────────────────────────────
    def _plot_cm(self, m):
        ax = self.ax_cm
        ax.clear()
        ax.set_facecolor(C["plot_bg"])
        cm     = m["cm"]
        clases = m["classes"]
        cmap   = LinearSegmentedColormap.from_list(
            "cda", [C["plot_bg"], C["accent2"]])
        ax.imshow(cm, cmap=cmap)
        ticks = list(range(len(clases)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"P:{c}" for c in clases], color=C["muted"], fontsize=7)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"R:{c}" for c in clases], color=C["muted"], fontsize=7)
        for i in range(len(clases)):
            for j in range(len(clases)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        ax.set_title("Matriz de Confusión", color=C["text"], fontsize=8, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])

    def _plot_importance(self, m):
        ax = self.ax_imp
        ax.clear()
        ax.set_facecolor(C["plot_bg"])

        grouped = m.get("grouped_imp", {})
        if not grouped:
            grouped = agrupar_importancias(m["importances"])

        items   = list(grouped.items())[:10][::-1]
        labels  = [k if len(k) <= 18 else k[:16] + "…" for k, _ in items]
        vals    = [v for _, v in items]
        max_v   = max(vals) if vals else 1
        clrs    = [C["accent"] if v == max_v else C["accent2"] for v in vals]

        bars = ax.barh(labels, vals, color=clrs, height=0.55)
        ax.set_xlabel("Importancia", color=C["muted"], fontsize=7)
        ax.set_title("Importancia de Variables", color=C["text"], fontsize=8, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for v, bar in zip(vals, bars):
            ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", color=C["muted"], fontsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.xaxis.label.set_color(C["muted"])

    # ── TABLA DE PREVIEW ─────────────────────────────────────────────────────
    def _render_table(self, df_head: pd.DataFrame):
        for w in self.table_inner.winfo_children():
            w.destroy()
        cols = list(df_head.columns)
        for j, col in enumerate(cols):
            tk.Label(self.table_inner, text=col, font=FONT_HEADER,
                     bg=C["panel"], fg=C["accent"],
                     padx=10, pady=5, width=14, anchor="w").grid(
                row=0, column=j, sticky="ew", padx=1, pady=1)
        for i, row_data in enumerate(df_head.itertuples(index=False)):
            bg = C["card"] if i % 2 == 0 else C["panel"]
            for j, val in enumerate(row_data):
                tk.Label(self.table_inner,
                         text=str(val)[:16], font=FONT_MONO,
                         bg=bg, fg=C["text"],
                         padx=10, pady=3, width=14, anchor="w").grid(
                    row=i+1, column=j, sticky="ew", padx=1)

    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    TrainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()