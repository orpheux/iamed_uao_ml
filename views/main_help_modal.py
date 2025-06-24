"""
main_help_modal.py
Modal de ayuda principal de la aplicación IAMED.
Proporciona información general del sistema y descripción de cada módulo.
"""

import customtkinter as ctk
from utils.constants import BUTTON_BG_COLOR, BUTTON_HOVER_COLOR


def help_main(master):
    """
    Modal de ayuda principal de la aplicación IAMED.
    Explica el funcionamiento general y describe cada módulo disponible.
    """
    # Crear ventana modal
    modal = ctk.CTkToplevel(master)
    modal.title("🏥 Ayuda - IAMED Sistema de Homologación")
    modal.geometry("700x600")
    modal.resizable(False, False)
    modal.transient(master)
    modal.grab_set()

    # Centrar modal
    modal.update_idletasks()
    x = (modal.winfo_screenwidth() // 2) - (700 // 2)
    y = (modal.winfo_screenheight() // 2) - (600 // 2)
    modal.geometry(f"700x600+{x}+{y}")

    # Frame principal con scroll
    scroll_frame = ctk.CTkScrollableFrame(modal, width=660, height=520)
    scroll_frame.pack(pady=20, padx=20, fill="both", expand=True)

    # Título principal
    titulo = ctk.CTkLabel(
        scroll_frame,
        text="🏥 IAMED - SISTEMA DE HOMOLOGACIÓN",
        font=("Arial", 20, "bold"),
        text_color=BUTTON_BG_COLOR
    )
    titulo.pack(pady=(10, 20))

    # Descripción general
    seccion_general = ctk.CTkFrame(
        scroll_frame, fg_color="#e6f3ff", corner_radius=10)
    seccion_general.pack(fill="x", pady=(0, 20), padx=10)

    ctk.CTkLabel(
        seccion_general,
        text="🎯 ¿QUÉ ES IAMED?",
        font=("Arial", 16, "bold"),
        text_color="#1e40af"
    ).pack(pady=(15, 10))

    texto_general = """IAMED es un sistema inteligente para la homologación de medicamentos que utiliza 
inteligencia artificial para encontrar medicamentos alternativos basándose en:

• Principio activo y composición
• Clasificación ATC (Anatomical Therapeutic Chemical)
• Vía de administración y forma farmacéutica
• Puntuación de similitud calculada por machine learning

El sistema procesa bases de datos de medicamentos y encuentra homólogos 
tanto individuales como de forma masiva para facilitar la gestión farmacéutica."""

    ctk.CTkLabel(
        seccion_general,
        text=texto_general,
        font=("Arial", 11),
        text_color="#1e40af",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Módulo 1: Cargar Archivos
    seccion_cargar = ctk.CTkFrame(
        scroll_frame, fg_color="#f0fff4", corner_radius=10)
    seccion_cargar.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_cargar,
        text="📂 MÓDULO: CARGAR ARCHIVOS",
        font=("Arial", 14, "bold"),
        text_color="#065f46"
    ).pack(pady=(15, 10))

    texto_cargar = """FUNCIÓN: Preparación de datos para el sistema

• Carga 4 archivos Excel de medicamentos:
  - Medicamentos vencidos
  - Medicamentos vigentes
  - Medicamentos en renovación
  - Medicamentos otros

• Procesa y valida automáticamente los datos
• Entrena el modelo de inteligencia artificial
• Genera análisis exploratorio (EDA) en PDF

Este módulo es PREREQUISITO para usar los demás módulos del sistema."""

    ctk.CTkLabel(
        seccion_cargar,
        text=texto_cargar,
        font=("Arial", 11),
        text_color="#065f46",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Módulo 2: Buscar Homólogo
    seccion_buscar = ctk.CTkFrame(
        scroll_frame, fg_color="#fff7ed", corner_radius=10)
    seccion_buscar.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_buscar,
        text="🔍 MÓDULO: BUSCAR HOMÓLOGO",
        font=("Arial", 14, "bold"),
        text_color="#c2410c"
    ).pack(pady=(15, 10))

    texto_buscar = """FUNCIÓN: Búsqueda individual de medicamentos alternativos

• Ingrese un código CUM específico
• Configure cantidad de resultados deseados
• Obtiene homólogos ordenados por similitud
• Muestra puntuación de confianza (0-100%)

RESULTADOS INCLUYEN:
• Información del medicamento consultado
• Lista de homólogos con puntuación
• Principio activo, clasificación ATC
• Forma farmacéutica y vía de administración

Ideal para consultas puntuales y verificación de alternativas."""

    ctk.CTkLabel(
        seccion_buscar,
        text=texto_buscar,
        font=("Arial", 11),
        text_color="#c2410c",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Módulo 3: Homologación Masiva
    seccion_masiva = ctk.CTkFrame(
        scroll_frame, fg_color="#fef2f2", corner_radius=10)
    seccion_masiva.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_masiva,
        text="📊 MÓDULO: HOMOLOGACIÓN MASIVA",
        font=("Arial", 14, "bold"),
        text_color="#dc2626"
    ).pack(pady=(15, 10))

    texto_masiva = """FUNCIÓN: Procesamiento en lote de múltiples medicamentos

• Carga archivo Excel con códigos CUM en primera columna
• Procesa automáticamente todos los registros
• Encuentra un homólogo por cada CUM
• Mantiene datos originales del archivo

ARCHIVO RESULTANTE INCLUYE:
• Todas las columnas originales
• CUM_HOMOLOGO: Código del medicamento alternativo
• NOMBRE_HOMOLOGO: Nombre del medicamento alternativo
• SCORE_SIMILITUD: Puntuación de confianza (0-1)

Perfecto para procesar listados completos de medicamentos."""

    ctk.CTkLabel(
        seccion_masiva,
        text=texto_masiva,
        font=("Arial", 11),
        text_color="#dc2626",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Flujo de trabajo recomendado
    seccion_flujo = ctk.CTkFrame(
        scroll_frame, fg_color="#f8fafc", corner_radius=10)
    seccion_flujo.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_flujo,
        text="🚀 FLUJO DE TRABAJO RECOMENDADO",
        font=("Arial", 14, "bold"),
        text_color="#374151"
    ).pack(pady=(15, 10))

    texto_flujo = """1. CONFIGURACIÓN INICIAL (Una sola vez):
   • Usar módulo "Cargar Archivos"
   • Subir los 4 archivos Excel requeridos
   • Entrenar el modelo de IA

2. USO DIARIO:
   • Para consultas individuales → "Buscar Homólogo"
   • Para lotes de medicamentos → "Homologación Masiva"

3. MANTENIMIENTO:
   • Actualizar archivos base periódicamente
   • Re-entrenar modelo con datos actualizados
   • Generar nuevos reportes EDA según necesidad

💡 CADA MÓDULO tiene su propio botón de ayuda (?) con información detallada."""

    ctk.CTkLabel(
        seccion_flujo,
        text=texto_flujo,
        font=("Arial", 11),
        text_color="#374151",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Botón cerrar DENTRO del scroll_frame
    btn_cerrar = ctk.CTkButton(
        scroll_frame,
        text="✖ Cerrar",
        command=modal.destroy,
        width=150,
        height=40,
        font=("Arial", 12, "bold"),
        fg_color=BUTTON_BG_COLOR,
        hover_color=BUTTON_HOVER_COLOR
    )
    btn_cerrar.pack(pady=20)

    # Focus en el modal
    modal.focus_set()
