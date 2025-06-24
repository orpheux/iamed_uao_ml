"""
buscar_homologo_help_modal.py
Modal de ayuda para la vista de búsqueda de homólogos.
"""

import customtkinter as ctk
from utils.constants import BUTTON_BG_COLOR, BUTTON_HOVER_COLOR


def help_buscar_homologo(parent):
    """
    Muestra un modal con ayuda contextual para la búsqueda de homólogos.
    """
    # Crear ventana modal
    modal = ctk.CTkToplevel(parent)
    modal.title("❓ Ayuda - Buscar Homólogo")
    modal.geometry("600x500")
    modal.resizable(False, False)
    modal.transient(parent)
    modal.grab_set()

    # Centrar modal
    modal.update_idletasks()
    x = (modal.winfo_screenwidth() // 2) - (600 // 2)
    y = (modal.winfo_screenheight() // 2) - (500 // 2)
    modal.geometry(f"600x500+{x}+{y}")

    # Frame principal con scroll
    scroll_frame = ctk.CTkScrollableFrame(modal, width=560, height=420)
    scroll_frame.pack(pady=20, padx=20, fill="both", expand=True)

    # Título principal
    titulo = ctk.CTkLabel(
        scroll_frame,
        text="🔍 BÚSQUEDA DE HOMÓLOGOS",
        font=("Arial", 18, "bold"),
        text_color=BUTTON_BG_COLOR
    )
    titulo.pack(pady=(10, 20))

    # Sección: ¿Qué es?
    seccion_que_es = ctk.CTkFrame(
        scroll_frame, fg_color="#f0f8ff", corner_radius=10)
    seccion_que_es.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_que_es,
        text="📋 ¿QUÉ ES LA BÚSQUEDA DE HOMÓLOGOS?",
        font=("Arial", 14, "bold"),
        text_color="#2c5282"
    ).pack(pady=(15, 10))

    texto_que_es = """Esta herramienta permite encontrar medicamentos alternativos (homólogos) 
para un medicamento específico identificado por su código CUM.

Un homólogo es un medicamento que tiene características similares en:
• Principio activo
• Clasificación ATC (Anatomical Therapeutic Chemical)
• Vía de administración
• Forma farmacéutica"""

    ctk.CTkLabel(
        seccion_que_es,
        text=texto_que_es,
        font=("Arial", 11),
        text_color="#2c5282",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Sección: Cómo usar
    seccion_como_usar = ctk.CTkFrame(
        scroll_frame, fg_color="#f0fff0", corner_radius=10)
    seccion_como_usar.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_como_usar,
        text="🔧 CÓMO USAR LA HERRAMIENTA",
        font=("Arial", 14, "bold"),
        text_color="#22543d"
    ).pack(pady=(15, 10))

    texto_como_usar = """1. INGRESE EL CÓDIGO CUM:
   • Escriba el código CUM del medicamento que desea consultar
   • Ejemplo: 19845234-1 o 20045678-2

2. CONFIGURE RESULTADOS:
   • Ajuste el número de homólogos que desea ver (por defecto: 5)
   • Rango recomendado: 3-10 resultados

3. EJECUTE LA BÚSQUEDA:
   • Presione Enter en el campo CUM, o
   • Haga clic en el botón ➤

4. REVISE LOS RESULTADOS:
   • Vea la información del medicamento consultado
   • Analice los homólogos encontrados ordenados por similitud"""

    ctk.CTkLabel(
        seccion_como_usar,
        text=texto_como_usar,
        font=("Arial", 11),
        text_color="#22543d",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Sección: Interpretación de resultados
    seccion_resultados = ctk.CTkFrame(
        scroll_frame, fg_color="#fff5f5", corner_radius=10)
    seccion_resultados.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_resultados,
        text="📊 INTERPRETACIÓN DE RESULTADOS",
        font=("Arial", 14, "bold"),
        text_color="#c53030"
    ).pack(pady=(15, 10))

    texto_resultados = """PUNTUACIÓN DE SIMILITUD (⭐):
• 90-100%: Homólogos excelentes (verde)
• 80-89%: Homólogos buenos (naranja)
• <80%: Homólogos con menor similitud (rojo)

INFORMACIÓN MOSTRADA:
• Medicamento consultado: Datos del CUM ingresado
• Lista de homólogos: Ordenados por similitud
• CUM, principio activo, clasificación ATC
• Forma farmacéutica, vía de administración
• Cantidad y unidad de medida

COLORES DE TARJETAS:
• Azul: Medicamento consultado
• Verde: Lista de homólogos encontrados
• Rojo: Mensajes de error o no encontrado"""

    ctk.CTkLabel(
        seccion_resultados,
        text=texto_resultados,
        font=("Arial", 11),
        text_color="#c53030",
        justify="left"
    ).pack(pady=(0, 15), padx=15)

    # Sección: Consejos
    seccion_consejos = ctk.CTkFrame(
        scroll_frame, fg_color="#fffaf0", corner_radius=10)
    seccion_consejos.pack(fill="x", pady=(0, 15), padx=10)

    ctk.CTkLabel(
        seccion_consejos,
        text="💡 CONSEJOS Y MEJORES PRÁCTICAS",
        font=("Arial", 14, "bold"),
        text_color="#d69e2e"    ).pack(pady=(15, 10))

    texto_consejos = """• Verifique que el código CUM esté completo y sea válido
• Use códigos CUM de medicamentos vigentes para mejores resultados
• Compare múltiples características, no solo la puntuación
• Considere la vía de administración según la necesidad del paciente
• Revise la forma farmacéutica para compatibilidad de uso
• Los homólogos con puntuación >90% son más confiables
• Si no encuentra resultados, verifique el código CUM
• Consulte siempre con un profesional de la salud"""

    ctk.CTkLabel(
        seccion_consejos,
        text=texto_consejos,
        font=("Arial", 11),
        text_color="#d69e2e",
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
