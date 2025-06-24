"""
homologacion_masiva_view.py
Vista para homologación masiva de medicamentos desde archivo Excel.
Permite cargar archivo, procesar homologación y descargar resultado.
"""

import os
import threading
from tkinter import filedialog, messagebox
import customtkinter as ctk

from utils.constants import (
    PRIMARY_COLOR,
    BUTTON_BG_COLOR,
    BUTTON_HOVER_COLOR,
    EXITO_COLOR,
    BUTTON_FONT
)
from services.homologacion_masiva_service import HomologacionMasivaService


class HomologacionMasivaView(ctk.CTkFrame):
    """Vista para homologación masiva de CUMs desde Excel"""

    def __init__(self, master):
        super().__init__(master)
        self.configure(fg_color="transparent")

        # Variables de estado
        self.archivo_cargado = None
        self.servicio_homologacion = None
        self.resultado_procesamiento = None
        self.en_proceso = False

        self._inicializar_servicio()
        self._crear_componentes()

    def _inicializar_servicio(self):
        """Inicializa el servicio de homologación masiva"""
        try:
            self.servicio_homologacion = HomologacionMasivaService()
        except Exception as e:
            messagebox.showerror(
                "Error", f"No se pudo inicializar el servicio: {str(e)}")

    def _crear_componentes(self):
        """Crea todos los componentes de la interfaz"""

        # Título principal
        titulo = ctk.CTkLabel(
            self,
            text="📊 Homologador masivo CUM",
            font=("Arial Black", 24, "bold"),
            text_color=PRIMARY_COLOR
        )
        titulo.pack(pady=(30, 10))

        # Subtítulo con instrucciones
        subtitulo = ctk.CTkLabel(
            self,
            text="por favor cargue un excel con los códigos a homologar en la columna A",
            font=("Arial", 14),
            text_color="gray"
        )
        subtitulo.pack(pady=(0, 30))

        # Frame principal de contenido
        self.frame_contenido = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_contenido.pack(expand=True, fill="both", padx=50)

        # Sección 1: Cargar archivo
        self._crear_seccion_cargar_archivo()

        # Sección 2: Estado del proceso (inicialmente oculta)
        self._crear_seccion_estado_proceso()

        # Sección 3: Resultado y descarga (inicialmente oculta)
        self._crear_seccion_resultado()

        # Botón de ayuda (esquina inferior derecha)
        self._crear_boton_ayuda()

    def _crear_seccion_cargar_archivo(self):
        """Crea la sección para cargar el archivo Excel"""

        # Frame para cargar archivo
        self.frame_cargar = ctk.CTkFrame(self.frame_contenido, height=200)
        self.frame_cargar.pack(fill="x", pady=(0, 20))
        self.frame_cargar.pack_propagate(False)

        # Campo de archivo con botón de selección
        frame_archivo = ctk.CTkFrame(self.frame_cargar, fg_color="transparent")
        frame_archivo.pack(expand=True)

        # Input del archivo (solo lectura)
        self.entry_archivo = ctk.CTkEntry(
            frame_archivo,
            placeholder_text="Seleccione un archivo Excel...",
            width=400,
            height=40,
            font=("Arial", 12),
            state="readonly"
        )
        self.entry_archivo.pack(side="left", padx=(0, 10))

        # Botón para seleccionar archivo
        self.btn_seleccionar = ctk.CTkButton(
            frame_archivo,
            text="📁",
            width=50,
            height=40,
            font=("Arial", 16),
            fg_color=BUTTON_BG_COLOR,
            hover_color=BUTTON_HOVER_COLOR,
            command=self._seleccionar_archivo
        )
        self.btn_seleccionar.pack(side="left")

        # Botón para iniciar homologación (inicialmente deshabilitado)
        self.btn_procesar = ctk.CTkButton(
            self.frame_cargar,
            text="🚀 Iniciar Homologación",
            width=200,
            height=50,
            font=BUTTON_FONT,
            fg_color=EXITO_COLOR,
            hover_color="#0f7a2e",
            state="disabled",
            command=self._iniciar_homologacion
        )
        self.btn_procesar.pack(pady=(20, 0))

    def _crear_seccion_estado_proceso(self):
        """Crea la sección que muestra el estado del proceso"""

        self.frame_estado = ctk.CTkFrame(self.frame_contenido, height=150)
        self.frame_estado.pack(fill="x", pady=(0, 20))
        self.frame_estado.pack_propagate(False)
        self.frame_estado.pack_forget()  # Inicialmente oculto

        # Label de estado
        self.label_estado = ctk.CTkLabel(
            self.frame_estado,
            text="⏳ Preparando homologación...",
            font=("Arial", 16, "bold"),
            text_color=PRIMARY_COLOR
        )
        self.label_estado.pack(pady=(30, 10))

        # Barra de progreso
        self.barra_progreso = ctk.CTkProgressBar(
            self.frame_estado,
            width=400,
            height=20
        )
        self.barra_progreso.pack(pady=(0, 10))
        self.barra_progreso.set(0)

        # Label de detalle del progreso
        self.label_progreso_detalle = ctk.CTkLabel(
            self.frame_estado,
            text="",
            font=("Arial", 12),
            text_color="gray"
        )
        self.label_progreso_detalle.pack()

    def _crear_seccion_resultado(self):
        """Crea la sección con los resultados y botón de descarga"""

        self.frame_resultado = ctk.CTkFrame(self.frame_contenido, height=200)
        self.frame_resultado.pack(fill="x")
        self.frame_resultado.pack_propagate(False)
        self.frame_resultado.pack_forget()  # Inicialmente oculto

        # Título de resultados
        self.label_resultado_titulo = ctk.CTkLabel(
            self.frame_resultado,
            text="✅ Homologación terminada",
            font=("Arial", 18, "bold"),
            text_color=EXITO_COLOR
        )
        self.label_resultado_titulo.pack(pady=(20, 10))

        # Frame para estadísticas
        frame_stats = ctk.CTkFrame(
            self.frame_resultado, fg_color="transparent")
        frame_stats.pack(pady=(0, 20))

        self.label_estadisticas = ctk.CTkLabel(
            frame_stats,
            text="",
            font=("Arial", 12),
            text_color="gray"
        )
        self.label_estadisticas.pack()

        # Botón de descarga
        self.btn_descargar = ctk.CTkButton(
            self.frame_resultado,
            text="💾 Descargar archivo homologado",
            width=250,
            height=50,
            font=BUTTON_FONT,
            fg_color=PRIMARY_COLOR,
            hover_color="#1a6c89",
            command=self._descargar_archivo
        )
        self.btn_descargar.pack()

    def _crear_boton_ayuda(self):
        """Crea el botón de ayuda flotante"""
        self.btn_ayuda = ctk.CTkButton(
            self,
            text="?",
            width=40,
            height=40,
            corner_radius=20,
            font=("Arial", 16, "bold"),
            fg_color="#17a2b8",
            hover_color="#138496",
            command=self._mostrar_ayuda
        )
        self.btn_ayuda.place(relx=0.97, rely=0.96, anchor="se")

    def _seleccionar_archivo(self):
        """Abre el diálogo para seleccionar archivo Excel"""
        if self.en_proceso:
            return

        archivo = filedialog.askopenfilename(
            title="Seleccionar archivo Excel",
            filetypes=[
                ("Archivos Excel", "*.xlsx *.xls"),
                ("Todos los archivos", "*.*")
            ]
        )

        if archivo:
            self._validar_y_cargar_archivo(archivo)

    def _validar_y_cargar_archivo(self, archivo_path):
        """Valida y carga el archivo Excel seleccionado"""
        if not self.servicio_homologacion:
            messagebox.showerror("Error", "Servicio no disponible")
            return

        # Mostrar nombre del archivo
        nombre_archivo = os.path.basename(archivo_path)
        self.entry_archivo.configure(state="normal")
        self.entry_archivo.delete(0, "end")
        self.entry_archivo.insert(0, nombre_archivo)
        self.entry_archivo.configure(state="readonly")

        # Validar archivo
        valido, mensaje = self.servicio_homologacion.validar_archivo_excel(
            archivo_path)

        if valido:
            self.archivo_cargado = archivo_path
            self.btn_procesar.configure(state="normal")
            messagebox.showinfo(
                "Éxito", f"Archivo cargado correctamente.\n{mensaje}")
        else:
            self.archivo_cargado = None
            self.btn_procesar.configure(state="disabled")
            self.entry_archivo.configure(state="normal")
            self.entry_archivo.delete(0, "end")
            self.entry_archivo.configure(state="readonly")
            messagebox.showerror("Error al leer el archivo", mensaje)

    def _iniciar_homologacion(self):
        """Inicia el proceso de homologación en un hilo separado"""
        if not self.archivo_cargado or self.en_proceso:
            return

        self.en_proceso = True

        # Cambiar UI al estado de procesamiento
        self.btn_procesar.configure(state="disabled")
        self.btn_seleccionar.configure(state="disabled")
        self.frame_estado.pack(fill="x", pady=(0, 20))
        self.frame_resultado.pack_forget()

        # Iniciar procesamiento en hilo separado
        thread = threading.Thread(target=self._procesar_homologacion_thread)
        thread.daemon = True
        thread.start()

    def _procesar_homologacion_thread(self):
        """Ejecuta la homologación en un hilo separado"""
        try:
            def callback_progreso(actual, total, mensaje):
                # Actualizar UI en el hilo principal
                self.after(0, self._actualizar_progreso,
                           actual, total, mensaje)

            # Verificar que el servicio esté inicializado
            if not self.servicio_homologacion:
                raise RuntimeError("El servicio de homologación no está disponible.")

            # Procesar homologación
            resultado = self.servicio_homologacion.procesar_homologacion(
                progreso_callback=callback_progreso
            )

            # Actualizar UI con el resultado
            self.after(0, self._finalizar_proceso, resultado)

        except Exception as e:
            self.after(0, self._error_proceso, str(e))

    def _actualizar_progreso(self, actual, total, mensaje):
        """Actualiza la barra de progreso y los mensajes"""
        progreso = actual / total if total > 0 else 0
        self.barra_progreso.set(progreso)

        self.label_estado.configure(text="🔄 Homologando, por favor espere...")
        self.label_progreso_detalle.configure(
            text=f"[{actual}/{total}] {mensaje}"
        )

    def _finalizar_proceso(self, resultado):
        """Finaliza el proceso y muestra los resultados"""
        self.en_proceso = False
        self.resultado_procesamiento = resultado

        if resultado['exito']:
            # Mostrar sección de resultado
            self.frame_estado.pack_forget()
            self.frame_resultado.pack(fill="x")

            # Actualizar estadísticas
            stats_text = (
                f"📊 Filas procesadas: {resultado.get('total_filas', 0)}\n"
                f"🔍 CUMs únicos: {resultado.get('cums_unicos', 0)}\n"
                f"✅ Homólogos encontrados: {resultado.get('homologos_encontrados', 0)}\n"
                f"📈 Tasa de éxito: {resultado.get('porcentaje_exito', 0):.1f}%"
            )
            self.label_estadisticas.configure(text=stats_text)

        else:
            # Mostrar error
            messagebox.showerror("Error en homologación",
                                 resultado.get('error', 'Error desconocido'))
            self.frame_estado.pack_forget()

        # Reactivar controles
        self.btn_seleccionar.configure(state="normal")
        self.btn_procesar.configure(state="normal")

    def _error_proceso(self, error_msg):
        """Maneja errores durante el proceso"""
        self.en_proceso = False
        messagebox.showerror(
            "Error", f"Error durante homologación: {error_msg}")

        self.frame_estado.pack_forget()
        self.btn_seleccionar.configure(state="normal")
        self.btn_procesar.configure(state="normal")

    def _descargar_archivo(self):
        """Abre diálogo para guardar el archivo homologado"""
        if not self.resultado_procesamiento or not self.servicio_homologacion:
            return

        archivo_salida = filedialog.asksaveasfilename(
            title="Guardar archivo homologado",
            defaultextension=".xlsx",
            filetypes=[("Archivo Excel", "*.xlsx"),
                       ("Todos los archivos", "*.*")]
        )

        if archivo_salida:
            exito, mensaje = self.servicio_homologacion.guardar_resultado(
                archivo_salida)

            if exito:
                messagebox.showinfo("Éxito", mensaje)
                # Resetear vista para nuevo proceso
                self._resetear_vista()
            else:
                messagebox.showerror("Error", mensaje)

    def _resetear_vista(self):
        """Resetea la vista para un nuevo proceso"""
        self.archivo_cargado = None
        self.resultado_procesamiento = None

        self.entry_archivo.configure(state="normal")
        self.entry_archivo.delete(0, "end")
        self.entry_archivo.configure(state="readonly")

        self.btn_procesar.configure(state="disabled")
        self.frame_estado.pack_forget()
        self.frame_resultado.pack_forget()

    def _mostrar_ayuda(self):
        """Muestra la ayuda contextual"""
        ayuda_texto = """
        🔍 AYUDA - HOMOLOGACIÓN MASIVA
        
        1. 📁 Seleccione un archivo Excel (.xlsx o .xls)
           - La primera columna debe contener los códigos CUM
           - Puede tener otras columnas que se mantendrán
        
        2. 🚀 Inicie la homologación
           - El proceso buscará homólogos para cada CUM
           - Se agregarán 3 columnas nuevas:
             • CUM_HOMOLOGO: Código del medicamento homólogo
             • NOMBRE_HOMOLOGO: Nombre del medicamento homólogo  
             • SCORE_SIMILITUD: Puntuación de similitud (0-1)
        
        3. 💾 Descargue el resultado
           - Seleccione dónde guardar el archivo procesado
           - El archivo mantendrá sus datos originales + homólogos
        
        📋 NOTAS:
        • Solo se encuentra 1 homólogo por CUM
        • CUMs sin homólogo mostrarán "SIN HOMÓLOGO"
        • El proceso puede tomar varios minutos según el tamaño
        """

        # Crear ventana de ayuda
        ventana_ayuda = ctk.CTkToplevel(self)
        ventana_ayuda.title("Ayuda - Homologación Masiva")
        ventana_ayuda.geometry("500x600")
        ventana_ayuda.resizable(False, False)

        # Centrar ventana
        ventana_ayuda.transient(self.winfo_toplevel())
        ventana_ayuda.grab_set()

        # Contenido de ayuda
        texto_ayuda = ctk.CTkTextbox(
            ventana_ayuda,
            font=("Arial", 11),
            wrap="word"
        )
        texto_ayuda.pack(expand=True, fill="both", padx=20, pady=20)
        texto_ayuda.insert("1.0", ayuda_texto)
        texto_ayuda.configure(state="disabled")

        # Botón cerrar
        btn_cerrar = ctk.CTkButton(
            ventana_ayuda,
            text="Cerrar",
            command=ventana_ayuda.destroy
        )
        btn_cerrar.pack(pady=(0, 20))
