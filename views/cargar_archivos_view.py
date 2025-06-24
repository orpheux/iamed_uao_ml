import os
import shutil
import threading
from tkinter import filedialog, messagebox
from typing import Dict, Optional
import customtkinter as ctk

from utils.constants import (
    PRIMARY_COLOR,
    BUTTON_BG_COLOR,
    BUTTON_HOVER_COLOR,
    EXITO_COLOR,
    BUTTON_FONT
)
from services.file_manager import cargar
from services.eda_generator import generar_eda_pdf
from services.encoding_service import process_medicamentos_encoding
from services.training_service import entrenar_modelo_homologacion



class CargarArchivosView(ctk.CTkFrame):
    """Vista elegante para carga y procesamiento de archivos de medicamentos"""
    
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        
        # Variables de estado
        self.rutas_archivos: Dict[str, Optional[str]] = {
            "medicamentos_vencidos": None,
            "medicamentos_vigentes": None,
            "medicamentos_renovacion": None,
            "medicamentos_otros": None
        }
        self.entradas = {}
        self.archivos_cargados = False
        self.modelo_entrenado = False
        self.en_proceso = False
          # Referencias a botones que se crean dinámicamente
        self.boton_entrenar: Optional[ctk.CTkButton] = None
        self.boton_eda: Optional[ctk.CTkButton] = None
        
        self._construir_interfaz()
    
    def _construir_interfaz(self):
        """Construye la interfaz elegante con secciones organizadas"""
        
        # Título principal
        titulo = ctk.CTkLabel(
            self,
            text="📂 Carga de archivos",
            font=("Arial Black", 24, "bold"),
            text_color=PRIMARY_COLOR
        )
        titulo.pack(pady=(30, 10))

        # Subtítulo con instrucciones
        subtitulo = ctk.CTkLabel(
            self,
            text="Seleccione los 4 archivos Excel requeridos para el entrenamiento del modelo",
            font=("Arial", 14),
            text_color="gray"
        )
        subtitulo.pack(pady=(0, 30))

        # Frame scrollable para todo el contenido
        self.frame_scroll = ctk.CTkScrollableFrame(
            self, 
            fg_color="transparent",
            scrollbar_button_color=PRIMARY_COLOR,
            scrollbar_button_hover_color="#1a6c89"
        )
        self.frame_scroll.pack(expand=True, fill="both", padx=50)

        # Frame principal de contenido dentro del scrollable
        self.frame_contenido = ctk.CTkFrame(self.frame_scroll, fg_color="transparent")
        self.frame_contenido.pack(expand=True, fill="both")

        # Sección 1: Selección de archivos
        self._crear_seccion_archivos()

        # Sección 2: Estado del proceso (inicialmente oculta)
        self._crear_seccion_estado_proceso()

        # Sección 3: Acciones adicionales (inicialmente oculta)
        self._crear_seccion_acciones()

        # Botón de ayuda
        self._crear_boton_ayuda()

    def _crear_seccion_archivos(self):
        """Crea la sección elegante para seleccionar archivos"""
        
        # Frame para la sección de archivos
        self.frame_archivos = ctk.CTkFrame(self.frame_contenido, height=300)
        self.frame_archivos.pack(fill="x", pady=(0, 20))
        self.frame_archivos.pack_propagate(False)

        # Título de la sección
        titulo_seccion = ctk.CTkLabel(
            self.frame_archivos,
            text="📋 Archivos requeridos",
            font=("Arial", 16, "bold"),
            text_color=PRIMARY_COLOR
        )
        titulo_seccion.pack(pady=(20, 15))

        # Configuración de archivos requeridos
        archivos_config = [
            ("medicamentos_vencidos", "💊 Medicamentos vencidos", "#e74c3c"),
            ("medicamentos_vigentes", "✅ Medicamentos vigentes", "#27ae60"),
            ("medicamentos_renovacion", "🔄 Medicamentos renovación", "#f39c12"),
            ("medicamentos_otros", "📦 Medicamentos otros", "#3498db")
        ]

        # Crear entradas para cada archivo
        for i, (clave, etiqueta, color) in enumerate(archivos_config):
            # Frame para cada archivo
            frame_archivo = ctk.CTkFrame(self.frame_archivos, fg_color="transparent")
            frame_archivo.pack(fill="x", padx=20, pady=5)

            # Etiqueta con emoji y color
            label_archivo = ctk.CTkLabel(
                frame_archivo,
                text=etiqueta,
                font=("Arial", 12, "bold"),
                text_color=color,
                width=200,
                anchor="w"
            )
            label_archivo.pack(side="left", padx=(0, 10))

            # Campo de entrada (readonly)
            entrada = ctk.CTkEntry(
                frame_archivo,
                placeholder_text="Seleccione archivo...",
                width=350,
                height=35,
                font=("Arial", 11),
                state="readonly"
            )
            entrada.pack(side="left", padx=(0, 10))
            self.entradas[clave] = entrada

            # Botón de selección
            boton_seleccionar = ctk.CTkButton(
                frame_archivo,
                text="�",
                width=45,
                height=35,
                font=("Arial", 14),
                fg_color=BUTTON_BG_COLOR,
                hover_color=BUTTON_HOVER_COLOR,
                command=lambda c=clave: self._seleccionar_archivo(c)
            )
            boton_seleccionar.pack(side="left")

        # Botón principal de carga
        self.boton_cargar = ctk.CTkButton(
            self.frame_archivos,
            text="🚀 Cargar archivos",
            width=200,
            height=50,
            font=BUTTON_FONT,
            fg_color=PRIMARY_COLOR,
            hover_color="#1a6c89",
            command=self._iniciar_carga_archivos
        )
        self.boton_cargar.pack(pady=(20, 20))

    def _crear_seccion_estado_proceso(self):
        """Crea la sección que muestra el estado del proceso con barra de progreso"""
        
        self.frame_estado = ctk.CTkFrame(self.frame_contenido, height=150)
        self.frame_estado.pack(fill="x", pady=(0, 20))
        self.frame_estado.pack_propagate(False)
        self.frame_estado.pack_forget()  # Inicialmente oculto

        # Label de estado principal
        self.label_estado = ctk.CTkLabel(
            self.frame_estado,
            text="⏳ Preparando carga...",
            font=("Arial", 16, "bold"),
            text_color=PRIMARY_COLOR
        )
        self.label_estado.pack(pady=(30, 10))

        # Barra de progreso elegante
        self.barra_progreso = ctk.CTkProgressBar(
            self.frame_estado,
            width=400,
            height=20,
            corner_radius=10
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

    def _crear_seccion_acciones(self):
        """Crea la sección con acciones adicionales"""
        
        self.frame_acciones = ctk.CTkFrame(self.frame_contenido, height=200)
        self.frame_acciones.pack(fill="x")
        self.frame_acciones.pack_propagate(False)
        self.frame_acciones.pack_forget()  # Inicialmente oculto

        # Título de la sección
        titulo_acciones = ctk.CTkLabel(
            self.frame_acciones,
            text="🛠️ Acciones disponibles",
            font=("Arial", 16, "bold"),
            text_color=PRIMARY_COLOR
        )
        titulo_acciones.pack(pady=(20, 15))

        # Frame para botones en fila
        frame_botones = ctk.CTkFrame(self.frame_acciones, fg_color="transparent")
        frame_botones.pack(pady=(0, 20))

        # Botón entrenar modelo
        self.boton_entrenar = ctk.CTkButton(
            frame_botones,
            text="🤖 Entrenar modelo",
            width=200,
            height=50,
            font=BUTTON_FONT,
            fg_color=EXITO_COLOR,
            hover_color="#0f7a2e",
            command=self._iniciar_entrenamiento
        )
        self.boton_entrenar.pack(side="left", padx=(0, 20))

        # Botón generar EDA
        self.boton_eda = ctk.CTkButton(
            frame_botones,
            text="📊 Generar EDA",
            width=200,
            height=50,
            font=BUTTON_FONT,
            fg_color="#9b59b6",
            hover_color="#8e44ad",
            command=self._iniciar_eda,
            state="disabled"
        )
        self.boton_eda.pack(side="left")

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

    def _seleccionar_archivo(self, clave):
        """Abre diálogo para seleccionar archivo Excel"""
        if self.en_proceso:
            return
            
        archivo = filedialog.askopenfilename(
            title=f"Seleccionar {clave.replace('_', ' ')}",
            filetypes=[("Archivos Excel", "*.xlsx"), ("Todos los archivos", "*.*")]
        )
        
        if archivo:
            if archivo.lower().endswith('.xlsx'):
                self.rutas_archivos[clave] = archivo
                nombre_archivo = os.path.basename(archivo)
                
                # Actualizar entrada
                self.entradas[clave].configure(state="normal")
                self.entradas[clave].delete(0, "end")
                self.entradas[clave].insert(0, nombre_archivo)
                self.entradas[clave].configure(state="readonly")
                
                # Verificar si todos los archivos están cargados
                self._verificar_archivos_completos()
            else:
                messagebox.showerror("Error", "Solo se permiten archivos .xlsx")

    def _verificar_archivos_completos(self):
        """Verifica si todos los archivos requeridos están seleccionados"""
        archivos_completos = all(ruta is not None for ruta in self.rutas_archivos.values())
        
        if archivos_completos:
            self.boton_cargar.configure(state="normal")
        else:
            self.boton_cargar.configure(state="disabled")

    def _iniciar_carga_archivos(self):
        """Inicia el proceso de carga de archivos"""
        if self.en_proceso:
            return
            
        # Validar que todos los archivos estén seleccionados
        faltantes = [k for k, v in self.rutas_archivos.items() if not v]
        if faltantes:
            nombres_faltantes = [k.replace('_', ' ').title() for k in faltantes]
            messagebox.showerror(
                "Archivos faltantes", 
                f"Faltan archivos por seleccionar:\n• " + "\n• ".join(nombres_faltantes)
            )
            return

        self.en_proceso = True
        self._mostrar_estado_proceso("📂 Cargando archivos...")
        
        # Deshabilitar controles
        self.boton_cargar.configure(state="disabled")
        
        # Iniciar carga en hilo separado
        thread = threading.Thread(target=self._cargar_archivos_thread)
        thread.daemon = True
        thread.start()

    def _cargar_archivos_thread(self):
        """Ejecuta la carga de archivos en un hilo separado"""
        try:
            def callback_progreso(paso, total, mensaje):
                self.after(0, self._actualizar_progreso_carga, paso, total, mensaje)

            # Simular progreso de carga
            self.after(0, callback_progreso, 1, 4, "Validando archivos...")
            
            # Ejecutar carga real
            resultados = cargar(self.rutas_archivos)
            
            if resultados is None:
                raise Exception("No se pudieron cargar los archivos")
            
            self.after(0, callback_progreso, 4, 4, "Archivos cargados exitosamente")
            self.after(0, self._finalizar_carga_archivos, resultados)
            
        except Exception as e:
            self.after(0, self._error_carga_archivos, str(e))

    def _actualizar_progreso_carga(self, paso, total, mensaje):
        """Actualiza el progreso de carga"""
        progreso = paso / total if total > 0 else 0
        self.barra_progreso.set(progreso)
        self.label_progreso_detalle.configure(text=f"[{paso}/{total}] {mensaje}")

    def _finalizar_carga_archivos(self, resultados):
        """Finaliza el proceso de carga exitoso"""
        self.en_proceso = False
        self.archivos_cargados = True
        
        # Actualizar estado final
        mensaje_exito = (
            "✅ Archivos cargados correctamente\n"
            f"📊 Vencidos: {resultados.get('vencidos', 0):,} • "
            f"Vigentes: {resultados.get('vigentes', 0):,} • "
            f"Renovación: {resultados.get('renovacion', 0):,} • "
            f"Otros: {resultados.get('otros', 0):,}"
        )
        
        self.label_estado.configure(text=mensaje_exito, text_color=EXITO_COLOR)
        self.barra_progreso.set(1.0)
        
        # Cambiar color del botón a éxito
        self.boton_cargar.configure(
            text="✅ Archivos cargados",
            fg_color=EXITO_COLOR,
            hover_color="#0f7a2e",
            state="normal"
        )
        
        # Mostrar acciones adicionales
        self.frame_acciones.pack(fill="x")

    def _error_carga_archivos(self, error_msg):
        """Maneja errores durante la carga"""
        self.en_proceso = False
        self.frame_estado.pack_forget()
        
        messagebox.showerror("Error en carga", f"Error al cargar archivos:\n{error_msg}")
        
        # Reactivar controles
        self.boton_cargar.configure(state="normal")

    def _mostrar_estado_proceso(self, mensaje):
        """Muestra la sección de estado con mensaje inicial"""
        self.frame_estado.pack(fill="x", pady=(0, 20))
        self.label_estado.configure(text=mensaje, text_color=PRIMARY_COLOR)
        self.barra_progreso.set(0)
        self.label_progreso_detalle.configure(text="Preparando...")

    def _iniciar_entrenamiento(self):
        """Inicia el proceso de entrenamiento del modelo"""
        if not self.archivos_cargados or self.en_proceso:
            return
            
        self.en_proceso = True
        self._mostrar_estado_proceso("🤖 Entrenando modelo...")
          # Deshabilitar botones
        if self.boton_entrenar:
            self.boton_entrenar.configure(state="disabled")
        if self.boton_eda:
            self.boton_eda.configure(state="disabled")
        
        # Iniciar entrenamiento en hilo separado
        thread = threading.Thread(target=self._entrenar_modelo_thread)
        thread.daemon = True
        thread.start()

    def _entrenar_modelo_thread(self):
        """Ejecuta el entrenamiento en un hilo separado"""
        try:
            def callback_progreso(paso, total, mensaje):
                self.after(0, self._actualizar_progreso_entrenamiento, paso, total, mensaje)

            # Pasos del entrenamiento
            self.after(0, callback_progreso, 1, 3, "Codificando datos...")
            
            input_path = "./data/medicamentos_preprocesados.parquet"
            output_path = "./data/dataset_entrenamiento_homologacion.parquet"
            model_path = "./models/iamed.pkl"

            # Codificación
            process_medicamentos_encoding(input_path, output_path)
            
            self.after(0, callback_progreso, 2, 3, "Entrenando modelo...")
            
            # Entrenamiento
            entrenar_modelo_homologacion(output_path, model_path)
            
            self.after(0, callback_progreso, 3, 3, "Entrenamiento completado")
            self.after(0, self._finalizar_entrenamiento)
            
        except Exception as e:
            self.after(0, self._error_entrenamiento, str(e))

    def _actualizar_progreso_entrenamiento(self, paso, total, mensaje):
        """Actualiza el progreso del entrenamiento"""
        progreso = paso / total if total > 0 else 0
        self.barra_progreso.set(progreso)
        self.label_progreso_detalle.configure(text=f"[{paso}/{total}] {mensaje}")

    def _finalizar_entrenamiento(self):
        """Finaliza el proceso de entrenamiento exitoso"""
        self.en_proceso = False
        self.modelo_entrenado = True
        
        # Actualizar estado final
        self.label_estado.configure(
            text="🎉 Modelo entrenado exitosamente", 
            text_color=EXITO_COLOR
        )
        self.barra_progreso.set(1.0)
          # Cambiar botón a éxito
        if self.boton_entrenar:
            self.boton_entrenar.configure(
                text="✅ Modelo entrenado",
                fg_color=EXITO_COLOR,
                hover_color="#0f7a2e",
                state="normal"
            )
        
        # Habilitar EDA
        if self.boton_eda:
            self.boton_eda.configure(state="normal")

    def _error_entrenamiento(self, error_msg):
        """Maneja errores durante el entrenamiento"""
        self.en_proceso = False
        messagebox.showerror("Error en entrenamiento", f"Error durante el entrenamiento:\n{error_msg}")
          # Reactivar controles
        if self.boton_entrenar:
            self.boton_entrenar.configure(state="normal")
        if self.boton_eda:
            self.boton_eda.configure(state="normal" if self.modelo_entrenado else "disabled")

    def _iniciar_eda(self):
        """Inicia la generación del EDA"""
        if self.en_proceso:
            return
            
        self.en_proceso = True
        self._mostrar_estado_proceso("📊 Generando EDA...")
          # Deshabilitar botones
        if self.boton_eda:
            self.boton_eda.configure(state="disabled")
        if self.boton_entrenar:
            self.boton_entrenar.configure(state="disabled")
        
        # Iniciar EDA en hilo separado
        thread = threading.Thread(target=self._generar_eda_thread)
        thread.daemon = True
        thread.start()

    def _generar_eda_thread(self):
        """Genera el EDA en un hilo separado"""
        try:
            def callback_progreso(paso, total, mensaje):
                self.after(0, self._actualizar_progreso_eda, paso, total, mensaje)

            self.after(0, callback_progreso, 1, 3, "Analizando datos...")
            
            # Generar EDA
            ruta_pdf = generar_eda_pdf()
            
            self.after(0, callback_progreso, 2, 3, "EDA generado, seleccionando ubicación...")
            
            # Solicitar ubicación de guardado
            self.after(0, self._solicitar_guardado_eda, ruta_pdf)
            
        except Exception as e:
            self.after(0, self._error_eda, str(e))

    def _actualizar_progreso_eda(self, paso, total, mensaje):
        """Actualiza el progreso del EDA"""
        progreso = paso / total if total > 0 else 0
        self.barra_progreso.set(progreso)
        self.label_progreso_detalle.configure(text=f"[{paso}/{total}] {mensaje}")

    def _solicitar_guardado_eda(self, ruta_pdf):
        """Solicita al usuario dónde guardar el EDA"""
        try:
            ruta_destino = filedialog.asksaveasfilename(
                title="Guardar EDA como...",
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("Todos los archivos", "*.*")],
                initialfile="Eda_Completo.pdf"
            )
            
            if ruta_destino:
                shutil.copy(str(ruta_pdf), str(ruta_destino))
                self._finalizar_eda_exitoso(ruta_destino)
            else:
                self._cancelar_eda()
                
        except Exception as e:
            self._error_eda(str(e))

    def _finalizar_eda_exitoso(self, ruta_destino):
        """Finaliza el EDA exitosamente"""
        self.en_proceso = False
        
        self.label_estado.configure(
            text=f"📄 EDA guardado exitosamente", 
            text_color=EXITO_COLOR
        )
        self.barra_progreso.set(1.0)
          # Cambiar botón a éxito
        if self.boton_eda:
            self.boton_eda.configure(
                text="✅ EDA generado",
                fg_color=EXITO_COLOR,
                hover_color="#0f7a2e",
                state="normal"
            )
        
        # Reactivar entrenamiento
        if self.boton_entrenar:
            self.boton_entrenar.configure(state="normal")
        
        messagebox.showinfo("EDA exportado", f"✅ EDA generado y guardado exitosamente en:\n{ruta_destino}")

    def _cancelar_eda(self):
        """Cancela la generación del EDA"""
        self.en_proceso = False
        self.frame_estado.pack_forget()
        
        if self.boton_eda:
            self.boton_eda.configure(state="normal")
        if self.boton_entrenar:
            self.boton_entrenar.configure(state="normal")

    def _error_eda(self, error_msg):
        """Maneja errores durante la generación del EDA"""
        self.en_proceso = False
        messagebox.showerror("Error generando EDA", f"Error durante la generación del EDA:\n{error_msg}")
        
        # Reactivar controles
        if self.boton_eda:
            self.boton_eda.configure(state="normal")  
        if self.boton_entrenar:
            self.boton_entrenar.configure(state="normal")

    def _mostrar_ayuda(self):
        """Muestra la ayuda contextual"""
        ayuda_texto = """
        🔍 AYUDA - CARGA DE ARCHIVOS
        
        1. 📁 Seleccione los 4 archivos Excel requeridos:
           • 💊 Medicamentos vencidos
           • ✅ Medicamentos vigentes  
           • 🔄 Medicamentos renovación
           • 📦 Medicamentos otros
        
        2. 🚀 Cargue los archivos
           • Se validarán y procesarán automáticamente
           • Verá estadísticas de cada archivo cargado
        
        3. 🤖 Entrene el modelo (opcional)
           • Codifica los datos y entrena el modelo de homologación
           • Necesario para usar la búsqueda de homólogos
        
        4. 📊 Genere EDA (opcional)
           • Crea un análisis exploratorio en PDF
           • Incluye gráficos y estadísticas detalladas
        
        📋 NOTAS:
        • Todos los archivos deben ser formato .xlsx
        • El proceso puede tomar varios minutos
        • Los archivos se procesan y almacenan localmente
        """
        
        # Crear ventana de ayuda
        ventana_ayuda = ctk.CTkToplevel(self)
        ventana_ayuda.title("Ayuda - Carga de Archivos")
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
