from tkinter import messagebox
import customtkinter as ctk
from services.search_service import SearchService
from views.buscar_homologo_help_modal import help_buscar_homologo
from utils.constants import (
    BUTTON_BG_COLOR,
    BUTTON_HOVER_COLOR,
    ROUND_BUTTON_RADIUS,
    ICON_FONT
)


class BuscarHomologoView(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.configure(fg_color="transparent")
        self.search_service = None
        self.boton_ayuda = None  # Inicializar atributo aquí
        self._inicializar_servicio()
        self._crear_componentes()

    def _inicializar_servicio(self):
        """Inicializa el servicio de búsqueda"""
        try:
            self.search_service = SearchService()
        except Exception as e:
            messagebox.showerror(
                "Error", f"No se pudo cargar el modelo: {str(e)}")

    def _crear_componentes(self):
        # Título
        titulo = ctk.CTkLabel(
            self, text="🔍 Buscar homólogo por CUM", font=("Arial", 20, "bold"))
        titulo.pack(pady=(20, 10))

        # Frame búsqueda
        frame_busqueda = ctk.CTkFrame(self, fg_color="transparent")
        frame_busqueda.pack(pady=(10, 5))

        self.entry_cum = ctk.CTkEntry(
            frame_busqueda, placeholder_text="Ingrese código CUM", width=200)
        self.entry_cum.pack(side="left", padx=(0, 5))
        self.entry_cum.bind("<Return>", lambda e: self._buscar())

        self.boton_buscar = ctk.CTkButton(
            frame_busqueda, text="➤", width=40, command=self._buscar)
        self.boton_buscar.pack(side="left")

        # Spinner de número de recomendaciones
        frame_config = ctk.CTkFrame(self, fg_color="transparent")
        frame_config.pack(pady=(10, 5))

        ctk.CTkLabel(frame_config, text="Número de resultados:").pack(
            side="left", padx=(0, 5))

        self.entry_top_n = ctk.CTkEntry(frame_config, width=50)
        self.entry_top_n.insert(0, "5")
        self.entry_top_n.pack(side="left")

        # Botón de ayuda
        self._crear_boton_ayuda()

        # Frame resultados con scrollbar
        self.scroll_frame = ctk.CTkScrollableFrame(self, height=400)
        self.scroll_frame.pack(pady=(10, 20), fill="both", expand=True)        # Mensaje inicial
        self._mostrar_mensaje_inicial()

    def _crear_boton_ayuda(self):
        """Crea el botón de ayuda para esta vista"""
        self.boton_ayuda = ctk.CTkButton(
            master=self.master,  # Usar self.master en lugar de self
            text="?",
            width=40,
            height=40,
            corner_radius=ROUND_BUTTON_RADIUS,
            font=ICON_FONT,
            fg_color=BUTTON_BG_COLOR,
            hover_color=BUTTON_HOVER_COLOR,
            text_color="white",
            command=self._mostrar_ayuda
        )
        self.boton_ayuda.place(relx=0.97, rely=0.02, anchor="ne")

    def _mostrar_ayuda(self):
        """Muestra el modal de ayuda para esta vista"""
        help_buscar_homologo(self)

    def _mostrar_mensaje_inicial(self):
        """Muestra mensaje inicial en el área de resultados"""
        mensaje = ctk.CTkLabel(
            self.scroll_frame,
            text="💡 Ingrese un código CUM y presione Enter o haga clic en ➤ para buscar homólogos",
            text_color="gray"
        )
        mensaje.pack(pady=20)

    def _buscar(self):
        """Ejecuta la búsqueda de homólogos"""
        if self.search_service is None:
            messagebox.showerror("Error", "Servicio de búsqueda no disponible")
            return

        codigo_cum = self.entry_cum.get().strip()

        # Validar entrada CUM
        if not codigo_cum:
            messagebox.showerror("Error", "Debe ingresar un código CUM.")
            return

        # Validar número de resultados
        try:
            top_n = int(self.entry_top_n.get())
            if top_n <= 0:
                raise ValueError("Debe ser mayor a 0")
        except ValueError:
            messagebox.showerror(
                "Error", "El número de resultados debe ser un entero positivo.")
            return

        # Limpiar resultados anteriores
        self._limpiar_resultados()

        # Mostrar indicador de carga
        loading_label = ctk.CTkLabel(
            self.scroll_frame,
            text="🔄 Buscando homólogos...",
            font=("Arial", 14)
        )
        loading_label.pack(pady=20)

        # Actualizar interfaz
        self.update_idletasks()

        try:
            # Ejecutar búsqueda
            resultado = self.search_service.buscar_homologos(
                codigo_cum, n_recomendaciones=top_n
            )

            # Limpiar indicador de carga
            loading_label.destroy()

            # Mostrar resultados
            self._mostrar_resultados(resultado)

        except Exception as e:
            loading_label.destroy()
            messagebox.showerror("Error durante la búsqueda", str(e))

    def _limpiar_resultados(self):
        """Limpia todos los widgets del área de resultados"""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

    def _mostrar_resultados(self, resultado):
        """Muestra los resultados de la búsqueda"""
        if not resultado['encontrado']:
            # Mostrar error si no se encontraron resultados
            error_frame = ctk.CTkFrame(
                self.scroll_frame, fg_color="red", corner_radius=10)
            error_frame.pack(pady=10, padx=10, fill="x")

            ctk.CTkLabel(
                error_frame,
                text=f"⚠️ {resultado.get('error', 'No se encontraron resultados')}",
                text_color="white",
                font=("Arial", 12, "bold")
            ).pack(pady=10)
            return

        # Mostrar información del medicamento origen
        self._mostrar_medicamento_origen(resultado['medicamento_origen'])

        # Mostrar homólogos encontrados
        recomendaciones = resultado['recomendaciones']
        if recomendaciones:
            self._mostrar_homologos(recomendaciones)

    def _mostrar_medicamento_origen(self, medicamento_origen):
        """Muestra información del medicamento de origen"""
        origen_frame = ctk.CTkFrame(
            self.scroll_frame, fg_color="#2aa9d6", corner_radius=10)
        origen_frame.pack(pady=(10, 5), padx=10, fill="x")

        # Título
        ctk.CTkLabel(
            origen_frame,
            text="📋 MEDICAMENTO CONSULTADO",
            text_color="white",
            font=("Arial", 14, "bold")
        ).pack(pady=(10, 5))

        # Información del medicamento
        info_text = f"🔸 Producto: {medicamento_origen.get('producto', 'N/A')}\n"
        info_text += f"🔸 Principio Activo: {medicamento_origen.get('principio_activo', 'N/A')}\n"
        
        # ATC con descripción
        atc_codigo = medicamento_origen.get('atc', 'N/A')
        atc_descripcion = medicamento_origen.get('atc_descripcion', '')
        if atc_descripcion:
            info_text += f"🔸 ATC: {atc_codigo} - {atc_descripcion}\n"
        else:
            info_text += f"🔸 ATC: {atc_codigo}\n"
            
        info_text += f"🔸 Vía: {medicamento_origen.get('via', 'N/A')}"

        ctk.CTkLabel(
            origen_frame,
            text=info_text,
            text_color="white",
            font=("Arial", 11),
            justify="left"
        ).pack(pady=(0, 10), padx=10)

    def _mostrar_homologos(self, recomendaciones):
        """Muestra la lista de homólogos encontrados"""
        # Título de resultados
        titulo_frame = ctk.CTkFrame(
            self.scroll_frame, fg_color="green", corner_radius=10)
        titulo_frame.pack(pady=(10, 5), padx=10, fill="x")

        ctk.CTkLabel(
            titulo_frame,
            text=f"✅ HOMÓLOGOS ENCONTRADOS ({len(recomendaciones)})",
            text_color="white",
            font=("Arial", 14, "bold")
        ).pack(pady=10)

        # Lista de homólogos
        for i, homologo in enumerate(recomendaciones, 1):
            self._crear_tarjeta_homologo(i, homologo)

    def _crear_tarjeta_homologo(self, numero, homologo):
        """Crea una tarjeta individual para cada homólogo"""
        # Frame principal del homólogo
        homologo_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=8)
        homologo_frame.pack(pady=5, padx=10, fill="x")

        # Encabezado con número y score
        header_frame = ctk.CTkFrame(homologo_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            header_frame,
            text=f"{numero}. {homologo.get('producto', 'N/A')}",
            font=("Arial", 12, "bold"),
            anchor="w"
        ).pack(side="left")

        score = homologo.get('score_similitud', 0)
        color_score = "#00aa00" if score >= 0.9 else "#ff6600" if score >= 0.8 else "#ff0000"

        ctk.CTkLabel(
            header_frame,
            text=f"⭐ {score:.1%}",
            font=("Arial", 11, "bold"),
            text_color=color_score
        ).pack(side="right")

        # Información detallada
        info_frame = ctk.CTkFrame(homologo_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        # ATC con descripción
        atc_codigo = homologo.get('atc', 'N/A')
        atc_descripcion = homologo.get('atc_descripcion', '')
        if atc_descripcion:
            atc_completo = f"{atc_codigo} - {atc_descripcion}"
        else:
            atc_completo = atc_codigo

        detalles = [
            f"🆔 CUM: {homologo.get('cum', 'N/A')}",
            f"🧪 Principio: {homologo.get('principio_activo', 'N/A')}",
            f"💊 ATC: {atc_completo}",
            f"💊 Forma: {homologo.get('forma_farmaceutica', 'N/A')}",
            f"🚪 Vía: {homologo.get('via', 'N/A')}",
            f"📏 Cantidad: {homologo.get('cantidad', 'N/A')} {homologo.get('unidad', '')}"
        ]

        for detalle in detalles:
            ctk.CTkLabel(
                info_frame,
                text=detalle,
                font=("Arial", 10),
                anchor="w",
                text_color="gray"
            ).pack(anchor="w", pady=1)
