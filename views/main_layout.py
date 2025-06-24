"""
main_layout.py
Vista principal de la aplicación IAMED.
Contiene el navbar, botón de ayuda y un área dinámica donde se cargan los módulos.
"""

import customtkinter as ctk
from PIL import Image

from utils.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    BUTTON_BG_COLOR,
    BUTTON_HOVER_COLOR,
    ROUND_BUTTON_RADIUS,
    ICON_FONT,
    LOGO_TX_PATH
)

from views.nav_bar import TopNavBar
from views.main_help_modal import help_main
from views.cargar_archivos_view import CargarArchivosView
from views.buscar_homologo_view import BuscarHomologoView
from views.homologacion_masiva_view import HomologacionMasivaView


class MainApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("IAMED")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)
        self.configure(fg_color="white")

        # Navbar
        TopNavBar(
            master=self,
            on_archivos=self.accion_archivos,
            on_busqueda=self.accion_busqueda,
            on_excel=self.accion_excel,
            on_home=self.accion_home
        )        # Frame donde van los módulos
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(expand=True, fill="both")

        # Construir botón de ayuda primero
        self._construir_boton_ayuda()
        
        # Contenido por defecto (logo central)
        self._mostrar_logo_central()

    def _mostrar_logo_central(self):
        logo_img = Image.open(LOGO_TX_PATH)
        ancho_objetivo = 400
        proporcion = ancho_objetivo / logo_img.width
        alto = int(logo_img.height * proporcion)

        logo_widget = ctk.CTkLabel(
            self.content_frame,
            image=ctk.CTkImage(light_image=logo_img,
                               size=(ancho_objetivo, alto)),
            text=""
        )
        logo_widget.place(relx=0.5, rely=0.5, anchor="center")
          # Mostrar botón de ayuda principal cuando estamos en la vista principal
        self.boton_ayuda.place(relx=0.97, rely=0.96, anchor="se")

    def _cambiar_contenido(self, ClaseVista):
        for widget in self.content_frame.winfo_children():
            widget.destroy()        # Ocultar botón de ayuda principal cuando navegamos a otros módulos
        self.boton_ayuda.place_forget()
        
        vista = ClaseVista(self.content_frame)
        vista.pack(expand=True, fill="both")

    def _construir_boton_ayuda(self):
        self.boton_ayuda = ctk.CTkButton(
            master=self,
            text="?",
            width=40,
            height=40,
            corner_radius=ROUND_BUTTON_RADIUS,
            font=ICON_FONT,
            fg_color=BUTTON_BG_COLOR,
            hover_color=BUTTON_HOVER_COLOR,
            text_color="white",
            command=self.accion_ayuda
        )
        # No colocar el botón aquí, se coloca en _mostrar_logo_central()

    def accion_archivos(self):
        print("→ Módulo: Cargar archivos")
        self._cambiar_contenido(CargarArchivosView)

    def accion_busqueda(self):
        print("→ Módulo: Buscar homólogo")
        self._cambiar_contenido(BuscarHomologoView)
    def accion_excel(self):
        print("→ Módulo: Homologación masiva")
        self._cambiar_contenido(HomologacionMasivaView)
    def accion_home(self):
        """Volver a la vista principal con el logo"""
        print("→ Volver al inicio")
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self._mostrar_logo_central()

    def accion_ayuda(self):
        print("→ Mostrar ayuda contextual")
        help_main(self)
