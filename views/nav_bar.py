"""
nav_bar.py
Navbar superior con logo, botones y separadores.
"""

import customtkinter as ctk
from PIL import Image

from utils.constants import (
    LOGO_PATH,
    BUTTON_FONT,
    NAV_BUTTON_WIDTH,
    NAV_BUTTON_HEIGHT,
    BUTTON_HOVER_COLOR
)


class TopNavBar(ctk.CTkFrame):
    def __init__(self, master, on_archivos, on_busqueda, on_excel, on_home=None):
        super().__init__(master, height=70, fg_color="white")
        self.pack(side="top", fill="x")

        self.columnconfigure((2, 4, 6), weight=1, uniform="buttons")

        # Logo (clickeable para volver al inicio)
        logo_img = ctk.CTkImage(
            light_image=Image.open(LOGO_PATH), size=(140, 50))
        self.logo = ctk.CTkLabel(self, image=logo_img, text="", cursor="hand2")
        self.logo.grid(row=0, column=0, padx=(20, 10))
        
        # Hacer el logo clickeable
        if on_home:
            self.logo.bind("<Button-1>", lambda e: on_home())

        # Botón 1: Cargar archivos
        self._crear_separador(1)
        self.btn_archivos = self._crear_boton("Cargar archivos", on_archivos)
        self.btn_archivos.grid(row=0, column=2, padx=10, sticky="nsew")

        # Botón 2: Buscar homólogo
        self._crear_separador(3)
        self.btn_busqueda = self._crear_boton("Buscar homólogo", on_busqueda)
        self.btn_busqueda.grid(row=0, column=4, padx=10, sticky="nsew")

        # Botón 3: Homologación masiva
        self._crear_separador(5)
        self.btn_excel = self._crear_boton("Homologación masiva", on_excel)
        self.btn_excel.grid(row=0, column=6, padx=10, sticky="nsew")

        # Línea inferior del navbar (más delgada)
        self.linea_inferior = ctk.CTkFrame(self, height=2, fg_color="#B0B0B0")
        self.linea_inferior.grid(row=1, column=0, columnspan=7, sticky="ew", padx=0)

    def _crear_boton(self, texto, comando):
        btn = ctk.CTkButton(
            self,
            text=texto,
            command=comando,
            width=NAV_BUTTON_WIDTH,
            height=NAV_BUTTON_HEIGHT,
            font=BUTTON_FONT,
            fg_color="#ffffff",
            hover_color=BUTTON_HOVER_COLOR,
            text_color="black",
            corner_radius=6
        )

        btn.bind("<Enter>", lambda e: btn.configure(text_color="white", fg_color=BUTTON_HOVER_COLOR))
        btn.bind("<Leave>", lambda e: btn.configure(text_color="black", fg_color="#ffffff"))

        return btn

    def _crear_separador(self, col):
        sep = ctk.CTkLabel(self, text="│", font=(
            "Arial", 22), text_color="#B0B0B0")
        sep.grid(row=0, column=col, padx=5)
