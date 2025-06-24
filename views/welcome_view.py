"""
welcome_view.py
Vista de bienvenida de la aplicación IAMED.
"""

import customtkinter as ctk
from PIL import Image

from utils.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    LOGO_PATH,
    BUTTON_BG_COLOR,
    BUTTON_HOVER_COLOR,
    ROUND_BUTTON_RADIUS,
    ICON_FONT
)


class WelcomeScreen(ctk.CTk):

    def __init__(self, on_continue=None):
        super().__init__()
        self.on_continue = on_continue

        self.title("IAMED - Bienvenido")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.resizable(False, False)

        ctk.set_appearance_mode("light")
        self.configure(bg_color="white")

        original = Image.open(LOGO_PATH)
        logo_size = original.size  # (ancho, alto)
        self.logo_img = ctk.CTkImage(light_image=original, size=logo_size)
        self.logo_label = ctk.CTkLabel(self, image=self.logo_img, text="")
        self.logo_label.place(relx=0.5, rely=0.4, anchor="center")

        self.boton_ingresar = ctk.CTkButton(
            master=self,
            text=">",
            width=60,
            height=60,
            corner_radius=ROUND_BUTTON_RADIUS,
            font=ICON_FONT,
            fg_color=BUTTON_BG_COLOR,
            hover_color=BUTTON_HOVER_COLOR,
            text_color="white",
            command=self.ingresar_app
        )
        self.boton_ingresar.place(relx=0.95, rely=0.93, anchor="se")

    def ingresar_app(self):
        self.destroy()
        if self.on_continue:
            self.on_continue()
