"""
main.py
Punto de entrada principal de la aplicación IAMED.
Encargado de orquestar las vistas.
"""

import customtkinter as ctk
from views.welcome_view import WelcomeScreen
from views.main_layout import MainApp


def iniciar_app():
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    welcome = WelcomeScreen(on_continue=iniciar_app)
    welcome.mainloop()
