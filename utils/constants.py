"""
constants.py
Constantes generales del proyecto IAMED.
Ubicado en utils/constants.py
"""

import os

# Tamaño base de la app
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600

# Rutas importantes
ASSETS_PATH = os.path.join("assets")
LOGO_PATH = os.path.join(ASSETS_PATH, "logo.png")
LOGO_TX_PATH = os.path.join(ASSETS_PATH, "logo_tx.png")
DATA_PATH = os.path.join("data")
OUTPUT_PATH = os.path.join("output")
MODEL_PATH = os.path.join("models", "iamed.pkl")

# Colores y temas (pueden ampliarse luego)
PRIMARY_COLOR = "#2aa9d6"
SECONDARY_COLOR = "#AED6F1"
LIGHT_BG = "#FFFFFF"
DARK_BG = "#1E1E1E"

BUTTON_BG_COLOR = PRIMARY_COLOR
BUTTON_HOVER_COLOR = "#1a6c89"
BUTTON_FONT = ("Arial", 16, "normal")
EXITO_COLOR = "#1BA94C"  # Verde oscuro elegante


# Configuración de UI
NAV_BUTTON_WIDTH = 140
NAV_BUTTON_HEIGHT = 40
ROUND_BUTTON_RADIUS = 30

# Fuentes
TITLE_FONT = ("Arial Black", 36)
BUTTON_FONT = ("Arial", 16)
ICON_FONT = ("Arial", 24)
PLACEHOLDER_FONT = ("Arial Black", 64)
