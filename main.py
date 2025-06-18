import tkinter as tk
from tkinter import messagebox
from functions import predict_species

root = tk.Tk()
root.title("Clasificador de Flores Iris")
root.geometry("400x400")

# Etiqueta principal
tk.Label(root, text="Clasificador de Iris", font=("Arial", 20)).pack(pady=10)

# Entradas de características
entries = {}
fields = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

for field in fields:
    frame = tk.Frame(root)
    frame.pack(pady=5)
    label = tk.Label(frame, text=field + ": ", width=20, anchor='w')
    label.pack(side='left')
    entry = tk.Entry(frame)
    entry.pack(side='left')
    entries[field] = entry

# Función de predicción y despliegue de resultado
def classify():
    try:
        values = [entries[field].get() for field in fields]
        if not all(values):
            raise ValueError("Faltan datos")
        result = predict_species(*values)
        messagebox.showinfo("Resultado", f"Clase predicha: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Verifica las entradas: {e}")

# Botón para clasificar
tk.Button(root, text="Clasificar", command=classify, font=("Arial", 14)).pack(pady=20)

root.mainloop()
