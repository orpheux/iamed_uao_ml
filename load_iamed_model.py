#!/usr/bin/env python3
"""
Script de demostración para cargar y usar el modelo IAMED
Sistema de homologación automática de medicamentos

Uso:
    python load_iamed_model.py
    
Autor: Sistema de Homologación IAMED
Fecha: 2025
"""

import pickle
import os


def cargar_modelo_iamed(ruta_modelo='./models/iamed.pkl'):
    """
    Función para cargar el modelo de homologación de medicamentos.

    Args:
        ruta_modelo (str): Ruta al archivo del modelo guardado

    Returns:
        dict: Diccionario con el modelo y sistema de recomendación

    Raises:
        FileNotFoundError: Si el modelo no existe
        Exception: Si hay error al cargar el modelo
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(
            f"❌ El modelo no existe en la ruta: {ruta_modelo}")

    print(f"📂 Cargando modelo IAMED desde: {ruta_modelo}")
    print(
        f"📊 Tamaño del archivo: {os.path.getsize(ruta_modelo) / (1024*1024):.2f} MB")

    try:
        with open(ruta_modelo, 'rb') as f:
            modelo_completo = pickle.load(f)

        print("✅ Modelo cargado exitosamente")
        print(f"📊 Versión: {modelo_completo.get('version', 'N/A')}")
        print(
            f"📅 Fecha entrenamiento: {modelo_completo.get('timestamp', 'N/A')}")
        print(f"📝 Descripción: {modelo_completo.get('descripcion', 'N/A')}")
        return modelo_completo

    except Exception as e:
        raise RuntimeError(f"❌ Error al cargar el modelo: {str(e)}") from e


def demo_recomendaciones(sistema_recomendacion, cums_prueba=None):
    """
    Función de demostración del sistema de recomendaciones.

    Args:
        sistema_recomendacion: Sistema de recomendación cargado
        cums_prueba (list): Lista de CUMs para probar
    """
    if cums_prueba is None:
        cums_prueba = ['2203-1', '15100-1', '19959949-14']

    print("\n" + "="*80)
    print("🧪 DEMOSTRACIÓN DEL SISTEMA DE RECOMENDACIONES")
    print("="*80)

    for i, cum in enumerate(cums_prueba, 1):
        print(f"\n🎬 PRUEBA {i}/{len(cums_prueba)}: CUM {cum}")
        print("-" * 60)

        try:
            resultado = sistema_recomendacion.recomendar_homologos(
                cum,
                n_recomendaciones=3,
                score_minimo=0.8
            )

            if resultado['encontrado']:
                print(
                    f"✅ Se encontraron {len(resultado['recomendaciones'])} recomendaciones")
                print(
                    f"📊 Candidatos evaluados: {resultado['candidatos_evaluados']}")

                for j, rec in enumerate(resultado['recomendaciones'], 1):
                    print(f"   {j}. {rec['producto']} (CUM: {rec['cum']})")
                    print(f"      ⭐ Score: {rec['score_similitud']:.1%}")
                    print(f"      📝 {rec['motivo']}")
            else:
                print(f"❌ No se encontraron recomendaciones")
                print(f"📝 Razón: {resultado.get('error', 'Sin información')}")

        except Exception as e:
            print(f"❌ Error procesando CUM {cum}: {str(e)}")


def main():
    """Función principal de demostración."""
    print("🔬 SISTEMA DE HOMOLOGACIÓN AUTOMÁTICA DE MEDICAMENTOS - IAMED")
    print("="*70)
    print("📊 Cargador y demostración del modelo entrenado")
    print("🎯 Versión: 1.0.0")
    print("="*70)

    try:
        # Cargar modelo
        modelo_completo = cargar_modelo_iamed()

        # Obtener sistema de recomendación
        sistema = modelo_completo['sistema_recomendacion']

        # Ejecutar demostración
        demo_recomendaciones(sistema)

        print("\n" + "="*80)
        print("🎉 DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("="*80)
        print("💡 Puedes usar el sistema de la siguiente manera:")
        print()
        print("   modelo = cargar_modelo_iamed()")
        print("   sistema = modelo['sistema_recomendacion']")
        print("   resultado = sistema.recomendar_homologos('TU-CUM-AQUI')")
        print("   sistema.mostrar_resultado_bonito(resultado)")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\n❌ ARCHIVO NO ENCONTRADO:")
        print(f"   {str(e)}")
        print(f"\n💡 SOLUCIÓN:")
        print(f"   1. Ejecuta primero el notebook p04_training.ipynb")
        print(f"   2. Asegúrate de que se haya creado el archivo ./models/iamed.pkl")
        print(f"   3. Ejecuta este script desde el directorio correcto")

    except Exception as e:
        print(f"\n❌ ERROR INESPERADO:")
        print(f"   {str(e)}")
        print(f"\n💡 CONTACTO:")
        print(f"   Revisa los logs y la documentación del sistema")


if __name__ == "__main__":
    main()
