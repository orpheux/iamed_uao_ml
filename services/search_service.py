"""
🔍 SERVICIO DE BÚSQUEDA DE HOMÓLOGOS
====================================
Servicio limpio que SOLO carga modelo y busca homólogos.
Flujo: pkl → buscar → resultados

Author: Sistema IAMED
Version: 1.0.0
"""

import os
import pickle
from typing import Dict, List

# Importar las clases necesarias del training_service para deserializar el modelo
from services.training_service import HomologacionClusteringModel, SistemaRecomendacionHomologos


class SearchService:
    """
    Servicio para buscar medicamentos homólogos usando modelo pre-entrenado.
    SOLO LEE MODELO Y BUSCA - NADA MÁS.
    """

    def __init__(self, model_path: str = "./models/iamed.pkl"):
        """
        Inicializa el servicio de búsqueda.

        Args:
            model_path (str): Ruta al modelo entrenado (.pkl)
        """
        self.model_path = model_path
        self.modelo_completo = None
        self.sistema_recomendacion = None
        self._cargar_modelo()

    def _cargar_modelo(self) -> None:
        """Carga el modelo entrenado desde el archivo pkl."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        print(f"📂 Cargando modelo desde: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            self.modelo_completo = pickle.load(f)

        self.sistema_recomendacion = self.modelo_completo['sistema_recomendacion']

        print("✅ Modelo cargado exitosamente")
        print(f"📊 Versión: {self.modelo_completo.get('version', 'N/A')}")
        print(f"📅 Entrenado: {self.modelo_completo.get('timestamp', 'N/A')}")
        print(f"📋 Stats: {self.modelo_completo.get('stats', {})}")

    def buscar_homologos(self, cum_code: str, n_recomendaciones: int = 5, score_minimo: float = 0.85) -> dict:
        """
        Busca medicamentos homólogos para un CUM específico.

        Args:
            cum_code (str): Código CUM del medicamento
            n_recomendaciones (int): Número máximo de recomendaciones
            score_minimo (float): Score mínimo para considerar homólogo

        Returns:
            dict: Resultado con homólogos encontrados
        """
        if self.sistema_recomendacion is None:
            raise RuntimeError("Modelo no cargado correctamente")

        return self.sistema_recomendacion.recomendar_homologos(
            cum_code, n_recomendaciones, score_minimo
        )

    def buscar_multiple(self, cums: List[str], n_recomendaciones: int = 5) -> Dict[str, dict]:
        """
        Busca homólogos para múltiples CUMs.

        Args:
            cums (List[str]): Lista de CUMs a buscar
            n_recomendaciones (int): Número máximo de recomendaciones por CUM

        Returns:
            Dict[str, dict]: Resultados por cada CUM
        """
        resultados = {}

        for cum_code in cums:
            print(f"🔍 Buscando homólogos para: {cum_code}")
            try:
                resultado = self.buscar_homologos(cum_code, n_recomendaciones)
                resultados[cum_code] = resultado

                if resultado['encontrado']:
                    print(
                        f"✅ Encontrados: {len(resultado['recomendaciones'])} homólogos")
                else:
                    print(f"❌ Sin homólogos: {resultado.get('error', 'N/A')}")

            except Exception as e:
                print(f"❌ Error en {cum_code}: {e}")
                resultados[cum_code] = {'error': str(e), 'encontrado': False}

        return resultados

    def mostrar_resultado(self, resultado: dict, mostrar_detalle: bool = True) -> None:
        """
        Muestra el resultado de búsqueda en formato legible.

        Args:
            resultado (dict): Resultado de búsqueda
            mostrar_detalle (bool): Si mostrar detalles completos
        """
        print("\n" + "="*60)
        print(f"🔍 BÚSQUEDA PARA CUM: {resultado['cum_origen']}")
        print("="*60)

        if not resultado['encontrado']:
            print(f"❌ ERROR: {resultado.get('error', 'Sin error específico')}")
            return

        # Medicamento origen
        origen = resultado['medicamento_origen']
        print( "📋 MEDICAMENTO ORIGEN:")
        print(f"   🔸 Producto: {origen['producto']}")
        print(f"   🔸 ATC: {origen['atc']}")
        print(f"   🔸 Vía: {origen['via']}")
        print(f"   🔸 Principio: {origen['principio_activo']}")

        # Recomendaciones
        recomendaciones = resultado['recomendaciones']
        print(f"\n✅ HOMÓLOGOS ENCONTRADOS: {len(recomendaciones)}")

        for i, rec in enumerate(recomendaciones, 1):
            print(f"\n{i}. 📦 {rec['producto']}")
            print(f"   🆔 CUM: {rec['cum']}")
            print(f"   ⭐ Score: {rec['score_similitud']:.1%}")

            if mostrar_detalle:
                print(f"   💊 ATC: {rec['atc']}")
                print(f"   🚪 Vía: {rec['via']}")
                print(f"   🧪 Principio: {rec['principio_activo']}")
                print(f"   💊 Forma: {rec['forma_farmaceutica']}")
                print(f"   📏 Cantidad: {rec['cantidad']} {rec['unidad']}")

            if i < len(recomendaciones):
                print("   " + "-"*40)


# Funciones utilitarias para uso directo
def buscar_homologos_directo(cum_code: str, model_path: str = "./models/iamed.pkl", n_recomendaciones: int = 5) -> dict:
    """
    Función directa: carga modelo, busca homólogos, devuelve resultado.

    Args:
        cum_code (str): CUM a buscar
        model_path (str): Ruta del modelo
        n_recomendaciones (int): Número de recomendaciones

    Returns:
        dict: Resultado de búsqueda
    """
    service = SearchService(model_path)
    return service.buscar_homologos(cum_code, n_recomendaciones)


def buscar_y_mostrar(cum_code: str, model_path: str = "./models/iamed.pkl", n_recomendaciones: int = 5) -> None:
    """
    Función directa: busca homólogos y muestra resultado en pantalla.

    Args:
        cum_code (str): CUM a buscar
        model_path (str): Ruta del modelo
        n_recomendaciones (int): Número de recomendaciones
    """
    buscador = SearchService(model_path)
    resultado = buscador.buscar_homologos(cum_code, n_recomendaciones)
    buscador.mostrar_resultado(resultado)


if __name__ == "__main__":
    # Ejemplos de uso
    MODEL_FILE = "./models/iamed.pkl"

    print("🔍 SERVICIO DE BÚSQUEDA DE HOMÓLOGOS")
    print("=" * 50)

    try:
        # Crear servicio de búsqueda
        search_service = SearchService(MODEL_FILE)

        # Ejemplos de CUMs para buscar
        cums_ejemplo = ["2203-1", "15100-1", "10045-1"]

        # Buscar homólogos para cada CUM
        for cum in cums_ejemplo:
            resultado_busqueda = search_service.buscar_homologos(cum, 3)
            search_service.mostrar_resultado(resultado_busqueda, mostrar_detalle=False)
            print("\n" + "="*60)

        print("\n🎉 ¡BÚSQUEDA COMPLETADA!")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Primero entrena el modelo con training_service.py")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
