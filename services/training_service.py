"""
🎯 SERVICIO DE ENTRENAMIENTO PARA HOMOLOGACIÓN DE MEDICAMENTOS
============================================================
Servicio basado EXACTAMENTE en p04_training.ipynb con KMeans completo.
Flujo: parquet → KMeans + clustering → modelo.pkl

Author: Sistema IAMED
Version: 1.0.0
"""

import os
import pickle
import warnings
from typing import Tuple, Optional

import polars as pl
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
warnings.filterwarnings('ignore')


class HomologacionClusteringModel:
    """
    Modelo de clustering especializado para homologación de medicamentos.
    BASADO EXACTAMENTE EN p04_training.ipynb

    Estrategia:
    1. Clustering primario por ATC+VÍA (variables críticas)
    2. Clustering secundario por características específicas con KMEANS
    3. Sistema de similitud con pesos jerárquicos
    """

    def __init__(self, pesos_jerarquicos: Optional[dict] = None):
        """Inicializa el modelo de clustering para homologación."""
        self.pesos_jerarquicos = pesos_jerarquicos or {
            'ATC': 0.40,                    # 40% - CRÍTICO
            'VIA_ADMINISTRACION': 0.30,     # 30% - CRÍTICO
            'FORMA_FARMACEUTICA': 0.20,     # 20% - IMPORTANTE
            'CANTIDAD_SIMILITUD': 0.10,     # 10% - IMPORTANTE
            'PRINCIPIO_BONUS': 0.00         # 0% base + bonus variable
        }

        # Modelos de clustering
        self.cluster_primario = None
        self.cluster_secundario = None
        self.scaler = StandardScaler()
        self.knn_model = None

        # Datos de entrenamiento
        self.df_validos = None
        self.df_referencia = None
        self.features_clustering = None
        self.datos_escalados = None

        # Mapeos y vocabularios
        self.combo_clusters = {}
        self.cluster_info = {}

    def preparar_features_clustering(self, df: pl.DataFrame) -> tuple:
        """Prepara las features optimizadas para clustering de homologación."""
        # FEATURES CRÍTICAS (deben coincidir exactamente para homologación)
        features_criticas = [
            'ATC_label',
            'VÍA ADMINISTRACIÓN_label',
            'PRINCIPIO ACTIVO_label'
        ]

        # FEATURES IMPORTANTES (permiten variación con penalización)
        features_importantes = [
            'FORMA FARMACÉUTICA_label',
            'CANTIDAD_log',
            'CANTIDAD_CUM_log',
            'RATIO_CANTIDAD',
            'CANTIDAD_bin',
            'UNIDAD MEDIDA_label'
        ]

        # FEATURES DE PROBABILIDAD (para scoring)
        features_probabilidad = [
            'ATC_prob_validos',
            'VÍA ADMINISTRACIÓN_prob_validos',
            'PRINCIPIO ACTIVO_prob_validos',
            'score_prob_critica'
        ]

        # Verificar disponibilidad
        features_criticas_disp = [
            f for f in features_criticas if f in df.columns]
        features_importantes_disp = [
            f for f in features_importantes if f in df.columns]
        features_prob_disp = [
            f for f in features_probabilidad if f in df.columns]

        # Combinar todas las features
        todas_features = features_criticas_disp + \
            features_importantes_disp + features_prob_disp

        return features_criticas_disp, features_importantes_disp, todas_features

    def clustering_por_combo_critico(self, df_validos: pl.DataFrame) -> tuple:
        """Realiza clustering primario agrupando por combinaciones críticas ATC+VÍA."""
        # Analizar combinaciones ATC+VÍA
        combos_info = (df_validos
                       .group_by('ATC_VIA_combo')
                       .agg([
                           pl.len().alias('count'),
                           pl.col('ATC').first().alias('ATC_ejemplo'),
                           pl.col('VÍA ADMINISTRACIÓN').first().alias(
                               'VIA_ejemplo'),
                           pl.col('PRINCIPIO ACTIVO').n_unique().alias(
                               'principios_unicos')
                       ])
                       .sort('count', descending=True))

        # Asignar cluster_id a cada combinación
        combo_clusters = {}
        cluster_info = {}

        for idx in range(combos_info.height):
            row = combos_info.row(idx)
            combo = row[0]
            count = row[1]
            atc = row[2]
            via = row[3]
            principios = row[4]

            combo_clusters[combo] = idx
            cluster_info[idx] = {
                'combo': combo,
                'count': count,
                'atc': atc,
                'via': via,
                'principios_unicos': principios,
                'cluster_id': idx
            }

        return combo_clusters, cluster_info

    def clustering_secundario_por_similitud(self, df_validos: pl.DataFrame, features_importantes: list) -> None:
        """Aplica clustering secundario dentro de cada grupo ATC+VÍA con KMEANS."""
        # Preparar datos para clustering secundario
        features_disponibles = [
            f for f in features_importantes if f in df_validos.columns]

        # Extraer datos numéricos para clustering (convertir a numpy para sklearn)
        datos_clustering = df_validos.select(features_disponibles).to_numpy()

        # Escalar datos
        datos_escalados = self.scaler.fit_transform(datos_clustering)

        # Determinar número óptimo de clusters usando método del codo
        # Máximo 50 clusters o 1 cada 10 medicamentos
        max_k = min(50, len(datos_escalados) // 10)

        if max_k >= 2:
            # Probar diferentes valores de k
            inertias = []
            # Limitar a máximo 20 para eficiencia
            k_range = range(2, min(max_k + 1, 21))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(datos_escalados)
                inertias.append(kmeans.inertia_)

            # Elegir k usando método del codo simplificado
            if len(inertias) > 2:
                # Calcular second differences para encontrar el "codo"
                diffs = np.diff(inertias)
                second_diffs = np.diff(diffs)
                k_optimo = k_range[np.argmin(second_diffs) + 1]
            else:
                k_optimo = k_range[0]

            # Entrenar modelo final KMEANS
            self.cluster_secundario = KMeans(
                n_clusters=k_optimo, random_state=42, n_init=10)
            clusters_secundarios = self.cluster_secundario.fit_predict(
                datos_escalados)

            # Evaluar calidad si hay suficientes datos
            if len(datos_escalados) > k_optimo:
                _ = silhouette_score(datos_escalados, clusters_secundarios)

        else:
            clusters_secundarios = np.zeros(len(datos_escalados))
            k_optimo = 1

        # Entrenar modelo KNN para búsqueda de similares
        self.knn_model = NearestNeighbors(n_neighbors=min(
            20, len(datos_escalados)), metric='cosine')
        self.knn_model.fit(datos_escalados)

        # Guardar datos para uso posterior
        self.datos_escalados = datos_escalados
        self.features_clustering = features_disponibles

    def fit(self, df_training: pl.DataFrame) -> None:
        """Entrena el modelo completo de clustering para homologación."""
        # Filtrar solo medicamentos válidos para entrenamiento
        self.df_validos = df_training.filter(pl.col('VALIDO') == 1)

        # Preparar features
        _, features_importantes, _ = self.preparar_features_clustering(self.df_validos)

        # Clustering primario por combinaciones críticas
        self.combo_clusters, self.cluster_info = self.clustering_por_combo_critico(self.df_validos)

        # Clustering secundario por similitud con KMEANS
        self.clustering_secundario_por_similitud(self.df_validos, features_importantes)

        # Guardar dataset de referencia (todos los datos para búsqueda)
        self.df_referencia = df_training

    def obtener_info_medicamento(self, cum: str) -> Optional[dict]:
        """Obtiene información completa de un medicamento por su CUM."""
        if self.df_referencia is None:
            return None

        medicamento = self.df_referencia.filter(pl.col('CUM') == cum)

        if medicamento.height == 0:
            return None

        # Convertir a diccionario
        info = medicamento.to_pandas().iloc[0].to_dict()
        return info


class SistemaRecomendacionHomologos:
    """Sistema completo de recomendación de medicamentos homólogos basado en p04_training."""

    def __init__(self, modelo_clustering: HomologacionClusteringModel):
        """Inicializa el sistema de recomendación."""
        self.modelo = modelo_clustering
        self.pesos = modelo_clustering.pesos_jerarquicos

    def calcular_score_similitud(self, medicamento_origen: dict, medicamento_candidato: dict) -> Tuple[float, dict]:
        """Calcula score de similitud entre medicamento origen y candidato."""
        detalle = {}
        score_total = 0.0

        # 1. ATC (40% - CRÍTICO)
        if medicamento_origen['ATC'] == medicamento_candidato['ATC']:
            score_atc = 1.0
            detalle['ATC'] = {'coincide': True,'score': 1.0, 'peso': self.pesos['ATC']}
        else:
            score_atc = 0.0
            detalle['ATC'] = {'coincide': False,'score': 0.0, 'peso': self.pesos['ATC']}

        score_total += score_atc * self.pesos['ATC']

        # 2. VÍA ADMINISTRACIÓN (30% - CRÍTICO)
        if medicamento_origen['VÍA ADMINISTRACIÓN'] == medicamento_candidato['VÍA ADMINISTRACIÓN']:
            score_via = 1.0
            detalle['VIA'] = {'coincide': True, 'score': 1.0,'peso': self.pesos['VIA_ADMINISTRACION']}
        else:
            score_via = 0.0
            detalle['VIA'] = {'coincide': False, 'score': 0.0,'peso': self.pesos['VIA_ADMINISTRACION']}

        score_total += score_via * self.pesos['VIA_ADMINISTRACION']

        # 3. PRINCIPIO ACTIVO (BONUS - MÁS FLEXIBLE)
        if medicamento_origen['PRINCIPIO ACTIVO'] == medicamento_candidato['PRINCIPIO ACTIVO']:
            bonus_principio = 0.15  # 15% bonus si coincide exactamente
            detalle['PRINCIPIO_BONUS'] = {'bonus': 0.15, 'tipo': 'exacto'}
        elif any(palabra in medicamento_candidato['PRINCIPIO ACTIVO'] for palabra in medicamento_origen['PRINCIPIO ACTIVO'].split() if len(palabra) > 3):
            bonus_principio = 0.10  # 10% bonus si coincide parcialmente
            detalle['PRINCIPIO_BONUS'] = {'bonus': 0.10, 'tipo': 'parcial'}
        else:
            bonus_principio = 0.0   # Sin bonus pero NO penaliza
            detalle['PRINCIPIO_BONUS'] = {'bonus': 0.0, 'tipo': 'diferente'}

        score_total += bonus_principio  # Se suma ENCIMA del 100%

        # 4. FORMA FARMACÉUTICA (20% - IMPORTANTE)
        if medicamento_origen['FORMA FARMACÉUTICA'] == medicamento_candidato['FORMA FARMACÉUTICA']:
            score_forma = 1.0
            detalle['FORMA'] = {'coincide': True, 'score': 1.0,
                                'peso': self.pesos['FORMA_FARMACEUTICA']}
        else:
            score_forma = 0.5  # Penalización menor para forma farmacéutica
            detalle['FORMA'] = {'coincide': False, 'score': 0.5,
                                'peso': self.pesos['FORMA_FARMACEUTICA']}

        score_total += score_forma * self.pesos['FORMA_FARMACEUTICA']

        # 5. SIMILITUD DE CANTIDAD (10% - IMPORTANTE)
        cantidad_origen = medicamento_origen.get('CANTIDAD', 0)
        cantidad_candidato = medicamento_candidato.get('CANTIDAD', 0)

        if cantidad_origen > 0 and cantidad_candidato > 0:
            ratio = min(cantidad_origen, cantidad_candidato) / \
                max(cantidad_origen, cantidad_candidato)
            if ratio < 0.5:
                score_cantidad = 0.1
            elif ratio < 0.8:
                score_cantidad = 0.4
            else:
                score_cantidad = ratio
        else:
            score_cantidad = 0.3

        detalle['CANTIDAD'] = {
            'origen': cantidad_origen,
            'candidato': cantidad_candidato,
            'score': score_cantidad,
            'peso': self.pesos['CANTIDAD_SIMILITUD']
        }

        score_total += score_cantidad * self.pesos['CANTIDAD_SIMILITUD']

        return score_total, detalle

    def recomendar_homologos(self, cum_origen: str, n_recomendaciones: int = 5, score_minimo: float = 0.85) -> dict:
        """Encuentra medicamentos homólogos para un CUM dado."""
        # 1. Obtener información del medicamento origen
        medicamento_origen = self.modelo.obtener_info_medicamento(cum_origen)

        if medicamento_origen is None:
            return {
                'cum_origen': cum_origen,
                'encontrado': False,
                'error': 'CUM no encontrado en el dataset',
                'recomendaciones': []
            }

        # 2. Verificar si tiene combinación ATC+VÍA válida
        combo_origen = f"{medicamento_origen['ATC_label']}_{medicamento_origen['VÍA ADMINISTRACIÓN_label']}"

        if combo_origen not in self.modelo.combo_clusters:
            return {
                'cum_origen': cum_origen,
                'encontrado': False,
                'error': f'No hay medicamentos válidos con la combinación ATC+VÍA: {medicamento_origen["ATC"]} + {medicamento_origen["VÍA ADMINISTRACIÓN"]}',
                'recomendaciones': []
            }

        # 3. Buscar candidatos con la misma combinación ATC+VÍA
        if self.modelo.df_validos is None:
            return {
                'cum_origen': cum_origen,
                'encontrado': False,
                'error': 'Modelo no entrenado',
                'recomendaciones': []
            }

        candidatos = (self.modelo.df_validos
                      .filter(
                          (pl.col('ATC') == medicamento_origen['ATC']) &
                          (pl.col('VÍA ADMINISTRACIÓN') == medicamento_origen['VÍA ADMINISTRACIÓN']) &
                          # Excluir el medicamento origen
                          (pl.col('CUM') != cum_origen)
                      ))

        if candidatos.height == 0:
            return {
                'cum_origen': cum_origen,
                'encontrado': False,
                'error': f'No hay otros medicamentos válidos con ATC: {medicamento_origen["ATC"]}, VÍA: {medicamento_origen["VÍA ADMINISTRACIÓN"]}',
                'recomendaciones': []
            }

        # 4. Calcular scores de similitud para cada candidato
        recomendaciones = []

        for idx in range(candidatos.height):
            candidato = candidatos.row(idx, named=True)
            candidato_dict = dict(candidato)

            # Calcular score
            score, _ = self.calcular_score_similitud(medicamento_origen, candidato_dict)

            # Solo incluir si supera el score mínimo
            if score >= score_minimo:
                recomendaciones.append({
                    'cum': candidato_dict['CUM'],
                    'producto': candidato_dict['PRODUCTO'],
                    'atc': candidato_dict['ATC'],
                    'via': candidato_dict['VÍA ADMINISTRACIÓN'],
                    'principio_activo': candidato_dict['PRINCIPIO ACTIVO'],
                    'forma_farmaceutica': candidato_dict['FORMA FARMACÉUTICA'],
                    'cantidad': candidato_dict['CANTIDAD'],
                    'unidad': candidato_dict['UNIDAD MEDIDA'],
                    'score_similitud': round(score, 4)
                })

        # 5. Ordenar por score y limitar número
        recomendaciones.sort(key=lambda x: x['score_similitud'], reverse=True)
        recomendaciones = recomendaciones[:n_recomendaciones]

        return {
            'cum_origen': cum_origen,
            'medicamento_origen': {
                'producto': medicamento_origen['PRODUCTO'],
                'atc': medicamento_origen['ATC'],
                'via': medicamento_origen['VÍA ADMINISTRACIÓN'],
                'principio_activo': medicamento_origen['PRINCIPIO ACTIVO'],
                'es_valido': medicamento_origen['VALIDO'] == 1
            },
            'encontrado': len(recomendaciones) > 0,
            'candidatos_evaluados': candidatos.height,
            'recomendaciones': recomendaciones,
            'parametros': {
                'score_minimo': score_minimo,
                'n_recomendaciones': n_recomendaciones
            }
        }


class TrainingService:
    """
    Servicio principal para entrenar el modelo completo de homologación.
    BASADO 100% EN p04_training.ipynb
    """

    def __init__(self):
        self.modelo_clustering = None
        self.sistema_recomendacion = None

    def entrenar_modelo_completo(self, df_training: pl.DataFrame) -> dict:
        """Entrena el modelo completo (clustering + sistema de recomendación)."""
        print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO DE CLUSTERING")
        print("=" * 70)

        # 1. Entrenar modelo de clustering con KMEANS
        print("🎯 Paso 1: Entrenando modelo de clustering...")
        self.modelo_clustering = HomologacionClusteringModel()
        self.modelo_clustering.fit(df_training)
        print("✅ Modelo de clustering entrenado exitosamente")

        # 2. Crear sistema de recomendación
        print("🎯 Paso 2: Creando sistema de recomendación...")
        self.sistema_recomendacion = SistemaRecomendacionHomologos(
            self.modelo_clustering)
        print("✅ Sistema de recomendación creado")

        # 3. Crear modelo completo
        print("🎯 Paso 3: Empaquetando modelo completo...")
        modelo_completo = {
            'modelo_clustering': self.modelo_clustering,
            'sistema_recomendacion': self.sistema_recomendacion,
            'timestamp': pd.Timestamp.now(),
            'descripcion': 'Sistema completo de homologación de medicamentos con clustering jerárquico',
            'version': '1.0.0',
            'stats': {
                'medicamentos_validos': self.modelo_clustering.df_validos.height if self.modelo_clustering.df_validos is not None else 0,
                'clusters_primarios': len(self.modelo_clustering.combo_clusters)
            }
        }

        print("✅ Modelo completo creado:")
        print(
            f"   📊 Medicamentos válidos: {modelo_completo['stats']['medicamentos_validos']:,}")
        print(
            f"   🎯 Clusters primarios: {modelo_completo['stats']['clusters_primarios']:,}")

        return modelo_completo

    def guardar_modelo(self, modelo_completo: dict, output_path: str) -> None:
        """Guarda el modelo entrenado en formato pickle."""
        print("💾 GUARDANDO MODELO ENTRENADO")
        print("=" * 40)

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"📁 Directorio: {os.path.dirname(output_path)}")
        print(f"📂 Archivo: {os.path.basename(output_path)}")

        # Guardar usando pickle
        with open(output_path, 'wb') as f:
            pickle.dump(modelo_completo, f)

        # Verificar que se guardó correctamente
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)  # MB
            print("✅ Modelo guardado exitosamente!")
            print(f"📊 Tamaño del archivo: {file_size:.2f} MB")
            print(f"📍 Ruta completa: {os.path.abspath(output_path)}")
        else:
            print("❌ ERROR: El modelo no se guardó correctamente")
            raise RuntimeError(
                f"No se pudo guardar el modelo en: {output_path}")

    def cargar_modelo(self, model_path: str) -> dict:
        """Carga un modelo previamente entrenado."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"El modelo no existe en la ruta: {model_path}")

        with open(model_path, 'rb') as f:
            modelo_completo = pickle.load(f)

        return modelo_completo

    def entrenar_desde_archivo(self, input_path: str, output_path: str) -> dict:
        """Entrena modelo completo desde archivo."""
        print("🎯 INICIANDO ENTRENAMIENTO DESDE ARCHIVO")
        print("=" * 50)
        print(f"📥 Archivo de entrada: {input_path}")
        print(f"📤 Archivo de salida: {output_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        # Cargar dataset
        print("📊 Cargando dataset de entrenamiento...")
        df_training = pl.read_parquet(input_path)
        print(
            f"✅ Dataset cargado: {df_training.height:,} registros, {df_training.width} columnas")

        # Entrenar modelo completo
        modelo_completo = self.entrenar_modelo_completo(df_training)

        # Guardar modelo
        self.guardar_modelo(modelo_completo, output_path)

        print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE!")
        print("="*50)

        return modelo_completo


# Funciones utilitarias para uso directo
def entrenar_modelo_homologacion(input_path: str, output_path: str) -> dict:
    """
    Función directa: lee parquet, entrena con KMEANS, guarda pkl.
    BASADA EN p04_training.ipynb
    """
    service = TrainingService()
    return service.entrenar_desde_archivo(input_path, output_path)


def cargar_modelo_homologacion(model_path: str) -> dict:
    """Carga modelo entrenado para hacer búsquedas."""
    service = TrainingService()
    return service.cargar_modelo(model_path)


def buscar_homologos(modelo_completo: dict, cum: str, n_recomendaciones: int = 5) -> dict:
    """Busca homólogos usando modelo cargado."""
    sistema = modelo_completo['sistema_recomendacion']
    return sistema.recomendar_homologos(cum, n_recomendaciones)


if __name__ == "__main__":
    # Ejemplo de uso basado en p04_training
    INPUT_FILE = "./data/dataset_entrenamiento_homologacion.parquet"
    OUTPUT_FILE = "./models/iamed.pkl"

    print("🎯 EJEMPLO DE ENTRENAMIENTO COMPLETO")
    print("=" * 50)

    try:
        # Entrenar modelo con KMEANS
        print("🚀 Iniciando entrenamiento...")
        modelo = entrenar_modelo_homologacion(INPUT_FILE, OUTPUT_FILE)
        print("✅ Modelo entrenado exitosamente!")
        print(f"📊 Stats: {modelo['stats']}")

        # Probar búsqueda
        print("\n🔍 Probando búsqueda de homólogos...")
        resultado = buscar_homologos(modelo, "2203-1", 3)
        print(f"✅ Homólogos encontrados: {len(resultado['recomendaciones'])}")

        for i, rec in enumerate(resultado['recomendaciones'], 1):
            print(
                f"   {i}. {rec['producto']} (Score: {rec['score_similitud']:.1%})")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que el archivo de datos existe en la ruta especificada")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        print("💡 Revisa que todos los datos y dependencias estén disponibles")
