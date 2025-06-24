
import os
import warnings
from typing import Optional
import polars as pl

warnings.filterwarnings('ignore')


class MedicamentosEncodingService:
    """
    Servicio completo de encoding para medicamentos con priorización jerárquica.

    Estrategia de Encoding:
    1. Variables CRÍTICAS: Encoding directo + penalización por no coincidencia
    2. Variables IMPORTANTES: Encoding con tolerancia y scoring gradual
    3. Variables NUMÉRICAS: Normalización + cálculo de similitud

    Examples:
        >>> service = MedicamentosEncodingService()
        >>> df_encoded = service.process_full_encoding(df_medicamentos)
    """

    def __init__(self):
        """Inicializa el servicio con configuraciones por defecto."""
        # Definir columnas según su propósito
        self.columnas_entrenamiento = [
            'ATC',                    # CRÍTICO - Código terapéutico
            'VÍA ADMINISTRACIÓN',     # CRÍTICO - Vía de administración
            'PRINCIPIO ACTIVO',       # CRÍTICO - Sustancia activa
            'FORMA FARMACÉUTICA',     # IMPORTANTE - Presentación
            'CANTIDAD CUM',           # IMPORTANTE - Dosis/concentración
            'CANTIDAD',               # IMPORTANTE - Cantidad total
            'UNIDAD MEDIDA',          # IMPORTANTE - Unidad de medida
        ]

        self.columnas_informativas = [
            'CUM',                    # Identificador único (entrada/salida)
            # Nombre comercial (solo para mostrar al usuario)
            'PRODUCTO',
            'EXPEDIENTE CUM',         # Información administrativa
            'DESCRIPCIÓN_ATC',        # Solo para consulta humana
        ]

        self.columna_filtro = 'VALIDO'    # Para filtrar medicamentos válidos

        # Pesos jerárquicos para el scoring final
        self.pesos_jerarquicos = {
            'ATC': 0.40,                    # 40% - CRÍTICO
            'VIA_ADMINISTRACION': 0.25,     # 25% - CRÍTICO
            'PRINCIPIO_ACTIVO': 0.20,       # 20% - CRÍTICO
            'FORMA_FARMACEUTICA': 0.10,     # 10% - IMPORTANTE
            # 5%  - IMPORTANTE (combinación de cantidad y unidad)
            'CANTIDAD_SIMILITUD': 0.05
        }

        # Encoders para cada variable
        self.encoders = {}
        self.scalers = {}
        self.vocabularios = {}
        self.encoding_stats = {}

    def crear_encoding_critico(self, df: pl.DataFrame, columna: str) -> pl.DataFrame:
        """
        Crea encoding para variables CRÍTICAS (ATC, VÍA, PRINCIPIO ACTIVO).

        Args:
            df (pl.DataFrame): Dataset de medicamentos
            columna (str): Nombre de la columna a encodear

        Returns:
            pl.DataFrame: DataFrame con nuevas columnas encoded
        """
        # Obtener vocabulario de medicamentos válidos
        valores_validos = (df.filter(pl.col('VALIDO') == 1).select(
            pl.col(columna).unique()).to_series().to_list())

        # Crear mapping de frecuencias en válidos
        freq_mapping = (df.filter(pl.col('VALIDO') == 1)
                        .group_by(columna)
                        .agg(pl.len().alias('freq_validos'))
                        .with_columns(
            (pl.col('freq_validos') / pl.col('freq_validos').sum()).alias('prob_validos')
        ))

        # Label encoding
        valores_unicos = df.select(
            pl.col(columna).unique()).to_series().to_list()
        label_map = {val: idx for idx,
                     val in enumerate(sorted(valores_unicos))}

        # Aplicar encodings
        df_encoded = (df
                      .with_columns([
                          # Label encoding
                          pl.col(columna).map_elements(
                              lambda x: label_map.get(x, -1)).alias(f'{columna}_label'),

                          # Binary encoding: está en válidos
                          pl.col(columna).is_in(valores_validos).cast(
                              pl.Int8).alias(f'{columna}_es_valido'),
                      ])
                      .join(freq_mapping, on=columna, how='left')
                      .with_columns([
                          pl.col('freq_validos').fill_null(0),
                          pl.col('prob_validos').fill_null(0)
                      ])
                      .rename({
                          'freq_validos': f'{columna}_freq_validos',
                          'prob_validos': f'{columna}_prob_validos'
                      }))        # Guardar encoder info
        self.encoders[columna] = {
            'tipo': 'CRITICO',
            'label_map': label_map,
            'valores_validos': valores_validos,
            'freq_mapping': freq_mapping}

        return df_encoded

    def crear_encoding_importante(self, df: pl.DataFrame, columna: str) -> pl.DataFrame:
        """
        Crea encoding para variables IMPORTANTES (FORMA FARMACÉUTICA).

        Args:
            df (pl.DataFrame): Dataset de medicamentos
            columna (str): Nombre de la columna a encodear

        Returns:
            pl.DataFrame: DataFrame con nuevas columnas encoded
        """
        # Similar al crítico pero con más tolerancia
        valores_unicos = df.select(
            pl.col(columna).unique()).to_series().to_list()
        label_map = {val: idx for idx,
                     val in enumerate(sorted(valores_unicos))}

        # Frecuencias generales (no solo válidos)
        freq_mapping = (df
                        .group_by(columna)
                        .agg(pl.len().alias('freq_total'))
                        .with_columns(
                            (pl.col('freq_total') /
                             pl.col('freq_total').sum()).alias('prob_total')
                        ))

        df_encoded = (df
                      .with_columns([
                          pl.col(columna).map_elements(
                              lambda x: label_map.get(x, -1)).alias(f'{columna}_label')
                      ])
                      .join(freq_mapping, on=columna, how='left')                      .rename({
                          'freq_total': f'{columna}_freq',
                          'prob_total': f'{columna}_prob'
                      }))

        self.encoders[columna] = {
            'tipo': 'IMPORTANTE',
            'label_map': label_map,
            'freq_mapping': freq_mapping
        }

        return df_encoded

    def crear_encoding_numerico(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Crea encoding para variables numéricas (CANTIDAD, CANTIDAD CUM).

        Args:
            df (pl.DataFrame): Dataset de medicamentos

        Returns:
            pl.DataFrame: DataFrame con variables numéricas normalizadas        """
        # Log transform para manejar outliers
        df_encoded = df.with_columns([
            # Log transform (agregamos 1 para evitar log(0))
            (pl.col('CANTIDAD') + 1).log().alias('CANTIDAD_log'),
            (pl.col('CANTIDAD CUM') + 1).log().alias('CANTIDAD_CUM_log'),

            # Ratios útiles
            (pl.col('CANTIDAD') / pl.col('CANTIDAD CUM')).alias('RATIO_CANTIDAD'),

            # Binning por rangos
            pl.when(pl.col('CANTIDAD') <= 10).then(0)
              .when(pl.col('CANTIDAD') <= 100).then(1)
              .when(pl.col('CANTIDAD') <= 500).then(2)
              .otherwise(3).alias('CANTIDAD_bin')])

        return df_encoded

    def aplicar_encoding_completo(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aplica todo el pipeline de encoding al dataset.

        Args:
            df (pl.DataFrame): Dataset original de medicamentos

        Returns:
            pl.DataFrame: Dataset completamente encoded para ML
        """
        # Aplicar encodings secuencialmente
        df_encoded = df.clone()

        # Variables críticas
        variables_criticas = ['ATC', 'VÍA ADMINISTRACIÓN', 'PRINCIPIO ACTIVO']
        for var in variables_criticas:
            if var in df.columns:
                df_encoded = self.crear_encoding_critico(df_encoded, var)

        # Variables importantes
        variables_importantes = ['FORMA FARMACÉUTICA', 'UNIDAD MEDIDA']
        for var in variables_importantes:
            if var in df.columns:
                df_encoded = self.crear_encoding_importante(
                    df_encoded, var)        # Variables numéricas
        df_encoded = self.crear_encoding_numerico(df_encoded)

        return df_encoded

    def crear_dataset_entrenamiento(self, df_encoded: pl.DataFrame) -> pl.DataFrame:
        """
        Crea el dataset final optimizado para clustering de homologación.

        Args:
            df_encoded (pl.DataFrame): Dataset con encoding aplicado

        Returns:
            pl.DataFrame: Dataset listo para entrenamiento con features seleccionadas
        """
        # FEATURES CRÍTICAS (peso total: 85%)
        features_criticas = [
            'ATC_label',                      # 40% - Identificador terapéutico
            'ATC_es_valido',                  # Crítico: debe estar en válidos
            'ATC_prob_validos',               # Probabilidad dentro de válidos

            'VÍA ADMINISTRACIÓN_label',       # 25% - Vía de administración
            'VÍA ADMINISTRACIÓN_es_valido',   # Crítico: debe estar en válidos
            'VÍA ADMINISTRACIÓN_prob_validos',

            'PRINCIPIO ACTIVO_label',         # 20% - Sustancia activa
            'PRINCIPIO ACTIVO_es_valido',     # Crítico: debe estar en válidos
            'PRINCIPIO ACTIVO_prob_validos'
        ]

        # FEATURES IMPORTANTES (peso total: 15%)
        features_importantes = [
            'FORMA FARMACÉUTICA_label',       # 10% - Forma de presentación
            'FORMA FARMACÉUTICA_prob',

            'CANTIDAD_log',                   # 5% - Dosificación
            'CANTIDAD_CUM_log',
            'RATIO_CANTIDAD',
            'CANTIDAD_bin',

            'UNIDAD MEDIDA_label',           # Para contexto de cantidad
            'UNIDAD MEDIDA_prob'
        ]

        # FEATURES INFORMATIVAS (mantener para referencia)
        features_informativas = [
            'CUM',
            'PRODUCTO',
            'ATC',
            'VÍA ADMINISTRACIÓN',
            'PRINCIPIO ACTIVO',
            'FORMA FARMACÉUTICA',
            'CANTIDAD',
            'CANTIDAD CUM',
            'UNIDAD MEDIDA',
            'VALIDO']

        # Seleccionar features para entrenamiento
        todas_features = features_criticas + features_importantes + features_informativas
        features_disponibles = [
            f for f in todas_features if f in df_encoded.columns]

        df_training = df_encoded.select(features_disponibles)

        # Crear features de interacción críticas
        df_training = df_training.with_columns([
            # Combinación ATC + VÍA (muy importante para homologación)
            (pl.col('ATC_label').cast(pl.Utf8) + "_" +
             pl.col('VÍA ADMINISTRACIÓN_label').cast(pl.Utf8))
            .alias('ATC_VIA_combo'),

            # Score de validez crítica (debe ser 3 para medicamentos perfectamente válidos)
            (pl.col('ATC_es_valido') + pl.col('VÍA ADMINISTRACIÓN_es_valido') +
             pl.col('PRINCIPIO ACTIVO_es_valido'))
            .alias('score_validez_critica'),

            # Score de probabilidad crítica (promedio de probabilidades en válidos)
            ((pl.col('ATC_prob_validos') + pl.col('VÍA ADMINISTRACIÓN_prob_validos') +
             pl.col('PRINCIPIO ACTIVO_prob_validos')) / 3)
            .alias('score_prob_critica'),

            # Flag: medicamento ideal para homologación (todas las críticas válidas)
            (pl.col('ATC_es_valido') & pl.col('VÍA ADMINISTRACIÓN_es_valido')
             & pl.col('PRINCIPIO ACTIVO_es_valido'))            .cast(pl.Int8).alias('es_ideal_homologacion')
        ])

        return df_training

    def process_full_encoding(self, df_input: pl.DataFrame, output_path: Optional[str] = None) -> pl.DataFrame:
        """
        Método principal que ejecuta todo el pipeline de encoding de medicamentos.

        Args:
            df_input (pl.DataFrame): Dataset original de medicamentos
            output_path (str, optional): Ruta para guardar el dataset procesado

        Returns:
            pl.DataFrame: Dataset completamente procesado y listo para entrenamiento
        """
        # 1. Aplicar encoding completo
        df_encoded = self.aplicar_encoding_completo(df_input)

        # 2. Crear dataset de entrenamiento
        df_training = self.crear_dataset_entrenamiento(df_encoded)

        # 3. Guardar si se especifica ruta
        if output_path:
            df_training.write_parquet(output_path)

        return df_training

    def load_and_process(self, input_path: str, output_path: Optional[str] = None) -> pl.DataFrame:
        """
        Carga un dataset desde archivo y aplica el proceso completo de encoding.

        Args:
            input_path (str): Ruta al archivo de medicamentos preprocesados
            output_path (str, optional): Ruta para guardar el resultado

        Returns:
            pl.DataFrame: Dataset procesado listo para entrenamiento

        Examples:
            >>> service = MedicamentosEncodingService()
            >>> df_final = service.load_and_process('../data/medicamentos_preprocesados.parquet', 
            ...                                   '../data/dataset_entrenamiento_homologacion.parquet')
        """
        print(f"📂 CARGANDO DATASET DESDE: {input_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

        # Cargar dataset
        df_medicamentos = pl.read_parquet(input_path)
        print(f"✅ Dataset cargado: {df_medicamentos.shape}")

        # Procesar
        df_final = self.process_full_encoding(df_medicamentos, output_path)

        return df_final


# Función utilitaria para uso directo
def process_medicamentos_encoding(input_path: str, output_path: Optional[str] = None) -> pl.DataFrame:
    """
    Función utilitaria para procesar encoding de medicamentos en una sola llamada.

    Args:
        input_path (str): Ruta al dataset de medicamentos preprocesados
        output_path (str, optional): Ruta de salida para el dataset procesado

    Returns:
        pl.DataFrame: Dataset listo para entrenamiento

    Examples:
        >>> df_training = process_medicamentos_encoding('../data/medicamentos_preprocesados.parquet',
        ...                                           '../data/dataset_entrenamiento_homologacion.parquet')
    """
    encoding_service = MedicamentosEncodingService()
    return encoding_service.load_and_process(input_path, output_path)


if __name__ == "__main__":
    # Ejemplo de uso del servicio
    print("🧪 EJEMPLO DE USO DEL SERVICIO DE ENCODING")
    print("=" * 50)

    # Rutas de ejemplo
    INPUT_FILE = "./data/medicamentos_preprocesados.parquet"
    OUTPUT_FILE = "./data/dataset_entrenamiento_homologacion.parquet"

    try:
        # Crear servicio y procesar
        service = MedicamentosEncodingService()
        df_resultado = service.load_and_process(INPUT_FILE, OUTPUT_FILE)

        print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📊 Resultado final: {df_resultado.shape}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de tener el archivo de medicamentos preprocesados")
