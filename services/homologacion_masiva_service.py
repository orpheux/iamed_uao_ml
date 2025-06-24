"""
🔄 SERVICIO DE HOMOLOGACIÓN MASIVA
==================================
Servicio para procesar archivos Excel con homologación masiva de CUMs.
Lee Excel con polars, aplica homologación a cada CUM y genera archivo resultado.

Author: Sistema IAMED
Version: 1.0.0
"""

import os
import polars as pl
from typing import Dict, Optional, Tuple, Callable
from services.search_service import SearchService
from services.training_service import HomologacionClusteringModel, SistemaRecomendacionHomologos


class HomologacionMasivaService:
    """
    Servicio para homologación masiva de CUMs desde archivos Excel.
    Utiliza polars para lectura eficiente y SearchService para homologación.
    """

    def __init__(self, model_path: str = "./models/iamed.pkl"):
        """
        Inicializa el servicio de homologación masiva.

        Args:
            model_path (str): Ruta al modelo entrenado (.pkl)
        """
        self.model_path = model_path
        self.search_service = None
        self.df_original = None
        self.df_homologado = None
        self._inicializar_servicio()

    def _inicializar_servicio(self) -> None:
        """Inicializa el servicio de búsqueda."""
        try:
            self.search_service = SearchService(self.model_path)
            print("✅ Servicio de homologación masiva inicializado")
        except Exception as e:
            raise RuntimeError(f"Error al inicializar servicio: {e}") from e

    def validar_archivo_excel(self, archivo_path: str) -> Tuple[bool, str]:
        """
        Valida que el archivo Excel sea válido y tenga el formato correcto.

        Args:
            archivo_path (str): Ruta del archivo Excel

        Returns:
            Tuple[bool, str]: (es_valido, mensaje_error)
        """
        try:
            # Verificar que existe el archivo
            if not os.path.exists(archivo_path):
                return False, "El archivo no existe"

            # Verificar extensión
            if not archivo_path.lower().endswith(('.xlsx', '.xls')):
                return False, "El archivo debe ser un Excel (.xlsx o .xls)"

            # Intentar leer el archivo - polars read_excel puede devolver dict o DataFrame
            try:
                df_result = pl.read_excel(archivo_path)
            except Exception as read_error:
                return False, f"Error al leer Excel: {str(read_error)}"

            # Si es un dict (múltiples hojas), tomar la primera hoja
            if isinstance(df_result, dict):
                df = list(df_result.values())[0]  # Primera hoja
            else:
                df = df_result

            # Verificar que no esté vacío
            if df.height == 0:
                return False, "El archivo Excel está vacío"

            # Verificar que tenga al menos una columna
            if df.width == 0:
                return False, "El archivo Excel no tiene columnas"

            # Guardar para uso posterior
            self.df_original = df

            print(
                f"✅ Archivo Excel válido: {df.height} filas, {df.width} columnas")
            return True, "Archivo válido"

        except Exception as e:
            return False, f"Error al leer archivo Excel: {str(e)}"

    def procesar_homologacion(self, columna_cum: Optional[str] = None, progreso_callback: Optional[Callable] = None) -> Dict:
        """
        Procesa la homologación masiva del archivo Excel.

        Args:
            columna_cum (str): Nombre de la columna que contiene los CUMs.
                              Si es None, usa la primera columna.
            progreso_callback: Función callback para reportar progreso

        Returns:
            Dict: Resultado del procesamiento
        """
        if self.df_original is None:
            return {
                'exito': False,
                'error': 'No hay archivo cargado. Primero valide un archivo Excel.'
            }

        try:
            df = self.df_original.clone()

            # Determinar columna de CUMs
            if columna_cum is None:
                columna_cum = df.columns[0]
                print(f"📍 Usando primera columna como CUMs: '{columna_cum}'")

            if columna_cum not in df.columns:
                return {
                    'exito': False,
                    'error': f'La columna "{columna_cum}" no existe en el archivo'
                }

            # Obtener lista de CUMs únicos (sin nulos)
            cums_serie = df.select(pl.col(columna_cum)).to_series()
            cums_unicos = cums_serie.drop_nulls().unique().to_list()

            print(f"🔍 Procesando {len(cums_unicos)} CUMs únicos")

            # Procesar homologación para cada CUM único
            homologaciones = {}
            total_cums = len(cums_unicos)

            for i, cum in enumerate(cums_unicos):
                if progreso_callback:
                    progreso_callback(i + 1, total_cums,
                                      f"Procesando CUM: {cum}")

                try:
                    # Convertir a string y limpiar
                    cum_str = str(cum).strip()
                    if not cum_str or cum_str.lower() in ['nan', 'none', '']:
                        # Buscar homólogo (solo 1 recomendación)
                        continue
                    if self.search_service is not None:
                        resultado_homologo = self.search_service.buscar_homologos(
                            cum_str, n_recomendaciones=1, score_minimo=0.85
                        )
                    else:
                        resultado_homologo = {
                            'encontrado': False, 'recomendaciones': []}

                    if resultado_homologo['encontrado'] and resultado_homologo['recomendaciones']:
                        homologo = resultado_homologo['recomendaciones'][0]
                        homologaciones[cum_str] = {
                            'cum_homologo': homologo['cum'],
                            'nombre_homologo': homologo['producto'],
                            'score': homologo['score_similitud']
                        }
                    else:
                        homologaciones[cum_str] = {
                            'cum_homologo': 'SIN HOMÓLOGO',
                            'nombre_homologo': 'NO ENCONTRADO',
                            'score': 0.0
                        }

                except Exception as e:
                    print(f"❌ Error procesando CUM {cum}: {e}")
                    homologaciones[str(cum)] = {
                        'cum_homologo': 'ERROR',
                        'nombre_homologo': f'Error: {str(e)}',
                        'score': 0.0
                    }

            # Crear DataFrame resultado con las nuevas columnas
            df_resultado = df.with_columns([
                pl.col(columna_cum).map_elements(
                    lambda x: homologaciones.get(str(x).strip(), {}).get(
                        'cum_homologo', 'SIN HOMÓLOGO'),
                    return_dtype=pl.String
                ).alias('CUM_HOMOLOGO'),

                pl.col(columna_cum).map_elements(
                    lambda x: homologaciones.get(str(x).strip(), {}).get(
                        'nombre_homologo', 'NO ENCONTRADO'),
                    return_dtype=pl.String
                ).alias('NOMBRE_HOMOLOGO'),

                pl.col(columna_cum).map_elements(
                    lambda x: homologaciones.get(
                        str(x).strip(), {}).get('score', 0.0),
                    return_dtype=pl.Float64
                ).alias('SCORE_SIMILITUD')
            ])

            self.df_homologado = df_resultado

            # Estadísticas
            total_procesados = len([h for h in homologaciones.values()
                                    if h['cum_homologo'] not in ['SIN HOMÓLOGO', 'ERROR']])

            return {
                'exito': True,
                'total_filas': df.height,
                'cums_unicos': len(cums_unicos),
                'homologos_encontrados': total_procesados,
                'porcentaje_exito': (total_procesados / len(cums_unicos) * 100) if cums_unicos else 0,
                'columna_procesada': columna_cum
            }

        except Exception as e:
            return {
                'exito': False,
                'error': f'Error durante homologación: {str(e)}'
            }

    def guardar_resultado(self, archivo_salida: str) -> Tuple[bool, str]:
        """
        Guarda el resultado de la homologación en un archivo Excel.

        Args:
            archivo_salida (str): Ruta donde guardar el archivo resultado

        Returns:
            Tuple[bool, str]: (exito, mensaje)
        """
        if self.df_homologado is None:
            return False, "No hay resultado de homologación para guardar"

        try:
            # Asegurar extensión .xlsx
            if not archivo_salida.lower().endswith('.xlsx'):
                archivo_salida += '.xlsx'

            # Guardar archivo Excel
            self.df_homologado.write_excel(archivo_salida)

            print(f"✅ Archivo guardado: {archivo_salida}")
            print(
                f"📊 Filas: {self.df_homologado.height}, Columnas: {self.df_homologado.width}")

            return True, f"Archivo guardado exitosamente en: {archivo_salida}"

        except Exception as e:
            return False, f"Error al guardar archivo: {str(e)}"

    def obtener_resumen_resultado(self) -> Dict:
        """
        Obtiene un resumen del resultado de homologación.

        Returns:
            Dict: Resumen con estadísticas
        """
        if self.df_homologado is None:
            return {'error': 'No hay resultado disponible'}

        try:
            # Contar homólogos encontrados vs no encontrados
            homologos_encontrados = self.df_homologado.filter(
                ~pl.col('CUM_HOMOLOGO').is_in(['SIN HOMÓLOGO', 'ERROR'])
            ).height

            sin_homologo = self.df_homologado.filter(
                pl.col('CUM_HOMOLOGO') == 'SIN HOMÓLOGO'
            ).height

            con_error = self.df_homologado.filter(
                pl.col('CUM_HOMOLOGO') == 'ERROR'
            ).height

            total_filas = self.df_homologado.height

            return {
                'total_filas': total_filas,
                'homologos_encontrados': homologos_encontrados,
                'sin_homologo': sin_homologo,
                'con_error': con_error,
                'porcentaje_exito': (homologos_encontrados / total_filas * 100) if total_filas > 0 else 0,
                'columnas_resultado': self.df_homologado.columns
            }

        except Exception as e:
            return {'error': f'Error calculando resumen: {str(e)}'}


# Funciones utilitarias para uso directo
def homologar_archivo_excel(
    archivo_entrada: str,
    archivo_salida: Optional[str] = None,
    columna_cum: Optional[str] = None,
    progreso_callback: Optional[Callable] = None
) -> Dict:
    """
    Función directa para homologar un archivo Excel completo.

    Args:
        archivo_entrada (str): Ruta del archivo Excel de entrada
        archivo_salida (str): Ruta del archivo Excel de salida (opcional)
        columna_cum (str): Nombre de la columna con CUMs (opcional, usa primera columna)
        progreso_callback: Función callback para progreso

    Returns:
        Dict: Resultado del procesamiento
    """
    homologacion_service = HomologacionMasivaService()

    # Validar archivo
    es_valido, mensaje_validacion = homologacion_service.validar_archivo_excel(archivo_entrada)
    if not es_valido:
        return {'exito': False, 'error': mensaje_validacion}

    # Procesar homologación
    resultado_funcion = homologacion_service.procesar_homologacion(
        columna_cum, progreso_callback)
    if not resultado_funcion['exito']:
        return resultado_funcion

    # Guardar resultado si se especifica archivo de salida
    if archivo_salida:
        exito_guardado, mensaje_guardado = homologacion_service.guardar_resultado(
            archivo_salida)
        resultado_funcion['archivo_guardado'] = exito_guardado
        resultado_funcion['mensaje_guardado'] = mensaje_guardado

    return resultado_funcion


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔄 SERVICIO DE HOMOLOGACIÓN MASIVA")
    print("=" * 50)

    try:
        # Crear servicio
        service = HomologacionMasivaService()

        # Archivo de ejemplo (ajustar ruta según necesidad)
        ARCHIVO_EJEMPLO = "./data/medicamentos_otros.xlsx"

        if os.path.exists(ARCHIVO_EJEMPLO):
            print(f"📂 Procesando archivo: {ARCHIVO_EJEMPLO}")

            # Validar archivo
            archivo_valido, mensaje = service.validar_archivo_excel(ARCHIVO_EJEMPLO)
            print(f"✅ Validación: {mensaje}")

            if archivo_valido:
                # Procesar homologación
                def mostrar_progreso(actual, total, msg):
                    porcentaje = (actual / total) * 100
                    print(f"📊 [{porcentaje:.1f}%] {msg}")

                resultado_proceso = service.procesar_homologacion(
                    progreso_callback=mostrar_progreso)
                print(f"🎯 Resultado: {resultado_proceso}")

                # Mostrar resumen
                resumen = service.obtener_resumen_resultado()
                print(f"📋 Resumen: {resumen}")
        else:
            print(f"❌ Archivo no encontrado: {ARCHIVO_EJEMPLO}")

    except Exception as e:
        print(f"❌ Error: {e}")
