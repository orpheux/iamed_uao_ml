"""Análisis Exploratorio de Datos (EDA) para el sistema de homologación de medicamentos.
Este módulo proporciona funciones para realizar un análisis exploratorio completo sobre un dataset de medicamentos,
con el objetivo de identificar variables clave, evaluar la calidad de los datos y proponer estrategias para la
homologación automática de medicamentos. Incluye herramientas para el análisis de validez, cobertura, combinaciones
de variables y generación de visualizaciones que facilitan la toma de decisiones en el diseño de sistemas de
recomendación y homologación.
Funciones principales:
    - eda_completo_medicamentos: Realiza el análisis exploratorio detallado y retorna un resumen estructurado.
    - crear_visualizaciones_eda: Genera visualizaciones gráficas a partir de los resultados del EDA.
    - ejecutar_eda_completo: Orquesta la ejecución del análisis y la visualización de resultados.
Ejemplo de uso:
    >>> df_prep = pl.read_parquet('./data/medicamentos_preprocesados.parquet')
    >>> ejecutar_eda_completo(df_prep)
Autor:
    Juan Bohorquez
Fecha de creación:
"""

import os
from typing import Dict
import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def eda_completo_medicamentos(df_preproc: pl.DataFrame) -> Dict:
    """
    Realiza un análisis exploratorio de datos completo para el dataset de medicamentos
    con el objetivo de diseñar un sistema de homologación automática.

    Args:
        df_preproc (pl.DataFrame): DataFrame preprocesado con datos de medicamentos

    Returns:
        Dict: Diccionario con todos los resultados del análisis exploratorio

    Raises:
        ValueError: Si el DataFrame está vacío o no contiene las columnas esperadas

    Notes:
        - Análisis orientado a identificar variables clave para homologación
        - Evaluación de calidad de datos y patrones de distribución
        - Identificación de medicamentos válidos vs inválidos
        - Análisis de cobertura para sistema de recomendación

    Examples:
        >>> resultados_eda = eda_completo_medicamentos(df_preproc)
        >>> print(f"Variables analizadas: {len(resultados_eda['variables_analizadas'])}")
        >>> print(f"Recomendación final: {resultados_eda['recomendaciones']['estrategia_recomendada']}")
    """

    print("="*100)
    print("🔍 ANÁLISIS EXPLORATORIO DE DATOS - DATASET DE MEDICAMENTOS")
    print("="*100)
    print("📋 Objetivo: Diseñar sistema de homologación automática de medicamentos")
    print("="*100)

    # Validación inicial
    if df_preproc.is_empty():
        raise ValueError("El DataFrame no puede estar vacío")

    resultados = {
        'dataset_info': {},
        'variables_analizadas': {},
        'analisis_validez': {},
        'analisis_cobertura': {},
        'recomendaciones': {}
    }

    # ============================================================================
    # 1. INFORMACIÓN GENERAL DEL DATASET
    # ============================================================================
    print("\n📊 1. INFORMACIÓN GENERAL DEL DATASET")
    print("="*80)

    filas, columnas = df_preproc.shape
    memoria_mb = df_preproc.estimated_size() / (1024**2)

    print(f"   📏 Dimensiones: {filas:,} filas × {columnas} columnas")
    print(f"   💾 Memoria utilizada: {memoria_mb:.2f} MB")
    print(f"   📁 Tamaño promedio por registro: {memoria_mb*1024/filas:.2f} KB")

    # Información de columnas
    print("\n📋 COLUMNAS DEL DATASET:")
    for i, col in enumerate(df_preproc.columns, 1):
        dtype = str(df_preproc[col].dtype)
        print(f"   {i:2d}. {col:<25} | {dtype}")

    resultados['dataset_info'] = {
        'filas': filas,
        'columnas': columnas,
        'memoria_mb': memoria_mb,
        'columnas_lista': df_preproc.columns
    }

    # ============================================================================
    # 2. ANÁLISIS DE CALIDAD DE DATOS
    # ============================================================================
    print("\n🔍 2. ANÁLISIS DE CALIDAD DE DATOS")
    print("="*80)

    # Valores nulos por columna
    nulos_por_columna = {}
    print("\n❌ VALORES NULOS POR COLUMNA:")

    hay_nulos = False
    for col in df_preproc.columns:
        try:
            nulos = df_preproc[col].null_count()
            porcentaje_nulos = (nulos / filas) * 100
            nulos_por_columna[col] = {
                'cantidad': nulos, 'porcentaje': porcentaje_nulos}

            if nulos > 0:
                print(
                    f"   🔸 {col:<25}: {nulos:>8,} ({porcentaje_nulos:>5.1f}%)")
                hay_nulos = True
        except Exception as e:
            print(f"   ⚠️ Error procesando {col}: {str(e)}")
            nulos_por_columna[col] = {'cantidad': 0, 'porcentaje': 0.0}

    if not hay_nulos:
        print("   ✅ No hay valores nulos en el dataset")

    # Valores únicos por columna
    print("\n🔢 CARDINALIDAD POR COLUMNA:")
    unicos_por_columna = {}

    for col in df_preproc.columns:
        try:
            unicos = df_preproc[col].n_unique()
            porcentaje_unicos = (unicos / filas) * 100
            unicos_por_columna[col] = {
                'cantidad': unicos, 'porcentaje': porcentaje_unicos}

            # Clasificar tipo de variable por cardinalidad
            if porcentaje_unicos > 95:
                tipo = "ID/ÚNICA"
            elif porcentaje_unicos > 50:
                tipo = "ALTA_CARD"
            elif porcentaje_unicos > 10:
                tipo = "MEDIA_CARD"
            else:
                tipo = "BAJA_CARD"

            print(
                f"   🔸 {col:<25}: {unicos:>8,} únicos ({porcentaje_unicos:>5.1f}%) [{tipo}]")
        except Exception as e:
            print(f"   ⚠️ Error procesando {col}: {str(e)}")
            unicos_por_columna[col] = {'cantidad': 0, 'porcentaje': 0.0}

    resultados['variables_analizadas']['calidad_datos'] = {
        'nulos_por_columna': nulos_por_columna,
        'unicos_por_columna': unicos_por_columna
    }

    # ============================================================================
    # 3. ANÁLISIS DE MEDICAMENTOS VÁLIDOS VS INVÁLIDOS
    # ============================================================================
    print("\n🎯 3. ANÁLISIS DE VALIDEZ DE MEDICAMENTOS")
    print("="*80)

    print("📋 Criterios de validez:")
    print("   ✅ ESTADO REGISTRO = 'Vigente'")
    print("   ✅ ESTADO CUM = 'Activo'")
    print("   ✅ MUESTRA MÉDICA = 'No'")

    # Identificar medicamentos válidos
    try:
        mascara_validos = (
            (df_preproc['ESTADO REGISTRO'] == 'Vigente') &
            (df_preproc['ESTADO CUM'] == 'Activo') &
            (df_preproc['MUESTRA MÉDICA'] == 'No')
        )

        validos = df_preproc.filter(mascara_validos)
        invalidos = df_preproc.filter(~mascara_validos)

        print("\n📊 DISTRIBUCIÓN DE VALIDEZ:")
        print(
            f"   ✅ Medicamentos VÁLIDOS: {len(validos):,} ({len(validos)/filas*100:.1f}%)")
        print(
            f"   ❌ Medicamentos INVÁLIDOS: {len(invalidos):,} ({len(invalidos)/filas*100:.1f}%)")

        # Análisis de motivos de invalidez
        print("\n🔍 ANÁLISIS DE MOTIVOS DE INVALIDEZ:")

        no_vigente = (df_preproc['ESTADO REGISTRO'] != 'Vigente').sum()
        no_activo = (df_preproc['ESTADO CUM'] != 'Activo').sum()
        es_muestra = (df_preproc['MUESTRA MÉDICA'] == 'Si').sum()

        print(f"   📋 Estado Registro ≠ 'Vigente': {no_vigente:,}")
        print(f"   📋 Estado CUM ≠ 'Activo': {no_activo:,}")
        print(f"   📋 Es Muestra Médica: {es_muestra:,}")

        resultados['analisis_validez'] = {
            'total_validos': len(validos),
            'total_invalidos': len(invalidos),
            'porcentaje_validos': len(validos)/filas*100,
            'motivos_invalidez': {
                'no_vigente': int(no_vigente),
                'no_activo': int(no_activo),
                'es_muestra': int(es_muestra)
            }
        }

    except Exception as e:
        print(f"   ⚠️ Error en análisis de validez: {str(e)}")
        # Crear datasets vacíos para evitar errores posteriores
        validos = df_preproc.head(0)
        invalidos = df_preproc.head(0)
        resultados['analisis_validez'] = {
            'total_validos': 0,
            'total_invalidos': 0,
            'porcentaje_validos': 0,
            'motivos_invalidez': {'no_vigente': 0, 'no_activo': 0, 'es_muestra': 0}
        }

    # ============================================================================
    # 4. ANÁLISIS DE VARIABLES CATEGÓRICAS PRINCIPALES
    # ============================================================================
    print("\n🏷️ 4. ANÁLISIS DE VARIABLES CATEGÓRICAS PRINCIPALES")
    print("="*80)

    variables_categoricas = [
        'PRINCIPIO ACTIVO', 'ATC', 'VÍA ADMINISTRACIÓN',
        'FORMA FARMACÉUTICA', 'UNIDAD MEDIDA', 'UNIDAD REFERENCIA', 'UNIDAD'
    ]

    analisis_categoricas = {}

    for variable in variables_categoricas:
        if variable not in df_preproc.columns:
            print(f"\n⚠️ Variable {variable} no encontrada en el dataset")
            continue

        print(f"\n📋 VARIABLE: {variable}")
        print("-" * 60)

        try:
            # Análisis general
            total_unicos = df_preproc[variable].n_unique()
            total_nulos = df_preproc[variable].null_count()

            # Análisis por válidos/inválidos
            unicos_validos = validos[variable].n_unique() if len(
                validos) > 0 else 0
            unicos_invalidos = invalidos[variable].n_unique() if len(
                invalidos) > 0 else 0

            print(f"   📊 Total únicos: {total_unicos:,}")
            print(f"   📊 Valores nulos: {total_nulos:,}")
            print(f"   📊 Únicos en válidos: {unicos_validos:,}")
            print(f"   📊 Únicos en inválidos: {unicos_invalidos:,}")

            # Top valores más frecuentes
            top_valores = df_preproc[variable].value_counts().sort(
                "count", descending=True).head(5)
            print("   🏆 TOP 5 valores más frecuentes:")

            for i, fila in enumerate(top_valores.iter_rows(named=True), 1):
                valor = fila[variable]
                conteo = fila['count']
                porcentaje = (conteo / filas) * 100
                valor_mostrar = str(
                    valor)[:40] + "..." if str(valor) and len(str(valor)) > 40 else str(valor)
                print(
                    f"      {i}. {valor_mostrar:<43} | {conteo:>8,} ({porcentaje:>5.1f}%)")

            analisis_categoricas[variable] = {
                'total_unicos': total_unicos,
                'total_nulos': total_nulos,
                'unicos_validos': unicos_validos,
                'unicos_invalidos': unicos_invalidos,
                'top_valores': [(fila[variable], fila['count']) for fila in top_valores.iter_rows(named=True)]
            }

        except Exception as e:
            print(f"   ⚠️ Error procesando variable {variable}: {str(e)}")
            analisis_categoricas[variable] = {
                'total_unicos': 0,
                'total_nulos': 0,
                'unicos_validos': 0,
                'unicos_invalidos': 0,
                'top_valores': []
            }

    resultados['variables_analizadas']['categoricas'] = analisis_categoricas

    # ============================================================================
    # 5. ANÁLISIS DE VARIABLES NUMÉRICAS
    # ============================================================================
    print("\n🔢 5. ANÁLISIS DE VARIABLES NUMÉRICAS")
    print("="*80)

    variables_numericas = ['CANTIDAD', 'CANTIDAD CUM']
    analisis_numericas = {}

    for variable in variables_numericas:
        if variable not in df_preproc.columns:
            print(f"\n⚠️ Variable {variable} no encontrada")
            continue

        print(f"\n📋 VARIABLE: {variable}")
        print("-" * 60)

        try:
            # Estadísticas descriptivas
            stats = df_preproc[variable].describe()
            nulos = df_preproc[variable].null_count()

            print(f"   📊 Valores nulos: {nulos:,}")
            print("   📊 Estadísticas descriptivas:")

            for i, row in enumerate(stats.iter_rows()):
                stat_name = row[0]  # Primera columna es 'statistic'
                value = row[1]      # Segunda columna es el valor
                if value is not None:
                    print(f"      {stat_name:<10}: {value:>12,.2f}")

            # Análisis de outliers (IQR method)
            if nulos < filas * 0.5:  # Si hay menos del 50% de nulos
                try:
                    q1 = df_preproc[variable].quantile(0.25)
                    q3 = df_preproc[variable].quantile(0.75)
                    if q1 is not None and q3 is not None:
                        iqr = q3 - q1
                        limite_inferior = q1 - 1.5 * iqr
                        limite_superior = q3 + 1.5 * iqr

                        outliers = df_preproc.filter(
                            (pl.col(variable) < limite_inferior) | (
                                pl.col(variable) > limite_superior)
                        ).shape[0]

                        print(
                            f"   📊 Outliers potenciales: {outliers:,} ({outliers/filas*100:.1f}%)")
                    else:
                        print(
                            "   📊 No se pudieron calcular outliers (cuartiles no disponibles)")
                except Exception:
                    print("   📊 No se pudieron calcular outliers")

            analisis_numericas[variable] = {
                'nulos': nulos,
                'estadisticas': {fila['statistic']: fila['value'] for fila in stats.iter_rows(named=True) if fila['value'] is not None}
            }

        except Exception as e:
            print(f"   ⚠️ Error procesando variable {variable}: {str(e)}")
            analisis_numericas[variable] = {'nulos': 0, 'estadisticas': {}}

    resultados['variables_analizadas']['numericas'] = analisis_numericas

    # ============================================================================
    # 6. ANÁLISIS DE COBERTURA PARA HOMOLOGACIÓN
    # ============================================================================
    print("\n🎯 6. ANÁLISIS DE COBERTURA PARA HOMOLOGACIÓN")
    print("="*80)

    print("📋 Objetivo: Determinar qué variables permiten mayor cobertura de homologación")

    # Variables clave para análisis de cobertura
    variables_clave = ['PRINCIPIO ACTIVO', 'ATC',
                       'VÍA ADMINISTRACIÓN', 'FORMA FARMACÉUTICA']
    cobertura_individual = {}

    for variable in variables_clave:
        if variable not in df_preproc.columns:
            continue

        try:
            # Conjuntos únicos
            valores_validos = set(validos[variable].unique().to_list()) if len(
                validos) > 0 else set()
            valores_invalidos = set(invalidos[variable].unique().to_list()) if len(
                invalidos) > 0 else set()
            valores_comunes = valores_validos.intersection(valores_invalidos)

            # Cobertura por valores únicos
            cobertura_valores = len(
                valores_comunes) / len(valores_invalidos) * 100 if len(valores_invalidos) > 0 else 0

            # Cobertura por registros
            if len(valores_comunes) > 0 and len(invalidos) > 0:
                invalidos_cubiertos = invalidos.filter(
                    pl.col(variable).is_in(list(valores_comunes)))
                cobertura_registros = len(
                    invalidos_cubiertos) / len(invalidos) * 100
            else:
                cobertura_registros = 0

            print(f"\n📊 COBERTURA - {variable}:")
            print(f"   🔸 Valores únicos en válidos: {len(valores_validos):,}")
            print(
                f"   🔸 Valores únicos en inválidos: {len(valores_invalidos):,}")
            print(f"   🔸 Valores comunes: {len(valores_comunes):,}")
            print(f"   🎯 Cobertura por valores: {cobertura_valores:.1f}%")
            print(f"   🎯 Cobertura por registros: {cobertura_registros:.1f}%")

            cobertura_individual[variable] = {
                'valores_validos': len(valores_validos),
                'valores_invalidos': len(valores_invalidos),
                'valores_comunes': len(valores_comunes),
                'cobertura_valores': cobertura_valores,
                'cobertura_registros': cobertura_registros
            }

        except Exception as e:
            print(f"\n⚠️ Error procesando cobertura de {variable}: {str(e)}")
            cobertura_individual[variable] = {
                'valores_validos': 0,
                'valores_invalidos': 0,
                'valores_comunes': 0,
                'cobertura_valores': 0,
                'cobertura_registros': 0
            }

    # ============================================================================
    # 7. ANÁLISIS DE COMBINACIONES CLAVE
    # ============================================================================
    print("\n🔗 7. ANÁLISIS DE COMBINACIONES DE VARIABLES")
    print("="*80)

    # Combinaciones importantes para filtrado
    combinaciones = [
        ['PRINCIPIO ACTIVO', 'VÍA ADMINISTRACIÓN'],
        ['ATC', 'VÍA ADMINISTRACIÓN'],
        ['ATC', 'VÍA ADMINISTRACIÓN', 'FORMA FARMACÉUTICA']
    ]

    cobertura_combinaciones = {}

    for combo in combinaciones:
        # Verificar que todas las columnas existen
        if not all(col in df_preproc.columns for col in combo):
            continue

        combo_str = " + ".join(combo)
        print(f"\n📊 COMBINACIÓN: {combo_str}")
        print("-" * 60)

        try:
            # Crear claves combinadas
            if len(validos) > 0:
                validos_combo = validos.with_columns(
                    pl.concat_str([pl.col(c).cast(pl.Utf8)
                                  for c in combo], separator="|").alias("CLAVE_COMBO")
                )
            else:
                validos_combo = pl.DataFrame()

            if len(invalidos) > 0:
                invalidos_combo = invalidos.with_columns(
                    pl.concat_str([pl.col(c).cast(pl.Utf8)
                                  for c in combo], separator="|").alias("CLAVE_COMBO")
                )
            else:
                invalidos_combo = pl.DataFrame()

            if len(validos_combo) > 0 and len(invalidos_combo) > 0:
                claves_validas = set(
                    validos_combo['CLAVE_COMBO'].unique().to_list())
                claves_invalidas = set(
                    invalidos_combo['CLAVE_COMBO'].unique().to_list())
                claves_comunes = claves_validas.intersection(claves_invalidas)

                # Medicamentos inválidos cubiertos
                if len(claves_comunes) > 0:
                    invalidos_cubiertos = invalidos_combo.filter(
                        pl.col('CLAVE_COMBO').is_in(list(claves_comunes))
                    )
                else:
                    invalidos_cubiertos = pl.DataFrame()

                cobertura_combo = len(
                    invalidos_cubiertos) / len(invalidos) * 100 if len(invalidos) > 0 else 0

                print(
                    f"   🔸 Combinaciones únicas en válidos: {len(claves_validas):,}")
                print(
                    f"   🔸 Combinaciones únicas en inválidos: {len(claves_invalidas):,}")
                print(f"   🔸 Combinaciones comunes: {len(claves_comunes):,}")
                print(f"   🎯 Cobertura: {cobertura_combo:.1f}%")

                cobertura_combinaciones[combo_str] = {
                    'claves_validas': len(claves_validas),
                    'claves_invalidas': len(claves_invalidas),
                    'claves_comunes': len(claves_comunes),
                    'cobertura': cobertura_combo
                }
            else:
                print("   ⚠️ No hay datos suficientes para analizar combinación")
                cobertura_combinaciones[combo_str] = {
                    'claves_validas': 0,
                    'claves_invalidas': 0,
                    'claves_comunes': 0,
                    'cobertura': 0
                }

        except Exception as e:
            print(f"   ⚠️ Error procesando combinación {combo_str}: {str(e)}")
            cobertura_combinaciones[combo_str] = {
                'claves_validas': 0,
                'claves_invalidas': 0,
                'claves_comunes': 0,
                'cobertura': 0
            }

    resultados['analisis_cobertura'] = {
        'individual': cobertura_individual,
        'combinaciones': cobertura_combinaciones
    }

    # ============================================================================
    # 8. RECOMENDACIONES BASADAS EN EL ANÁLISIS
    # ============================================================================
    print("\n💡 8. RECOMENDACIONES PARA EL MODELO")
    print("="*80)

    # Identificar mejor estrategia basada en cobertura
    if cobertura_individual:
        mejor_individual = max(cobertura_individual.items(
        ), key=lambda x: x[1]['cobertura_registros'])
    else:
        mejor_individual = ("N/A", {'cobertura_registros': 0})

    if cobertura_combinaciones:
        mejor_combinacion = max(
            cobertura_combinaciones.items(), key=lambda x: x[1]['cobertura'])
    else:
        mejor_combinacion = None

    print("\n🏆 VARIABLE CON MEJOR COBERTURA INDIVIDUAL:")
    print(
        f"   {mejor_individual[0]}: {mejor_individual[1]['cobertura_registros']:.1f}%")

    if mejor_combinacion:
        print("\n🏆 COMBINACIÓN CON MEJOR COBERTURA:")
        print(
            f"   {mejor_combinacion[0]}: {mejor_combinacion[1]['cobertura']:.1f}%")

    # Clasificar variables por importancia
    variables_criticas = []
    variables_importantes = []
    variables_opcionales = []

    for var, info in cobertura_individual.items():
        if info['cobertura_registros'] >= 80:
            variables_criticas.append(var)
        elif info['cobertura_registros'] >= 50:
            variables_importantes.append(var)
        else:
            variables_opcionales.append(var)

    print("\n🔥 CLASIFICACIÓN DE VARIABLES:")
    print(f"   CRÍTICAS (≥80% cobertura): {variables_criticas}")
    print(f"   IMPORTANTES (≥50% cobertura): {variables_importantes}")
    print(f"   OPCIONALES (<50% cobertura): {variables_opcionales}")

    # Estrategia recomendada
    if mejor_combinacion and mejor_combinacion[1]['cobertura'] > 80:
        estrategia = f"Usar filtro: {mejor_combinacion[0]}"
    elif mejor_individual[1]['cobertura_registros'] > 70:
        estrategia = f"Usar filtro: {mejor_individual[0]} + variables complementarias"
    else:
        estrategia = "Usar estrategia híbrida con múltiples variables"

    print("\n🎯 ESTRATEGIA RECOMENDADA:")
    print(f"   {estrategia}")

    resultados['recomendaciones'] = {
        'mejor_individual': mejor_individual,
        'mejor_combinacion': mejor_combinacion,
        'variables_criticas': variables_criticas,
        'variables_importantes': variables_importantes,
        'variables_opcionales': variables_opcionales,
        'estrategia_recomendada': estrategia
    }

    # ============================================================================
    # 9. RESUMEN EJECUTIVO
    # ============================================================================
    print("\n" + "="*100)
    print("📋 RESUMEN EJECUTIVO DEL EDA")
    print("="*100)

    print("\n📊 DATOS CLAVE:")
    print(f"   • Dataset: {filas:,} medicamentos, {columnas} variables")
    print(
        f"   • Medicamentos válidos: {len(validos):,} ({len(validos)/filas*100:.1f}%)")
    print(
        f"   • Medicamentos inválidos: {len(invalidos):,} ({len(invalidos)/filas*100:.1f}%)")

    print("\n🎯 HALLAZGOS PRINCIPALES:")
    print(
        f"   • Mejor variable individual: {mejor_individual[0]} ({mejor_individual[1]['cobertura_registros']:.1f}% cobertura)")
    if mejor_combinacion:
        print(
            f"   • Mejor combinación: {mejor_combinacion[0]} ({mejor_combinacion[1]['cobertura']:.1f}% cobertura)")
    print(f"   • Variables críticas identificadas: {len(variables_criticas)}")

    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Implementar filtros basados en variables críticas")
    print("   2. Diseñar sistema de scoring con variables importantes")
    print("   3. Desarrollar algoritmo de homologación automática")
    print("   4. Validar resultados con casos de prueba")

    print("\n✅ EDA COMPLETADO EXITOSAMENTE")

    return resultados

# ============================================================================
# FUNCIÓN PARA CREAR VISUALIZACIONES
# ============================================================================


def crear_visualizaciones_eda(df_preproc: pl.DataFrame, resultados: Dict):
    """
    Crea visualizaciones basadas en los resultados del EDA.

    Args:
        df_preproc (pl.DataFrame): DataFrame original
        resultados (Dict): Resultados del EDA

    Notes:
        - Genera gráficos de distribución, cobertura y calidad de datos
        - Optimizado para análisis de homologación de medicamentos
    """

    print("\n📊 CREANDO VISUALIZACIONES DEL EDA...")

    try:
        # Configurar estilo
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('EDA - Dataset de Medicamentos para Homologación',
                     fontsize=16, fontweight='bold')

        # Obtener medicamentos válidos e inválidos
        try:
            mascara_validos = (
                (df_preproc['ESTADO REGISTRO'] == 'Vigente') &
                (df_preproc['ESTADO CUM'] == 'Activo') &
                (df_preproc['MUESTRA MÉDICA'] == 'No')
            )

            validos = df_preproc.filter(mascara_validos)
            invalidos = df_preproc.filter(~mascara_validos)
        except Exception:
            validos = df_preproc.head(0)
            invalidos = df_preproc.head(0)

        # 1. Distribución Válidos vs Inválidos
        if len(validos) > 0 or len(invalidos) > 0:
            labels = ['Válidos', 'Inválidos']
            sizes = [len(validos), len(invalidos)]
            colors = ['lightgreen', 'lightcoral']
            axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%',
                           colors=colors, startangle=90)
            axes[0, 0].set_title('Distribución de Medicamentos')
        else:
            axes[0, 0].text(0.5, 0.5, 'Sin datos de validez',
                            ha='center', va='center')
            axes[0, 0].set_title('Distribución de Medicamentos')

        # 2. Cobertura por Variable Individual
        if 'individual' in resultados['analisis_cobertura'] and resultados['analisis_cobertura']['individual']:
            variables = list(
                resultados['analisis_cobertura']['individual'].keys())
            coberturas = [resultados['analisis_cobertura']['individual'][var]['cobertura_registros']
                          for var in variables]

            colors = ['green' if c >= 80 else 'orange' if c >=
                      50 else 'red' for c in coberturas]
            bars = axes[0, 1].bar(range(len(variables)),
                                  coberturas, color=colors, alpha=0.7)
            axes[0, 1].set_xticks(range(len(variables)))
            axes[0, 1].set_xticklabels([v[:10] + '...' if len(v) > 10 else v for v in variables],
                                       rotation=45, ha='right')
            axes[0, 1].set_ylabel('Cobertura (%)')
            axes[0, 1].set_title('Cobertura por Variable')
            axes[0, 1].set_ylim(0, 100)

            # Agregar valores en las barras
            for bar, valor in zip(bars, coberturas):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{valor:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            axes[0, 1].text(0.5, 0.5, 'Sin datos de cobertura',
                            ha='center', va='center')
            axes[0, 1].set_title('Cobertura por Variable')

        # 3. Top Variables por Cardinalidad
        if 'categoricas' in resultados['variables_analizadas'] and resultados['variables_analizadas']['categoricas']:
            var_names = list(resultados['variables_analizadas']['categoricas'].keys())[
                :7]  # Top 7
            cardinalidades = [resultados['variables_analizadas']['categoricas'][var]['total_unicos']
                              for var in var_names]

            # Ordenar por cardinalidad
            sorted_data = sorted(zip(var_names, cardinalidades),
                                 key=lambda x: x[1], reverse=True)
            var_names_sorted = [x[0] for x in sorted_data]
            cardinalidades_sorted = [x[1] for x in sorted_data]

            axes[0, 2].barh(range(len(var_names_sorted)),
                            cardinalidades_sorted, color='skyblue', alpha=0.7)
            axes[0, 2].set_yticks(range(len(var_names_sorted)))
            axes[0, 2].set_yticklabels(
                [v[:15] + '...' if len(v) > 15 else v for v in var_names_sorted])
            axes[0, 2].set_xlabel('Valores Únicos')
            axes[0, 2].set_title('Cardinalidad por Variable')
            axes[0, 2].set_xscale('log')
        else:
            axes[0, 2].text(0.5, 0.5, 'Sin datos categóricos',
                            ha='center', va='center')
            axes[0, 2].set_title('Cardinalidad por Variable')

        # 4. Cobertura por Combinaciones
        if 'combinaciones' in resultados['analisis_cobertura'] and resultados['analisis_cobertura']['combinaciones']:
            combos = list(resultados['analisis_cobertura']
                          ['combinaciones'].keys())
            coberturas_combo = [resultados['analisis_cobertura']['combinaciones'][combo]['cobertura']
                                for combo in combos]

            colors = ['green' if c >= 80 else 'orange' if c >=
                      50 else 'red' for c in coberturas_combo]
            bars = axes[1, 0].bar(range(len(combos)),
                                  coberturas_combo, color=colors, alpha=0.7)
            axes[1, 0].set_xticks(range(len(combos)))
            axes[1, 0].set_xticklabels([c.replace(' + ', '\n+\n')[:20] + '...' if len(c) > 20 else c.replace(' + ', '\n+\n')
                                        for c in combos], rotation=0, ha='center', fontsize=8)
            axes[1, 0].set_ylabel('Cobertura (%)')
            axes[1, 0].set_title('Cobertura por Combinaciones')
            axes[1, 0].set_ylim(0, 100)

            # Agregar valores en las barras
            for bar, valor in zip(bars, coberturas_combo):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{valor:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'Sin datos de combinaciones',
                            ha='center', va='center')
            axes[1, 0].set_title('Cobertura por Combinaciones')

        # 5. Distribución de CANTIDAD (si existe)
        if 'CANTIDAD' in df_preproc.columns and len(validos) > 0:
            try:
                cantidad_data = validos['CANTIDAD'].drop_nulls().to_list()
                if cantidad_data and len(cantidad_data) > 0:
                    # Filtrar valores positivos y usar log scale
                    cantidad_positiva = [x for x in cantidad_data if x > 0]
                    if cantidad_positiva:
                        cantidad_log = [np.log10(x) for x in cantidad_positiva]
                        axes[1, 1].hist(
                            cantidad_log, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                        axes[1, 1].set_xlabel('log₁₀(CANTIDAD)')
                        axes[1, 1].set_ylabel('Frecuencia')
                        axes[1, 1].set_title(
                            'Distribución de CANTIDAD (Válidos)')
                    else:
                        axes[1, 1].text(
                            0.5, 0.5, 'Sin datos positivos de cantidad', ha='center', va='center')
                        axes[1, 1].set_title('Distribución de CANTIDAD')
                else:
                    axes[1, 1].text(
                        0.5, 0.5, 'Sin datos de cantidad', ha='center', va='center')
                    axes[1, 1].set_title('Distribución de CANTIDAD')
            except Exception:
                axes[1, 1].text(
                    0.5, 0.5, 'Error procesando cantidad', ha='center', va='center')
                axes[1, 1].set_title('Distribución de CANTIDAD')
        else:
            axes[1, 1].text(
                0.5, 0.5, 'Variable CANTIDAD no disponible', ha='center', va='center')
            axes[1, 1].set_title('Distribución de CANTIDAD')

        # 6. Calidad de Datos (% Nulos)
        if 'calidad_datos' in resultados['variables_analizadas'] and resultados['variables_analizadas']['calidad_datos']:
            nulos_data = resultados['variables_analizadas']['calidad_datos']['nulos_por_columna']
            variables_nulos = list(nulos_data.keys())[:10]  # Top 10
            porcentajes_nulos = [nulos_data[var]['porcentaje']
                                 for var in variables_nulos]

            colors = ['red' if p > 20 else 'orange' if p >
                      5 else 'green' for p in porcentajes_nulos]
            bars = axes[1, 2].barh(
                range(len(variables_nulos)), porcentajes_nulos, color=colors, alpha=0.7)
            axes[1, 2].set_yticks(range(len(variables_nulos)))
            axes[1, 2].set_yticklabels(
                [v[:15] + '...' if len(v) > 15 else v for v in variables_nulos])
            axes[1, 2].set_xlabel('% Valores Nulos')
            axes[1, 2].set_title('Calidad de Datos')

            # Agregar valores en las barras
            for bar, valor in zip(bars, porcentajes_nulos):
                width = bar.get_width()
                if width > 0:
                    axes[1, 2].text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                                    f'{valor:.1f}%', ha='left', va='center', fontsize=8)
        else:
            axes[1, 2].text(0.5, 0.5, 'Sin datos de calidad',
                            ha='center', va='center')
            axes[1, 2].set_title('Calidad de Datos')

        plt.tight_layout()
        plt.show()

        print("✅ Visualizaciones creadas exitosamente")

    except Exception as e:
        print(f"⚠️ Error creando visualizaciones: {str(e)}")
        print("Continuando sin visualizaciones...")

# ============================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ============================================================================


def ejecutar_eda_completo(df_preproc: pl.DataFrame):
    """
    Función principal que ejecuta todo el EDA y crea las visualizaciones.

    Args:
        df_preproc (pl.DataFrame): DataFrame preprocesado

    Returns:
        Dict: Resultados completos del análisis
    """

    try:
        # Ejecutar EDA
        resultados = eda_completo_medicamentos(df_preproc)

        # Crear visualizaciones
        crear_visualizaciones_eda(df_preproc, resultados)

        return resultados

    except Exception as e:
        print(f"❌ Error ejecutando EDA: {str(e)}")
        return {}

# ============================================================================
# CÓDIGO DE EJECUCIÓN
# ============================================================================


os.system('cls' if os.name == 'nt' else 'clear')

df_prep = pl.read_parquet('./data/medicamentos_preprocesados.parquet')

# Para ejecutar el EDA completo:
ejecutar_eda_completo(df_prep)
