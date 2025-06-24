"""
file_manager.py
Servicios relacionados con la carga, validación y lectura de archivos.
"""

import os
import shutil
import polars as pl


def cargar(rutas_archivos: dict):
    """
    Carga, valida y mueve los archivos seleccionados por el usuario.

    Args:
        rutas_archivos (dict): Diccionario con claves como 'medicametos_vencidos'
                               y rutas absolutas como valores.
    """
    print("🚀 Iniciando carga de archivos...")

    try:
        # Asegurar que exista la carpeta ./data
        ruta_destino = "./data"
        os.makedirs(ruta_destino, exist_ok=True)

        # Copiar los archivos al destino
        for nombre, ruta in rutas_archivos.items():
            nuevo_path = os.path.join(ruta_destino, f"{nombre}.xlsx")
            shutil.copy2(ruta, nuevo_path)
            # actualiza con nueva ubicación
            rutas_archivos[nombre] = nuevo_path

        # Leer todos los archivos con polars
        df_medicamentos_vencidos = pl.read_excel(rutas_archivos['medicamentos_vencidos'])
        df_medicamentos_vigentes = pl.read_excel(rutas_archivos['medicamentos_vigentes'])
        df_medicamentos_renovacion = pl.read_excel(rutas_archivos['medicamentos_renovacion'])
        df_medicamentos_otros = pl.read_excel(rutas_archivos['medicamentos_otros'])

        print("✅ Archivos leídos correctamente:")
        print(f"   📊 Medicamentos vencidos: {df_medicamentos_vencidos.shape[0]:,} filas")
        print(f"   📊 Medicamentos vigentes: {df_medicamentos_vigentes.shape[0]:,} filas")
        print(f"   📊 Medicamentos renovación: {df_medicamentos_renovacion.shape[0]:,} filas")
        print(f"   📊 Medicamentos otros: {df_medicamentos_otros.shape[0]:,} filas")

        # Validar columnas y tipos
        dataframes = [
            ("df_medicamentos_vencidos", df_medicamentos_vencidos),
            ("df_medicamentos_vigentes", df_medicamentos_vigentes),
            ("df_medicamentos_renovacion", df_medicamentos_renovacion),
            ("df_medicamentos_otros", df_medicamentos_otros)
        ]

        print("\n🔍 COLUMNAS POR DATASET:")
        print("=" * 80)
        for nombre, df in dataframes:
            print(f"\n📋 {nombre}:")
            print(" | ".join(df.columns))

        columnas_iguales = all(
            df.columns == dataframes[0][1].columns for _, df in dataframes[1:])
        tipos_iguales = all(
            df.dtypes == dataframes[0][1].dtypes for _, df in dataframes[1:])

        print(f"\n✅ ¿Todas las columnas son iguales?: {columnas_iguales}")
        print(f"✅ ¿Todos los tipos de datos son iguales?: {tipos_iguales}")

        if columnas_iguales and tipos_iguales:
            print("🎉 Los DataFrames son compatibles para unificación")
        else:
            raise ValueError("⚠️ Los archivos contienen encabezados o tipos de datos diferentes. No se pueden unir.")


        if columnas_iguales and tipos_iguales:
            # Añadir columna identificadora del dataset origen
            df_medicamentos_vencidos = df_medicamentos_vencidos.with_columns(
                pl.lit("medicametos_vencidos").alias("DATASET")
            )
            df_medicamentos_vigentes = df_medicamentos_vigentes.with_columns(
                pl.lit("medicamentos_vigentes").alias("DATASET")
            )
            df_medicamentos_renovacion = df_medicamentos_renovacion.with_columns(
                pl.lit("medicamentos_renovacion").alias("DATASET")
            )
            df_medicamentos_otros = df_medicamentos_otros.with_columns(
                pl.lit("medicamentos_otros").alias("DATASET")
            )

            # Unir todos los DataFrames
            df_unido = pl.concat([
                df_medicamentos_vencidos,
                df_medicamentos_vigentes,
                df_medicamentos_renovacion,
                df_medicamentos_otros
            ])

            print("✅ DataFrames unidos correctamente")
            print(f"📊 Dataset unificado: {df_unido.shape[0]:,} filas × {df_unido.shape[1]} columnas")

        else:
            raise ValueError("❌ No se puede unir. Revisar los archivos manualmente e intente cargarlos de nuevo.")

        df_unido = df_unido.with_columns(
            (pl.col("EXPEDIENTE CUM").cast(pl.Utf8) + pl.lit("-") + pl.col("CONSECUTIVO").cast(pl.Utf8)).alias("CUM")
        )

        df_unido = df_unido.with_columns(
            ((pl.col('ESTADO REGISTRO') == 'Vigente') &
            (pl.col('ESTADO CUM') == 'Activo') &
            (pl.col('MUESTRA MÉDICA') == 'No')).cast(pl.Int8).alias('VALIDO')
        )

        # Reordenar columnas finales
        df_preproc = df_unido.select([
            'CUM',
            'PRODUCTO',
            'EXPEDIENTE CUM',
            'ATC',
            'DESCRIPCIÓN_ATC',
            'VÍA ADMINISTRACIÓN',
            'PRINCIPIO ACTIVO',
            'FORMA FARMACÉUTICA',
            'CANTIDAD CUM',
            'CANTIDAD',
            'UNIDAD MEDIDA',
            'VALIDO',
            'ESTADO REGISTRO',
            'ESTADO CUM',
            'MUESTRA MÉDICA'
        ]).unique().sort("CUM")

        print("✅ Transformaciones completadas")
        print(f"📊 Dataset final: {df_preproc.shape[0]:,} filas × {df_preproc.shape[1]} columnas")

        df_preproc.write_parquet("./data/medicamentos_preprocesados.parquet")

        return {
            "vencidos":     df_medicamentos_vencidos.shape[0],
            "vigentes":     df_medicamentos_vigentes.shape[0],
            "renovacion":   df_medicamentos_renovacion.shape[0],
            "otros":        df_medicamentos_otros.shape[0]
        }

    except Exception as e:
        print(f"❌ Error leyendo archivos: {e}")
