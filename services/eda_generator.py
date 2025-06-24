
from pathlib import Path
import io
import polars as pl
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def generar_eda_pdf():
    """
    Genera el informe EDA completo en PDF y lo guarda en ./output/Eda_Completo.pdf.
    Cada página incluye el logo en la esquina superior derecha. El contenido del EDA
    se toma directamente desde los print() definidos por Sebastián.
    """
    ruta_parquet = Path("./data/medicamentos_preprocesados.parquet")
    ruta_logo    = Path("./assets/logo.png")
    ruta_salida  = Path("./output/Eda_Completo.pdf")
    ruta_salida.parent.mkdir(exist_ok=True)

    if not ruta_parquet.exists():
        raise FileNotFoundError("❌ Archivo de entrada no encontrado en ./data/medicamentos_preprocesados.parquet")
    if not ruta_logo.exists():
        raise FileNotFoundError("❌ Logo no encontrado en ./assets/logo.png")

    df_preproc = pl.read_parquet(str(ruta_parquet))
    filas, columnas = df_preproc.shape

    # Captura de prints
    buffer = io.StringIO()    
    print_fn = lambda *args, **kwargs: print(*args, **kwargs, file=buffer)

    # ========= EDA ==========

    print_fn("="*60)
    print_fn("📊 INFORMACIÓN GENERAL DEL DATASET")
    print_fn("="*60)

    memoria_mb = df_preproc.estimated_size() / (1024**2)
    print_fn(f"📏 Dimensiones: {filas:,} filas × {columnas} columnas")
    print_fn(f"💾 Memoria utilizada: {memoria_mb:.2f} MB")
    print_fn(f"📁 Tamaño promedio por registro: {memoria_mb*1024/filas:.2f} KB")

    print_fn("\n📋 COLUMNAS DEL DATASET:")
    for i, col in enumerate(df_preproc.columns, 1):
        dtype = str(df_preproc[col].dtype)
        print_fn(f"   {i:2d}. {col:<25} | {dtype}")

    print_fn("\n🔍 TIPOS DE DATOS:")
    tipos_datos = {}
    for dtype in df_preproc.dtypes:
        dtype_str = str(dtype)
        tipos_datos[dtype_str] = tipos_datos.get(dtype_str, 0) + 1
    print_fn(tipos_datos)

    # ANÁLISIS DE CALIDAD DE DATOS
    print_fn("\n🔍 ANÁLISIS DE CALIDAD DE DATOS")
    print_fn("="*80)

    # Valores nulos por columna
    print_fn("\n❌ VALORES NULOS POR COLUMNA:")
    hay_nulos = False
    nulos_por_columna = {}

    for col in df_preproc.columns:
        try:
            nulos = df_preproc[col].null_count()
            porcentaje_nulos = (nulos / filas) * 100
            nulos_por_columna[col] = {'cantidad': nulos, 'porcentaje': porcentaje_nulos}

            if nulos > 0:
                print_fn(f"   🔸 {col:<25}: {nulos:>8,} ({porcentaje_nulos:>5.1f}%)")
                hay_nulos = True
        except Exception as e:
            print_fn(f"   ⚠️ Error procesando {col}: {str(e)}")
            nulos_por_columna[col] = {'cantidad': 0, 'porcentaje': 0.0}

    if not hay_nulos:
        print_fn("   ✅ No hay valores nulos en el dataset")

    # Valores únicos por columna
    print_fn("\n🔢 CARDINALIDAD POR COLUMNA:")
    unicos_por_columna = {}

    for col in df_preproc.columns:
        try:
            unicos = df_preproc[col].n_unique()
            porcentaje_unicos = (unicos / filas) * 100
            unicos_por_columna[col] = {'cantidad': unicos, 'porcentaje': porcentaje_unicos}

            # Clasificar tipo de variable por cardinalidad
            if porcentaje_unicos > 95:
                tipo = "ID/ÚNICA"
            elif porcentaje_unicos > 50:
                tipo = "ALTA_CARD"
            elif porcentaje_unicos > 10:
                tipo = "MEDIA_CARD"
            else:
                tipo = "BAJA_CARD"

            print_fn(f"   🔸 {col:<25}: {unicos:>8,} únicos ({porcentaje_unicos:>5.1f}%) [{tipo}]")
        except Exception as e:
            print_fn(f"   ⚠️ Error procesando {col}: {str(e)}")
            unicos_por_columna[col] = {'cantidad': 0, 'porcentaje': 0.0}

    # ANÁLISIS DE MEDICAMENTOS VÁLIDOS VS INVÁLIDOS
    print_fn("\n🎯 ANÁLISIS DE VALIDEZ DE MEDICAMENTOS")
    print_fn("="*80)

    print_fn("📋 Criterios de validez:")
    print_fn("   ✅ ESTADO REGISTRO = 'Vigente'")
    print_fn("   ✅ ESTADO CUM = 'Activo'")
    print_fn("   ✅ MUESTRA MÉDICA = 'No'")

    # Identificar medicamentos válidos
    try:
        mascara_validos = (
            (df_preproc['ESTADO REGISTRO'] == 'Vigente') &
            (df_preproc['ESTADO CUM'] == 'Activo') &
            (df_preproc['MUESTRA MÉDICA'] == 'No')
        )

        validos = df_preproc.filter(mascara_validos)
        invalidos = df_preproc.filter(~mascara_validos)

        print_fn("\n📊 DISTRIBUCIÓN DE VALIDEZ:")
        print_fn(f"   ✅ Medicamentos VÁLIDOS: {len(validos):,} ({len(validos)/filas*100:.1f}%)")
        print_fn(f"   ❌ Medicamentos INVÁLIDOS: {len(invalidos):,} ({len(invalidos)/filas*100:.1f}%)")        # Análisis de motivos de invalidez
        print_fn("\n🔍 ANÁLISIS DE MOTIVOS DE INVALIDEZ:")

        no_vigente = df_preproc.filter(pl.col('ESTADO REGISTRO') != 'Vigente').height
        no_activo = df_preproc.filter(pl.col('ESTADO CUM') != 'Activo').height
        es_muestra = df_preproc.filter(pl.col('MUESTRA MÉDICA') == 'Si').height

        print_fn(f"   📋 Estado Registro ≠ 'Vigente': {no_vigente:,}")
        print_fn(f"   📋 Estado CUM ≠ 'Activo': {no_activo:,}")
        print_fn(f"   📋 Es Muestra Médica: {es_muestra:,}")

        # Análisis de solapamiento
        print_fn("\n📈 ANÁLISIS DE SOLAPAMIENTO:")
        solo_estado = df_preproc.filter(
            (pl.col('ESTADO REGISTRO') != 'Vigente') & 
            (pl.col('ESTADO CUM') == 'Activo') & 
            (pl.col('MUESTRA MÉDICA') == 'No')
        ).height
        solo_cum = df_preproc.filter(
            (pl.col('ESTADO REGISTRO') == 'Vigente') & 
            (pl.col('ESTADO CUM') != 'Activo') & 
            (pl.col('MUESTRA MÉDICA') == 'No')
        ).height
        solo_muestra = df_preproc.filter(
            (pl.col('ESTADO REGISTRO') == 'Vigente') & 
            (pl.col('ESTADO CUM') == 'Activo') & 
            (pl.col('MUESTRA MÉDICA') == 'Si')
        ).height
        
        print_fn(f"   📋 Solo por Estado Registro: {solo_estado:,}")
        print_fn(f"   📋 Solo por Estado CUM: {solo_cum:,}")
        print_fn(f"   📋 Solo por Muestra Médica: {solo_muestra:,}")

    except Exception as e:
        print_fn(f"   ⚠️ Error en análisis de validez: {str(e)}")
        validos = df_preproc.head(0)
        invalidos = df_preproc.head(0)

    # ANÁLISIS DE VARIABLES CATEGÓRICAS PRINCIPALES
    print_fn("\n🏷️ ANÁLISIS DE VARIABLES CATEGÓRICAS PRINCIPALES")
    print_fn("="*80)

    variables_categoricas = [
        'PRINCIPIO ACTIVO', 'ATC', 'VÍA ADMINISTRACIÓN',
        'FORMA FARMACÉUTICA', 'UNIDAD MEDIDA', 'CANTIDAD CUM'
    ]

    for variable in variables_categoricas:
        if variable not in df_preproc.columns:
            print_fn(f"\n⚠️ Variable {variable} no encontrada en el dataset")
            continue

        print_fn(f"\n{'='*80}")
        print_fn(f"📊 ANÁLISIS DE: {variable}")
        print_fn(f"{'='*80}")
        
        try:
            # Análisis general
            total_unicos = df_preproc[variable].n_unique()
            total_nulos = df_preproc[variable].null_count()

            # Análisis por válidos/inválidos
            unicos_validos = validos[variable].n_unique() if len(validos) > 0 else 0
            unicos_invalidos = invalidos[variable].n_unique() if len(invalidos) > 0 else 0

            print_fn(f"   📊 Total únicos: {total_unicos:,}")
            print_fn(f"   📊 Valores nulos: {total_nulos:,}")
            print_fn(f"   📊 Únicos en válidos: {unicos_validos:,}")
            print_fn(f"   📊 Únicos en inválidos: {unicos_invalidos:,}")

            # Top valores más frecuentes
            top_valores = df_preproc[variable].value_counts().sort("count", descending=True).head(5)
            print_fn("   🏆 TOP 5 valores más frecuentes:")

            for i, fila in enumerate(top_valores.iter_rows(named=True), 1):
                valor = fila[variable]
                conteo = fila['count']
                porcentaje = (conteo / filas) * 100
                valor_mostrar = str(valor)[:40] + "..." if str(valor) and len(str(valor)) > 40 else str(valor)
                print_fn(f"      {i}. {valor_mostrar:<43} | {conteo:>8,} ({porcentaje:>5.1f}%)")

        except Exception as e:
            print_fn(f"   ⚠️ Error procesando variable {variable}: {str(e)}")

    # ANÁLISIS DE COBERTURA PARA HOMOLOGACIÓN
    print_fn("\n🎯 ANÁLISIS DE COBERTURA PARA HOMOLOGACIÓN")
    print_fn("="*80)

    print_fn("📋 Objetivo: Determinar qué variables permiten mayor cobertura de homologación")

    # Recalcular válidos e inválidos correctamente
    print_fn("\n🔍 Verificando filtros de validez...")

    # Crear filtros correctos para medicamentos válidos
    validos = df_preproc.filter(
        (pl.col('ESTADO REGISTRO') == 'Vigente') &
        (pl.col('ESTADO CUM') == 'Activo') &
        (pl.col('MUESTRA MÉDICA') == 'No')
    )

    # Crear filtros correctos para medicamentos inválidos
    invalidos = df_preproc.filter(
        (pl.col('ESTADO REGISTRO') != 'Vigente') |
        (pl.col('ESTADO CUM') != 'Activo') |
        (pl.col('MUESTRA MÉDICA') != 'No')
    )

    print_fn(f"✅ Medicamentos VÁLIDOS encontrados: {len(validos):,}")
    print_fn(f"❌ Medicamentos INVÁLIDOS encontrados: {len(invalidos):,}")
    print_fn(f"📊 Total verificado: {len(validos) + len(invalidos):,}")

    # Variables clave para análisis de cobertura
    variables_clave = ['PRINCIPIO ACTIVO', 'ATC', 'VÍA ADMINISTRACIÓN', 'FORMA FARMACÉUTICA']
    cobertura_individual = {}

    for variable in variables_clave:
        if variable not in df_preproc.columns:
            continue

        try:
            # Conjuntos únicos
            valores_validos = set(validos[variable].unique().to_list()) if len(validos) > 0 else set()
            valores_invalidos = set(invalidos[variable].unique().to_list()) if len(invalidos) > 0 else set()
            valores_comunes = valores_validos.intersection(valores_invalidos)

            # Cobertura por valores únicos
            cobertura_valores = len(valores_comunes) / len(valores_invalidos) * 100 if len(valores_invalidos) > 0 else 0

            # Cobertura por registros
            if len(valores_comunes) > 0 and len(invalidos) > 0:
                invalidos_cubiertos = invalidos.filter(pl.col(variable).is_in(list(valores_comunes)))
                cobertura_registros = len(invalidos_cubiertos) / len(invalidos) * 100
            else:
                cobertura_registros = 0

            print_fn(f"\n📊 COBERTURA - {variable}:")
            print_fn(f"   🔸 Valores únicos en válidos: {len(valores_validos):,}")
            print_fn(f"   🔸 Valores únicos en inválidos: {len(valores_invalidos):,}")
            print_fn(f"   🔸 Valores comunes: {len(valores_comunes):,}")
            print_fn(f"   🎯 Cobertura por valores: {cobertura_valores:.1f}%")
            print_fn(f"   🎯 Cobertura por registros: {cobertura_registros:.1f}%")

            cobertura_individual[variable] = {
                'valores_validos': len(valores_validos),
                'valores_invalidos': len(valores_invalidos),
                'valores_comunes': len(valores_comunes),
                'cobertura_valores': cobertura_valores,
                'cobertura_registros': cobertura_registros
            }

        except Exception as e:
            print_fn(f"\n⚠️ Error procesando cobertura de {variable}: {str(e)}")
            cobertura_individual[variable] = {
                'valores_validos': 0,
                'valores_invalidos': 0,
                'valores_comunes': 0,
                'cobertura_valores': 0,
                'cobertura_registros': 0
            }

    # ANÁLISIS DE COMBINACIONES DE VARIABLES
    print_fn("\n🔗 ANÁLISIS DE COMBINACIONES DE VARIABLES")
    print_fn("="*80)

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
        print_fn(f"\n📊 COMBINACIÓN: {combo_str}")
        print_fn("-" * 60)

        try:
            # Crear claves combinadas
            if len(validos) > 0:
                validos_combo = validos.with_columns(
                    pl.concat_str([pl.col(c).cast(pl.Utf8) for c in combo], separator="|").alias("CLAVE_COMBO")
                )
            else:
                validos_combo = pl.DataFrame()

            if len(invalidos) > 0:
                invalidos_combo = invalidos.with_columns(
                    pl.concat_str([pl.col(c).cast(pl.Utf8) for c in combo], separator="|").alias("CLAVE_COMBO")
                )
            else:
                invalidos_combo = pl.DataFrame()

            if len(validos_combo) > 0 and len(invalidos_combo) > 0:
                claves_validas = set(validos_combo['CLAVE_COMBO'].unique().to_list())
                claves_invalidas = set(invalidos_combo['CLAVE_COMBO'].unique().to_list())
                claves_comunes = claves_validas.intersection(claves_invalidas)

                # Medicamentos inválidos cubiertos
                if len(claves_comunes) > 0:
                    invalidos_cubiertos = invalidos_combo.filter(
                        pl.col('CLAVE_COMBO').is_in(list(claves_comunes))
                    )
                else:
                    invalidos_cubiertos = pl.DataFrame()

                cobertura_combo = len(invalidos_cubiertos) / len(invalidos) * 100 if len(invalidos) > 0 else 0

                print_fn(f"   🔸 Combinaciones únicas en válidos: {len(claves_validas):,}")
                print_fn(f"   🔸 Combinaciones únicas en inválidos: {len(claves_invalidas):,}")
                print_fn(f"   🔸 Combinaciones comunes: {len(claves_comunes):,}")
                print_fn(f"   🎯 Cobertura: {cobertura_combo:.1f}%")

                cobertura_combinaciones[combo_str] = {
                    'claves_validas': len(claves_validas),
                    'claves_invalidas': len(claves_invalidas),
                    'claves_comunes': len(claves_comunes),
                    'cobertura': cobertura_combo
                }
            else:
                print_fn("   ⚠️ No hay datos suficientes para analizar combinación")
                cobertura_combinaciones[combo_str] = {
                    'claves_validas': 0,
                    'claves_invalidas': 0,
                    'claves_comunes': 0,
                    'cobertura': 0
                }

        except Exception as e:
            print_fn(f"   ⚠️ Error procesando combinación {combo_str}: {str(e)}")
            cobertura_combinaciones[combo_str] = {
                'claves_validas': 0,
                'claves_invalidas': 0,
                'claves_comunes': 0,
                'cobertura': 0
            }

    # RECOMENDACIONES BASADAS EN EL ANÁLISIS
    print_fn("\n💡 RECOMENDACIONES PARA EL MODELO")
    print_fn("="*80)

    # Identificar mejor estrategia basada en cobertura
    if cobertura_individual:
        mejor_individual = max(cobertura_individual.items(), key=lambda x: x[1]['cobertura_registros'])
    else:
        mejor_individual = ("N/A", {'cobertura_registros': 0})

    if cobertura_combinaciones:
        mejor_combinacion = max(cobertura_combinaciones.items(), key=lambda x: x[1]['cobertura'])
    else:
        mejor_combinacion = None

    print_fn("\n🏆 VARIABLE CON MEJOR COBERTURA INDIVIDUAL:")
    print_fn(f"   {mejor_individual[0]}: {mejor_individual[1]['cobertura_registros']:.1f}%")

    if mejor_combinacion:
        print_fn("\n🏆 COMBINACIÓN CON MEJOR COBERTURA:")
        print_fn(f"   {mejor_combinacion[0]}: {mejor_combinacion[1]['cobertura']:.1f}%")

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

    print_fn("\n🔥 CLASIFICACIÓN DE VARIABLES:")
    print_fn(f"   CRÍTICAS (≥80% cobertura): {variables_criticas}")
    print_fn(f"   IMPORTANTES (≥50% cobertura): {variables_importantes}")
    print_fn(f"   OPCIONALES (<50% cobertura): {variables_opcionales}")

    # Estrategia recomendada
    if mejor_combinacion and mejor_combinacion[1]['cobertura'] > 80:
        estrategia = f"Usar filtro: {mejor_combinacion[0]}"
    elif mejor_individual[1]['cobertura_registros'] > 70:
        estrategia = f"Usar filtro: {mejor_individual[0]} + variables complementarias"
    else:
        estrategia = "Usar estrategia híbrida con múltiples variables"

    print_fn("\n🎯 ESTRATEGIA RECOMENDADA:")
    print_fn(f"   {estrategia}")

    # RESUMEN EJECUTIVO
    print_fn("\n" + "="*100)
    print_fn("📋 RESUMEN EJECUTIVO DEL EDA")
    print_fn("="*100)

    print_fn("\n📊 DATOS CLAVE:")
    print_fn(f"   • Dataset: {filas:,} medicamentos, {columnas} variables")
    print_fn(f"   • Medicamentos válidos: {len(validos):,} ({len(validos)/filas*100:.1f}%)")
    print_fn(f"   • Medicamentos inválidos: {len(invalidos):,} ({len(invalidos)/filas*100:.1f}%)")

    print_fn("\n🎯 HALLAZGOS PRINCIPALES:")
    print_fn(f"   • Mejor variable individual: {mejor_individual[0]} ({mejor_individual[1]['cobertura_registros']:.1f}% cobertura)")
    if mejor_combinacion:
        print_fn(f"   • Mejor combinación: {mejor_combinacion[0]} ({mejor_combinacion[1]['cobertura']:.1f}% cobertura)")
    print_fn(f"   • Variables críticas identificadas: {len(variables_criticas)}")

    contenido = buffer.getvalue().split("\n")

    # Crear PDF
    c = canvas.Canvas(str(ruta_salida), pagesize=A4)
    width, height = A4
    x_margin, y_margin = 50, 60
    line_height = 14
    y_position = height - y_margin - 80  # Espacio para el logo

    try:
        logo = ImageReader(str(ruta_logo))
        logo_disponible = True
    except Exception as e:
        print(f"⚠️ No se pudo cargar el logo: {e}")
        logo_disponible = False
        logo = None

    def encabezado():
        if logo_disponible and logo is not None:
            # Posición del logo en esquina superior derecha
            logo_width = 80
            logo_height = 40
            logo_x = width - logo_width - 20
            logo_y = height - logo_height - 20
            c.drawImage(logo, logo_x, logo_y, width=logo_width, height=logo_height, preserveAspectRatio=True)

    encabezado()

    for linea in contenido:
        if y_position <= y_margin + 20:  # Margen inferior más espacio
            c.showPage()
            encabezado()
            y_position = height - y_margin - 80  # Resetear posición considerando logo
        
        c.setFont("Helvetica", 10)
        # Truncar líneas muy largas para que no se salgan de la página
        linea_truncada = linea[:100] if len(linea) > 100 else linea
        c.drawString(x_margin, y_position, linea_truncada)
        y_position -= line_height

    c.save()
    print(f"✅ PDF generado exitosamente: {ruta_salida}")
    return ruta_salida
