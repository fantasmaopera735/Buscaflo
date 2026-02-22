# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import calendar
from collections import Counter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

RUTA_CSV = 'flotodo.csv'

st.set_page_config(
    page_title="Flotodo - Suite Ultimate",
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Flotodo - Suite Ultimate")

@st.cache_resource
def cargar_datos_flotodo(_ruta_csv):
    """
    Carga datos de Flotodo con soporte para formatos de 5 o 6 columnas.
    Formato esperado:
    - 6 columnas: Fecha;Tipo;Centena;Fijo;1erCorrido;2doCorrido
    - 5 columnas: Fecha;Tipo;Fijo;1erCorrido;2doCorrido (sin Centena)
    """
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"No se encontro el archivo {_ruta_csv}")
            st.stop()
        
        # Leer archivo con manejo robusto de errores
        lineas_validas = []
        lineas_invalidas = []
        
        with open(_ruta_csv, 'r', encoding='latin-1') as f:
            lineas = f.readlines()
        
        # Detectar número de columnas del header
        header = lineas[0].strip()
        num_cols_header = len(header.split(';'))
        
        for i, linea in enumerate(lineas[1:], start=2):  # Empezar desde línea 2 (después del header)
            campos = linea.strip().split(';')
            num_campos = len(campos)
            
            # Aceptar líneas con el mismo número de campos que el header
            if num_campos == num_cols_header:
                lineas_validas.append(linea)
            elif num_campos == num_cols_header - 1:
                # Línea con un campo menos, puede faltar el último campo
                campos.append('0')  # Añadir campo vacío
                lineas_validas.append(';'.join(campos) + '\n')
            elif num_campos == num_cols_header + 1:
                # Línea con un campo extra, combinar últimos dos campos
                campos[-2] = campos[-2] + campos[-1]
                campos.pop()
                lineas_validas.append(';'.join(campos) + '\n')
            else:
                lineas_invalidas.append((i, num_campos, linea.strip()[:50]))
        
        # Mostrar advertencia si hay líneas inválidas
        if lineas_invalidas:
            st.warning(f"Se omitieron {len(lineas_invalidas)} líneas con formato incorrecto.")
            with st.expander("Ver líneas omitidas"):
                for num_linea, num_campos, contenido in lineas_invalidas[:10]:
                    st.text(f"Línea {num_linea}: {num_campos} campos - {contenido}...")
                if len(lineas_invalidas) > 10:
                    st.text(f"... y {len(lineas_invalidas) - 10} más")
        
        # Reconstruir CSV con líneas válidas
        contenido_limpio = header + '\n' + ''.join(lineas_validas)
        
        # Crear DataFrame desde el contenido limpio
        from io import StringIO
        df = pd.read_csv(StringIO(contenido_limpio), sep=';', encoding='latin-1')
        
        # Normalizar nombres de columnas
        columnas_originales = df.columns.tolist()
        columnas_normalizadas = {col: col.strip() for col in columnas_originales}
        df.columns = [columnas_normalizadas.get(col, col) for col in columnas_originales]
        
        # Detectar columnas automaticamente
        col_fecha = None
        col_tipo = None
        col_centena = None
        col_fijo = None
        col_1er = None
        col_2do = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'fecha' in col_lower:
                col_fecha = col
            if 'tipo' in col_lower or 'sorteo' in col_lower:
                col_tipo = col
            if 'centena' in col_lower or 'cent' in col_lower:
                col_centena = col
            if col_lower == 'fijo':
                col_fijo = col
            if 'primer' in col_lower or '1er' in col_lower or '1correo' in col_lower or 'primero' in col_lower:
                col_1er = col
            if 'segundo' in col_lower or '2do' in col_lower or '2correo' in col_lower:
                col_2do = col
        
        # Si no se encontraron por nombre, intentar por posicion
        # Soporta tanto 5 como 6 columnas
        if col_fecha is None and len(df.columns) >= 1:
            col_fecha = df.columns[0]
        if col_tipo is None and len(df.columns) >= 2:
            col_tipo = df.columns[1]
        
        # Detectar si hay columna de Centena (6 columnas) o no (5 columnas)
        if len(df.columns) >= 6:
            # Formato con Centena: Fecha, Tipo, Centena, Fijo, 1er, 2do
            if col_centena is None:
                col_centena = df.columns[2]
            if col_fijo is None:
                col_fijo = df.columns[3]
            if col_1er is None:
                col_1er = df.columns[4]
            if col_2do is None:
                col_2do = df.columns[5]
        elif len(df.columns) >= 5:
            # Formato sin Centena: Fecha, Tipo, Fijo, 1er, 2do
            if col_fijo is None:
                col_fijo = df.columns[2]
            if col_1er is None:
                col_1er = df.columns[3]
            if col_2do is None:
                col_2do = df.columns[4]
        
        # Renombrar columnas a nombres estandar
        mapeo = {}
        if col_fecha:
            mapeo[col_fecha] = 'Fecha'
        if col_tipo:
            mapeo[col_tipo] = 'Tipo_Sorteo'
        if col_centena:
            mapeo[col_centena] = 'Centena'
        if col_fijo:
            mapeo[col_fijo] = 'Fijo'
        if col_1er:
            mapeo[col_1er] = 'Primer_Corrido'
        if col_2do:
            mapeo[col_2do] = 'Segundo_Corrido'
        
        df = df.rename(columns=mapeo)
        
        # Si no hay columna Centena, crearla con valores 0
        if 'Centena' not in df.columns:
            df['Centena'] = 0
        
        # Convertir fecha con multiples formatos
        def convertir_fecha(valor):
            """Intenta convertir fecha con diferentes formatos"""
            if pd.isna(valor):
                return pd.NaT
            
            valor_str = str(valor).strip()
            
            # Lista de formatos a intentar
            formatos = [
                '%d/%m/%Y',    # 01/12/2024
                '%d-%m-%Y',    # 01-12-2024
                '%Y-%m-%d',    # 2024-12-01
                '%d/%m/%y',    # 01/12/24
                '%d-%m-%y',    # 01-12-24
                '%Y/%m/%d',    # 2024/12/01
            ]
            
            for fmt in formatos:
                try:
                    return pd.to_datetime(valor_str, format=fmt)
                except:
                    continue
            
            # Si ningun formato funciona, intentar con pandas automatico
            try:
                return pd.to_datetime(valor_str, dayfirst=True)
            except:
                return pd.NaT
        
        # Aplicar conversion de fecha
        df['Fecha'] = df['Fecha'].apply(convertir_fecha)
        
        # Contar fechas invalidas
        fechas_invalidas = df['Fecha'].isna().sum()
        if fechas_invalidas > 0:
            st.warning(f"Se encontraron {fechas_invalidas} fechas con formato invalido que fueron omitidas.")
        
        df.dropna(subset=['Fecha'], inplace=True)
        
        # Normalizar tipo de sorteo
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map({
                'TARDE': 'T', 'T': 'T',
                'NOCHE': 'N', 'N': 'N'
            }).fillna('OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        # Crear DataFrame de Fijos para análisis
        if 'Fijo' in df.columns:
            df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
            df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
            df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce')
            df_fijos = df_fijos.dropna(subset=['Numero'])
            df_fijos['Numero'] = df_fijos['Numero'].astype(int)
        else:
            st.error("No se encontro la columna Fijo")
            st.stop()
        
        # Orden: N (Noche) > T (Tarde)
        draw_order_map = {'T': 0, 'N': 1, 'OTRO': 2}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(2)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        
        return df_fijos, df
        
    except pd.errors.EmptyDataError:
        st.error("El archivo CSV esta vacio.")
        st.stop()
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

def extraer_digitos_sesion(centena, fijo, primer_corr, segundo_corr):
    """Extrae los digitos de una sesion organizados por tipo"""
    digitos = {
        'centena': [],
        'fijo_dec': [],
        'fijo_uni': [],
        'corrido1_dec': [],
        'corrido1_uni': [],
        'corrido2_dec': [],
        'corrido2_uni': [],
        'todos': []
    }
    
    # Centena (1 digito)
    try:
        c = int(float(centena))
        digitos['centena'].append(c)
        digitos['todos'].append(c)
    except:
        pass
    
    # Fijo (2 digitos)
    try:
        f = int(float(fijo))
        digitos['fijo_dec'].append(f // 10)
        digitos['fijo_uni'].append(f % 10)
        digitos['todos'].extend([f // 10, f % 10])
    except:
        pass
    
    # Primer Corrido (2 digitos)
    try:
        p1 = int(float(primer_corr))
        digitos['corrido1_dec'].append(p1 // 10)
        digitos['corrido1_uni'].append(p1 % 10)
        digitos['todos'].extend([p1 // 10, p1 % 10])
    except:
        pass
    
    # Segundo Corrido (2 digitos)
    try:
        p2 = int(float(segundo_corr))
        digitos['corrido2_dec'].append(p2 // 10)
        digitos['corrido2_uni'].append(p2 % 10)
        digitos['todos'].extend([p2 // 10, p2 % 10])
    except:
        pass
    
    return digitos

def analizar_dia_completo(df_completo, fecha):
    """Analiza un dia completo y devuelve los digitos faltantes"""
    df_dia = df_completo[df_completo['Fecha'].dt.date == fecha.date()].copy()
    
    if df_dia.empty:
        return None, "Sin datos para esa fecha"
    
    todos_digitos = []
    sesiones_encontradas = []
    detalle_digitos = []
    
    # Digitos por tipo
    digitos_centena = []
    digitos_fijo = []
    digitos_corr1 = []
    digitos_corr2 = []
    
    for _, row in df_dia.iterrows():
        centena = row.get('Centena', 0)
        fijo = row.get('Fijo', 0)
        primer_corr = row.get('Primer_Corrido', 0)
        segundo_corr = row.get('Segundo_Corrido', 0)
        
        digitos = extraer_digitos_sesion(centena, fijo, primer_corr, segundo_corr)
        todos_digitos.extend(digitos['todos'])
        
        digitos_centena.extend(digitos['centena'])
        digitos_fijo.extend(digitos['fijo_dec'] + digitos['fijo_uni'])
        digitos_corr1.extend(digitos['corrido1_dec'] + digitos['corrido1_uni'])
        digitos_corr2.extend(digitos['corrido2_dec'] + digitos['corrido2_uni'])
        
        sesiones_encontradas.append(row['Tipo_Sorteo'])
        
        detalle_digitos.append({
            'Sesion': row['Tipo_Sorteo'],
            'Centena': centena,
            'Fijo': fijo,
            '1er_Corrido': primer_corr,
            '2do_Corrido': segundo_corr,
            'Digitos': digitos['todos']
        })
    
    # Calcular faltantes
    todos_dig = set(range(10))
    presentes = set(todos_digitos)
    faltantes = todos_dig - presentes
    
    return {
        'digitos_presentes': sorted(list(presentes)),
        'digitos_faltantes': sorted(list(faltantes)),
        'sesiones': sesiones_encontradas,
        'total_digitos': len(todos_digitos),
        'detalle': detalle_digitos,
        'digitos_lista': todos_digitos,
        'por_tipo': {
            'centena': digitos_centena,
            'fijo': digitos_fijo,
            'corrido1': digitos_corr1,
            'corrido2': digitos_corr2
        }
    }, None

def estadisticas_digitos_separadas(df_completo, dias_atras=180):
    """Calcula estadisticas por digito separadas por tipo de columna"""
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    
    df_filtrado = df_completo[df_completo['Fecha'] >= fecha_inicio].copy()
    
    # Contadores por tipo
    contadores = {
        'general': Counter(),
        'centena': Counter(),
        'fijo': Counter(),
        'corrido1': Counter(),
        'corrido2': Counter()
    }
    
    # Ultima aparicion por tipo
    ultima_aparicion = {
        'general': {d: None for d in range(10)},
        'centena': {d: None for d in range(10)},
        'fijo': {d: None for d in range(10)},
        'corrido1': {d: None for d in range(10)},
        'corrido2': {d: None for d in range(10)}
    }
    
    fechas_unicas = sorted(df_filtrado['Fecha'].dt.date.unique())
    
    for fecha in fechas_unicas:
        fecha_dt = datetime.combine(fecha, datetime.min.time())
        resultado, _ = analizar_dia_completo(df_filtrado, fecha_dt)
        
        if resultado:
            # General
            for d in resultado['digitos_presentes']:
                contadores['general'][d] += 1
                ultima_aparicion['general'][d] = fecha
            
            # Por tipo
            for d in resultado['por_tipo']['centena']:
                contadores['centena'][d] += 1
                ultima_aparicion['centena'][d] = fecha
            
            for d in resultado['por_tipo']['fijo']:
                contadores['fijo'][d] += 1
                ultima_aparicion['fijo'][d] = fecha
            
            for d in resultado['por_tipo']['corrido1']:
                contadores['corrido1'][d] += 1
                ultima_aparicion['corrido1'][d] = fecha
            
            for d in resultado['por_tipo']['corrido2']:
                contadores['corrido2'][d] += 1
                ultima_aparicion['corrido2'][d] = fecha
    
    # Calcular estadisticas
    fecha_hoy_date = fecha_hoy.date()
    stats = {}
    
    for tipo in ['general', 'centena', 'fijo', 'corrido1', 'corrido2']:
        datos = []
        total_sorteos = sum(contadores[tipo].values()) if contadores[tipo] else 1
        
        for d in range(10):
            freq = contadores[tipo].get(d, 0)
            ultima = ultima_aparicion[tipo][d]
            
            dias_sin = (fecha_hoy_date - ultima).days if ultima else 999
            porcentaje = round((freq / total_sorteos) * 100, 1) if total_sorteos > 0 else 0
            
            datos.append({
                'Digito': d,
                'Frecuencia': freq,
                'Porcentaje': f"{porcentaje}%",
                'Dias_Sin_Aparecer': dias_sin if ultima else 'N/A',
                'Ultima': ultima.strftime('%d/%m') if ultima else 'N/A'
            })
        
        stats[tipo] = pd.DataFrame(datos)
    
    return stats

def backtest_digito_faltante(df_completo, dias_atras=90):
    """Evalua la efectividad de la estrategia del digito faltante"""
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    
    fechas_unicas = sorted(df_completo['Fecha'].dt.date.unique())
    
    resultados = []
    aciertos = 0
    total_evaluados = 0
    
    for i, fecha in enumerate(fechas_unicas):
        fecha_dt = datetime.combine(fecha, datetime.min.time())
        
        if i >= len(fechas_unicas) - 1:
            continue
        
        fecha_siguiente = fechas_unicas[i + 1]
        
        if fecha_dt < fecha_inicio:
            continue
        
        resultado_dia, error = analizar_dia_completo(df_completo, fecha_dt)
        if error or not resultado_dia['digitos_faltantes']:
            continue
        
        faltantes = resultado_dia['digitos_faltantes']
        
        df_siguiente = df_completo[df_completo['Fecha'].dt.date == fecha_siguiente]
        if df_siguiente.empty:
            continue
        
        fijos_siguiente = df_siguiente['Fijo'].tolist()
        digitos_fijos_siguiente = set()
        for f in fijos_siguiente:
            try:
                f_int = int(float(f))
                digitos_fijos_siguiente.add(f_int // 10)
                digitos_fijos_siguiente.add(f_int % 10)
            except:
                pass
        
        coincidencias = [d for d in faltantes if d in digitos_fijos_siguiente]
        acierto = len(coincidencias) > 0
        
        if acierto:
            aciertos += 1
        total_evaluados += 1
        
        resultados.append({
            'Fecha': fecha,
            'Faltantes': ','.join(map(str, faltantes)),
            'Fijos_Sig': ','.join([f"{int(float(f)):02d}" for f in fijos_siguiente]),
            'Coincidencia': 'SI' if acierto else 'NO',
            'Digitos_Coinc': ','.join(map(str, coincidencias)) if coincidencias else '-'
        })
    
    efectividad = (aciertos / total_evaluados * 100) if total_evaluados > 0 else 0
    
    return {
        'resultados': resultados,
        'total_evaluados': total_evaluados,
        'aciertos': aciertos,
        'efectividad': round(efectividad, 2)
    }

def analizar_siguientes(df_fijos, numero_busqueda, ventana_sorteos):
    indices = df_fijos[df_fijos['Numero'] == numero_busqueda].index.tolist()
    if not indices:
        return None, 0
    lista_s = []
    for idx in indices:
        i = idx + 1
        f = idx + ventana_sorteos + 1
        if i < len(df_fijos):
            lista_s.extend(df_fijos.iloc[i:f]['Numero'].tolist())
    if not lista_s:
        return None, len(indices)
    c = Counter(lista_s)
    r = pd.DataFrame.from_dict(c, orient='index', columns=['Frecuencia'])
    r['Probabilidad'] = (r['Frecuencia'] / len(lista_s) * 100).round(2)
    r['Numero'] = [f"{int(x):02d}" for x in r.index]
    return r.sort_values('Frecuencia', ascending=False), len(indices)

def analizar_almanaque(df_fijos, dia_inicio, dia_fin, meses_atras):
    fecha_hoy = datetime.now()
    bloques_validos = []
    nombres_bloques = []
    
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_i = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            if f_i > f_f:
                continue
            df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
        except:
            continue
    
    if not bloques_validos:
        return {'success': False, 'mensaje': 'Sin datos'}
    
    df_total = pd.concat(bloques_validos)
    df_total['Decena'] = df_total['Numero'] // 10
    df_total['Unidad'] = df_total['Numero'] % 10
    
    cnt_d = df_total['Decena'].value_counts().reindex(range(10), fill_value=0)
    cnt_u = df_total['Unidad'].value_counts().reindex(range(10), fill_value=0)
    
    def clasificar(serie):
        df_t = serie.sort_values(ascending=False).reset_index()
        df_t.columns = ['Digito', 'Frecuencia']
        conds = [(df_t.index < 3), (df_t.index < 6)]
        vals = ['Caliente', 'Tibio']
        df_t['Estado'] = np.select(conds, vals, default='Frio')
        mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
        return df_t, mapa
    
    df_dec, mapa_d = clasificar(cnt_d)
    df_uni, mapa_u = clasificar(cnt_u)
    
    ranking = []
    for n, v in df_total['Numero'].value_counts().items():
        d, u = n // 10, n % 10
        p = f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"
        ranking.append({'Numero': f"{n:02d}", 'Frecuencia': v, 'Perfil': p})
    df_rank = pd.DataFrame(ranking).sort_values('Frecuencia', ascending=False) if ranking else pd.DataFrame()
    
    tend = df_rank['Perfil'].value_counts().reset_index()
    tend.columns = ['Perfil', 'Frecuencia']
    top_p = tend.iloc[0]['Perfil'] if not tend.empty else "N/A"
    
    pers_num = []
    nums_unicos = df_total['Numero'].unique()
    for n in nums_unicos:
        c = sum(1 for b in bloques_validos if n in b['Numero'].values)
        if c == len(bloques_validos):
            perfil_val = df_rank[df_rank['Numero'] == f"{n:02d}"]['Perfil']
            p = perfil_val.values[0] if not perfil_val.empty else "?"
            pers_num.append({'Numero': f"{n:02d}", 'Perfil': p})
    df_pers_num = pd.DataFrame(pers_num) if pers_num else pd.DataFrame()
    
    sets_perfiles = []
    for df_b in bloques_validos:
        perfiles_en_bloque = set()
        for row in df_b.itertuples():
            d, u = row.Numero // 10, row.Numero % 10
            ed, eu = mapa_d.get(d, '?'), mapa_u.get(u, '?')
            perfiles_en_bloque.add(f"{ed} + {eu}")
        sets_perfiles.append(perfiles_en_bloque)
    
    persistentes_perfiles = set.intersection(*sets_perfiles) if sets_perfiles else set()
    persistentes_num_set = set(p['Numero'] for p in pers_num) if pers_num else set()
    
    hoy = datetime.now()
    estado_periodo = ""
    df_historial_actual = pd.DataFrame()
    
    try:
        fin_mes_actual = calendar.monthrange(hoy.year, hoy.month)[1]
        fecha_ini_evaluacion = datetime(hoy.year, hoy.month, min(dia_inicio, fin_mes_actual))
        fecha_fin_teorica = datetime(hoy.year, hoy.month, min(dia_fin, fin_mes_actual))
        
        if hoy < fecha_ini_evaluacion:
            estado_periodo = f"PERIODO NO INICIADO (Comienza el {fecha_ini_evaluacion.strftime('%d/%m')})"
        else:
            fecha_fin_real = min(hoy, fecha_fin_teorica)
            df_evaluacion = df_fijos[(df_fijos['Fecha'] >= fecha_ini_evaluacion) & (df_fijos['Fecha'] <= fecha_fin_real)].copy()
            
            if not df_evaluacion.empty:
                historial_data = []
                for row in df_evaluacion.itertuples():
                    num = row.Numero
                    d, u = num // 10, num % 10
                    ed, eu = mapa_d.get(d, '?'), mapa_u.get(u, '?')
                    perfil_completo = f"{ed} + {eu}"
                    cumple_regla = f"{num:02d}" in persistentes_num_set or perfil_completo in persistentes_perfiles
                    motivo = "Num. Persistente" if f"{num:02d}" in persistentes_num_set else ("Perfil Persistente" if perfil_completo in persistentes_perfiles else "")
                    historial_data.append({
                        'Fecha': row.Fecha,
                        'Tipo': row.Tipo_Sorteo,
                        'Numero': f"{num:02d}",
                        'Perfil': perfil_completo,
                        'Cumple': 'SI' if cumple_regla else 'NO',
                        'Tipo_Cumple': motivo if cumple_regla else '-'
                    })
                df_historial_actual = pd.DataFrame(historial_data)
                orden_sorteo = {'N': 0, 'T': 1, 'OTRO': 2}
                df_historial_actual['orden'] = df_historial_actual['Tipo'].map(orden_sorteo).fillna(2)
                df_historial_actual = df_historial_actual.sort_values(['Fecha', 'orden'], ascending=[False, True]).reset_index(drop=True)
            estado_periodo = f"PERIODO ACTIVO (hasta: {hoy.strftime('%d/%m')})"
    except Exception as e:
        estado_periodo = f"Error: {str(e)}"
    
    df_faltantes = pd.DataFrame()
    if "ACTIVO" in estado_periodo:
        esperados = set(df_rank.head(20)['Numero'].tolist()) if not df_rank.empty else set()
        esperados.update(persistentes_num_set)
        if not df_historial_actual.empty:
            salidos = set(df_historial_actual['Numero'].unique())
            faltantes_nums = esperados - salidos
        else:
            faltantes_nums = esperados
        if faltantes_nums:
            df_faltantes = pd.DataFrame([{'Numero': n, 'Estado': 'FALTANTE'} for n in sorted(list(faltantes_nums))])
    
    return {
        'success': True,
        'df_total': df_total,
        'df_dec': df_dec,
        'df_uni': df_uni,
        'df_rank': df_rank,
        'nombres_bloques': nombres_bloques,
        'df_pers_num': df_pers_num,
        'tend': tend,
        'top_p': top_p,
        'persistentes_perfiles': persistentes_perfiles,
        'df_historial_actual': df_historial_actual,
        'df_faltantes': df_faltantes,
        'estado_periodo': estado_periodo
    }

def backtesting_estrategia(df_fijos, mes_objetivo, anio_objetivo, dia_ini, dia_fin, meses_atras):
    try:
        fecha_ref = datetime(anio_objetivo, mes_objetivo, 1)
        bloques_train = []
        nombres_train = []
        
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_ref - relativedelta(months=offset)
            try:
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_i = datetime(f_obj.year, f_obj.month, min(dia_ini, last_day))
                f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
                if f_i > f_f:
                    continue
                df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
                if not df_b.empty:
                    bloques_train.append(df_b)
                    nombres_train.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
            except:
                continue
        
        if not bloques_train:
            return None, "Sin datos para entrenamiento"
        
        df_train = pd.concat(bloques_train)
        df_train['Dec'] = df_train['Numero'] // 10
        df_train['Uni'] = df_train['Numero'] % 10
        cnt_d = df_train['Dec'].value_counts().reindex(range(10), fill_value=0)
        cnt_u = df_train['Uni'].value_counts().reindex(range(10), fill_value=0)
        
        def get_lists(serie):
            df_t = serie.sort_values(ascending=False).reset_index()
            df_t.columns = ['Digito', 'Frecuencia']
            conds = [(df_t.index < 3), (df_t.index < 6)]
            vals = ['Caliente', 'Tibio']
            df_t['Estado'] = np.select(conds, vals, default='Frio')
            mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
            hot = [r['Digito'] for _, r in df_t.iterrows() if r['Estado'] == 'Caliente']
            warm = [r['Digito'] for _, r in df_t.iterrows() if r['Estado'] == 'Tibio']
            cold = [r['Digito'] for _, r in df_t.iterrows() if r['Estado'] == 'Frio']
            return mapa, hot, warm, cold
        
        mapa_d, hot_d, warm_d, cold_d = get_lists(cnt_d)
        mapa_u, hot_u, warm_u, cold_u = get_lists(cnt_u)
        
        sets_perfiles = []
        for df_b in bloques_train:
            perfiles = set()
            for row in df_b.itertuples():
                d, u = row.Numero // 10, row.Numero % 10
                perfiles.add(f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}")
            sets_perfiles.append(perfiles)
        perfiles_persistentes = set.intersection(*sets_perfiles) if sets_perfiles else set()
        
        f_prueba_ini = datetime(anio_objetivo, mes_objetivo, min(dia_ini, 28))
        f_prueba_fin = datetime(anio_objetivo, mes_objetivo, min(dia_fin, calendar.monthrange(anio_objetivo, mes_objetivo)[1]))
        df_test = df_fijos[(df_fijos['Fecha'] >= f_prueba_ini) & (df_fijos['Fecha'] <= f_prueba_fin)]
        
        if df_test.empty:
            return None, "Sin datos en el mes de prueba"
        
        resultados = []
        aciertos = 0
        sufrientes = 0
        
        for row in df_test.itertuples():
            num = row.Numero
            d, u = num // 10, num % 10
            perfil = f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"
            es_pers = perfil in perfiles_persistentes
            if es_pers:
                aciertos += 1
            es_sufriente = (d in cold_d) or (u in cold_u)
            if es_sufriente:
                sufrientes += 1
            resultados.append({
                'Fecha': row.Fecha,
                'Tipo': row.Tipo_Sorteo,
                'Numero': num,
                'Decena': f"{d} ({mapa_d.get(d, '?')})",
                'Unidad': f"{u} ({mapa_u.get(u, '?')})",
                'Resultado': 'ESTRUCTURA' if es_pers else ('SUFRIENTE' if es_sufriente else 'OTRO')
            })
        
        df_detalle = pd.DataFrame(resultados)
        orden_sorteo = {'N': 0, 'T': 1, 'OTRO': 2}
        df_detalle['orden'] = df_detalle['Tipo'].map(orden_sorteo).fillna(2)
        df_detalle = df_detalle.sort_values(['Fecha', 'orden'], ascending=[False, True]).reset_index(drop=True)
        
        total = len(df_detalle)
        porc = (aciertos / total * 100) if total > 0 else 0
        
        return {
            "Entrenamiento": ", ".join(nombres_train),
            "Prueba": f"{f_prueba_ini.strftime('%B %Y')} ({dia_ini}-{dia_fin})",
            "Perfiles": perfiles_persistentes,
            "Total": total,
            "Aciertos": aciertos,
            "Efectividad": round(porc, 2),
            "Sufrientes": sufrientes,
            "Detalle": df_detalle,
            "hot_d": hot_d, "warm_d": warm_d, "cold_d": cold_d,
            "hot_u": hot_u, "warm_u": warm_u, "cold_u": cold_u
        }
    except Exception as e:
        return None, f"Error: {str(e)}"

def analizar_estabilidad(df_fijos, dias_analisis=365):
    fecha_limite = datetime.now() - timedelta(days=dias_analisis)
    df_hist = df_fijos[df_fijos['Fecha'] >= fecha_limite].copy()
    if df_hist.empty:
        return None
    datos = []
    hoy = datetime.now()
    for num in range(100):
        df_num = df_hist[df_hist['Numero'] == num].sort_values('Fecha')
        if len(df_num) < 2:
            max_gap = 9999
            avg_gap = 9999
            std_gap = 0
            gap_actual = (hoy - df_num['Fecha'].max()).days if not df_num.empty else dias_analisis
            estado = "SIN DATOS"
            ultima = df_num['Fecha'].max() if not df_num.empty else None
        else:
            fechas = df_num['Fecha'].tolist()
            gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
            max_gap = max(gaps) if gaps else 9999
            avg_gap = np.mean(gaps) if gaps else 0
            std_gap = np.std(gaps) if gaps else 0
            ultima_salida = fechas[-1]
            gap_actual = (hoy - ultima_salida).days
            if gap_actual > max_gap:
                max_gap = gap_actual
            if gap_actual == 0:
                estado = "EN RACHA"
            elif gap_actual <= avg_gap:
                estado = "NORMAL"
            elif gap_actual <= avg_gap * 2.0:
                estado = "VENCIDO"
            else:
                estado = "MUY VENCIDO"
            ultima = ultima_salida
        datos.append({
            'Numero': f"{num:02d}",
            'Gap Actual': gap_actual,
            'Gap Max': max_gap,
            'Gap Prom': round(avg_gap, 1),
            'Irregularidad': round(std_gap, 1),
            'Estado': estado,
            'Ultima': ultima.strftime('%d/%m/%Y') if ultima else "N/A"
        })
    return pd.DataFrame(datos).sort_values(['Gap Max', 'Irregularidad'], ascending=[True, True]).reset_index(drop=True)

def generar_sugerencia(df, dias, gap):
    fh = datetime.now()
    df_t = df[df['Fecha'] >= fh - timedelta(days=dias)].copy()
    if df_t.empty:
        return pd.DataFrame()
    df_t['Dec'] = df_t['Numero'] // 10
    df_t['Uni'] = df_t['Numero'] % 10
    td = df_t['Dec'].value_counts().head(3).index.tolist()
    tu = df_t['Uni'].value_counts().head(3).index.tolist()
    res = []
    for n in [d*10+u for d in td for u in tu]:
        df_n = df[df['Numero'] == n]
        if not df_n.empty:
            g = (fh - df_n['Fecha'].max()).days
            if g >= gap:
                res.append({'Numero': f"{n:02d}", 'Gap': g, 'Estado': "Muy" if g > gap*1.5 else "Op"})
    return pd.DataFrame(res).sort_values('Gap', ascending=False) if res else pd.DataFrame()

def buscar_seq(df_fijos, part, type_, seq):
    try:
        p = [x.strip().upper() for x in seq.replace(',', ' ').split() if x.strip()]
    except:
        return None, "Error"
    if len(p) == 0 or len(p) > 5:
        return None, "Invalido"
    if type_ == 'digito':
        v = set(range(10))
    elif type_ == 'paridad':
        v = {'P', 'I'}
    elif type_ == 'altura':
        v = {'A', 'B'}
    else:
        return None, "Tipo desconocido"
    try:
        if type_ == 'digito':
            p = [int(x) for x in p]
            if any(x not in v for x in p):
                return None, "0-9"
        else:
            if any(x not in v for x in p):
                return None, f"Usa: {', '.join(v)}"
    except:
        return None, "Error conversion"
    l = []
    for x in df_fijos['Numero']:
        val = x // 10 if part == 'Decena' else x % 10
        if type_ == 'digito':
            l.append(val)
        elif type_ == 'paridad':
            l.append('P' if val % 2 == 0 else 'I')
        elif type_ == 'altura':
            l.append('A' if val >= 5 else 'B')
    lp = len(p)
    dat = {}
    for i in range(len(l) - lp):
        if l[i:i+lp] == p:
            sig = l[i + lp]
            r = df_fijos.iloc[i + lp]
            e = f"{r['Numero']:02d} ({r['Fecha'].strftime('%d/%m/%Y')})"
            if sig not in dat:
                dat[sig] = {'c': 0, 'e': []}
            dat[sig]['c'] += 1
            if len(dat[sig]['e']) < 3 and e not in dat[sig]['e']:
                dat[sig]['e'].append(e)
    if not dat:
        return None, "No encontrado"
    total = sum(v['c'] for v in dat.values())
    rows = []
    for k, v in dat.items():
        prob = (v['c'] / total * 100) if total > 0 else 0
        rows.append({'Siguiente': k, 'Frecuencia': v['c'], 'Ejemplos': ", ".join(v['e']), 'Prob': round(prob, 2)})
    return pd.DataFrame(rows).sort_values('Frecuencia', ascending=False), None

# --- FUNCION TRANSFERENCIA (FLOTODO: T->N y N->T) ---
def analizar_transferencia_flotodo(df_completo, dias_atras=180):
    """
    Analiza transferencia decena->unidad con LOGICA DE CICLOS para Flotodo:
    - T->N: Decena Tarde -> Unidad Noche (mismo dia)
    - N->T: Decena Noche -> Unidad Tarde (dia siguiente)
    """
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    
    df_filtrado = df_completo[df_completo['Fecha'] >= fecha_inicio].copy()
    fechas_unicas = sorted(df_filtrado['Fecha'].dt.date.unique())
    
    # Registrar eventos con detalle
    eventos = {
        'T->N': [],
        'N->T': []
    }
    
    for i, fecha in enumerate(fechas_unicas):
        df_dia = df_filtrado[df_filtrado['Fecha'].dt.date == fecha]
        
        fijo_T = df_dia[df_dia['Tipo_Sorteo'] == 'T']['Numero'].values
        fijo_N = df_dia[df_dia['Tipo_Sorteo'] == 'N']['Numero'].values
        
        # T->N (mismo dia)
        if len(fijo_T) > 0 and len(fijo_N) > 0:
            decena_T = int(fijo_T[0]) // 10
            unidad_N = int(fijo_N[0]) % 10
            if decena_T == unidad_N:
                eventos['T->N'].append({'fecha': fecha, 'digito': decena_T})
        
        # N->T (dia siguiente)
        if len(fijo_N) > 0 and i < len(fechas_unicas) - 1:
            fecha_siguiente = fechas_unicas[i + 1]
            df_siguiente = df_filtrado[df_filtrado['Fecha'].dt.date == fecha_siguiente]
            fijo_T_sig = df_siguiente[df_siguiente['Tipo_Sorteo'] == 'T']['Numero'].values
            
            if len(fijo_T_sig) > 0:
                decena_N = int(fijo_N[0]) // 10
                unidad_T_sig = int(fijo_T_sig[0]) % 10
                if decena_N == unidad_T_sig:
                    eventos['N->T'].append({'fecha': fecha, 'digito': decena_N})
    
    # Calcular estadisticas con logica de ciclos
    fecha_hoy_date = fecha_hoy.date()
    stats = []
    
    for tipo, eventos_lista in eventos.items():
        if len(eventos_lista) >= 2:
            # Calcular gaps entre eventos
            gaps = []
            for j in range(1, len(eventos_lista)):
                gap = (eventos_lista[j]['fecha'] - eventos_lista[j-1]['fecha']).days
                gaps.append(gap)
            
            # Promedio historico (todos los gaps)
            promedio_historico = round(np.mean(gaps), 1) if gaps else 0
            ausencia_maxima = max(gaps) if gaps else 0
            
            # SECUENCIA RECIENTE (ultimos 2-3 gaps)
            if len(gaps) >= 2:
                secuencia_reciente = round(np.mean(gaps[-2:]), 1)  # Promedio ultimos 2 gaps
                ultimo_gap = gaps[-1]
            else:
                secuencia_reciente = gaps[0] if gaps else promedio_historico
                ultimo_gap = gaps[-1] if gaps else 0
            
            # Detectar tipo de secuencia
            if secuencia_reciente < promedio_historico * 0.7:  # 30% mas rapido
                tipo_secuencia = "ACELERADO"
                prediccion_dias = secuencia_reciente
            elif secuencia_reciente > promedio_historico * 1.3:  # 30% mas lento
                tipo_secuencia = "LENTO"
                prediccion_dias = secuencia_reciente
            else:
                tipo_secuencia = "NORMAL"
                prediccion_dias = promedio_historico
            
            # Ultimo evento
            ultimo_evento = eventos_lista[-1]
            ultima_fecha = ultimo_evento['fecha']
            ultimo_digito = ultimo_evento['digito']
            dias_sin_evento = (fecha_hoy_date - ultima_fecha).days
            
            # LOGICA DE CICLOS CON SECUENCIA
            if dias_sin_evento > promedio_historico * 3:
                estado_ciclo = "REINICIAR - Esperar primera vez"
                alerta = False
                repeticion = 0
                dias_estimados = prediccion_dias
            elif dias_sin_evento >= prediccion_dias:
                estado_ciclo = "ALERTA - Puede repetir"
                alerta = True
                repeticion = len(eventos_lista)
                dias_estimados = max(0, round(prediccion_dias - dias_sin_evento, 0))
            else:
                estado_ciclo = "EN CICLO - Aun no toca"
                alerta = False
                repeticion = len(eventos_lista)
                dias_estimados = round(prediccion_dias - dias_sin_evento, 0)
            
            frecuencia = len(eventos_lista)
            
        elif len(eventos_lista) == 1:
            ultimo_evento = eventos_lista[0]
            ultima_fecha = ultimo_evento['fecha']
            ultimo_digito = ultimo_evento['digito']
            dias_sin_evento = (fecha_hoy_date - ultima_fecha).days
            promedio_historico = 0
            secuencia_reciente = 0
            ausencia_maxima = dias_sin_evento
            frecuencia = 1
            tipo_secuencia = "SIN DATOS"
            estado_ciclo = "PRIMERA VEZ - Esperar para segunda"
            alerta = False
            repeticion = 1
            prediccion_dias = 0
            dias_estimados = 0
        else:
            frecuencia = 0
            dias_sin_evento = 999
            promedio_historico = 0
            secuencia_reciente = 0
            ausencia_maxima = 999
            ultima_fecha = None
            ultimo_digito = None
            tipo_secuencia = "SIN DATOS"
            estado_ciclo = "SIN EVENTOS - Esperar primera vez"
            alerta = False
            repeticion = 0
            prediccion_dias = 0
            dias_estimados = 0
        
        stats.append({
            'Transferencia': tipo,
            'Frecuencia': frecuencia,
            'Promedio_Historico': promedio_historico,
            'Secuencia_Reciente': secuencia_reciente if isinstance(secuencia_reciente, (int, float)) else 0,
            'Tipo_Secuencia': tipo_secuencia,
            'Prediccion_Dias': prediccion_dias if isinstance(prediccion_dias, (int, float)) else 0,
            'Ausencia_Max': ausencia_maxima,
            'Dias_Sin_Evento': dias_sin_evento if ultima_fecha else 'N/A',
            'Dias_Estimados': dias_estimados,
            'Ultima_Fecha': ultima_fecha.strftime('%d/%m/%Y') if ultima_fecha else 'N/A',
            'Ultimo_Digito': ultimo_digito if ultimo_digito is not None else 'N/A',
            'Estado_Ciclo': estado_ciclo,
            'Alerta': alerta,
            'Proxima_Repeticion': repeticion + 1 if repeticion > 0 else 1
        })
    
    return pd.DataFrame(stats), eventos

def main():
    df_fijos, df_completo = cargar_datos_flotodo(RUTA_CSV)
    if df_fijos.empty:
        st.error("DataFrame vacio")
        st.stop()
    
    with st.sidebar.expander("Columnas detectadas", expanded=False):
        st.write("Columnas en el CSV:")
        for col in df_completo.columns:
            st.write(f"- {col}")
    
    st.sidebar.header("Panel")
    with st.sidebar.expander("Datos", expanded=True):
        def mostrar(l, i, ic):
            if not l.empty:
                f = l['Fecha'].max()
                n = l[l['Fecha'] == f]['Numero'].values[0]
                st.metric(f"{ic} {i}", f"{f.strftime('%d/%m')}", delta=f"{n:02d}")
        mostrar(df_fijos[df_fijos['Tipo_Sorteo'] == 'T'], "Tarde", "T")
        mostrar(df_fijos[df_fijos['Tipo_Sorteo'] == 'N'], "Noche", "N")
    
    with st.sidebar.expander("Agregar Sorteo", expanded=False):
        f = st.date_input("Fecha:", datetime.now().date(), format="DD/MM/YYYY", label_visibility="collapsed")
        s = st.radio("Sesion:", ["Tarde (T)", "Noche (N)"], horizontal=True, label_visibility="collapsed")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cent = st.number_input("Cent", 0, 9, 0, label_visibility="collapsed")
        with c2:
            fj = st.number_input("Fijo", 0, 99, 0, format="%02d", label_visibility="collapsed")
        with c3:
            c1v = st.number_input("1er", 0, 99, 0, format="%02d", label_visibility="collapsed")
        with c4:
            p2 = st.number_input("2do", 0, 99, 0, format="%02d", label_visibility="collapsed")
        
        if st.button("Guardar", type="primary", use_container_width=True):
            cd = s.split('(')[1].replace(')', '')
            try:
                with open(RUTA_CSV, 'a', encoding='latin-1') as file:
                    file.write(f"{f.strftime('%d/%m/%Y')};{cd};{cent};{fj};{c1v};{p2}\n")
                st.success("Guardado")
                time.sleep(1)
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with st.sidebar.expander("Borrar Sorteo", expanded=False):
        st.caption("Selecciona el sorteo a borrar:")
        
        df_ultimos = df_completo.tail(20).copy()
        df_ultimos['Mostrar'] = df_ultimos['Fecha'].dt.strftime('%d/%m/%Y') + ' - ' + df_ultimos['Tipo_Sorteo'] + ' - Fijo:' + df_ultimos['Fijo'].astype(str).str.zfill(2)
        
        opciones = [''] + df_ultimos['Mostrar'].tolist()
        seleccion = st.selectbox("Sorteo:", opciones, label_visibility="collapsed")
        
        if st.button("Borrar", type="secondary", use_container_width=True):
            if seleccion:
                idx_borrar = df_ultimos[df_ultimos['Mostrar'] == seleccion].index
                if len(idx_borrar) > 0:
                    idx_borrar = idx_borrar[0]
                    with open(RUTA_CSV, 'r', encoding='latin-1') as file:
                        lineas = file.readlines()
                    
                    linea_borrar = idx_borrar + 1
                    if linea_borrar < len(lineas):
                        del lineas[linea_borrar]
                        
                        with open(RUTA_CSV, 'w', encoding='latin-1') as file:
                            file.writelines(lineas)
                        
                        st.success(f"Borrado: {seleccion}")
                        time.sleep(1)
                        st.cache_resource.clear()
                        st.rerun()
            else:
                st.warning("Selecciona un sorteo")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Recargar"):
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.subheader("Modo")
    modo = st.sidebar.radio("Filtro:", ["General", "Tarde", "Noche"])
    
    if modo == "Tarde":
        dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].copy()
        t = "Tarde"
    elif modo == "Noche":
        dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].copy()
        t = "Noche"
    else:
        dfa = df_fijos.copy()
        t = "General"
    
    if dfa.empty:
        st.warning(f"No hay datos para: {t}")
        st.stop()
    
    tabs = st.tabs(["Transferencia", "Digito Faltante", "Patrones", "Almanaque", "Propuesta", "Secuencia", "Laboratorio", "Estabilidad"])
    
    # TAB: TRANSFERENCIA
    with tabs[0]:
        st.subheader("Transferencia Decena -> Unidad")
        st.markdown("**Analiza cuando la decena de un sorteo pasa como unidad al siguiente (con ciclos)**")
        st.info("T->N: Decena Tarde -> Unidad Noche (mismo dia) | N->T: Decena Noche -> Unidad Tarde (dia siguiente)")
        
        st.markdown("### Logica de Ciclos")
        st.markdown("""
        - **1ra vez**: El evento ocurre -> Se marca el ciclo (NO se apuesta)
        - **2da vez**: Puede repetir -> **ALERTA: apostar**
        - **3ra vez**: Puede repetir -> **ALERTA: apostar**
        - **Si se aleja 3x del promedio**: Reiniciar ciclo, esperar 1ra vez
        - **ACELERADO**: Secuencia actual mas rapida que el promedio -> usar secuencia reciente
        """)
        
        dias_stats = st.slider("Dias de historial:", 30, 365, 90, key="trans_stats")
        
        if st.button("Analizar Transferencias", type="primary", key="btn_trans"):
            with st.spinner("Analizando..."):
                df_stats, eventos = analizar_transferencia_flotodo(df_completo, dias_stats)
            
            for _, row in df_stats.iterrows():
                # Mostrar tipo de secuencia con DOBLE PREDICCION
                tipo_sec = row['Tipo_Secuencia']
                promedio_hist = row['Promedio_Historico']
                secuencia_rec = row['Secuencia_Reciente']
                prediccion = row['Prediccion_Dias']
                
                st.markdown(f"### **{row['Transferencia']}**")
                
                # Siempre mostrar AMBAS predicciones
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric("Por Promedio Historico", f"{promedio_hist} dias")
                with col_pred2:
                    st.metric("Por Tendencia de Racha", f"{secuencia_rec} dias")
                
                if tipo_sec == "ACELERADO":
                    st.warning(f"**ACELERADO**: La racha va mas rapido que el promedio. Usar prediccion de racha ({secuencia_rec} dias)")
                    st.markdown(f"**Prediccion recomendada**: cada **{prediccion}** dias (segun racha actual)")
                elif tipo_sec == "LENTO":
                    st.info(f"**LENTO**: La racha va mas lento que el promedio. Considerar prediccion de racha ({secuencia_rec} dias)")
                else:
                    st.success(f"**NORMAL**: Secuencia estable, usar promedio historico ({promedio_hist} dias)")
                
                if row['Alerta']:
                    # Obtener decena actual segun tipo
                    if row['Transferencia'] == 'T->N':
                        ultimo_T = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].iloc[-1] if len(df_fijos[df_fijos['Tipo_Sorteo'] == 'T']) > 0 else None
                        if ultimo_T is not None:
                            decena_actual = int(ultimo_T['Numero']) // 10
                            nums_sugeridos = [f"{d*10 + decena_actual:02d}" for d in range(10)]
                            fecha_T = ultimo_T['Fecha'].strftime('%d/%m/%Y')
                            
                            st.success(f"**{row['Transferencia']}** - ALERTA: Puede repetir ({row['Proxima_Repeticion']}a vez)")
                            st.markdown(f"Ultimo evento: {row['Ultima_Fecha']} (digito {row['Ultimo_Digito']})")
                            st.markdown(f"Sin evento hace: {row['Dias_Sin_Evento']} dias | Prediccion: cada {row['Prediccion_Dias']} dias")
                            st.markdown(f"**Fijo Tarde del {fecha_T}: {int(ultimo_T['Numero']):02d}** -> Decena: **{decena_actual}**")
                            st.markdown(f"**Jugar en NOCHE numeros terminados en {decena_actual}:** {', '.join(nums_sugeridos)}")
                    
                    elif row['Transferencia'] == 'N->T':
                        ultimo_N = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].iloc[-1] if len(df_fijos[df_fijos['Tipo_Sorteo'] == 'N']) > 0 else None
                        if ultimo_N is not None:
                            decena_actual = int(ultimo_N['Numero']) // 10
                            nums_sugeridos = [f"{d*10 + decena_actual:02d}" for d in range(10)]
                            fecha_N = ultimo_N['Fecha'].strftime('%d/%m/%Y')
                            
                            st.success(f"**{row['Transferencia']}** - ALERTA: Puede repetir ({row['Proxima_Repeticion']}a vez)")
                            st.markdown(f"Ultimo evento: {row['Ultima_Fecha']} (digito {row['Ultimo_Digito']})")
                            st.markdown(f"Sin evento hace: {row['Dias_Sin_Evento']} dias | Prediccion: cada {row['Prediccion_Dias']} dias")
                            st.markdown(f"**Fijo Noche del {fecha_N}: {int(ultimo_N['Numero']):02d}** -> Decena: **{decena_actual}**")
                            st.markdown(f"**Jugar en TARDE (dia siguiente) numeros terminados en {decena_actual}:** {', '.join(nums_sugeridos)}")
                else:
                    st.info(f"Estado: {row['Estado_Ciclo']}")
                    if row['Dias_Estimados'] > 0:
                        st.markdown(f"Estimado para proximo evento: **{row['Dias_Estimados']} dias**")
                
                st.markdown("---")
    
    # TAB: Digito Faltante
    with tabs[1]:
        st.subheader("Digito Faltante - Estrategia")
        st.markdown("**Analiza que digitos (0-9) faltaron en un dia y si aparecieron como Fijo al dia siguiente**")
        st.info("Flotodo tiene 2 sesiones (T, N) = 14 digitos por dia (7 por sesion)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Analisis de Fecha")
            fecha_analisis = st.date_input("Fecha a analizar:", datetime.now().date(), key="df_fecha")
            fecha_dt = datetime.combine(fecha_analisis, datetime.min.time())
            
            resultado, error = analizar_dia_completo(df_completo, fecha_dt)
            
            if error:
                st.warning(error)
            else:
                st.markdown(f"**Sesiones encontradas:** {', '.join(resultado['sesiones'])}")
                st.markdown(f"**Total digitos analizados:** {resultado['total_digitos']}")
                
                with st.expander("Ver detalle de cada sesion"):
                    df_detalle = pd.DataFrame(resultado['detalle'])
                    st.dataframe(df_detalle, hide_index=True)
                
                st.markdown(f"**Digitos presentes:** {', '.join(map(str, resultado['digitos_presentes']))}")
                
                if resultado['digitos_faltantes']:
                    st.markdown(f"### Digitos FALTANTES: {', '.join(map(str, resultado['digitos_faltantes']))}")
                    
                    st.markdown("**Numeros posibles con digitos faltantes:**")
                    numeros_posibles = []
                    for d in resultado['digitos_faltantes']:
                        for u in range(10):
                            numeros_posibles.append(f"{d}{u}")
                        for dec in range(10):
                            if dec != d:
                                numeros_posibles.append(f"{dec}{d}")
                    numeros_posibles = sorted(list(set(numeros_posibles)))
                    st.write(', '.join(numeros_posibles[:25]) + '...')
                else:
                    st.success("No hay digitos faltantes - aparecieron todos del 0-9")
                
                # Mostrar digitos por tipo
                st.markdown("---")
                st.markdown("### Digitos por Tipo en esta Fecha")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Centena:** {resultado['por_tipo']['centena']}")
                    st.markdown(f"**Fijo:** {resultado['por_tipo']['fijo']}")
                with c2:
                    st.markdown(f"**1er Corrido:** {resultado['por_tipo']['corrido1']}")
                    st.markdown(f"**2do Corrido:** {resultado['por_tipo']['corrido2']}")
        
        with col2:
            st.markdown("### Backtest de Efectividad")
            dias_bt = st.slider("Dias a evaluar:", 30, 180, 90, key="df_dias")
            
            if st.button("Ejecutar Backtest", type="primary", key="df_bt"):
                with st.spinner("Analizando..."):
                    bt_result = backtest_digito_faltante(df_completo, dias_bt)
                
                st.metric("Efectividad", f"{bt_result['efectividad']}%")
                st.metric("Aciertos", f"{bt_result['aciertos']}/{bt_result['total_evaluados']}")
                
                if bt_result['resultados']:
                    with st.expander("Ver detalle de evaluaciones"):
                        df_bt = pd.DataFrame(bt_result['resultados'])
                        st.dataframe(df_bt, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Estadisticas por Digito (Separadas por Tipo)")
        dias_stats = st.slider("Dias de historial:", 30, 365, 180, key="df_stats")
        
        if st.button("Calcular Estadisticas", key="df_calc"):
            with st.spinner("Calculando..."):
                stats = estadisticas_digitos_separadas(df_completo, dias_stats)
            
            # Tabs para cada tipo
            tab_stats = st.tabs(["General (Todo)", "Centena", "Fijo", "1er Corrido", "2do Corrido"])
            
            with tab_stats[0]:
                st.markdown("**Estadisticas Generales (unificando todos los digitos)**")
                st.dataframe(stats['general'], hide_index=True)
            
            with tab_stats[1]:
                st.markdown("**Estadisticas de Centena** (1 digito por sesion)")
                st.dataframe(stats['centena'], hide_index=True)
            
            with tab_stats[2]:
                st.markdown("**Estadisticas de Fijo** (decena + unidad)")
                st.dataframe(stats['fijo'], hide_index=True)
            
            with tab_stats[3]:
                st.markdown("**Estadisticas de 1er Corrido** (decena + unidad)")
                st.dataframe(stats['corrido1'], hide_index=True)
            
            with tab_stats[4]:
                st.markdown("**Estadisticas de 2do Corrido** (decena + unidad)")
                st.dataframe(stats['corrido2'], hide_index=True)
    
    with tabs[2]:
        st.subheader(f"Patrones: {t}")
        c1, c2 = st.columns(2)
        with c1:
            n = st.number_input("Disparador:", 0, 99, 40, format="%02d", key="p1")
        with c2:
            v = st.slider("Ventana:", 1, 30, 15, key="p2")
        if st.button("Analizar", key="b1"):
            st.session_state['sb1'] = True
        if st.session_state.get('sb1'):
            r, tot = analizar_siguientes(dfa, n, v)
            if r is None:
                st.warning(f"El numero {n:02d} no ha salido")
            else:
                st.success(f"Encontrado {tot} veces")
                st.dataframe(r.head(20), hide_index=True)
    
    with tabs[3]:
        st.subheader(f"Almanaque: {t}")
        with st.form("alm_form"):
            c1, c2 = st.columns(2)
            with c1:
                di = st.number_input("Dia Ini:", 1, 31, 16, key="a1")
                dfi = st.number_input("Dia Fin:", 1, 31, 20, key="a2")
            with c2:
                ma = st.slider("Meses Atras:", 1, 12, 4, key="a3")
            submitted = st.form_submit_button("Analizar", type="primary")
            if submitted:
                if di > dfi:
                    st.error("Dia ini no puede ser mayor")
                else:
                    with st.spinner("Analizando..."):
                        res = analizar_almanaque(dfa, int(di), int(dfi), int(ma))
                    if not res['success']:
                        st.error(res['mensaje'])
                    else:
                        if res['nombres_bloques']:
                            st.success(f"Periodos: {', '.join(res['nombres_bloques'])}")
                        st.info(f"Estado: {res['estado_periodo']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Decenas")
                            st.dataframe(res['df_dec'], hide_index=True)
                        with col2:
                            st.markdown("### Unidades")
                            st.dataframe(res['df_uni'], hide_index=True)
                        if not res['df_historial_actual'].empty:
                            st.markdown("### Resultados Actuales")
                            hv = res['df_historial_actual'].copy()
                            hv['Fecha'] = hv['Fecha'].dt.strftime('%d/%m/%Y')
                            st.dataframe(hv, hide_index=True)
                        if not res['df_faltantes'].empty:
                            st.markdown("### Faltantes")
                            st.dataframe(res['df_faltantes'], hide_index=True)
                        with st.expander("Persistencia"):
                            if not res['df_pers_num'].empty:
                                st.dataframe(res['df_pers_num'], hide_index=True)
                            if res['persistentes_perfiles']:
                                st.write("Perfiles:", list(res['persistentes_perfiles']))
                        with st.expander("Ranking"):
                            if not res['df_rank'].empty:
                                st.dataframe(res['df_rank'].head(20), hide_index=True)
    
    with tabs[4]:
        st.subheader(f"Propuesta: {t}")
        c1, c2 = st.columns(2)
        with c1:
            dt = st.number_input("Dias:", 5, 60, 15, key="pr1")
        with c2:
            dg = st.number_input("Gap:", 1, 90, 10, key="pr2")
        if st.button("Generar", key="bpr"):
            st.session_state['spr'] = True
        if st.session_state.get('spr'):
            p = generar_sugerencia(dfa, dt, dg)
            if p.empty:
                st.warning("No hay sugerencias")
            else:
                st.dataframe(p, hide_index=True)
    
    with tabs[5]:
        st.subheader(f"Secuencia: {t}")
        c1, c2, c3 = st.columns(3)
        with c1:
            part = st.selectbox("Parte:", ["Decena", "Unidad"], key="s1")
        with c2:
            type_ = st.selectbox("Tipo:", ["Digito (0-9)", "Paridad (P/I)", "Altura (A/B)"], key="s2")
        with c3:
            seq = st.text_input("Seq:", key="s3")
        if st.button("Buscar", key="bseq"):
            st.session_state['sseq'] = True
        if st.session_state.get('sseq') and seq:
            tc = type_.lower().replace('(', '').replace(')', '').split(' ')[0]
            r, e = buscar_seq(dfa, part, tc, seq)
            if e:
                st.warning(e)
            else:
                st.dataframe(r, hide_index=True)
    
    with tabs[6]:
        st.subheader("Laboratorio")
        meses_lab = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
        hoy = datetime.now()
        mes_def = hoy.month - 1 if hoy.month > 1 else 12
        anio_def = hoy.year if hoy.month > 1 else hoy.year - 1
        c1, c2, c3 = st.columns(3)
        with c1:
            nm = st.selectbox("Mes:", list(meses_lab.values()), index=list(meses_lab.keys()).index(mes_def), key="l1")
            mn = [k for k, v in meses_lab.items() if v == nm][0]
        with c2:
            an = st.number_input("Anio:", 2020, 2030, anio_def, key="l2")
        with c3:
            di = st.number_input("Dia Ini:", 1, 31, 1, key="l3")
            df_ = st.number_input("Dia Fin:", 1, 31, 15, key="l4")
        mat = st.slider("Meses atras:", 2, 6, 3, key="l5")
        if st.button("Ejecutar", type="primary"):
            with st.spinner("Analizando..."):
                res = backtesting_estrategia(dfa, mn, an, di, df_, mat)
            if isinstance(res, dict):
                st.success(f"Efectividad: {res['Efectividad']}%")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Estrategia")
                    st.write(f"Entrenamiento: {res['Entrenamiento']}")
                    st.write(f"Calientes D: {res['hot_d']}")
                    st.write(f"Tibios D: {res['warm_d']}")
                    st.write(f"Frios D: {res['cold_d']}")
                    st.write(f"Calientes U: {res['hot_u']}")
                    st.write(f"Tibios U: {res['warm_u']}")
                    st.write(f"Frios U: {res['cold_u']}")
                with c2:
                    st.markdown("### Resultados")
                    st.metric("Total", res['Total'])
                    st.metric("Aciertos", res['Aciertos'])
                    st.metric("Sufrientes", res['Sufrientes'])
                    dv = res['Detalle'].copy()
                    dv['Fecha'] = dv['Fecha'].dt.strftime('%d/%m/%Y')
                    st.dataframe(dv, hide_index=True)
            else:
                st.error(res)
    
    with tabs[7]:
        st.subheader(f"Estabilidad: {t}")
        dias = st.slider("Dias historial:", 90, 3650, 365, step=30, key="e1")
        if st.button("Calcular", key="be"):
            with st.spinner("Analizando..."):
                est = analizar_estabilidad(dfa, dias)
            if est is None:
                st.error("Sin datos")
            else:
                st.dataframe(est.head(30), hide_index=True)

if __name__ == "__main__":
    main()
