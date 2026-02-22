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
    page_icon="🎲",
    layout="wide"
)

st.title("🎲 Flotodo - Suite Ultimate")

@st.cache_resource
def cargar_datos_flotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"❌ Error: No se encontró el archivo {_ruta_csv}.")
            st.stop()
        
        df = pd.read_csv(_ruta_csv, sep=';', encoding='latin-1')
        
        columnas_requeridas = ['Fecha']
        if not all(col in df.columns for col in columnas_requeridas):
            if 'Fecha' not in df.columns and len(df.columns) > 0:
                df.columns = ['Fecha', 'Tipo_Sorteo', 'Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido'] + list(df.columns[6:])
        
        rename_map = {}
        cols = df.columns.tolist()
        for col in cols:
            col_lower = col.lower().strip()
            if 'fecha' in col_lower:
                rename_map[col] = 'Fecha'
            elif 'tarde' in col_lower or 'noche' in col_lower or 'sorteo' in col_lower:
                rename_map[col] = 'Tipo_Sorteo'
            elif 'centena' in col_lower or 'cent' in col_lower:
                rename_map[col] = 'Centena'
            elif col_lower == 'fijo':
                rename_map[col] = 'Fijo'
            elif 'corrido' in col_lower and '1' in col_lower:
                rename_map[col] = 'Primer_Corrido'
            elif 'corrido' in col_lower and '2' in col_lower:
                rename_map[col] = 'Segundo_Corrido'
        
        df.rename(columns=rename_map, inplace=True)
        
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)
        
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map({
                'TARDE': 'T', 'T': 'T',
                'NOCHE': 'N', 'N': 'N'
            }).fillna('OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        if 'Fijo' in df.columns:
            df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
            df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
            df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce')
            df_fijos = df_fijos.dropna(subset=['Numero'])
            df_fijos['Numero'] = df_fijos['Numero'].astype(int)
        else:
            st.error("❌ No se encontró la columna 'Fijo' en el archivo CSV.")
            st.stop()
        
        draw_order_map = {'T': 0, 'N': 1, 'OTRO': 2}
        df_fijos['draw_order'] = df_fijos['Tipo_Sorteo'].map(draw_order_map).fillna(2)
        df_fijos['sort_key'] = df_fijos['Fecha'] + pd.to_timedelta(df_fijos['draw_order'], unit='h')
        df_fijos = df_fijos.sort_values(by='sort_key').reset_index(drop=True)
        
        return df_fijos, df
        
    except pd.errors.EmptyDataError:
        st.error("❌ El archivo CSV está vacío.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.stop()

# --- FUNCIÓN EXTRAER DÍGITOS POR SESIÓN ---
def extraer_digitos_sesion(centena, fijo, primer_corr, segundo_corr):
    """Extrae los dígitos de una sesión organizados por tipo"""
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
    
    # Centena (1 dígito)
    try:
        c = int(float(centena))
        digitos['centena'].append(c)
        digitos['todos'].append(c)
    except:
        pass
    
    # Fijo (2 dígitos)
    try:
        f = int(float(fijo))
        digitos['fijo_dec'].append(f // 10)
        digitos['fijo_uni'].append(f % 10)
        digitos['todos'].extend([f // 10, f % 10])
    except:
        pass
    
    # Primer Corrido (2 dígitos)
    try:
        p1 = int(float(primer_corr))
        digitos['corrido1_dec'].append(p1 // 10)
        digitos['corrido1_uni'].append(p1 % 10)
        digitos['todos'].extend([p1 // 10, p1 % 10])
    except:
        pass
    
    # Segundo Corrido (2 dígitos)
    try:
        p2 = int(float(segundo_corr))
        digitos['corrido2_dec'].append(p2 // 10)
        digitos['corrido2_uni'].append(p2 % 10)
        digitos['todos'].extend([p2 // 10, p2 % 10])
    except:
        pass
    
    return digitos

# --- FUNCIÓN ANALIZAR DÍA COMPLETO ---
def analizar_dia_completo(df_completo, fecha):
    """Analiza un día completo y devuelve los dígitos faltantes"""
    df_dia = df_completo[df_completo['Fecha'].dt.date == fecha.date()].copy()
    
    if df_dia.empty:
        return None, "Sin datos para esa fecha"
    
    todos_digitos = []
    sesiones_encontradas = []
    detalle_digitos = []
    
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
            'Sesión': row['Tipo_Sorteo'],
            'Centena': centena,
            'Fijo': fijo,
            '1er_Corrido': primer_corr,
            '2do_Corrido': segundo_corr,
            'Dígitos': digitos['todos']
        })
    
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

# --- FUNCIÓN ESTADÍSTICAS POR DÍGITO SEPARADAS ---
def estadisticas_digitos_separadas(df_completo, dias_atras=180):
    """Calcula estadísticas por dígito separadas por tipo de columna"""
    fecha_hoy = datetime.now()
    fecha_inicio = fecha_hoy - timedelta(days=dias_atras)
    
    df_filtrado = df_completo[df_completo['Fecha'] >= fecha_inicio].copy()
    
    contadores = {
        'general': Counter(),
        'centena': Counter(),
        'fijo': Counter(),
        'corrido1': Counter(),
        'corrido2': Counter()
    }
    
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
            for d in resultado['digitos_presentes']:
                contadores['general'][d] += 1
                ultima_aparicion['general'][d] = fecha
            
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
                'Dígito': d,
                'Frecuencia': freq,
                'Porcentaje': f"{porcentaje}%",
                'Días Sin Aparecer': dias_sin if ultima else 'N/A',
                'Última': ultima.strftime('%d/%m') if ultima else 'N/A'
            })
        
        stats[tipo] = pd.DataFrame(datos)
    
    return stats

# --- FUNCIÓN BACKTEST DÍGITO FALTANTE ---
def backtest_digito_faltante(df_completo, dias_atras=90):
    """Evalúa la efectividad de la estrategia del dígito faltante"""
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
            'Dígitos_Coinc': ','.join(map(str, coincidencias)) if coincidencias else '-'
        })
    
    efectividad = (aciertos / total_evaluados * 100) if total_evaluados > 0 else 0
    
    return {
        'resultados': resultados,
        'total_evaluados': total_evaluados,
        'aciertos': aciertos,
        'efectividad': round(efectividad, 2)
    }

# --- FUNCIÓN PATRONES ---
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
    r['Probabilidad (%)'] = (r['Frecuencia'] / len(lista_s) * 100).round(2)
    r['Número'] = [f"{int(x):02d}" for x in r.index]
    return r.sort_values('Frecuencia', ascending=False), len(indices)

# --- FUNCIÓN ALMANAQUE (ORIGINAL) ---
def analizar_almanaque(df_fijos, dia_inicio, dia_fin, meses_atras, strict_mode=True):
    fecha_hoy = datetime.now()
    
    bloques_validos = []
    nombres_bloques = []
    debug_info = []
    
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_i = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            
            if f_i > f_f: 
                continue
                
            df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
            
            debug_info.append(f"{f_obj.strftime('%B %Y')}: {len(df_b)} registros en rango {dia_inicio}-{dia_fin}.")
            
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
                
        except Exception as e:
            debug_info.append(f"Error en {f_obj.strftime('%B %Y')}: {str(e)}")
            continue

    mensaje_advertencia = ""
    fallback_usado = False
    
    if not bloques_validos:
        if strict_mode:
            return {
                'success': False,
                'mensaje': "Sin datos en modo estricto.",
                'debug_info': debug_info
            }
        
        debug_info.append("Modo Fallback activado.")
        
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_hoy - relativedelta(months=offset)
            try:
                f_i = datetime(f_obj.year, f_obj.month, 1)
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_f = datetime(f_obj.year, f_obj.month, last_day)
                
                df_b = df_fijos[(df_fijos['Fecha'] >= f_i) & (df_fijos['Fecha'] <= f_f)]
                
                debug_info.append(f"FALLBACK {f_obj.strftime('%B %Y')}: {len(df_b)} registros.")
                
                if not df_b.empty:
                    bloques_validos.append(df_b)
                    nombres_bloques.append(f"{f_obj.strftime('%b')} (Todo el mes)")
                    
            except Exception as e:
                debug_info.append(f"Error Fallback: {str(e)}")
                continue
        
        if not bloques_validos:
            return {
                'success': False,
                'mensaje': "Sin datos.",
                'debug_info': debug_info
            }
        
        fallback_usado = True
        mensaje_advertencia = "(Usando datos del mes completo)."
    
    df_total = pd.concat(bloques_validos)
    df_total['Decena'] = df_total['Numero'] // 10
    df_total['Unidad'] = df_total['Numero'] % 10
    
    cnt_d = df_total['Decena'].value_counts().reindex(range(10), fill_value=0)
    cnt_u = df_total['Unidad'].value_counts().reindex(range(10), fill_value=0)
    
    def clasificar(serie):
        df_t = serie.sort_values(ascending=False).reset_index()
        df_t.columns = ['Digito', 'Frecuencia']
        conds = [(df_t.index < 3), (df_t.index < 6)]
        vals = ['🔥 Caliente', '🟡 Tibio']
        df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
        mapa = {r['Digito']: r['Estado'] for _, r in df_t.iterrows()}
        return df_t, mapa

    df_dec, mapa_d = clasificar(cnt_d)
    df_uni, mapa_u = clasificar(cnt_u)
    
    hot_d = df_dec[df_dec['Estado'] == '🔥 Caliente']['Digito'].tolist()
    hot_u = df_uni[df_uni['Estado'] == '🔥 Caliente']['Digito'].tolist()
    lista_3x3 = [{'Número': f"{d*10+u:02d}", 'Veces': len(df_total[df_total['Numero'] == d*10+u])} 
                 for d in hot_d for u in hot_u]
    df_3x3 = pd.DataFrame(lista_3x3).sort_values('Veces', ascending=False) if lista_3x3 else pd.DataFrame(columns=['Número', 'Veces'])

    ranking = []
    for n, v in df_total['Numero'].value_counts().items():
        d, u = n // 10, n % 10
        p = f"{mapa_d.get(d, '?')} + {mapa_u.get(u, '?')}"
        ranking.append({'Número': f"{n:02d}", 'Frecuencia': v, 'Perfil': p})
    df_rank = pd.DataFrame(ranking).sort_values('Frecuencia', ascending=False) if ranking else pd.DataFrame(columns=['Número', 'Frecuencia', 'Perfil'])
    
    tend = df_rank['Perfil'].value_counts().reset_index()
    tend.columns = ['Perfil', 'Frecuencia']
    top_p = tend.iloc[0]['Perfil'] if not tend.empty else "N/A"

    tend_nums = []
    if top_p != "N/A" and " + " in top_p:
        p_dec, p_uni = top_p.split(" + ")
        decs_obj = df_dec[df_dec['Estado'] == p_dec]['Digito'].tolist()
        unis_obj = df_uni[df_uni['Estado'] == p_uni]['Digito'].tolist()
        for d in decs_obj:
            for u in unis_obj:
                tend_nums.append({'Número': f"{d*10+u:02d}", 'Sugerencia': f"{p_dec} x {p_uni}"})
    df_tend_nums = pd.DataFrame(tend_nums)

    pers_num = []
    nums_unicos = df_total['Numero'].unique()
    for n in nums_unicos:
        c = sum(1 for b in bloques_validos if n in b['Numero'].values)
        if c == len(bloques_validos):
            perfil_val = df_rank[df_rank['Número'] == f"{n:02d}"]['Perfil']
            p = perfil_val.values[0] if not perfil_val.empty else "Desconocido"
            pers_num.append({'Número': f"{n:02d}", 'Perfil': p})
    
    df_pers_num = pd.DataFrame(pers_num).sort_values('Número').reset_index(drop=True) if pers_num else pd.DataFrame(columns=['Número', 'Perfil'])

    sets_perfiles = []
    for df_b in bloques_validos:
        perfiles_en_bloque = set()
        for row in df_b.itertuples():
            d, u = row.Numero // 10, row.Numero % 10
            ed, eu = mapa_d.get(d, '?'), mapa_u.get(u, '?')
            perfiles_en_bloque.add(f"{ed} + {eu}")
        sets_perfiles.append(perfiles_en_bloque)
    
    persistentes_perfiles = set.intersection(*sets_perfiles) if sets_perfiles else set()
    persistentes_num_set = set(p['Número'] for p in pers_num) if pers_num else set()

    hoy = datetime.now()
    estado_periodo = ""
    df_historial_actual = pd.DataFrame()
    
    try:
        fin_mes_actual = calendar.monthrange(hoy.year, hoy.month)[1]
        fecha_ini_evaluacion = datetime(hoy.year, hoy.month, min(dia_inicio, fin_mes_actual))
        fecha_fin_teorica = datetime(hoy.year, hoy.month, min(dia_fin, fin_mes_actual))
        
        if hoy < fecha_ini_evaluacion:
            estado_periodo = f"⚪ PERIODO NO INICIADO (Comienza el {fecha_ini_evaluacion.strftime('%d/%m')})"
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
                    
                    cumple_regla = False
                    motivo = ""
                    
                    if f"{num:02d}" in persistentes_num_se
