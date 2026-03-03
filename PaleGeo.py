# -*- coding: utf-8 -*-
"""
PaleGeo - Análisis de Pales por Grupos
Hoja: Geotodo (M=Mañana, T=Tarde, N=Noche)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

if __name__ == "__main__":
    st.set_page_config(page_title="PaleGeo", page_icon="🎯", layout="wide")

GS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'
GS_SHEET = 'Geotodo'
COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

MAPEO = {
    't': 'Tarde', 'tarde': 'Tarde',
    'n': 'Noche', 'noche': 'Noche',
    'm': 'Mañana', 'mañana': 'Mañana', 'manana': 'Mañana'
}

GRUPOS = {
    'CERRADOS': {'digitos': {0, 6, 8, 9}, 'numeros': []},
    'ABIERTOS': {'digitos': {2, 3, 5}, 'numeros': []},
    'RECTOS': {'digitos': {1, 4, 7}, 'numeros': []}
}

for i in range(100):
    s = f"{i:02d}"
    d1, d2 = int(s[0]), int(s[1])
    for g, d in GRUPOS.items():
        if d1 in d['digitos'] and d2 in d['digitos']:
            d['numeros'].append(i)

INFO = {g: {'digitos': ','.join(map(str, d['digitos'])), 'cant': len(d['numeros'])} for g, d in GRUPOS.items()}

@st.cache_resource
def conectar():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        try:
            if 'gcp_service_account' in st.secrets:
                from google.oauth2.service_account import Credentials
                creds_dict = dict(st.secrets['gcp_service_account'])
                creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
                return gspread.authorize(creds)
        except:
            pass
        
        for f in ['credentials.json', 'credenciales.json']:
            if os.path.exists(f):
                creds = ServiceAccountCredentials.from_json_keyfile_name(f, scope)
                return gspread.authorize(creds)
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def cargar_datos(gc, archivo_id, nombre_hoja):
    if gc:
        try:
            spreadsheet = gc.open_by_key(archivo_id)
            worksheet = spreadsheet.worksheet(nombre_hoja)
            return pd.DataFrame(worksheet.get_all_records())
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def parsear_fecha(f):
    if pd.isna(f):
        return None
    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(str(f).strip(), fmt)
        except:
            continue
    return None

def normalizar_sesion(s):
    if pd.isna(s):
        return None
    return MAPEO.get(str(s).strip().lower(), s)

def clasificar(n):
    try:
        s = f"{int(n):02d}"
        d1, d2 = int(s[0]), int(s[1])
        for g, d in GRUPOS.items():
            if d1 in d['digitos'] and d2 in d['digitos']:
                return g
        return None
    except:
        return None

def detectar_pales(fila):
    grupos = defaultdict(list)
    for col in [COL_FIJO, COL_CORR1, COL_CORR2]:
        if col in fila.index and pd.notna(fila[col]):
            try:
                n = int(fila[col])
                g = clasificar(n)
                if g:
                    grupos[g].append(f"{n:02d}")
            except:
                pass
    return {g: nums for g, nums in grupos.items() if len(nums) >= 2}

def calcular_estadisticas_grupo(pales_grupo, fecha_max):
    """Calcula estadísticas detalladas para un grupo"""
    if not pales_grupo:
        return {
            'frecuencia': 0,
            'promedio_dias': 0,
            'ausencia_maxima': 0,
            'dias_sin_aparecer': 0,
            'ultima_fecha': None,
            'gaps': []
        }
    
    # Ordenar por fecha
    fechas = sorted([p['fecha'] for p in pales_grupo if pd.notna(p['fecha'])])
    
    if not fechas:
        return {
            'frecuencia': len(pales_grupo),
            'promedio_dias': 0,
            'ausencia_maxima': 0,
            'dias_sin_aparecer': 0,
            'ultima_fecha': None,
            'gaps': []
        }
    
    # Calcular gaps (días entre apariciones)
    gaps = []
    for i in range(1, len(fechas)):
        gap = (fechas[i] - fechas[i-1]).days
        if gap > 0:
            gaps.append(gap)
    
    frecuencia = len(fechas)
    promedio_dias = round(np.mean(gaps), 1) if gaps else 0
    ausencia_maxima = max(gaps) if gaps else 0
    ultima_fecha = fechas[-1]
    dias_sin_aparecer = (fecha_max - ultima_fecha).days if ultima_fecha and pd.notna(fecha_max) else 0
    
    return {
        'frecuencia': frecuencia,
        'promedio_dias': promedio_dias,
        'ausencia_maxima': ausencia_maxima,
        'dias_sin_aparecer': dias_sin_aparecer,
        'ultima_fecha': ultima_fecha,
        'gaps': gaps
    }

def analizar_combinaciones(todos_pales, grupo_seleccionado):
    """Analiza combinaciones dentro de un grupo"""
    comb_count = defaultdict(lambda: {'count': 0, 'fechas': []})
    
    for p in todos_pales:
        if p['grupo'] == grupo_seleccionado:
            nums = sorted([int(n) for n in p['numeros']])
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    comb = f"{nums[i]:02d}-{nums[j]:02d}"
                    comb_count[comb]['count'] += 1
                    fecha = p['fecha'].strftime('%d/%m/%Y') if pd.notna(p['fecha']) else '-'
                    comb_count[comb]['fechas'].append(fecha)
    
    return comb_count

def main():
    st.title("🎯 PaleGeo - Pales por Grupos")
    st.markdown("**Hoja: Geotodo** | Sesiones: Mañana, Tarde y Noche")
    
    gc = conectar()
    if not gc:
        st.error("Sin conexión")
        return
    
    df = cargar_datos(gc, GS_ID, GS_SHEET)
    if df is None or len(df) == 0:
        st.error("Sin datos")
        return
    
    st.success(f"✅ {len(df)} registros")
    
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df['Sesion'] = df[COL_SESION].apply(normalizar_sesion)
    fecha_max = df['Fecha_Parsed'].max()
    
    todos_pales = []
    for idx, fila in df.iterrows():
        pales = detectar_pales(fila)
        for g, nums in pales.items():
            todos_pales.append({
                'fecha': fila['Fecha_Parsed'],
                'sesion': fila['Sesion'],
                'grupo': g,
                'numeros': nums
            })
    
    # Calcular estadísticas por grupo
    stats_por_grupo = {}
    for grupo in GRUPOS.keys():
        pales_g = [p for p in todos_pales if p['grupo'] == grupo]
        stats_por_grupo[grupo] = calcular_estadisticas_grupo(pales_g, fecha_max)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🎯 Combinaciones", "📜 Historial", "ℹ️ Grupos"])
    
    # === TAB 1: RESUMEN CON ESTADÍSTICAS ===
    with tab1:
        st.header("📊 Estadísticas por Grupo")
        st.markdown("*Un pale ocurre cuando 2+ números del mismo grupo aparecen en un mismo sorteo*")
        
        # Tabla resumen
        resumen_data = []
        for grupo in ['CERRADOS', 'ABIERTOS', 'RECTOS']:
            s = stats_por_grupo[grupo]
            ultima = s['ultima_fecha'].strftime('%d/%m/%Y') if s['ultima_fecha'] else '-'
            resumen_data.append({
                'Grupo': grupo,
                'Dígitos': INFO[grupo]['digitos'],
                'Frecuencia': s['frecuencia'],
                'Promedio Días': s['promedio_dias'],
                'Ausencia Máx': s['ausencia_maxima'],
                'Días Sin Aparecer': s['dias_sin_aparecer'],
                'Última Fecha': ultima
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Detalle por grupo
        for grupo in GRUPOS.keys():
            pales_g = [p for p in todos_pales if p['grupo'] == grupo]
            s = stats_por_grupo[grupo]
            
            with st.expander(f"{'🔒' if grupo == 'CERRADOS' else '🔓' if grupo == 'ABIERTOS' else '📏'} {grupo} - Ver detalle"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pales encontrados", s['frecuencia'])
                    st.metric("Promedio entre salidas", f"{s['promedio_dias']} días")
                with col2:
                    st.metric("Ausencia máxima", f"{s['ausencia_maxima']} días")
                    st.metric("Días sin aparecer", s['dias_sin_aparecer'])
                with col3:
                    if s['gaps']:
                        st.metric("Gap mínimo", f"{min(s['gaps'])} días")
                        st.metric("Gap máximo", f"{max(s['gaps'])} días")
                
                if pales_g:
                    st.markdown("**Últimos 10 pales:**")
                    for p in pales_g[-10:][::-1]:
                        fecha = p['fecha'].strftime('%d/%m/%Y') if pd.notna(p['fecha']) else '-'
                        st.write(f"• {fecha} ({p['sesion']}): {', '.join(p['numeros'])}")
    
    # === TAB 2: COMBINACIONES ===
    with tab2:
        st.header("🎯 Análisis de Combinaciones")
        st.markdown("Combinaciones de números dentro de cada grupo que forman pales.")
        
        grupo_sel = st.selectbox("Selecciona grupo:", list(GRUPOS.keys()), key="grupo_comb")
        
        comb_data = analizar_combinaciones(todos_pales, grupo_sel)
        
        if comb_data:
            df_comb = pd.DataFrame([
                {'Combinación': c, 'Frecuencia': d['count'], 'Últimas fechas': ', '.join(d['fechas'][-3:])}
                for c, d in sorted(comb_data.items(), key=lambda x: x[1]['count'], reverse=True)
            ])
            st.dataframe(df_comb, use_container_width=True, hide_index=True)
        else:
            st.info(f"No hay combinaciones para {grupo_sel}")
    
    # === TAB 3: HISTORIAL ===
    with tab3:
        st.header("📜 Historial de Pales")
        st.markdown("Orden: Noche → Tarde → Mañana dentro de cada día")
        
        grupo_filtro = st.selectbox("Filtrar por grupo:", ['Todos'] + list(GRUPOS.keys()), key="filtro_hist")
        
        orden = {'Noche': 1, 'Tarde': 2, 'Mañana': 3}
        
        if grupo_filtro == 'Todos':
            pales_filtrados = todos_pales
        else:
            pales_filtrados = [p for p in todos_pales if p['grupo'] == grupo_filtro]
        
        pales_ord = sorted(pales_filtrados, key=lambda x: (x['fecha'], orden.get(x['sesion'], 99)), reverse=True)
        
        for p in pales_ord[:50]:
            fecha = p['fecha'].strftime('%d/%m/%Y') if pd.notna(p['fecha']) else '-'
            st.markdown(f"**{fecha}** ({p['sesion']}) - {p['grupo']}: {', '.join(p['numeros'])}")
        
        st.caption(f"Mostrando {min(50, len(pales_ord))} de {len(pales_ord)} pales")
    
    # === TAB 4: GRUPOS ===
    with tab4:
        st.header("ℹ️ Información de Grupos")
        
        for grupo, datos in GRUPOS.items():
            st.subheader(f"{'🔒' if grupo == 'CERRADOS' else '🔓' if grupo == 'ABIERTOS' else '📏'} {grupo}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Dígitos:** {INFO[grupo]['digitos']}")
            with col2:
                st.write(f"**Cantidad de números:** {INFO[grupo]['cant']}")
            
            nums_str = ", ".join([f"{n:02d}" for n in sorted(datos['numeros'])])
            st.write(f"**Números:** {nums_str}")
            st.markdown("---")

if __name__ == "__main__":
    main()