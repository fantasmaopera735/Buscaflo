# -*- coding: utf-8 -*-
"""
PaleFlo - Análisis de Pales por Grupos
Hoja: Sorteos (T=Tarde, N=Noche)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import calendar
import gspread
from oauth2client.service_account import ServiceAccountCredentials

if __name__ == "__main__":
    st.set_page_config(
        page_title="PaleFlo - Análisis de Pales",
        page_icon="🎯",
        layout="wide"
    )

# ============================================================================
# CONSTANTES
# ============================================================================

GS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'
GS_SHEET = 'Sorteos'

COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

MAPEO_SESIONES = {'t': 'Tarde', 'tarde': 'Tarde', 'n': 'Noche', 'noche': 'Noche'}

GRUPOS = {
    'CERRADOS': {'digitos': {0, 6, 8, 9}, 'numeros': []},
    'ABIERTOS': {'digitos': {2, 3, 5}, 'numeros': []},
    'RECTOS': {'digitos': {1, 4, 7}, 'numeros': []}
}

for i in range(100):
    num_str = f"{i:02d}"
    d1, d2 = int(num_str[0]), int(num_str[1])
    for grupo, datos in GRUPOS.items():
        if d1 in datos['digitos'] and d2 in datos['digitos']:
            datos['numeros'].append(i)

INFO_GRUPOS = {g: {'digitos': ','.join(map(str, d['digitos'])), 'cantidad': len(d['numeros'])} for g, d in GRUPOS.items()}

# ============================================================================
# CONEXIÓN
# ============================================================================

@st.cache_resource
def conectar_google_sheets():
    """Conecta con Google Sheets (local o Streamlit Cloud)"""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        try:
            if 'gcp_service_account' in st.secrets:
                from google.oauth2.service_account import Credentials as GoogleCredentials
                creds_dict = dict(st.secrets['gcp_service_account'])
                credentials = GoogleCredentials.from_service_account_info(creds_dict, scopes=scope)
                client = gspread.authorize(credentials)
                return client
        except Exception:
            pass
        
        import os
        creds_path = None
        posibles_rutas = [
            'credentials.json', 'credenciales.json',
            os.path.join(os.path.dirname(__file__), 'credentials.json'),
            os.path.join(os.path.dirname(__file__), 'credenciales.json'),
            os.path.join(os.getcwd(), 'credentials.json'),
            os.path.join(os.getcwd(), 'credenciales.json')
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                creds_path = ruta
                break
        
        if not creds_path:
            st.error("No se encontró 'credentials.json' ni 'credenciales.json'")
            return None
        
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Error conectando a Google Sheets: {e}")
        return None

def cargar_datos(_gc, archivo_id, nombre_hoja):
    """Carga datos desde Google Sheets"""
    if _gc is not None:
        try:
            spreadsheet = _gc.open_by_key(archivo_id)
            worksheet = spreadsheet.worksheet(nombre_hoja)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
    return None

# ============================================================================
# FUNCIONES
# ============================================================================

def parsear_fecha(fecha_str):
    if pd.isna(fecha_str):
        return None
    fecha_str = str(fecha_str).strip()
    formatos = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y']
    for fmt in formatos:
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    return None

def normalizar_sesion(sesion_raw):
    if pd.isna(sesion_raw):
        return None
    return MAPEO_SESIONES.get(str(sesion_raw).strip().lower(), sesion_raw)

def clasificar_numero(numero):
    try:
        num_str = f"{int(numero):02d}"
        d1, d2 = int(num_str[0]), int(num_str[1])
        for grupo, datos in GRUPOS.items():
            if d1 in datos['digitos'] and d2 in datos['digitos']:
                return grupo
        return None
    except:
        return None

def calcular_estado(dias_sin_aparecer, promedio):
    if promedio == 0:
        return "N/A"
    if dias_sin_aparecer <= promedio:
        return "✅ NORMAL"
    elif dias_sin_aparecer <= promedio * 1.5:
        return "⚠️ VENCIDA"
    else:
        return "🚨 MUY VENCIDA"

def calcular_temperatura(dias_sin_aparecer, promedio):
    if promedio == 0:
        return "N/A"
    ratio = dias_sin_aparecer / promedio
    if ratio <= 0.25:
        return "🔥 CALIENTE"
    elif ratio <= 1.0:
        return "🌡️ TIBIO"
    elif ratio <= 1.5:
        return "❄️ FRIO"
    else:
        return "🧊 MUY FRIO"

def detectar_pales_sesion(fila):
    grupos_en_sesion = defaultdict(list)
    for col in [COL_FIJO, COL_CORR1, COL_CORR2]:
        if col in fila.index and pd.notna(fila[col]):
            try:
                numero = int(fila[col])
                grupo = clasificar_numero(numero)
                if grupo:
                    grupos_en_sesion[grupo].append(numero)
            except:
                pass
    return {g: nums for g, nums in grupos_en_sesion.items() if len(nums) >= 2}

def analizar_pales(df):
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df['Sesion_Normalizada'] = df[COL_SESION].apply(normalizar_sesion)
    df = df.sort_values('Fecha_Parsed').reset_index(drop=True)
    fecha_max = df['Fecha_Parsed'].max()
    
    resultados = {}
    for grupo in GRUPOS.keys():
        fechas_pale = []
        for idx, row in df.iterrows():
            pales = detectar_pales_sesion(row)
            if grupo in pales:
                fechas_pale.append({'fecha': row['Fecha_Parsed'], 'sesion': row['Sesion_Normalizada'], 'numeros': pales[grupo]})
        
        if fechas_pale:
            fechas = [f['fecha'] for f in fechas_pale if pd.notna(f['fecha'])]
            fechas.sort()
            gaps = [(fechas[i] - fechas[i-1]).days for i in range(1, len(fechas)) if (fechas[i] - fechas[i-1]).days > 0]
            frecuencia = len(fechas)
            promedio_dias = np.mean(gaps) if gaps else 0
            ausencia_maxima = max(gaps) if gaps else 0
            ultima_fecha = fechas[-1] if fechas else None
            dias_sin_aparecer = (fecha_max - ultima_fecha).days if ultima_fecha else 0
        else:
            frecuencia, promedio_dias, ausencia_maxima = 0, 0, 0
            ultima_fecha, dias_sin_aparecer = None, 0
            fechas = []
        
        resultados[grupo] = {
            'frecuencia': frecuencia,
            'promedio_dias': round(promedio_dias, 1),
            'ausencia_maxima': ausencia_maxima,
            'dias_sin_aparecer': dias_sin_aparecer,
            'ultima_fecha': ultima_fecha,
            'estado': calcular_estado(dias_sin_aparecer, promedio_dias),
            'temperatura': calcular_temperatura(dias_sin_aparecer, promedio_dias),
            'numeros_grupo': GRUPOS[grupo]['numeros'],
            'fechas_pale': fechas_pale
        }
    
    return resultados, df, fecha_max

def analizar_combinaciones(df):
    fecha_max = df['Fecha_Parsed'].max()
    combinaciones_stats = {}
    
    for grupo in GRUPOS.keys():
        combinaciones = {}
        for idx, row in df.iterrows():
            pales = detectar_pales_sesion(row)
            if grupo in pales and len(pales[grupo]) >= 2:
                nums = sorted(pales[grupo])
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        comb = (nums[i], nums[j])
                        if comb not in combinaciones:
                            combinaciones[comb] = []
                        combinaciones[comb].append({'fecha': row['Fecha_Parsed'], 'sesion': row.get('Sesion_Normalizada')})
        
        combinaciones_resultado = []
        for comb, apariciones in combinaciones.items():
            fechas = [a['fecha'] for a in apariciones if pd.notna(a['fecha'])]
            fechas.sort()
            gaps = [(fechas[i] - fechas[i-1]).days for i in range(1, len(fechas)) if (fechas[i] - fechas[i-1]).days > 0]
            ultima = fechas[-1] if fechas else None
            dias_sin = (fecha_max - ultima).days if ultima else 0
            
            combinaciones_resultado.append({
                'combinacion': comb,
                'frecuencia': len(fechas),
                'promedio_dias': round(np.mean(gaps), 1) if gaps else 0,
                'dias_sin_aparecer': dias_sin,
                'estado': calcular_estado(dias_sin, np.mean(gaps) if gaps else 0)
            })
        
        combinaciones_resultado.sort(key=lambda x: x['frecuencia'], reverse=True)
        combinaciones_stats[grupo] = combinaciones_resultado
    
    return combinaciones_stats

def obtener_historial_pales(df, cantidad=50):
    sesion_orden = {'Noche': 1, 'N': 1, 'Tarde': 2, 'T': 2}
    df['Sesion_Orden'] = df['Sesion_Normalizada'].map(sesion_orden).fillna(999)
    
    df_con_pales = []
    for idx, row in df.iterrows():
        pales = detectar_pales_sesion(row)
        if pales:
            row_dict = row.to_dict()
            row_dict['Pales'] = pales
            df_con_pales.append(row_dict)
    
    if not df_con_pales:
        return pd.DataFrame()
    
    df_pales = pd.DataFrame(df_con_pales)
    return df_pales.sort_values(['Fecha_Parsed', 'Sesion_Orden'], ascending=[False, True]).head(cantidad)

# ============================================================================
# INTERFAZ
# ============================================================================

def main():
    st.title("🎯 PaleFlo - Análisis de Pales por Grupos")
    st.markdown("**Hoja: Sorteos** | Sesiones: Tarde (T) y Noche (N)")
    st.markdown("---")
    
    st.sidebar.header("📋 Grupos")
    for grupo, info in INFO_GRUPOS.items():
        st.sidebar.markdown(f"**{grupo}**: Dígitos {info['digitos']} ({info['cantidad']} números)")
    
    with st.spinner("Cargando datos..."):
        gc = conectar_google_sheets()
        df = cargar_datos(gc, GS_ID, GS_SHEET)
    
    if df is None or len(df) == 0:
        st.error("No se pudieron cargar los datos.")
        st.stop()
    
    st.success(f"✅ Datos cargados: {len(df)} registros")
    
    resultados_pales, df_procesado, fecha_max = analizar_pales(df)
    st.info(f"📅 Fecha más reciente: {fecha_max.strftime('%d/%m/%Y') if pd.notna(fecha_max) else 'N/A'}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumen", "🎯 Combinaciones", "📜 Historial", "ℹ️ Grupos"])
    
    def highlight_estado(val):
        if 'MUY VENCIDA' in str(val): return 'background-color: #ff6b6b; color: white'
        elif 'VENCIDA' in str(val): return 'background-color: #ffd93d; color: black'
        elif 'NORMAL' in str(val): return 'background-color: #6bcb77; color: white'
        return ''
    
    def highlight_temperatura(val):
        if 'CALIENTE' in str(val): return 'background-color: #ff4757; color: white'
        elif 'TIBIO' in str(val): return 'background-color: #ffa502; color: white'
        elif 'FRIO' in str(val): return 'background-color: #3742fa; color: white'
        elif 'MUY FRIO' in str(val): return 'background-color: #2f3542; color: white'
        return ''
    
    with tab1:
        st.header("📊 Resumen de Pales por Grupo")
        resumen_data = []
        for grupo in ['CERRADOS', 'ABIERTOS', 'RECTOS']:
            r = resultados_pales[grupo]
            resumen_data.append({
                'Grupo': grupo,
                'Dígitos': INFO_GRUPOS[grupo]['digitos'],
                'Frecuencia': r['frecuencia'],
                'Promedio Días': r['promedio_dias'],
                'Días Sin Aparecer': r['dias_sin_aparecer'],
                'Estado': r['estado'],
                'Temperatura': r['temperatura']
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen.style.applymap(highlight_estado, subset=['Estado']).applymap(highlight_temperatura, subset=['Temperatura']), use_container_width=True, hide_index=True)
        
        st.subheader("🔍 Detalle por Grupo")
        grupo_sel = st.selectbox("Selecciona grupo:", ['CERRADOS', 'ABIERTOS', 'RECTOS'])
        r = resultados_pales[grupo_sel]
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Frecuencia", r['frecuencia'])
        with col2: st.metric("Días Sin Aparecer", r['dias_sin_aparecer'])
        with col3: st.metric("Estado", r['estado'])
        
        if r['fechas_pale']:
            st.subheader(f"📅 Últimos Pales de {grupo_sel}")
            for fp in r['fechas_pale'][-10:][::-1]:
                fecha_str = fp['fecha'].strftime('%d/%m/%Y') if pd.notna(fp['fecha']) else '-'
                nums_str = ", ".join([f"{n:02d}" for n in fp['numeros']])
                st.write(f"• {fecha_str} ({fp['sesion']}): {nums_str}")
    
    with tab2:
        st.header("🎯 Combinaciones por Grupo")
        combinaciones = analizar_combinaciones(df_procesado)
        grupo_comb = st.selectbox("Grupo:", ['CERRADOS', 'ABIERTOS', 'RECTOS'], key="grupo_comb")
        
        combs = combinaciones[grupo_comb]
        if combs:
            df_combs = pd.DataFrame([
                {'Combinación': f"{c['combinacion'][0]:02d}-{c['combinacion'][1]:02d}", 'Frecuencia': c['frecuencia'], 'Días Sin Aparecer': c['dias_sin_aparecer'], 'Estado': c['estado']}
                for c in combs
            ])
            st.dataframe(df_combs.style.applymap(highlight_estado, subset=['Estado']), use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("📜 Historial de Pales")
        cantidad = st.number_input("Registros:", 10, 200, 50)
        df_hist = obtener_historial_pales(df_procesado, cantidad)
        
        if not df_hist.empty:
            datos_mostrar = []
            for idx, row in df_hist.iterrows():
                fecha_str = row['Fecha_Parsed'].strftime('%d/%m/%Y') if pd.notna(row['Fecha_Parsed']) else '-'
                sesion = row.get('Sesion_Normalizada', '-')
                pales_str = " | ".join([f"{g}: {', '.join([f'{n:02d}' for n in nums])}" for g, nums in row['Pales'].items()])
                datos_mostrar.append({'Fecha': fecha_str, 'Sesión': sesion, 'Pales': pales_str})
            
            df_final = pd.DataFrame(datos_mostrar)
            df_final.index = df_final.index + 1
            st.dataframe(df_final, use_container_width=True)
    
    with tab4:
        st.header("ℹ️ Información de Grupos")
        for grupo, datos in GRUPOS.items():
            st.subheader(f"🔒 {grupo}" if grupo == 'CERRADOS' else f"🔓 {grupo}" if grupo == 'ABIERTOS' else f"📏 {grupo}")
            st.write(f"**Dígitos:** {', '.join(map(str, datos['digitos']))}")
            nums_str = ", ".join([f"{n:02d}" for n in datos['numeros']])
            st.write(f"**Números ({len(datos['numeros'])}):** {nums_str}")
            st.markdown("---")

if __name__ == "__main__":
    main()