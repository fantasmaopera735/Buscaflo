# -*- coding: utf-8 -*-
"""
SumaFlo - Análisis de Sumas de Dígitos del Fijo
Hoja: Sorteos (T=Tarde, N=Noche)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import calendar
import gspread
from oauth2client.service_account import ServiceAccountCredentials

if __name__ == "__main__":
    st.set_page_config(
        page_title="SumaFlo - Análisis de Sumas del Fijo",
        page_icon="🔢",
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

MAPEO_SESIONES = {
    't': 'Tarde', 'tarde': 'Tarde',
    'n': 'Noche', 'noche': 'Noche'
}

SUMA_NUMEROS = {}
for i in range(100):
    num_str = f"{i:02d}"
    suma = int(num_str[0]) + int(num_str[1])
    if suma not in SUMA_NUMEROS:
        SUMA_NUMEROS[suma] = []
    SUMA_NUMEROS[suma].append(i)

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

def calcular_suma_digitos(numero):
    try:
        num_str = f"{int(numero):02d}"
        return int(num_str[0]) + int(num_str[1])
    except:
        return None

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

def calcular_estado(dias_sin_aparecer, promedio):
    if promedio == 0:
        return "N/A"
    if dias_sin_aparecer <= promedio:
        return "✅ NORMAL"
    elif dias_sin_aparecer <= promedio * 1.5:
        return "⚠️ VENCIDA"
    else:
        return "🚨 MUY VENCIDA"

def analizar_sumas(df):
    resultados = {}
    df['Sesion_Normalizada'] = df[COL_SESION].apply(normalizar_sesion)
    df['Suma'] = df[COL_FIJO].apply(calcular_suma_digitos)
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df = df.sort_values('Fecha_Parsed').reset_index(drop=True)
    fecha_max = df['Fecha_Parsed'].max()
    
    for suma in range(19):
        df_suma = df[df['Suma'] == suma].copy()
        if len(df_suma) == 0:
            resultados[suma] = {
                'frecuencia': 0, 'promedio_dias': 0, 'ausencia_maxima': 0,
                'dias_sin_aparecer': 0, 'ultima_fecha': None, 'estado': 'N/A',
                'temperatura': 'N/A', 'numeros': SUMA_NUMEROS.get(suma, [])
            }
            continue
        
        fechas = df_suma['Fecha_Parsed'].dropna().sort_values().tolist()
        gaps = [(fechas[i] - fechas[i-1]).days for i in range(1, len(fechas)) if (fechas[i] - fechas[i-1]).days > 0]
        
        ultima_fecha = fechas[-1] if fechas else None
        dias_sin_aparecer = (fecha_max - ultima_fecha).days if ultima_fecha else 0
        
        resultados[suma] = {
            'frecuencia': len(fechas),
            'promedio_dias': round(np.mean(gaps), 1) if gaps else 0,
            'ausencia_maxima': max(gaps) if gaps else 0,
            'dias_sin_aparecer': dias_sin_aparecer,
            'ultima_fecha': ultima_fecha,
            'estado': calcular_estado(dias_sin_aparecer, np.mean(gaps) if gaps else 0),
            'temperatura': calcular_temperatura(dias_sin_aparecer, np.mean(gaps) if gaps else 0),
            'numeros': SUMA_NUMEROS.get(suma, [])
        }
    
    return resultados, df, fecha_max

def analizar_pronostico_dia(df):
    patrones_tarde_noche = defaultdict(lambda: defaultdict(int))
    conteo_tarde = defaultdict(int)
    conteo_noche = defaultdict(int)
    
    for fecha, grupo in df.groupby('Fecha_Parsed'):
        if pd.isna(fecha):
            continue
        sumas_sesion = {}
        for _, row in grupo.iterrows():
            sesion = row['Sesion_Normalizada']
            suma = row['Suma']
            if pd.notna(suma) and sesion:
                sesion_lower = str(sesion).lower()
                if sesion_lower in ['tarde', 't']:
                    sumas_sesion['tarde'] = int(suma)
                    conteo_tarde[int(suma)] += 1
                elif sesion_lower in ['noche', 'n']:
                    sumas_sesion['noche'] = int(suma)
                    conteo_noche[int(suma)] += 1
        
        if 'tarde' in sumas_sesion and 'noche' in sumas_sesion:
            patrones_tarde_noche[sumas_sesion['tarde']][sumas_sesion['noche']] += 1
    
    return {
        'tarde_noche': {k: dict(v) for k, v in patrones_tarde_noche.items()},
        'totales_tarde_noche': {k: sum(v.values()) for k, v in patrones_tarde_noche.items()},
        'conteo_tarde': dict(conteo_tarde),
        'conteo_noche': dict(conteo_noche)
    }

def obtener_historial_sumas(df, cantidad=50):
    sesion_orden = {'Noche': 1, 'N': 1, 'Tarde': 2, 'T': 2}
    df['Sesion_Orden'] = df['Sesion_Normalizada'].map(sesion_orden).fillna(999)
    return df.sort_values(['Fecha_Parsed', 'Sesion_Orden'], ascending=[False, True]).head(cantidad)

def analizar_almanaque(df, dia_inicio, dia_fin, cantidad_meses):
    hoy = datetime.now()
    meses_analizar = []
    
    for i in range(1, cantidad_meses + 1):
        mes_target = hoy.month - i
        año_target = hoy.year
        while mes_target <= 0:
            mes_target += 12
            año_target -= 1
        try:
            fecha_ini_mes = datetime(año_target, mes_target, dia_inicio)
            dias_en_mes = calendar.monthrange(año_target, mes_target)[1]
            fecha_fin_mes = datetime(año_target, mes_target, min(dia_fin, dias_en_mes))
            meses_analizar.append({
                'fecha_ini': fecha_ini_mes, 'fecha_fin': fecha_fin_mes,
                'nombre_mes': fecha_ini_mes.strftime('%B %Y')
            })
        except ValueError:
            continue
    
    resultados_meses = {}
    sumas_por_mes = {}
    
    for mes_info in meses_analizar:
        df_mes = df[(df['Fecha_Parsed'] >= mes_info['fecha_ini']) & (df['Fecha_Parsed'] <= mes_info['fecha_fin'])].copy()
        sumas_mes = df_mes['Suma'].value_counts().to_dict()
        sumas_por_mes[mes_info['nombre_mes']] = sumas_mes
        resultados_meses[mes_info['nombre_mes']] = {
            'total_sorteos': len(df_mes),
            'sumas': sumas_mes
        }
    
    sumas_persistentes = set()
    if sumas_por_mes:
        conjuntos_sumas = [set(sumas.keys()) for sumas in sumas_por_mes.values()]
        sumas_persistentes = set.intersection(*conjuntos_sumas) if conjuntos_sumas else set()
    
    return {
        'meses_analizados': meses_analizar,
        'resultados_por_mes': resultados_meses,
        'sumas_persistentes': sumas_persistentes,
        'sumas_por_mes': sumas_por_mes
    }

# ============================================================================
# INTERFAZ
# ============================================================================

def main():
    st.title("🔢 SumaFlo - Análisis de Sumas del Fijo")
    st.markdown("**Hoja: Sorteos** | Sesiones: Tarde (T) y Noche (N)")
    st.markdown("---")
    
    with st.spinner("Cargando datos..."):
        gc = conectar_google_sheets()
        df = cargar_datos(gc, GS_ID, GS_SHEET)
    
    if df is None or len(df) == 0:
        st.error("No se pudieron cargar los datos.")
        st.stop()
    
    st.success(f"✅ Datos cargados: {len(df)} registros")
    
    resultados_sumas, df_procesado, fecha_max = analizar_sumas(df)
    st.info(f"📅 Fecha más reciente: {fecha_max.strftime('%d/%m/%Y') if pd.notna(fecha_max) else 'N/A'}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen", "📅 Almanaque", "🔮 Pronóstico", "📜 Historial", "🔥 Temperatura"
    ])
    
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
        st.header("📊 Resumen General de Sumas")
        resumen_data = []
        for suma in range(19):
            r = resultados_sumas[suma]
            resumen_data.append({
                'Suma': suma,
                'Números': ', '.join([f"{n:02d}" for n in r['numeros'][:5]]),
                'Frecuencia': r['frecuencia'],
                'Promedio Días': r['promedio_dias'],
                'Días Sin Aparecer': r['dias_sin_aparecer'],
                'Estado': r['estado'],
                'Temperatura': r['temperatura']
            })
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen.style.applymap(highlight_estado, subset=['Estado']).applymap(highlight_temperatura, subset=['Temperatura']), use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("📅 Almanaque")
        col1, col2, col3 = st.columns(3)
        with col1: dia_inicio = st.number_input("Día inicial:", 1, 31, 1)
        with col2: dia_fin = st.number_input("Día final:", 1, 31, 15)
        with col3: cantidad_meses = st.slider("Meses:", 1, 12, 3)
        
        if st.button("🔍 Analizar", type="primary"):
            almanaque = analizar_almanaque(df_procesado, dia_inicio, dia_fin, cantidad_meses)
            for mes, datos in almanaque['resultados_por_mes'].items():
                with st.expander(f"📆 {mes} - {datos['total_sorteos']} sorteos"):
                    if datos['sumas']:
                        df_mes = pd.DataFrame([{'Suma': s, 'Frecuencia': f} for s, f in sorted(datos['sumas'].items(), key=lambda x: x[1], reverse=True)])
                        st.dataframe(df_mes, use_container_width=True, hide_index=True)
            
            if almanaque['sumas_persistentes']:
                st.success(f"✅ Sumas persistentes: {sorted(almanaque['sumas_persistentes'])}")
    
    with tab3:
        st.header("🔮 Pronóstico Tarde → Noche")
        pronostico = analizar_pronostico_dia(df_procesado)
        
        suma_tarde = st.selectbox("Suma en la Tarde:", options=list(range(19)), key="suma_tarde")
        total_tarde = pronostico['conteo_tarde'].get(suma_tarde, 0)
        st.info(f"📊 Suma {suma_tarde} apareció {total_tarde} veces en la Tarde")
        
        noche_probs = pronostico['tarde_noche'].get(suma_tarde, {})
        total_patron = pronostico['totales_tarde_noche'].get(suma_tarde, 0)
        
        if noche_probs:
            st.markdown("### 🌙 Probable en la Noche")
            df_noche = pd.DataFrame([{'Suma': s, 'Frecuencia': f, 'Porcentaje': f"{f/total_patron*100:.1f}%"} for s, f in sorted(noche_probs.items(), key=lambda x: x[1], reverse=True)])
            st.dataframe(df_noche, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos para este patrón.")
    
    with tab4:
        st.header("📜 Historial")
        cantidad = st.number_input("Registros:", 10, 200, 50)
        df_hist = obtener_historial_sumas(df_procesado, cantidad)
        df_mostrar = df_hist[['Fecha_Parsed', 'Sesion_Normalizada', COL_FIJO, 'Suma']].copy()
        df_mostrar['Fecha'] = df_mostrar['Fecha_Parsed'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '-')
        df_mostrar['Fijo'] = df_mostrar[COL_FIJO].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else '-')
        df_final = df_mostrar[['Fecha', 'Sesion_Normalizada', 'Fijo', 'Suma']].reset_index(drop=True)
        df_final.index = df_final.index + 1
        st.dataframe(df_final, use_container_width=True)
    
    with tab5:
        st.header("🔥 Temperatura y Estado")
        temp_data = [{'Suma': s, 'Temperatura': resultados_sumas[s]['temperatura'], 'Estado': resultados_sumas[s]['estado'], 'Días Sin Aparecer': resultados_sumas[s]['dias_sin_aparecer']} for s in range(19)]
        df_temp = pd.DataFrame(temp_data)
        st.dataframe(df_temp.style.applymap(highlight_estado, subset=['Estado']).applymap(highlight_temperatura, subset=['Temperatura']), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()