# -*- coding: utf-8 -*-
"""
SumaFlo - Análisis de Sumas de Dígitos del Fijo
Aplicación Streamlit para análisis de lotería cubana
Hoja: Sorteos
Versión 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import calendar
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configuración de página (solo cuando se ejecuta directamente)
import sys
if __name__ == "__main__":
    st.set_page_config(
        page_title="SumaFlo - Análisis de Sumas del Fijo",
        page_icon="🔢",
        layout="wide"
    )

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

GS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'
GS_SHEET = 'Sorteos'

# Columnas de Google Sheets (hoja Sorteos)
COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

# Orden de sesiones para mostrar (más reciente a menos reciente en un mismo día)
ORDEN_SESIONES_DISPLAY = ['Noche', 'Tarde', 'Mañana']

# Mapeo de sesiones (T=Tarde, N=Noche)
MAPEO_SESIONES = {
    't': 'Tarde',
    'tarde': 'Tarde',
    'n': 'Noche',
    'noche': 'Noche',
    'm': 'Mañana',
    'mañana': 'Mañana',
    'manana': 'Mañana'
}

# Números que componen cada suma (00-99)
SUMA_NUMEROS = {}
for i in range(100):
    num_str = f"{i:02d}"
    suma = int(num_str[0]) + int(num_str[1])
    if suma not in SUMA_NUMEROS:
        SUMA_NUMEROS[suma] = []
    SUMA_NUMEROS[suma].append(i)

# ============================================================================
# FUNCIONES DE CONEXIÓN
# ============================================================================

@st.cache_resource
def conectar_google_sheets():
    """Conecta con Google Sheets"""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Buscar archivo de credenciales en múltiples ubicaciones
        import os
        creds_path = None
        
        # Lista de posibles ubicaciones y nombres
        posibles_rutas = [
            'credentials.json',
            'credenciales.json',
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

@st.cache_data(ttl=300)
def cargar_datos(_gc, archivo_id, nombre_hoja):
    """Carga datos desde Google Sheets (hoja Sorteos)"""
    if _gc is not None:
        try:
            spreadsheet = _gc.open_by_key(archivo_id)
            worksheet = spreadsheet.worksheet(nombre_hoja)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            # Verificar que existan las columnas necesarias
            columnas_requeridas = [COL_FECHA, COL_SESION, COL_FIJO]
            for col in columnas_requeridas:
                if col not in df.columns:
                    st.error(f"Columna '{col}' no encontrada. Columnas disponibles: {list(df.columns)}")
                    return None
            
            return df
        except Exception as e:
            st.error(f"Error cargando datos desde Google Sheets: {e}")
            return None
    else:
        st.error("No se pudo conectar a Google Sheets.")
        return None

# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def parsear_fecha(fecha_str):
    """Parsea fecha en múltiples formatos"""
    if pd.isna(fecha_str):
        return None
    
    fecha_str = str(fecha_str).strip()
    
    formatos = [
        '%d/%m/%Y',      # 2/9/2017
        '%d-%m-%Y',      # 02-09-2017
        '%Y-%m-%d',      # 2017-09-02
        '%d/%m/%y',      # 2/9/17
        '%d-%m-%y',      # 02-09-17
        '%d/%m/%Y',      # 19/5/2008
    ]
    
    for fmt in formatos:
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    
    return None

def normalizar_sesion(sesion_raw):
    """Normaliza el nombre de la sesión (T->Tarde, N->Noche)"""
    if pd.isna(sesion_raw):
        return None
    sesion_lower = str(sesion_raw).strip().lower()
    return MAPEO_SESIONES.get(sesion_lower, sesion_raw)

def calcular_suma_digitos(numero):
    """Calcula la suma de los dígitos de un número"""
    try:
        num_str = f"{int(numero):02d}"
        return int(num_str[0]) + int(num_str[1])
    except:
        return None

def calcular_temperatura(dias_sin_aparecer, promedio):
    """Calcula la temperatura de una suma"""
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
    """Calcula el estado de una suma"""
    if promedio == 0:
        return "N/A"
    
    if dias_sin_aparecer <= promedio:
        return "✅ NORMAL"
    elif dias_sin_aparecer <= promedio * 1.5:
        return "⚠️ VENCIDA"
    else:
        return "🚨 MUY VENCIDA"

def analizar_sumas(df):
    """Analiza las sumas de los dígitos del Fijo"""
    resultados = {}
    
    # Normalizar sesión y calcular suma
    df['Sesion_Normalizada'] = df[COL_SESION].apply(normalizar_sesion)
    df['Suma'] = df[COL_FIJO].apply(calcular_suma_digitos)
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    
    # Ordenar por fecha
    df = df.sort_values('Fecha_Parsed').reset_index(drop=True)
    
    # Fecha más reciente
    fecha_max = df['Fecha_Parsed'].max()
    
    # Analizar cada suma (0-18)
    for suma in range(19):
        df_suma = df[df['Suma'] == suma].copy()
        
        if len(df_suma) == 0:
            resultados[suma] = {
                'frecuencia': 0,
                'promedio_dias': 0,
                'ausencia_maxima': 0,
                'dias_sin_aparecer': 0,
                'ultima_fecha': None,
                'estado': 'N/A',
                'temperatura': 'N/A',
                'numeros': SUMA_NUMEROS.get(suma, []),
                'fechas_aparicion': [],
                'gap_actual': 0,
                'gaps': []
            }
            continue
        
        # Fechas de aparición ordenadas
        fechas = df_suma['Fecha_Parsed'].dropna().sort_values().tolist()
        
        # Calcular gaps
        gaps = []
        for i in range(1, len(fechas)):
            gap = (fechas[i] - fechas[i-1]).days
            if gap > 0:
                gaps.append(gap)
        
        # Estadísticas
        frecuencia = len(fechas)
        promedio_dias = np.mean(gaps) if gaps else 0
        ausencia_maxima = max(gaps) if gaps else 0
        
        # Días sin aparecer
        ultima_fecha = fechas[-1] if fechas else None
        dias_sin_aparecer = (fecha_max - ultima_fecha).days if ultima_fecha else 0
        
        resultados[suma] = {
            'frecuencia': frecuencia,
            'promedio_dias': round(promedio_dias, 1),
            'ausencia_maxima': ausencia_maxima,
            'dias_sin_aparecer': dias_sin_aparecer,
            'ultima_fecha': ultima_fecha,
            'estado': calcular_estado(dias_sin_aparecer, promedio_dias),
            'temperatura': calcular_temperatura(dias_sin_aparecer, promedio_dias),
            'numeros': SUMA_NUMEROS.get(suma, []),
            'fechas_aparicion': fechas,
            'gap_actual': dias_sin_aparecer,
            'gaps': gaps
        }
    
    return resultados, df, fecha_max

def analizar_numeros_por_suma(df, suma, numeros):
    """Analiza los números individuales que componen una suma"""
    resultados = []
    
    df_suma = df[df['Suma'] == suma].copy()
    fecha_max = df['Fecha_Parsed'].max()
    
    for num in numeros:
        df_num = df_suma[df_suma[COL_FIJO] == num].copy()
        
        if len(df_num) == 0:
            resultados.append({
                'Numero': f"{num:02d}",
                'Frecuencia': 0,
                'Ultima_Fecha': None,
                'Dias_Sin_Aparecer': '-'
            })
            continue
        
        fechas = df_num['Fecha_Parsed'].dropna().sort_values().tolist()
        ultima_fecha = fechas[-1] if fechas else None
        dias_sin = (fecha_max - ultima_fecha).days if ultima_fecha else 0
        
        resultados.append({
            'Numero': f"{num:02d}",
            'Frecuencia': len(fechas),
            'Ultima_Fecha': ultima_fecha.strftime('%d/%m/%Y') if ultima_fecha else '-',
            'Dias_Sin_Aparecer': dias_sin
        })
    
    return pd.DataFrame(resultados).sort_values('Frecuencia', ascending=False)

def analizar_pronostico_dia(df):
    """Analiza correlaciones entre sumas del mismo día"""
    # Conteos
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
        
        # Tarde → Noche
        if 'tarde' in sumas_sesion and 'noche' in sumas_sesion:
            key = sumas_sesion['tarde']
            patrones_tarde_noche[key][sumas_sesion['noche']] += 1
    
    def convertir_dict(d):
        return {k: dict(v) for k, v in d.items()}
    
    totales_tarde_noche = {k: sum(v.values()) for k, v in patrones_tarde_noche.items()}
    
    return {
        'tarde_noche': convertir_dict(patrones_tarde_noche),
        'totales_tarde_noche': totales_tarde_noche,
        'conteo_tarde': dict(conteo_tarde),
        'conteo_noche': dict(conteo_noche)
    }

def obtener_historial_sumas(df, cantidad=50):
    """Obtiene historial ordenado por día (Noche → Tarde)"""
    sesion_orden = {'Noche': 1, 'N': 1, 'Tarde': 2, 'T': 2}
    df['Sesion_Orden'] = df['Sesion_Normalizada'].map(sesion_orden).fillna(999)
    
    df_historial = df.sort_values(
        ['Fecha_Parsed', 'Sesion_Orden'], 
        ascending=[False, True]
    ).head(cantidad)
    
    return df_historial

def analizar_almanaque(df, fecha_inicio, fecha_fin, cantidad_meses):
    """Analiza sumas por meses (empezando desde mes anterior)"""
    hoy = datetime.now()
    meses_analizar = []
    
    for i in range(1, cantidad_meses + 1):
        mes_target = hoy.month - i
        año_target = hoy.year
        
        while mes_target <= 0:
            mes_target += 12
            año_target -= 1
        
        try:
            fecha_ini_mes = datetime(año_target, mes_target, fecha_inicio)
            dias_en_mes = calendar.monthrange(año_target, mes_target)[1]
            dia_fin = min(fecha_fin, dias_en_mes)
            fecha_fin_mes = datetime(año_target, mes_target, dia_fin)
            
            meses_analizar.append({
                'mes': mes_target,
                'año': año_target,
                'fecha_ini': fecha_ini_mes,
                'fecha_fin': fecha_fin_mes,
                'nombre_mes': fecha_ini_mes.strftime('%B %Y')
            })
        except ValueError:
            continue
    
    resultados_meses = {}
    sumas_por_mes = {}
    fechas_por_suma_mes = {}
    
    for mes_info in meses_analizar:
        df_mes = df[
            (df['Fecha_Parsed'] >= mes_info['fecha_ini']) & 
            (df['Fecha_Parsed'] <= mes_info['fecha_fin'])
        ].copy()
        
        sumas_mes = df_mes['Suma'].value_counts().to_dict()
        sumas_por_mes[mes_info['nombre_mes']] = sumas_mes
        
        fechas_suma = {}
        for suma in range(19):
            df_suma = df_mes[df_mes['Suma'] == suma]
            fechas = df_suma['Fecha_Parsed'].dropna().sort_values().tolist()
            if fechas:
                fechas_suma[suma] = [f.strftime('%d/%m/%Y') for f in fechas]
        fechas_por_suma_mes[mes_info['nombre_mes']] = fechas_suma
        
        resultados_meses[mes_info['nombre_mes']] = {
            'total_sorteos': len(df_mes),
            'sumas': sumas_mes,
            'suma_mas_frecuente': max(sumas_mes.items(), key=lambda x: x[1]) if sumas_mes else (None, 0),
            'fechas_sumas': fechas_suma
        }
    
    sumas_persistentes = set()
    if sumas_por_mes:
        conjuntos_sumas = [set(sumas.keys()) for sumas in sumas_por_mes.values()]
        sumas_persistentes = set.intersection(*conjuntos_sumas) if conjuntos_sumas else set()
    
    fechas_persistentes = {}
    for suma in sumas_persistentes:
        fechas_total = []
        for mes in resultados_meses.keys():
            if suma in resultados_meses[mes]['fechas_sumas']:
                fechas_total.extend(resultados_meses[mes]['fechas_sumas'][suma])
        fechas_persistentes[suma] = fechas_total
    
    conteo_meses_suma = defaultdict(int)
    for mes, sumas in sumas_por_mes.items():
        for suma in sumas.keys():
            conteo_meses_suma[suma] += 1
    
    total_sumas = defaultdict(int)
    for sumas in sumas_por_mes.values():
        for suma, count in sumas.items():
            total_sumas[suma] += count
    
    return {
        'meses_analizados': meses_analizar,
        'resultados_por_mes': resultados_meses,
        'sumas_persistentes': sumas_persistentes,
        'fechas_persistentes': fechas_persistentes,
        'conteo_meses_suma': dict(conteo_meses_suma),
        'total_sumas': dict(total_sumas),
        'cantidad_meses': cantidad_meses
    }

def calcular_numeros_salidores(df, cantidad=10, rango_dias=None):
    """Determina qué números son más salidores"""
    fecha_max = df['Fecha_Parsed'].max()
    
    if rango_dias:
        fecha_inicio = fecha_max - timedelta(days=rango_dias)
        df_filtrado = df[df['Fecha_Parsed'] >= fecha_inicio].copy()
    else:
        df_filtrado = df.copy()
    
    freq_numeros = df_filtrado[COL_FIJO].value_counts()
    top_salidores = freq_numeros.head(cantidad).to_dict()
    freq_sumas = df_filtrado['Suma'].value_counts().to_dict()
    
    return top_salidores, freq_sumas

def analizar_transiciones_sumas(df):
    """Analiza transiciones entre sumas consecutivas"""
    transiciones = defaultdict(int)
    suma_anterior = None
    
    df = df.sort_values(['Fecha_Parsed', 'Sesion_Orden']).reset_index(drop=True)
    
    for _, row in df.iterrows():
        suma_actual = row['Suma']
        if pd.notna(suma_actual):
            if suma_anterior is not None:
                par = (int(suma_anterior), int(suma_actual))
                transiciones[par] += 1
            suma_anterior = suma_actual
    
    return transiciones

def analizar_sumas_atraen_rechazan(df):
    """Analiza qué sumas se atraen y cuáles se rechazan"""
    transiciones = analizar_transiciones_sumas(df)
    total_trans = sum(transiciones.values())
    
    matriz = np.zeros((19, 19))
    for (s1, s2), count in transiciones.items():
        if 0 <= s1 < 19 and 0 <= s2 < 19:
            matriz[s1][s2] = count
    
    probabilidad_esperada = total_trans / (19 * 19) if total_trans > 0 else 0
    
    atraen = []
    for (s1, s2), count in transiciones.items():
        if count > probabilidad_esperada * 1.5:
            atraen.append({
                'Suma_Origen': s1,
                'Suma_Destino': s2,
                'Frecuencia': count,
                'Tipo': 'ATRACCIÓN'
            })
    
    rechazan = []
    for s1 in range(19):
        for s2 in range(19):
            if (s1, s2) not in transiciones or transiciones[(s1, s2)] == 0:
                rechazan.append({
                    'Suma_Origen': s1,
                    'Suma_Destino': s2,
                    'Frecuencia': 0,
                    'Tipo': 'RECHAZO TOTAL'
                })
    
    atraen = sorted(atraen, key=lambda x: x['Frecuencia'], reverse=True)[:20]
    rechazan = sorted(rechazan, key=lambda x: x['Frecuencia'])[:20]
    
    return atraen, rechazan, matriz

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

def main():
    st.title("🔢 SumaFlo - Análisis de Sumas del Fijo")
    st.markdown("**Hoja: Sorteos** | Fuente: Google Sheets")
    st.markdown("---")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        gc = conectar_google_sheets()
        df = cargar_datos(gc, GS_ID, GS_SHEET)
    
    if df is None or len(df) == 0:
        st.error("No se pudieron cargar los datos.")
        st.stop()
    
    st.success(f"✅ Datos cargados: {len(df)} registros")
    
    # Procesar datos
    resultados_sumas, df_procesado, fecha_max = analizar_sumas(df)
    st.info(f"📅 Fecha más reciente: {fecha_max.strftime('%d/%m/%Y') if pd.notna(fecha_max) else 'N/A'}")
    
    # Crear pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resumen de Sumas",
        "🏆 Más Salidores",
        "📅 Almanaque",
        "🔮 Pronóstico del Día",
        "📜 Historial",
        "🧲 Atracción/Rechazo"
    ])
    
    # Funciones de estilo
    def highlight_estado(val):
        if 'MUY VENCIDA' in str(val):
            return 'background-color: #ff6b6b; color: white'
        elif 'VENCIDA' in str(val):
            return 'background-color: #ffd93d; color: black'
        elif 'NORMAL' in str(val):
            return 'background-color: #6bcb77; color: white'
        return ''
    
    def highlight_temperatura(val):
        if 'CALIENTE' in str(val):
            return 'background-color: #ff4757; color: white'
        elif 'TIBIO' in str(val):
            return 'background-color: #ffa502; color: white'
        elif 'FRIO' in str(val):
            return 'background-color: #3742fa; color: white'
        elif 'MUY FRIO' in str(val):
            return 'background-color: #2f3542; color: white'
        return ''
    
    def obtener_fiabilidad(cantidad):
        if cantidad >= 50:
            return "✅ MUY FIABLE", "green"
        elif cantidad >= 20:
            return "🟢 FIABLE", "green"
        elif cantidad >= 10:
            return "🟡 MODERADA", "orange"
        else:
            return "🔴 POCO FIABLE", "red"
    
    # ==================== PESTAÑA 1: RESUMEN ====================
    with tab1:
        st.header("📊 Resumen General de Sumas")
        
        resumen_data = []
        for suma in range(19):
            r = resultados_sumas[suma]
            resumen_data.append({
                'Suma': suma,
                'Números': ', '.join([f"{n:02d}" for n in r['numeros'][:5]]) + ('...' if len(r['numeros']) > 5 else ''),
                'Frecuencia': r['frecuencia'],
                'Promedio Días': r['promedio_dias'],
                'Ausencia Máx': r['ausencia_maxima'],
                'Días Sin Aparecer': r['dias_sin_aparecer'],
                'Última Fecha': r['ultima_fecha'].strftime('%d/%m/%Y') if r['ultima_fecha'] else '-',
                'Estado': r['estado'],
                'Temperatura': r['temperatura']
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        styled_df = df_resumen.style.applymap(highlight_estado, subset=['Estado']).applymap(highlight_temperatura, subset=['Temperatura'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🔍 Ver Detalle de una Suma")
        
        suma_seleccionada = st.selectbox(
            "Selecciona una Suma para ver detalles:",
            options=list(range(19)),
            format_func=lambda x: f"Suma {x} - Números: {SUMA_NUMEROS[x]}"
        )
        
        r = resultados_sumas[suma_seleccionada]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frecuencia Total", r['frecuencia'])
            st.metric("Promedio de Días", r['promedio_dias'])
        with col2:
            st.metric("Ausencia Máxima", f"{r['ausencia_maxima']} días")
            st.metric("Días Sin Aparecer", r['dias_sin_aparecer'])
        with col3:
            st.metric("Estado", r['estado'])
            st.metric("Temperatura", r['temperatura'])
        
        st.subheader(f"📋 Números que componen la Suma {suma_seleccionada}")
        df_numeros = analizar_numeros_por_suma(df_procesado, suma_seleccionada, r['numeros'])
        st.dataframe(df_numeros, use_container_width=True, hide_index=True)
    
    # ==================== PESTAÑA 2: MÁS SALIDORES ====================
    with tab2:
        st.header("🏆 Sumas y Números Más Salidores")
        
        col1, col2 = st.columns(2)
        with col1:
            cantidad = st.number_input("Cantidad a mostrar:", min_value=5, max_value=50, value=10)
        with col2:
            usar_rango = st.checkbox("Filtrar por rango de días", value=False)
            rango_dias = st.number_input("Rango de días:", min_value=7, max_value=365, value=30) if usar_rango else None
        
        top_numeros, top_sumas = calcular_numeros_salidores(df_procesado, cantidad, rango_dias)
        
        st.subheader("📊 Sumas Más Salidores")
        if top_sumas:
            df_top_sumas = pd.DataFrame([
                {'Suma': s, 'Frecuencia': f} for s, f in sorted(top_sumas.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_top_sumas, use_container_width=True, hide_index=True)
        
        st.subheader("🔢 Números Más Salidores")
        if top_numeros:
            df_top_num = pd.DataFrame([
                {'Número': f"{n:02d}", 'Suma': calcular_suma_digitos(n), 'Frecuencia': f} 
                for n, f in top_numeros.items()
            ])
            st.dataframe(df_top_num, use_container_width=True, hide_index=True)
    
    # ==================== PESTAÑA 3: ALMANAQUE ====================
    with tab3:
        st.header("📅 Almanaque - Análisis por Meses")
        
        st.markdown("""
        **IMPORTANTE:** El análisis empieza desde el **mes anterior** al actual.
        
        **Ejemplo:** Si estamos en Marzo 2026 y seleccionas 3 meses:
        - Se analizarán: Febrero 2026, Enero 2026, Diciembre 2025
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            dia_inicio = st.number_input("Día inicial:", min_value=1, max_value=31, value=1)
        with col2:
            dia_fin = st.number_input("Día final:", min_value=1, max_value=31, value=15)
        with col3:
            cantidad_meses = st.slider("Cantidad de meses:", min_value=1, max_value=12, value=3)
        
        if st.button("🔍 Analizar Almanaque", type="primary"):
            with st.spinner("Analizando..."):
                almanaque = analizar_almanaque(df_procesado, dia_inicio, dia_fin, cantidad_meses)
            
            st.success(f"📊 Analizando {almanaque['cantidad_meses']} meses")
            
            fechas_info = " | ".join([f"{m['nombre_mes']}: {m['fecha_ini'].strftime('%d/%m')} al {m['fecha_fin'].strftime('%d/%m')}" 
                                     for m in almanaque['meses_analizados']])
            st.info(f"📅 Fechas analizadas: {fechas_info}")
            
            st.subheader("📋 Resultados por Mes")
            for mes, datos in almanaque['resultados_por_mes'].items():
                with st.expander(f"📆 {mes} - {datos['total_sorteos']} sorteos"):
                    if datos['sumas']:
                        df_mes = pd.DataFrame([
                            {'Suma': s, 'Frecuencia': f} 
                            for s, f in sorted(datos['sumas'].items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(df_mes, use_container_width=True, hide_index=True)
            
            st.subheader("🔄 Sumas Persistentes")
            st.markdown("*Sumas que aparecen al menos una vez en CADA mes analizado*")
            
            if almanaque['sumas_persistentes']:
                st.success(f"✅ {len(almanaque['sumas_persistentes'])} sumas persistentes encontradas:")
                
                # Selector de ordenamiento
                col_ord1, col_ord2 = st.columns([1, 2])
                with col_ord1:
                    orden = st.radio(
                        "Ordenar por frecuencia:",
                        options=["Mayor a menor", "Menor a mayor"],
                        horizontal=True,
                        key="orden_persistentes_flo"
                    )
                
                # Crear lista de sumas con sus totales para ordenar
                sumas_con_total = [
                    (suma, almanaque['total_sumas'].get(suma, 0))
                    for suma in almanaque['sumas_persistentes']
                ]
                
                # Ordenar según selección
                if orden == "Mayor a menor":
                    sumas_ordenadas = sorted(sumas_con_total, key=lambda x: x[1], reverse=True)
                else:
                    sumas_ordenadas = sorted(sumas_con_total, key=lambda x: x[1], reverse=False)
                
                # Mostrar tabla resumen
                with col_ord2:
                    df_persistentes = pd.DataFrame([
                        {'Suma': s, 'Apariciones Totales': total, 
                         'Meses Presente': almanaque['conteo_meses_suma'].get(s, 0)}
                        for s, total in sumas_ordenadas
                    ])
                    st.dataframe(df_persistentes, use_container_width=True, hide_index=True)
                
                # Detalle de cada suma
                st.markdown("### 📋 Detalle de Sumas Persistentes")
                for suma, total in sumas_ordenadas:
                    fechas = almanaque['fechas_persistentes'].get(suma, [])
                    fechas_str = ", ".join(fechas) if fechas else ""
                    
                    with st.expander(f"**Suma {suma}** - {total} apariciones en {almanaque['cantidad_meses']} meses"):
                        st.write(f"**Fechas de salida:** {fechas_str}")
            else:
                st.warning("⚠️ No hay sumas que aparezcan en todos los meses analizados.")
    
    # ==================== PESTAÑA 4: PRONÓSTICO ====================
    with tab4:
        st.header("🔮 Pronóstico del Día")
        
        st.markdown("""
        **🚦 Indicador de Fiabilidad:**
        | Cantidad | Fiabilidad | Recomendación |
        |----------|------------|---------------|
        | ≥ 50 sorteos | ✅ **MUY FIABLE** | Alta confianza |
        | 20-49 sorteos | 🟢 **FIABLE** | Buena confianza |
        | 10-19 sorteos | 🟡 **MODERADA** | Usar con precaución |
        | < 10 sorteos | 🔴 **POCO FIABLE** | No recomendado |
        """)
        
        with st.spinner("Analizando patrones..."):
            pronostico = analizar_pronostico_dia(df_procesado)
        
        st.subheader("🌤️ Si en la Tarde sale...")
        suma_tarde = st.selectbox("Selecciona la suma de la Tarde:", options=list(range(19)), key="suma_tarde")
        
        total_tarde = pronostico['conteo_tarde'].get(suma_tarde, 0)
        st.info(f"📊 **Suma {suma_tarde} apareció {total_tarde} veces en la Tarde**")
        
        noche_probs = pronostico['tarde_noche'].get(suma_tarde, {})
        total_patron = pronostico['totales_tarde_noche'].get(suma_tarde, 0)
        
        if noche_probs:
            fiabilidad, color = obtener_fiabilidad(total_patron)
            if color == "green":
                st.success(f"**Fiabilidad: {fiabilidad}** | {total_patron} sorteos encontrados")
            elif color == "orange":
                st.warning(f"**Fiabilidad: {fiabilidad}** | {total_patron} sorteos encontrados")
            else:
                st.error(f"**Fiabilidad: {fiabilidad}** | {total_patron} sorteos encontrados")
            
            st.markdown("### 🌙 Probable en la Noche")
            df_noche = pd.DataFrame([
                {'Suma': s, 'Frecuencia': f, 'Porcentaje': f"{f/total_patron*100:.1f}%"} 
                for s, f in sorted(noche_probs.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_noche, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos suficientes para este patrón.")
    
    # ==================== PESTAÑA 5: HISTORIAL ====================
    with tab5:
        st.header("📜 Historial de Sumas")
        st.markdown("**Orden:** Por fecha descendente. Dentro de cada día: Noche → Tarde")
        
        cantidad_historial = st.number_input("Cantidad de registros:", min_value=10, max_value=200, value=50)
        
        df_historial = obtener_historial_sumas(df_procesado, cantidad_historial)
        
        df_mostrar = df_historial[['Fecha_Parsed', 'Sesion_Normalizada', COL_FIJO, 'Suma']].copy()
        df_mostrar['Fecha'] = df_mostrar['Fecha_Parsed'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '-')
        df_mostrar['Fijo'] = df_mostrar[COL_FIJO].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else '-')
        df_mostrar['Sesión'] = df_mostrar['Sesion_Normalizada']
        df_mostrar['Suma'] = df_mostrar['Suma'].apply(lambda x: int(x) if pd.notna(x) else '-')
        
        df_final = df_mostrar[['Fecha', 'Sesión', 'Fijo', 'Suma']].reset_index(drop=True)
        df_final.index = df_final.index + 1
        
        st.dataframe(df_final, use_container_width=True)
    
    # ==================== PESTAÑA 6: ATRACCIÓN/RECHAZO ====================
    with tab6:
        st.header("🧲 Análisis de Sumas que se Atraen y Rechazan")
        
        atraen, rechazan, matriz = analizar_sumas_atraen_rechazan(df_procesado)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧲 Sumas que se Atraen")
            if atraen:
                df_atraen = pd.DataFrame(atraen)
                df_atraen['Combinación'] = df_atraen.apply(lambda x: f"Suma {x['Suma_Origen']} → {x['Suma_Destino']}", axis=1)
                st.dataframe(df_atraen[['Combinación', 'Frecuencia']], use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron atracciones significativas.")
        
        with col2:
            st.subheader("❌ Sumas que se Rechazan")
            if rechazan:
                df_rechazan = pd.DataFrame(rechazan)
                df_rechazan['Combinación'] = df_rechazan.apply(lambda x: f"Suma {x['Suma_Origen']} → {x['Suma_Destino']}", axis=1)
                st.dataframe(df_rechazan[['Combinación', 'Frecuencia']], use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron rechazos significativos.")

if __name__ == "__main__":
    main()