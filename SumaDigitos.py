# -*- coding: utf-8 -*-
"""
SumaDigitos - Análisis de Sumas de Dígitos del Fijo
Aplicación Streamlit para análisis de lotería cubana
Versión 2.1
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
        page_title="SumaDigitos - Análisis de Sumas del Fijo",
        page_icon="🔢",
        layout="wide"
    )

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

GS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'
GS_SHEET = 'Geotodo'

# Columnas de Google Sheets (hoja Geotodo)
COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

# Orden de sesiones para mostrar (más reciente a menos reciente en un mismo día)
ORDEN_SESIONES_DISPLAY = ['Noche', 'Tarde', 'Mañana']

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
    """Conecta con Google Sheets (local o Streamlit Cloud)"""
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        # Opción 1: Intentar leer desde st.secrets (Streamlit Cloud)
        try:
            if 'gcp_service_account' in st.secrets:
                from google.oauth2.service_account import Credentials as GoogleCredentials
                creds_dict = dict(st.secrets['gcp_service_account'])
                credentials = GoogleCredentials.from_service_account_info(creds_dict, scopes=scope)
                client = gspread.authorize(credentials)
                return client
        except Exception:
            pass
        
        # Opción 2: Buscar archivo de credenciales (local)
        import os
        creds_path = None
        
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
    """Carga datos desde Google Sheets (hoja Geotodo)"""
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
    ]
    
    for fmt in formatos:
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    
    return None

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
    
    # Calcular suma para cada Fijo
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
        
        # Calcular gaps (días entre apariciones)
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
    """
    Analiza correlaciones entre sumas del mismo día.
    
    Patrones:
    1. Si en Mañana sale suma X → qué sumas suelen salir en Tarde/Noche
    2. Si en Mañana sale X y Tarde sale Y → qué suma suele salir en Noche
    
    Retorna también el conteo total de veces que aparece cada suma en cada sesión.
    """
    # Preparar datos
    df['Suma'] = df[COL_FIJO].apply(calcular_suma_digitos)
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    
    # Agrupar por fecha y sesión
    patrones_manana_tarde = defaultdict(lambda: defaultdict(int))
    patrones_manana_noche = defaultdict(lambda: defaultdict(int))
    patrones_tarde_noche = defaultdict(lambda: defaultdict(int))
    patrones_manana_tarde_a_noche = defaultdict(lambda: defaultdict(int))
    
    # Conteo total de veces que aparece cada suma en cada sesión
    conteo_manana = defaultdict(int)
    conteo_tarde = defaultdict(int)
    conteo_noche = defaultdict(int)
    
    # Agrupar por fecha
    for fecha, grupo in df.groupby('Fecha_Parsed'):
        if pd.isna(fecha):
            continue
        
        # Obtener sumas por sesión (normalizar nombre de sesión)
        sumas_sesion = {}
        for _, row in grupo.iterrows():
            sesion_raw = str(row[COL_SESION]).strip()
            suma = row['Suma']
            if pd.notna(suma):
                # Normalizar nombre de sesión
                sesion_lower = sesion_raw.lower()
                if 'mañana' in sesion_lower or 'manana' in sesion_lower or sesion_lower == 'm':
                    sumas_sesion['mañana'] = int(suma)
                    conteo_manana[int(suma)] += 1
                elif 'tarde' in sesion_lower or sesion_lower == 't':
                    sumas_sesion['tarde'] = int(suma)
                    conteo_tarde[int(suma)] += 1
                elif 'noche' in sesion_lower or sesion_lower == 'n':
                    sumas_sesion['noche'] = int(suma)
                    conteo_noche[int(suma)] += 1
                else:
                    # Si no coincide, usar el nombre original en minúsculas
                    sumas_sesion[sesion_lower] = int(suma)
        
        # Analizar patrones
        # Mañana → Tarde
        if 'mañana' in sumas_sesion and 'tarde' in sumas_sesion:
            key = sumas_sesion['mañana']
            patrones_manana_tarde[key][sumas_sesion['tarde']] += 1
        
        # Mañana → Noche
        if 'mañana' in sumas_sesion and 'noche' in sumas_sesion:
            key = sumas_sesion['mañana']
            patrones_manana_noche[key][sumas_sesion['noche']] += 1
        
        # Tarde → Noche
        if 'tarde' in sumas_sesion and 'noche' in sumas_sesion:
            key = sumas_sesion['tarde']
            patrones_tarde_noche[key][sumas_sesion['noche']] += 1
        
        # Mañana + Tarde → Noche
        if 'mañana' in sumas_sesion and 'tarde' in sumas_sesion and 'noche' in sumas_sesion:
            key = (sumas_sesion['mañana'], sumas_sesion['tarde'])
            patrones_manana_tarde_a_noche[key][sumas_sesion['noche']] += 1
    
    # Convertir defaultdicts anidados a diccionarios normales
    def convertir_dict(d):
        return {k: dict(v) for k, v in d.items()}
    
    # Calcular totales de cada patrón
    totales_manana_tarde = {k: sum(v.values()) for k, v in patrones_manana_tarde.items()}
    totales_manana_noche = {k: sum(v.values()) for k, v in patrones_manana_noche.items()}
    totales_tarde_noche = {k: sum(v.values()) for k, v in patrones_tarde_noche.items()}
    totales_manana_tarde_a_noche = {k: sum(v.values()) for k, v in patrones_manana_tarde_a_noche.items()}
    
    return {
        'manana_tarde': convertir_dict(patrones_manana_tarde),
        'manana_noche': convertir_dict(patrones_manana_noche),
        'tarde_noche': convertir_dict(patrones_tarde_noche),
        'manana_tarde_a_noche': convertir_dict(patrones_manana_tarde_a_noche),
        # Totales de cada patrón
        'totales_manana_tarde': totales_manana_tarde,
        'totales_manana_noche': totales_manana_noche,
        'totales_tarde_noche': totales_tarde_noche,
        'totales_manana_tarde_a_noche': totales_manana_tarde_a_noche,
        # Conteo por sesión
        'conteo_manana': dict(conteo_manana),
        'conteo_tarde': dict(conteo_tarde),
        'conteo_noche': dict(conteo_noche)
    }

def obtener_historial_sumas(df, cantidad=50):
    """
    Obtiene historial de sumas ordenado por día, de más reciente a menos reciente.
    
    Dentro de cada día: Noche → Tarde → Mañana
    Los días se ordenan de más reciente a menos reciente.
    """
    # Crear columna de orden para sesión (Noche=1, Tarde=2, Mañana=3)
    sesion_orden = {'Noche': 1, 'Tarde': 2, 'Mañana': 3}
    df['Sesion_Orden'] = df[COL_SESION].map(sesion_orden)
    
    # Manejar sesiones que no estén en el orden definido
    df['Sesion_Orden'] = df['Sesion_Orden'].fillna(999)
    
    # Ordenar: fecha descendente, luego sesión ascendente (Noche primero)
    df_historial = df.sort_values(
        ['Fecha_Parsed', 'Sesion_Orden'], 
        ascending=[False, True]
    ).head(cantidad)
    
    return df_historial

def analizar_almanaque(df, fecha_inicio, fecha_fin, cantidad_meses):
    """
    Analiza sumas en un rango de fechas para múltiples meses.
    
    IMPORTANTE: Empieza desde el mes ANTERIOR al actual.
    """
    hoy = datetime.now()
    
    # Calcular los meses a analizar (empezando desde el mes ANTERIOR)
    meses_analizar = []
    
    for i in range(1, cantidad_meses + 1):  # i empieza en 1, no en 0
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
    
    # Analizar cada mes
    resultados_meses = {}
    sumas_por_mes = {}
    fechas_por_suma_mes = {}  # Nuevo: guardar fechas de cada suma por mes
    
    for mes_info in meses_analizar:
        df_mes = df[
            (df['Fecha_Parsed'] >= mes_info['fecha_ini']) & 
            (df['Fecha_Parsed'] <= mes_info['fecha_fin'])
        ].copy()
        
        # Contar sumas del mes
        sumas_mes = df_mes['Suma'].value_counts().to_dict()
        sumas_por_mes[mes_info['nombre_mes']] = sumas_mes
        
        # Guardar fechas de cada suma
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
    
    # Encontrar sumas persistentes (aparecen en TODOS los meses)
    sumas_persistentes = set()
    if sumas_por_mes:
        conjuntos_sumas = [set(sumas.keys()) for sumas in sumas_por_mes.values()]
        sumas_persistentes = set.intersection(*conjuntos_sumas) if conjuntos_sumas else set()
    
    # Recopilar fechas de sumas persistentes
    fechas_persistentes = {}
    for suma in sumas_persistentes:
        fechas_total = []
        for mes in resultados_meses.keys():
            if suma in resultados_meses[mes]['fechas_sumas']:
                fechas_total.extend(resultados_meses[mes]['fechas_sumas'][suma])
        fechas_persistentes[suma] = fechas_total
    
    # Contar en cuántos meses aparece cada suma
    conteo_meses_suma = defaultdict(int)
    for mes, sumas in sumas_por_mes.items():
        for suma in sumas.keys():
            conteo_meses_suma[suma] += 1
    
    # Total de apariciones por suma
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
    """Analiza las transiciones entre sumas consecutivas"""
    transiciones = defaultdict(int)
    suma_anterior = None
    
    df = df.sort_values(['Fecha_Parsed', COL_SESION]).reset_index(drop=True)
    
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
    st.title("🔢 SumaDigitos - Análisis de Sumas del Fijo")
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
    st.info(f"📅 Fecha más reciente: {fecha_max.strftime('%d/%m/%Y')}")
    
    # Crear pestañas
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Resumen de Sumas",
        "🏆 Más Salidores",
        "📅 Almanaque",
        "🔮 Pronóstico del Día",
        "📜 Historial",
        "🧲 Atracción/Rechazo",
        "🔥 Temperatura y Estado"
    ])
    
    # ==================== PESTAÑA 1: RESUMEN ====================
    with tab1:
        st.header("📊 Resumen General de Sumas")
        st.markdown("""
        Esta tabla muestra el análisis de cada suma de dígitos del Fijo (00-99).
        La suma se calcula sumando ambos dígitos: ejemplo, 26 → 2+6 = 8.
        """)
        
        # Crear DataFrame de resumen
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
        
        # Mostrar tabla con formato condicional
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
        
        styled_df = df_resumen.style.applymap(highlight_estado, subset=['Estado']).applymap(highlight_temperatura, subset=['Temperatura'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Selector para ver detalle de una suma
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
        
        # Números que componen esta suma
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
            rango_dias = st.number_input("Rango de días (hacia atrás):", min_value=7, max_value=365, value=30) if usar_rango else None
        
        top_numeros, top_sumas = calcular_numeros_salidores(df_procesado, cantidad, rango_dias)
        
        st.subheader("📊 Sumas Más Salidores")
        if top_sumas:
            df_top_sumas = pd.DataFrame([
                {'Suma': s, 'Frecuencia': f} for s, f in sorted(top_sumas.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_top_sumas, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos para mostrar.")
        
        st.subheader("🔢 Números Más Salidores")
        if top_numeros:
            df_top_num = pd.DataFrame([
                {'Número': f"{n:02d}", 'Suma': calcular_suma_digitos(n), 'Frecuencia': f} 
                for n, f in top_numeros.items()
            ])
            st.dataframe(df_top_num, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos para mostrar.")
        
        st.info(f"ℹ️ Mostrando los {cantidad} elementos más frecuentes." + (f" Rango: últimos {rango_dias} días." if rango_dias else " Rango: todos los datos."))
    
    # ==================== PESTAÑA 3: ALMANAQUE ====================
    with tab3:
        st.header("📅 Almanaque - Análisis por Meses")
        
        st.markdown("""
        Analiza las sumas en un rango de días específico a lo largo de varios meses.
        
        **IMPORTANTE:** El análisis empieza desde el **mes anterior** al actual.
        
        **Ejemplo:** Si estamos en Marzo 2026 y seleccionas día inicial 1, día final 15, y 3 meses:
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
            
            # Resultados por mes
            st.subheader("📋 Resultados por Mes")
            
            for mes, datos in almanaque['resultados_por_mes'].items():
                with st.expander(f"📆 {mes} - {datos['total_sorteos']} sorteos"):
                    if datos['sumas']:
                        df_mes = pd.DataFrame([
                            {'Suma': s, 'Frecuencia': f} 
                            for s, f in sorted(datos['sumas'].items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(df_mes, use_container_width=True, hide_index=True)
                        st.write(f"**Suma más frecuente:** {datos['suma_mas_frecuente'][0]} ({datos['suma_mas_frecuente'][1]} veces)")
                    else:
                        st.write("Sin datos para este período.")
            
            # Sumas persistentes con fechas
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
                        key="orden_persistentes"
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
            
            # Todas las sumas con su presencia
            st.subheader("📊 Presencia de Sumas por Mes")
            
            df_presencia = pd.DataFrame([
                {'Suma': s, 'Apariciones Total': almanaque['total_sumas'].get(s, 0), 
                 'Meses Presente': almanaque['conteo_meses_suma'].get(s, 0)}
                for s in range(19)
            ])
            df_presencia = df_presencia.sort_values('Apariciones Total', ascending=False)
            st.dataframe(df_presencia, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.caption("""
        ℹ️ **Ayuda del Almanaque:**
        - Selecciona el rango de días (ej: 1 al 15) y la cantidad de meses a analizar.
        - El sistema buscará ese mismo rango de días en los meses ANTERIORES (no incluye el mes actual).
        - Las **sumas persistentes** son las que aparecen al menos una vez en cada mes.
        - Se muestran las fechas exactas de cada salida de las sumas persistentes.
        """)
    
    # ==================== PESTAÑA 4: PRONÓSTICO DEL DÍA ====================
    with tab4:
        st.header("🔮 Pronóstico del Día")
        st.markdown("""
        Análisis de correlaciones entre sumas del mismo día:
        - **Mañana → Tarde/Noche**: Si sale suma X en la mañana, qué suele salir después
        - **Mañana + Tarde → Noche**: Patrón combinado para predecir la noche
        """)
        
        # Leyenda de fiabilidad
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
        
        # Función para determinar fiabilidad
        def obtener_fiabilidad(cantidad):
            if cantidad >= 50:
                return "✅ MUY FIABLE", "green"
            elif cantidad >= 20:
                return "🟢 FIABLE", "green"
            elif cantidad >= 10:
                return "🟡 MODERADA", "orange"
            else:
                return "🔴 POCO FIABLE", "red"
        
        # Seleccionar suma de la mañana
        st.subheader("🌅 Si en la Mañana sale...")
        suma_manana = st.selectbox(
            "Selecciona la suma que salió en la mañana:",
            options=list(range(19)),
            key="suma_manana"
        )
        
        # Mostrar conteo total de veces que apareció esa suma en la mañana
        total_manana = pronostico['conteo_manana'].get(suma_manana, 0)
        st.info(f"📊 **Suma {suma_manana} apareció {total_manana} veces en la mañana**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌤️ Probable en la Tarde")
            tarde_probs = pronostico['manana_tarde'].get(suma_manana, {})
            total_patron_tarde = pronostico['totales_manana_tarde'].get(suma_manana, 0)
            
            if tarde_probs:
                # Mostrar fiabilidad
                fiabilidad, color = obtener_fiabilidad(total_patron_tarde)
                if color == "green":
                    st.success(f"**Fiabilidad: {fiabilidad}** | {total_patron_tarde} sorteos encontrados")
                elif color == "orange":
                    st.warning(f"**Fiabilidad: {fiabilidad}** | {total_patron_tarde} sorteos encontrados")
                else:
                    st.error(f"**Fiabilidad: {fiabilidad}** | {total_patron_tarde} sorteos encontrados")
                
                df_tarde = pd.DataFrame([
                    {'Suma': s, 'Frecuencia': f, 'Porcentaje': f"{f/total_patron_tarde*100:.1f}%"} 
                    for s, f in sorted(tarde_probs.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(df_tarde, use_container_width=True, hide_index=True)
            else:
                st.warning("No hay datos suficientes para este patrón.")
        
        with col2:
            st.markdown("### 🌙 Probable en la Noche")
            noche_probs = pronostico['manana_noche'].get(suma_manana, {})
            total_patron_noche = pronostico['totales_manana_noche'].get(suma_manana, 0)
            
            if noche_probs:
                # Mostrar fiabilidad
                fiabilidad, color = obtener_fiabilidad(total_patron_noche)
                if color == "green":
                    st.success(f"**Fiabilidad: {fiabilidad}** | {total_patron_noche} sorteos encontrados")
                elif color == "orange":
                    st.warning(f"**Fiabilidad: {fiabilidad}** | {total_patron_noche} sorteos encontrados")
                else:
                    st.error(f"**Fiabilidad: {fiabilidad}** | {total_patron_noche} sorteos encontrados")
                
                df_noche = pd.DataFrame([
                    {'Suma': s, 'Frecuencia': f, 'Porcentaje': f"{f/total_patron_noche*100:.1f}%"} 
                    for s, f in sorted(noche_probs.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(df_noche, use_container_width=True, hide_index=True)
            else:
                st.warning("No hay datos suficientes para este patrón.")
        
        st.markdown("---")
        
        # Patrón combinado Mañana + Tarde → Noche
        st.subheader("🎯 Si en Mañana y Tarde salen...")
        col1, col2 = st.columns(2)
        
        with col1:
            suma_m = st.selectbox("Suma Mañana:", options=list(range(19)), key="suma_m_combo")
        with col2:
            suma_t = st.selectbox("Suma Tarde:", options=list(range(19)), key="suma_t_combo")
        
        clave = (suma_m, suma_t)
        noche_comb = pronostico['manana_tarde_a_noche'].get(clave, {})
        total_patron_comb = pronostico['totales_manana_tarde_a_noche'].get(clave, 0)
        
        if noche_comb:
            # Mostrar fiabilidad
            fiabilidad, color = obtener_fiabilidad(total_patron_comb)
            if color == "green":
                st.success(f"**Fiabilidad: {fiabilidad}** | {total_patron_comb} sorteos encontrados (Mañana={suma_m} + Tarde={suma_t})")
            elif color == "orange":
                st.warning(f"**Fiabilidad: {fiabilidad}** | {total_patron_comb} sorteos encontrados (Mañana={suma_m} + Tarde={suma_t})")
            else:
                st.error(f"**Fiabilidad: {fiabilidad}** | {total_patron_comb} sorteos encontrados (Mañana={suma_m} + Tarde={suma_t})")
            
            st.markdown("### 🌙 Probable en la Noche")
            df_comb = pd.DataFrame([
                {'Suma': s, 'Frecuencia': f, 'Porcentaje': f"{f/total_patron_comb*100:.1f}%"} 
                for s, f in sorted(noche_comb.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_comb, use_container_width=True, hide_index=True)
        else:
            st.error(f"**Fiabilidad: 🔴 SIN DATOS** | No hay datos históricos para Mañana={suma_m} + Tarde={suma_t}")
        
        st.markdown("---")
        st.caption("""
        ℹ️ **Ayuda del Pronóstico:**
        - Este análisis busca patrones históricos en los sorteos del mismo día.
        - **Cantidad de sorteos**: Indica cuántas veces se encontró ese patrón. Mayor cantidad = mayor fiabilidad.
        - **Porcentaje**: Indica qué tan frecuente es cada resultado dentro del patrón.
        - **Recomendación**: Patrones con menos de 10 ocurrencias pueden no ser fiables.
        - Los patrones combinados (Mañana+Tarde→Noche) son más específicos pero tienen menos ocurrencias.
        """)
    
    # ==================== PESTAÑA 5: HISTORIAL ====================
    with tab5:
        st.header("📜 Historial de Sumas")
        st.markdown("""
        Lista de sumas ordenadas por día, de más reciente a menos reciente.
        
        **Orden dentro de cada día:** Noche → Tarde → Mañana
        """)
        
        cantidad_historial = st.number_input("Cantidad de registros:", min_value=10, max_value=200, value=50)
        
        df_historial = obtener_historial_sumas(df_procesado, cantidad_historial)
        
        # Preparar tabla
        df_mostrar = df_historial[['Fecha_Parsed', COL_SESION, COL_FIJO, 'Suma']].copy()
        df_mostrar['Fecha'] = df_mostrar['Fecha_Parsed'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '-')
        df_mostrar['Fijo'] = df_mostrar[COL_FIJO].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else '-')
        df_mostrar['Sesión'] = df_mostrar[COL_SESION]
        df_mostrar['Suma'] = df_mostrar['Suma'].apply(lambda x: int(x) if pd.notna(x) else '-')
        
        df_final = df_mostrar[['Fecha', 'Sesión', 'Fijo', 'Suma']].reset_index(drop=True)
        df_final.index = df_final.index + 1
        
        st.dataframe(df_final, use_container_width=True)
        
        st.subheader("📈 Estadísticas del Historial")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Registros mostrados", len(df_final))
        with col2:
            sumas_unicas = df_mostrar['Suma'].nunique()
            st.metric("Sumas diferentes", sumas_unicas)
        with col3:
            suma_mas_frec = df_mostrar['Suma'].value_counts().idxmax() if len(df_mostrar) > 0 else '-'
            st.metric("Suma más frecuente", suma_mas_frec)
        
        st.info("ℹ️ Orden: Por fecha descendente. Dentro de cada día: Noche → Tarde → Mañana (del último sorteo al primero del día)")
    
    # ==================== PESTAÑA 6: ATRACCIÓN/RECHAZO ====================
    with tab6:
        st.header("🧲 Análisis de Sumas que se Atraen y Rechazan")
        st.markdown("""
        - **Atracción**: Sumas que tienden a aparecer consecutivamente más de lo esperado.
        - **Rechazo**: Sumas que raramente o nunca aparecen consecutivamente.
        """)
        
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
    
    # ==================== PESTAÑA 7: TEMPERATURA Y ESTADO ====================
    with tab7:
        st.header("🔥 Indicador de Temperatura y Estado")
        
        st.markdown("""
        **Criterios de Temperatura:**
        - 🔥 **CALIENTE**: Apareció recientemente (≤25% del promedio)
        - 🌡️ **TIBIO**: Normal (entre 25% y 100% del promedio)
        - ❄️ **FRIO**: Más tiempo del promedio (100% - 150% del promedio)
        - 🧊 **MUY FRIO**: Mucho tiempo sin aparecer (>150% del promedio)
        
        **Criterios de Estado:**
        - ✅ **NORMAL**: Días sin aparecer ≤ promedio
        - ⚠️ **VENCIDA**: Días sin aparecer > promedio
        - 🚨 **MUY VENCIDA**: Días sin aparecer > 1.5 × promedio
        """)
        
        st.subheader("📊 Resumen de Estados y Temperaturas")
        
        resumen_estado_temp = []
        for suma in range(19):
            r = resultados_sumas[suma]
            resumen_estado_temp.append({
                'Suma': suma,
                'Estado': r['estado'],
                'Temperatura': r['temperatura'],
                'Días Sin Aparecer': r['dias_sin_aparecer'],
                'Promedio': r['promedio_dias']
            })
        
        df_estado_temp = pd.DataFrame(resumen_estado_temp)
        styled_et = df_estado_temp.style.applymap(highlight_estado, subset=['Estado']).applymap(highlight_temperatura, subset=['Temperatura'])
        st.dataframe(styled_et, use_container_width=True, hide_index=True)
        
        st.subheader("⚠️ Alertas")
        
        alertas = []
        for suma in range(19):
            r = resultados_sumas[suma]
            if 'MUY VENCIDA' in r['estado']:
                alertas.append({
                    'Tipo': '🚨 CRÍTICA',
                    'Suma': suma,
                    'Mensaje': f"La Suma {suma} lleva {r['dias_sin_aparecer']} días sin aparecer. Promedio: {r['promedio_dias']} días."
                })
            elif 'VENCIDA' in r['estado'] and r['dias_sin_aparecer'] > r['promedio_dias'] * 1.3:
                alertas.append({
                    'Tipo': '⚠️ ATENCIÓN',
                    'Suma': suma,
                    'Mensaje': f"La Suma {suma} está vencida ({r['dias_sin_aparecer']} días vs promedio {r['promedio_dias']})."
                })
        
        if alertas:
            for alerta in alertas:
                st.warning(f"{alerta['Tipo']} - Suma {alerta['Suma']}: {alerta['Mensaje']}")
        else:
            st.success("✅ No hay alertas en este momento.")

if __name__ == "__main__":
    main()