# -*- coding: utf-8 -*-
"""
PaleFlo - Análisis de Pales por Grupos
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
        page_title="PaleFlo - Análisis de Pales",
        page_icon="🎯",
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

# Mapeo de sesiones
MAPEO_SESIONES = {
    't': 'Tarde',
    'tarde': 'Tarde',
    'n': 'Noche',
    'noche': 'Noche'
}

# Definición de grupos para Pales
GRUPOS = {
    'CERRADOS': {'digitos': {0, 6, 8, 9}, 'numeros': []},
    'ABIERTOS': {'digitos': {2, 3, 5}, 'numeros': []},
    'RECTOS': {'digitos': {1, 4, 7}, 'numeros': []}
}

# Generar números de cada grupo (00-99 donde ambos dígitos pertenecen al mismo grupo)
for i in range(100):
    num_str = f"{i:02d}"
    d1, d2 = int(num_str[0]), int(num_str[1])
    for grupo, datos in GRUPOS.items():
        if d1 in datos['digitos'] and d2 in datos['digitos']:
            datos['numeros'].append(i)

# Información de grupos
INFO_GRUPOS = {
    'CERRADOS': {'digitos': '0, 6, 8, 9', 'cantidad': len(GRUPOS['CERRADOS']['numeros'])},
    'ABIERTOS': {'digitos': '2, 3, 5', 'cantidad': len(GRUPOS['ABIERTOS']['numeros'])},
    'RECTOS': {'digitos': '1, 4, 7', 'cantidad': len(GRUPOS['RECTOS']['numeros'])}
}

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
        '%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y'
    ]
    
    for fmt in formatos:
        try:
            return datetime.strptime(fecha_str, fmt)
        except ValueError:
            continue
    
    return None

def normalizar_sesion(sesion_raw):
    """Normaliza el nombre de la sesión"""
    if pd.isna(sesion_raw):
        return None
    sesion_lower = str(sesion_raw).strip().lower()
    return MAPEO_SESIONES.get(sesion_lower, sesion_raw)

def clasificar_numero(numero):
    """Clasifica un número en su grupo, si ambos dígitos pertenecen al mismo grupo"""
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
    """Calcula el estado"""
    if promedio == 0:
        return "N/A"
    
    if dias_sin_aparecer <= promedio:
        return "✅ NORMAL"
    elif dias_sin_aparecer <= promedio * 1.5:
        return "⚠️ VENCIDA"
    else:
        return "🚨 MUY VENCIDA"

def calcular_temperatura(dias_sin_aparecer, promedio):
    """Calcula la temperatura"""
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
    """Detecta si hay pale en una sesión (2+ números del mismo grupo)"""
    grupos_en_sesion = defaultdict(list)
    
    for col in [COL_FIJO, COL_CORR1, COL_CORR2]:
        if col in fila.index and pd.notna(fila[col]):
            numero = int(fila[col]) if str(fila[col]).isdigit() else None
            if numero is not None:
                grupo = clasificar_numero(numero)
                if grupo:
                    grupos_en_sesion[grupo].append(numero)
    
    # Retornar grupos con pale (2+ números)
    pales = {g: nums for g, nums in grupos_en_sesion.items() if len(nums) >= 2}
    return pales

def analizar_pales(df):
    """Analiza los pales por grupo"""
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df['Sesion_Normalizada'] = df[COL_SESION].apply(normalizar_sesion)
    df = df.sort_values('Fecha_Parsed').reset_index(drop=True)
    
    fecha_max = df['Fecha_Parsed'].max()
    
    resultados = {}
    
    for grupo in GRUPOS.keys():
        # Encontrar sesiones con pale de este grupo
        fechas_pale = []
        
        for idx, row in df.iterrows():
            pales = detectar_pales_sesion(row)
            if grupo in pales:
                fechas_pale.append({
                    'fecha': row['Fecha_Parsed'],
                    'sesion': row['Sesion_Normalizada'],
                    'numeros': pales[grupo]
                })
        
        # Calcular estadísticas
        if fechas_pale:
            fechas = [f['fecha'] for f in fechas_pale if pd.notna(f['fecha'])]
            fechas.sort()
            
            gaps = []
            for i in range(1, len(fechas)):
                gap = (fechas[i] - fechas[i-1]).days
                if gap > 0:
                    gaps.append(gap)
            
            frecuencia = len(fechas)
            promedio_dias = np.mean(gaps) if gaps else 0
            ausencia_maxima = max(gaps) if gaps else 0
            ultima_fecha = fechas[-1] if fechas else None
            dias_sin_aparecer = (fecha_max - ultima_fecha).days if ultima_fecha else 0
        else:
            frecuencia = 0
            promedio_dias = 0
            ausencia_maxima = 0
            ultima_fecha = None
            dias_sin_aparecer = 0
            fechas = []
            gaps = []
        
        resultados[grupo] = {
            'frecuencia': frecuencia,
            'promedio_dias': round(promedio_dias, 1),
            'ausencia_maxima': ausencia_maxima,
            'dias_sin_aparecer': dias_sin_aparecer,
            'ultima_fecha': ultima_fecha,
            'estado': calcular_estado(dias_sin_aparecer, promedio_dias),
            'temperatura': calcular_temperatura(dias_sin_aparecer, promedio_dias),
            'numeros_grupo': GRUPOS[grupo]['numeros'],
            'fechas_pale': fechas_pale,
            'gaps': gaps
        }
    
    return resultados, df, fecha_max

def analizar_combinaciones(df):
    """Analiza combinaciones dentro de cada grupo"""
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    fecha_max = df['Fecha_Parsed'].max()
    
    combinaciones_stats = {}
    
    for grupo in GRUPOS.keys():
        numeros = GRUPOS[grupo]['numeros']
        combinaciones = {}
        
        for idx, row in df.iterrows():
            pales = detectar_pales_sesion(row)
            if grupo in pales and len(pales[grupo]) >= 2:
                # Crear combinación ordenada
                nums = sorted(pales[grupo])
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        comb = (nums[i], nums[j])
                        if comb not in combinaciones:
                            combinaciones[comb] = []
                        combinaciones[comb].append({
                            'fecha': row['Fecha_Parsed'],
                            'sesion': row['Sesion_Normalizada'] if COL_SESION in row.index else None
                        })
        
        # Calcular estadísticas por combinación
        combinaciones_resultado = []
        for comb, apariciones in combinaciones.items():
            fechas = [a['fecha'] for a in apariciones if pd.notna(a['fecha'])]
            fechas.sort()
            
            gaps = []
            for i in range(1, len(fechas)):
                gap = (fechas[i] - fechas[i-1]).days
                if gap > 0:
                    gaps.append(gap)
            
            frecuencia = len(fechas)
            promedio = np.mean(gaps) if gaps else 0
            ultima = fechas[-1] if fechas else None
            dias_sin = (fecha_max - ultima).days if ultima else 0
            
            combinaciones_resultado.append({
                'combinacion': comb,
                'frecuencia': frecuencia,
                'promedio_dias': round(promedio, 1),
                'ausencia_maxima': max(gaps) if gaps else 0,
                'dias_sin_aparecer': dias_sin,
                'ultima_fecha': ultima,
                'estado': calcular_estado(dias_sin, promedio),
                'fechas': fechas
            })
        
        # Ordenar por frecuencia
        combinaciones_resultado.sort(key=lambda x: x['frecuencia'], reverse=True)
        combinaciones_stats[grupo] = combinaciones_resultado
    
    return combinaciones_stats

def analizar_pales_entre_grupos(df):
    """Analiza cuando aparecen números de diferentes grupos en la misma sesión"""
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    
    coexistencias = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        grupos_en_sesion = set()
        
        for col in [COL_FIJO, COL_CORR1, COL_CORR2]:
            if col in row.index and pd.notna(row[col]):
                try:
                    numero = int(row[col])
                    grupo = clasificar_numero(numero)
                    if grupo:
                        grupos_en_sesion.add(grupo)
                except:
                    pass
        
        # Registrar combinaciones de grupos
        grupos_lista = sorted(list(grupos_en_sesion))
        for i in range(len(grupos_lista)):
            for j in range(i+1, len(grupos_lista)):
                par = (grupos_lista[i], grupos_lista[j])
                coexistencias[par]['total'] += 1
    
    return dict(coexistencias)

def analizar_almanaque_pales(df, dia_inicio, dia_fin, cantidad_meses):
    """Analiza pales por meses"""
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
            dia_fin_ajustado = min(dia_fin, dias_en_mes)
            fecha_fin_mes = datetime(año_target, mes_target, dia_fin_ajustado)
            
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
    pales_por_mes = {}
    
    for mes_info in meses_analizar:
        df_mes = df[
            (df['Fecha_Parsed'] >= mes_info['fecha_ini']) & 
            (df['Fecha_Parsed'] <= mes_info['fecha_fin'])
        ].copy()
        
        # Contar pales por grupo en este mes
        pales_mes = {'CERRADOS': 0, 'ABIERTOS': 0, 'RECTOS': 0}
        fechas_pales_mes = {'CERRADOS': [], 'ABIERTOS': [], 'RECTOS': []}
        
        for idx, row in df_mes.iterrows():
            pales = detectar_pales_sesion(row)
            for grupo in pales.keys():
                if grupo in pales_mes:
                    pales_mes[grupo] += 1
                    fechas_pales_mes[grupo].append(row['Fecha_Parsed'].strftime('%d/%m/%Y') if pd.notna(row['Fecha_Parsed']) else '')
        
        pales_por_mes[mes_info['nombre_mes']] = pales_mes
        resultados_meses[mes_info['nombre_mes']] = {
            'total_sorteos': len(df_mes),
            'pales': pales_mes,
            'fechas_pales': fechas_pales_mes
        }
    
    # Encontrar grupos persistentes
    grupos_persistentes = set()
    if pales_por_mes:
        for grupo in GRUPOS.keys():
            aparece_en_todos = all(
                pales_por_mes[mes].get(grupo, 0) > 0 
                for mes in pales_por_mes.keys()
            )
            if aparece_en_todos:
                grupos_persistentes.add(grupo)
    
    return {
        'meses_analizados': meses_analizar,
        'resultados_por_mes': resultados_meses,
        'pales_por_mes': pales_por_mes,
        'grupos_persistentes': grupos_persistentes,
        'cantidad_meses': cantidad_meses
    }

def obtener_historial_pales(df, cantidad=50):
    """Obtiene historial de pales ordenado por fecha"""
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df['Sesion_Normalizada'] = df[COL_SESION].apply(normalizar_sesion)
    
    sesion_orden = {'Noche': 1, 'N': 1, 'Tarde': 2, 'T': 2}
    df['Sesion_Orden'] = df['Sesion_Normalizada'].map(sesion_orden).fillna(999)
    
    # Filtrar solo sesiones con pales
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
    df_pales = df_pales.sort_values(['Fecha_Parsed', 'Sesion_Orden'], ascending=[False, True]).head(cantidad)
    
    return df_pales

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

def main():
    st.title("🎯 PaleFlo - Análisis de Pales por Grupos")
    st.markdown("**Hoja: Sorteos** | Fuente: Google Sheets")
    st.markdown("---")
    
    # Mostrar info de grupos
    st.sidebar.header("📋 Información de Grupos")
    for grupo, info in INFO_GRUPOS.items():
        st.sidebar.markdown(f"**{grupo}**")
        st.sidebar.markdown(f"Dígitos: {info['digitos']}")
        st.sidebar.markdown(f"Números: {info['cantidad']}")
        st.sidebar.markdown("---")
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        gc = conectar_google_sheets()
        df = cargar_datos(gc, GS_ID, GS_SHEET)
    
    if df is None or len(df) == 0:
        st.error("No se pudieron cargar los datos.")
        st.stop()
    
    st.success(f"✅ Datos cargados: {len(df)} registros")
    
    # Procesar datos
    resultados_pales, df_procesado, fecha_max = analizar_pales(df)
    st.info(f"📅 Fecha más reciente: {fecha_max.strftime('%d/%m/%Y') if pd.notna(fecha_max) else 'N/A'}")
    
    # Crear pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Resumen por Grupo",
        "🎯 Combinaciones",
        "📅 Almanaque",
        "📜 Historial de Pales",
        "🔗 Grupos Coexistentes"
    ])
    
    # ==================== PESTAÑA 1: RESUMEN ====================
    with tab1:
        st.header("📊 Resumen de Pales por Grupo")
        
        # Tabla resumen
        resumen_data = []
        for grupo in ['CERRADOS', 'ABIERTOS', 'RECTOS']:
            r = resultados_pales[grupo]
            resumen_data.append({
                'Grupo': grupo,
                'Dígitos': INFO_GRUPOS[grupo]['digitos'],
                'Números': INFO_GRUPOS[grupo]['cantidad'],
                'Frecuencia': r['frecuencia'],
                'Promedio Días': r['promedio_dias'],
                'Ausencia Máx': r['ausencia_maxima'],
                'Días Sin Aparecer': r['dias_sin_aparecer'],
                'Última Fecha': r['ultima_fecha'].strftime('%d/%m/%Y') if r['ultima_fecha'] else '-',
                'Estado': r['estado'],
                'Temperatura': r['temperatura']
            })
        
        df_resumen = pd.DataFrame(resumen_data)
        
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
        
        # Detalle por grupo
        st.markdown("---")
        st.subheader("🔍 Detalle por Grupo")
        
        grupo_sel = st.selectbox("Selecciona un grupo:", options=['CERRADOS', 'ABIERTOS', 'RECTOS'])
        
        r = resultados_pales[grupo_sel]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frecuencia", r['frecuencia'])
            st.metric("Promedio de Días", r['promedio_dias'])
        with col2:
            st.metric("Ausencia Máxima", f"{r['ausencia_maxima']} días")
            st.metric("Días Sin Aparecer", r['dias_sin_aparecer'])
        with col3:
            st.metric("Estado", r['estado'])
            st.metric("Temperatura", r['temperatura'])
        
        # Números del grupo
        st.subheader(f"📋 Números del grupo {grupo_sel}")
        nums_str = ", ".join([f"{n:02d}" for n in r['numeros_grupo']])
        st.write(f"**{len(r['numeros_grupo'])} números:** {nums_str}")
        
        # Últimos pales del grupo
        if r['fechas_pale']:
            st.subheader(f"📅 Últimos Pales de {grupo_sel}")
            for fp in r['fechas_pale'][-10:][::-1]:
                fecha_str = fp['fecha'].strftime('%d/%m/%Y') if pd.notna(fp['fecha']) else '-'
                nums_str = ", ".join([f"{n:02d}" for n in fp['numeros']])
                st.write(f"• {fecha_str} ({fp['sesion']}): {nums_str}")
    
    # ==================== PESTAÑA 2: COMBINACIONES ====================
    with tab2:
        st.header("🎯 Análisis de Combinaciones por Grupo")
        st.markdown("Combinaciones de números dentro de cada grupo que forman pales.")
        
        combinaciones = analizar_combinaciones(df_procesado)
        
        grupo_comb = st.selectbox("Selecciona grupo:", options=['CERRADOS', 'ABIERTOS', 'RECTOS'], key="grupo_comb")
        
        combs = combinaciones[grupo_comb]
        
        if combs:
            st.subheader(f"Combinaciones en {grupo_comb}")
            
            # Crear DataFrame
            df_combs = pd.DataFrame([
                {
                    'Combinación': f"{c['combinacion'][0]:02d} - {c['combinacion'][1]:02d}",
                    'Frecuencia': c['frecuencia'],
                    'Promedio Días': c['promedio_dias'],
                    'Ausencia Máx': c['ausencia_maxima'],
                    'Días Sin Aparecer': c['dias_sin_aparecer'],
                    'Estado': c['estado']
                }
                for c in combs
            ])
            
            # Selector de ordenamiento
            orden = st.radio("Ordenar por:", ["Frecuencia (mayor)", "Frecuencia (menor)", "Días sin aparecer (mayor)"], horizontal=True)
            
            if orden == "Frecuencia (mayor)":
                df_combs = df_combs.sort_values('Frecuencia', ascending=False)
            elif orden == "Frecuencia (menor)":
                df_combs = df_combs.sort_values('Frecuencia', ascending=True)
            else:
                df_combs = df_combs.sort_values('Días Sin Aparecer', ascending=False)
            
            st.dataframe(df_combs.style.applymap(highlight_estado, subset=['Estado']), use_container_width=True, hide_index=True)
        else:
            st.info(f"No hay combinaciones registradas para {grupo_comb}")
    
    # ==================== PESTAÑA 3: ALMANAQUE ====================
    with tab3:
        st.header("📅 Almanaque de Pales por Meses")
        
        st.markdown("""
        **IMPORTANTE:** El análisis empieza desde el **mes anterior** al actual.
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
                almanaque = analizar_almanaque_pales(df_procesado, dia_inicio, dia_fin, cantidad_meses)
            
            st.success(f"📊 Analizando {almanaque['cantidad_meses']} meses")
            
            fechas_info = " | ".join([f"{m['nombre_mes']}: {m['fecha_ini'].strftime('%d/%m')} al {m['fecha_fin'].strftime('%d/%m')}" 
                                     for m in almanaque['meses_analizados']])
            st.info(f"📅 Fechas analizadas: {fechas_info}")
            
            # Resultados por mes
            st.subheader("📋 Pales por Mes")
            
            df_meses = pd.DataFrame([
                {'Mes': mes, 'CERRADOS': datos['pales']['CERRADOS'], 
                 'ABIERTOS': datos['pales']['ABIERTOS'], 'RECTOS': datos['pales']['RECTOS'],
                 'Total Sorteos': datos['total_sorteos']}
                for mes, datos in almanaque['resultados_por_mes'].items()
            ])
            st.dataframe(df_meses, use_container_width=True, hide_index=True)
            
            # Grupos persistentes
            st.subheader("🔄 Grupos Persistentes")
            st.markdown("*Grupos con al menos 1 pale en CADA mes analizado*")
            
            if almanaque['grupos_persistentes']:
                st.success(f"✅ {len(almanaque['grupos_persistentes'])} grupos persistentes: {', '.join(sorted(almanaque['grupos_persistentes']))}")
            else:
                st.warning("⚠️ No hay grupos que hayan tenido pales en todos los meses analizados.")
    
    # ==================== PESTAÑA 4: HISTORIAL ====================
    with tab4:
        st.header("📜 Historial de Pales")
        st.markdown("Sesiones donde aparecieron pales, ordenadas de más reciente a menos reciente.")
        
        cantidad_hist = st.number_input("Cantidad de registros:", min_value=10, max_value=200, value=50)
        
        df_hist = obtener_historial_pales(df_procesado, cantidad_hist)
        
        if not df_hist.empty:
            # Preparar tabla
            datos_mostrar = []
            for idx, row in df_hist.iterrows():
                fecha_str = row['Fecha_Parsed'].strftime('%d/%m/%Y') if pd.notna(row['Fecha_Parsed']) else '-'
                sesion = row.get('Sesion_Normalizada', row.get(COL_SESION, '-'))
                fijo = f"{int(row[COL_FIJO]):02d}" if pd.notna(row[COL_FIJO]) else '-'
                
                pales_str = " | ".join([f"{g}: {', '.join([f'{n:02d}' for n in nums])}" for g, nums in row['Pales'].items()])
                
                datos_mostrar.append({
                    'Fecha': fecha_str,
                    'Sesión': sesion,
                    'Fijo': fijo,
                    'Pales': pales_str
                })
            
            df_final = pd.DataFrame(datos_mostrar)
            df_final.index = df_final.index + 1
            st.dataframe(df_final, use_container_width=True)
        else:
            st.info("No hay pales registrados en los datos.")
    
    # ==================== PESTAÑA 5: COEXISTENCIA ====================
    with tab5:
        st.header("🔗 Coexistencia entre Grupos")
        st.markdown("Análisis de cuando números de diferentes grupos aparecen en la misma sesión.")
        
        coexistencias = analizar_pales_entre_grupos(df_procesado)
        
        if coexistencias:
            st.subheader("📊 Combinaciones entre Grupos")
            
            df_coex = pd.DataFrame([
                {'Combinación': f"{par[0]} + {par[1]}", 'Veces': datos['total']}
                for par, datos in sorted(coexistencias.items(), key=lambda x: x[1]['total'], reverse=True)
            ])
            st.dataframe(df_coex, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de coexistencia entre grupos.")

if __name__ == "__main__":
    main()