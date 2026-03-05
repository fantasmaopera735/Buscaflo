# -*- coding: utf-8 -*-
"""
SumaDigitos - Análisis de Sumas para Geotodo (Mañana, Tarde y Noche)
=====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- CONFIGURACIÓN GOOGLE SHEETS ---
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
SPREADSHEET_ID = "1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw"

@st.cache_resource
def get_gsheet_client():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)
        return gspread.authorize(creds)
    except:
        import os
        creds_path = 'credentials.json'
        if os.path.exists(creds_path):
            creds = Credentials.from_service_account_file(creds_path, scopes=SCOPE)
            return gspread.authorize(creds)
        raise Exception("No se encontraron credenciales")

def get_worksheet(sheet_name):
    client = get_gsheet_client()
    return client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)

# --- FUNCIONES DE SUMA ---
def suma_digitos(numero):
    """Calcula la suma de los dígitos de un número de 2 dígitos (00-99)"""
    try:
        n = int(float(numero))
        return (n // 10) + (n % 10)
    except:
        return 0

def obtener_numeros_que_suman(suma_objetivo):
    """Devuelve todos los números de 2 dígitos que suman el valor objetivo"""
    numeros = []
    for n in range(100):
        if suma_digitos(n) == suma_objetivo:
            numeros.append(f"{n:02d}")
    return numeros

# --- CARGA DE DATOS ---
@st.cache_data(ttl=60)
def cargar_datos():
    try:
        wks = get_worksheet("Geotodo")
        data = wks.get_all_records()
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        
        def convertir_fecha(valor):
            if pd.isna(valor) or str(valor).strip() == '':
                return pd.NaT
            valor_str = str(valor).strip()
            formatos = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y']
            for fmt in formatos:
                try:
                    return pd.to_datetime(valor_str, format=fmt)
                except:
                    continue
            try:
                return pd.to_datetime(valor_str, dayfirst=True)
            except:
                return pd.NaT
        
        df['Fecha'] = df['Fecha'].apply(convertir_fecha)
        df = df.dropna(subset=['Fecha'])
        
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].apply(lambda x: 
                'M' if x in ['M', 'MAÑANA', 'MANANA', 'MAÑANA/', 'MANANA/'] else
                'T' if x in ['T', 'TARDE', 'TARDE/'] else
                'N' if x in ['N', 'NOCHE', '/NOCHE', 'NOCHE/'] else
                'M' if 'MAÑANA' in x or 'MANANA' in x else
                'T' if 'TARDE' in x else 'N' if 'NOCHE' in x else 'OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        col_fijo = 'Fijo'
        col_corr1 = None
        col_corr2 = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if '1er' in col_lower or 'primer' in col_lower:
                col_corr1 = col
            elif '2do' in col_lower or 'segundo' in col_lower:
                col_corr2 = col
        
        if col_corr1 is None and len(df.columns) >= 5:
            col_corr1 = df.columns[4]
        if col_corr2 is None and len(df.columns) >= 6:
            col_corr2 = df.columns[5]
        
        df['Suma_Fijo'] = df[col_fijo].apply(suma_digitos)
        if col_corr1:
            df['Suma_Corr1'] = df[col_corr1].apply(suma_digitos)
        else:
            df['Suma_Corr1'] = 0
        if col_corr2:
            df['Suma_Corr2'] = df[col_corr2].apply(suma_digitos)
        else:
            df['Suma_Corr2'] = 0
        
        df['Suma_Total'] = df['Suma_Fijo'] + df['Suma_Corr1'] + df['Suma_Corr2']
        df['Suma_Corridos'] = df['Suma_Corr1'] + df['Suma_Corr2']
        
        df['Fijo_Num'] = df[col_fijo].apply(lambda x: f"{int(float(x)):02d}" if pd.notna(x) else "00")
        if col_corr1:
            df['Corr1_Num'] = df[col_corr1].apply(lambda x: f"{int(float(x)):02d}" if pd.notna(x) else "00")
        else:
            df['Corr1_Num'] = "00"
        if col_corr2:
            df['Corr2_Num'] = df[col_corr2].apply(lambda x: f"{int(float(x)):02d}" if pd.notna(x) else "00")
        else:
            df['Corr2_Num'] = "00"
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# --- FUNCIONES DE ESTADÍSTICAS ---
def calcular_estadisticas_suma(df, tipo_sesion=None):
    """Calcula estadísticas completas para cada tipo de suma."""
    if tipo_sesion and tipo_sesion != 'Todas':
        df = df[df['Tipo_Sorteo'] == tipo_sesion].copy()
    
    resultados = {}
    hoy = datetime.now().date()
    
    config_sumas = {
        'Suma_Fijo': range(19),      # 0-18
        'Suma_Total': range(55),     # 0-54
        'Suma_Corridos': range(37)   # 0-36
    }
    
    for tipo_suma, rango_sumas in config_sumas.items():
        stats = []
        
        for suma in rango_sumas:
            df_suma = df[df[tipo_suma] == suma].sort_values('Fecha')
            fechas = df_suma['Fecha'].dt.date.tolist()
            
            if len(fechas) == 0:
                stats.append({
                    'Suma': suma,
                    'Frecuencia': 0,
                    'Ausencia_Maxima': 999,
                    'Promedio_Dias': 0,
                    'Dias_Sin_Aparecer': 999,
                    'Ultima_Fecha': None
                })
            else:
                frecuencia = len(fechas)
                ultima_fecha = fechas[-1]
                dias_sin_aparecer = (hoy - ultima_fecha).days
                
                if len(fechas) >= 2:
                    gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
                    ausencia_maxima = max(gaps)
                    promedio_dias = round(np.mean(gaps), 1)
                else:
                    ausencia_maxima = dias_sin_aparecer
                    promedio_dias = dias_sin_aparecer
                
                stats.append({
                    'Suma': suma,
                    'Frecuencia': frecuencia,
                    'Ausencia_Maxima': ausencia_maxima,
                    'Promedio_Dias': promedio_dias,
                    'Dias_Sin_Aparecer': dias_sin_aparecer,
                    'Ultima_Fecha': ultima_fecha
                })
        
        resultados[tipo_suma] = pd.DataFrame(stats)
    
    return resultados

def generar_almanaque_sumas(df, tipo_suma, dia_inicio, dia_fin, meses_atras, tipo_sesion=None):
    """Genera almanaque de SUMAS para el período especificado."""
    if tipo_sesion and tipo_sesion != 'Todas':
        df = df[df['Tipo_Sorteo'] == tipo_sesion].copy()
    
    fecha_hoy = datetime.now()
    mes_actual = fecha_hoy.month
    anio_actual = fecha_hoy.year
    
    bloques_validos = []
    nombres_bloques = []
    
    # Obtener bloques de meses anteriores
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        if f_obj.month == mes_actual and f_obj.year == anio_actual:
            continue
        
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_i = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            if f_i > f_f:
                continue
            df_b = df[(df['Fecha'] >= f_i) & (df['Fecha'] <= f_f)]
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
        except:
            continue
    
    if not bloques_validos:
        # Fallback: usar mes completo
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_hoy - relativedelta(months=offset)
            if f_obj.month == mes_actual and f_obj.year == anio_actual:
                continue
            try:
                f_i = datetime(f_obj.year, f_obj.month, 1)
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_f = datetime(f_obj.year, f_obj.month, last_day)
                df_b = df[(df['Fecha'] >= f_i) & (df['Fecha'] <= f_f)]
                if not df_b.empty:
                    bloques_validos.append(df_b)
                    nombres_bloques.append(f"{f_obj.strftime('%b')} (Todo)")
            except:
                continue
    
    if not bloques_validos:
        return {'success': False, 'mensaje': 'Sin datos históricos'}
    
    df_total = pd.concat(bloques_validos)
    
    # Definir rango de sumas según tipo
    config_rangos = {
        'Suma_Fijo': range(19),      # 0-18
        'Suma_Total': range(55),     # 0-54
        'Suma_Corridos': range(37)   # 0-36
    }
    rango_sumas = config_rangos.get(tipo_suma, range(19))
    
    # Contar frecuencias de SUMAS (no números)
    conteo_sumas = df_total[tipo_suma].value_counts()
    
    # Crear ranking de sumas
    ranking = []
    for suma in rango_sumas:
        freq = conteo_sumas.get(suma, 0)
        ranking.append({
            'Suma': suma,
            'Frecuencia': freq
        })
    
    df_rank = pd.DataFrame(ranking).sort_values('Frecuencia', ascending=False)
    
    # Clasificar sumas por tercios
    def clasificar_sumas(df_rank):
        df_t = df_rank.copy()
        total = len(df_t)
        tercio = total // 3
        conds = [
            df_t.index < tercio,
            df_t.index < 2 * tercio
        ]
        vals = ['🔥 Caliente', '🟡 Tibio']
        df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
        return df_t
    
    df_rank = clasificar_sumas(df_rank)
    
    # Sumas persistentes (aparecen en TODOS los bloques)
    sumas_persistentes = []
    for suma in rango_sumas:
        aparece_en_todos = all(suma in bloque[tipo_suma].values for bloque in bloques_validos)
        if aparece_en_todos:
            sumas_persistentes.append(suma)
    
    # Sumas calientes
    sumas_calientes = df_rank[df_rank['Estado'] == '🔥 Caliente']['Suma'].tolist()
    
    # Historial del período actual
    hoy = datetime.now()
    estado_periodo = ""
    df_historial_actual = pd.DataFrame()
    
    try:
        fin_mes = calendar.monthrange(hoy.year, hoy.month)[1]
        fecha_ini = datetime(hoy.year, hoy.month, min(dia_inicio, fin_mes))
        fecha_fin = datetime(hoy.year, hoy.month, min(dia_fin, fin_mes))
        
        if hoy.date() < fecha_ini.date():
            estado_periodo = f"⚪ Período inicia el {fecha_ini.strftime('%d/%m')}"
        else:
            fecha_fin_real = min(hoy, fecha_fin)
            df_eval = df[(df['Fecha'] >= fecha_ini) & (df['Fecha'] <= fecha_fin_real)]
            
            if not df_eval.empty:
                historial = []
                for _, row in df_eval.iterrows():
                    suma_val = row[tipo_suma]
                    estado = df_rank[df_rank['Suma'] == suma_val]['Estado'].values
                    estado_val = estado[0] if len(estado) > 0 else '❓'
                    es_persistente = suma_val in sumas_persistentes
                    
                    historial.append({
                        'Fecha': row['Fecha'],
                        'Sesión': row['Tipo_Sorteo'],
                        'Suma': suma_val,
                        'Estado': estado_val,
                        'Persistente': '✅ SÍ' if es_persistente else '❌ NO',
                        'Fijo': row.get('Fijo_Num', '00'),
                        'Corr1': row.get('Corr1_Num', '00'),
                        'Corr2': row.get('Corr2_Num', '00')
                    })
                df_historial_actual = pd.DataFrame(historial)
                orden_sesion = {'N': 0, 'T': 1, 'M': 2}
                df_historial_actual['orden'] = df_historial_actual['Sesión'].map(orden_sesion).fillna(3)
                df_historial_actual = df_historial_actual.sort_values(['Fecha', 'orden'], ascending=[False, True])
                df_historial_actual = df_historial_actual.drop(columns=['orden'])
            
            estado_periodo = f"🟢 Período activo (hasta {fecha_fin_real.strftime('%d/%m')})"
    except Exception as e:
        estado_periodo = f"Error: {str(e)}"
    
    # Sumas faltantes
    df_faltantes = pd.DataFrame()
    if not df_historial_actual.empty and sumas_calientes:
        salidas = set(df_historial_actual['Suma'].unique())
        faltantes = [s for s in sumas_calientes if s not in salidas]
        if faltantes:
            df_faltantes = pd.DataFrame([{'Suma Faltante': s, 'Estado': '⏳ Pendiente'} for s in sorted(faltantes)])
    
    return {
        'success': True,
        'df_rank': df_rank,
        'sumas_persistentes': sumas_persistentes,
        'sumas_calientes': sumas_calientes,
        'nombres_bloques': nombres_bloques,
        'df_historial_actual': df_historial_actual,
        'df_faltantes': df_faltantes,
        'estado_periodo': estado_periodo
    }

def buscar_suma_detalle(df, tipo_suma, suma_buscar, tipo_sesion=None):
    """Busca una suma específica y muestra todas las veces que apareció."""
    if tipo_sesion and tipo_sesion != 'Todas':
        df = df[df['Tipo_Sorteo'] == tipo_sesion].copy()
    
    df_filtrado = df[df[tipo_suma] == suma_buscar].sort_values('Fecha', ascending=False)
    
    if df_filtrado.empty:
        return None
    
    resultados = []
    for _, row in df_filtrado.iterrows():
        resultados.append({
            'Fecha': row['Fecha'].strftime('%d/%m/%Y'),
            'Sesión': row['Tipo_Sorteo'],
            'Fijo': row['Fijo_Num'],
            'Suma F': row['Suma_Fijo'],
            'Corr1': row['Corr1_Num'],
            'Suma C1': row['Suma_Corr1'],
            'Corr2': row['Corr2_Num'],
            'Suma C2': row['Suma_Corr2'],
            'Suma Total': row['Suma_Total'],
            'Suma Corr': row['Suma_Corridos']
        })
    
    return pd.DataFrame(resultados)


# --- FUNCIÓN PRINCIPAL ---
def main():
    """Función principal que ejecuta la aplicación."""
    st.set_page_config(page_title="SumaDigitos - Análisis de Sumas", page_icon="🔢", layout="wide")

    st.title("🔢 SumaDigitos - Análisis de Sumas (Mañana, Tarde y Noche)")
    st.markdown("""
    **Suma de dígitos:** Para un número de 2 dígitos, se suman sus componentes.
    - Ejemplo: 69 → 6+9 = 15 | 00 → 0+0 = 0 | 99 → 9+9 = 18

    **Tipos de suma:**
    - **Suma Fijo:** Suma de dígitos del Fijo (rango: 0-18)
    - **Suma Total:** Suma_Fijo + Suma_Corr1 + Suma_Corr2 (rango: 0-54)
    - **Suma Corridos:** Suma_Corr1 + Suma_Corr2 (rango: 0-36)

    **Sesiones:** M = Mañana | T = Tarde | N = Noche
    """)

    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = cargar_datos()

    if df.empty:
        st.error("No se pudieron cargar los datos")
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    tipo_sesion = st.sidebar.selectbox("Sesión", ['Todas', 'M', 'T', 'N'], 
                                       format_func=lambda x: {'Todas': 'Todas', 'M': 'Mañana', 'T': 'Tarde', 'N': 'Noche'}.get(x, x))

    # Pestañas principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Tabla de Estadísticas", "📅 Almanaque Suma Total", "📅 Almanaque Suma Corridos", "📅 Almanaque Suma Fijo", "🔍 Buscar Suma"])

    # === TAB 1: TABLA DE ESTADÍSTICAS COMPLETA ===
    with tab1:
        st.header("📊 Tabla Completa de Estadísticas")
        
        with st.spinner("Calculando estadísticas..."):
            stats = calcular_estadisticas_suma(df, tipo_sesion if tipo_sesion != 'Todas' else None)
        
        # SUMA TOTAL - Tabla principal
        st.subheader("➕ Suma Total (Fijo + Corrido1 + Corrido2) - Rango: 0-54")
        df_total = stats['Suma_Total'].copy()
        df_total = df_total[df_total['Frecuencia'] > 0]
        df_total['Ultima_Fecha'] = df_total['Ultima_Fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if x else 'N/A')
        df_total = df_total.rename(columns={
            'Suma': 'Suma',
            'Frecuencia': 'Frecuencia',
            'Ausencia_Maxima': 'Ausencia Máx',
            'Promedio_Dias': 'Promedio Días',
            'Dias_Sin_Aparecer': 'Días Sin Aparecer',
            'Ultima_Fecha': 'Última Fecha'
        })
        df_total = df_total.sort_values('Frecuencia', ascending=False)
        st.dataframe(df_total, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # SUMA CORRIDOS - Tabla
        st.subheader("🎲 Suma Corridos (Corrido1 + Corrido2) - Rango: 0-36")
        df_corr = stats['Suma_Corridos'].copy()
        df_corr = df_corr[df_corr['Frecuencia'] > 0]
        df_corr['Ultima_Fecha'] = df_corr['Ultima_Fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if x else 'N/A')
        df_corr = df_corr.rename(columns={
            'Suma': 'Suma',
            'Frecuencia': 'Frecuencia',
            'Ausencia_Maxima': 'Ausencia Máx',
            'Promedio_Dias': 'Promedio Días',
            'Dias_Sin_Aparecer': 'Días Sin Aparecer',
            'Ultima_Fecha': 'Última Fecha'
        })
        df_corr = df_corr.sort_values('Frecuencia', ascending=False)
        st.dataframe(df_corr, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # SUMA FIJO - Tabla
        st.subheader("🔢 Suma Fijo - Rango: 0-18")
        df_fijo = stats['Suma_Fijo'].copy()
        df_fijo['Ultima_Fecha'] = df_fijo['Ultima_Fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if x else 'N/A')
        df_fijo = df_fijo.rename(columns={
            'Suma': 'Suma',
            'Frecuencia': 'Frecuencia',
            'Ausencia_Maxima': 'Ausencia Máx',
            'Promedio_Dias': 'Promedio Días',
            'Dias_Sin_Aparecer': 'Días Sin Aparecer',
            'Ultima_Fecha': 'Última Fecha'
        })
        df_fijo = df_fijo.sort_values('Frecuencia', ascending=False)
        st.dataframe(df_fijo, use_container_width=True, hide_index=True)

    # === TAB 2: ALMANAQUE SUMA TOTAL ===
    with tab2:
        st.header("📅 Almanaque Suma Total (Fijo + Corrido1 + Corrido2)")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            dia_ini = st.number_input("Día inicio", 1, 31, 1, key="alm_total_ini")
        with col_cfg2:
            dia_fin = st.number_input("Día fin", 1, 31, 15, key="alm_total_fin")
        with col_cfg3:
            meses_atras = st.number_input("Meses atrás", 1, 24, 6, key="alm_total_meses")
        
        if st.button("Generar Almanaque Suma Total", key="btn_alm_total"):
            with st.spinner("Generando almanaque..."):
                resultado = generar_almanaque_sumas(df, 'Suma_Total', dia_ini, dia_fin, meses_atras,
                                                    tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if not resultado['success']:
                st.warning(resultado.get('mensaje', 'Sin datos'))
            else:
                st.info(f"📅 Períodos analizados: {', '.join(resultado['nombres_bloques'])}")
                
                # Mostrar sumas persistentes
                if resultado['sumas_persistentes']:
                    st.success(f"🔥 **Sumas Persistentes (aparecen en todos los períodos):** {resultado['sumas_persistentes']}")
                else:
                    st.info("No hay sumas que aparezcan en todos los períodos")
                
                # Ranking de sumas
                st.subheader("📊 Ranking de Sumas (por frecuencia)")
                df_rank = resultado['df_rank'].copy()
                df_rank = df_rank[df_rank['Frecuencia'] > 0]
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
                
                # Historial actual
                if not resultado['df_historial_actual'].empty:
                    st.subheader("📋 Historial Período Actual")
                    st.markdown(f"**{resultado['estado_periodo']}**")
                    st.dataframe(resultado['df_historial_actual'], use_container_width=True, hide_index=True)
                
                # Sumas faltantes
                if not resultado['df_faltantes'].empty:
                    st.subheader("⏳ Sumas Calientes Faltantes en Período Actual")
                    st.dataframe(resultado['df_faltantes'], use_container_width=True, hide_index=True)

    # === TAB 3: ALMANAQUE SUMA CORRIDOS ===
    with tab3:
        st.header("📅 Almanaque Suma Corridos (Corrido1 + Corrido2)")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            dia_ini_c = st.number_input("Día inicio", 1, 31, 1, key="alm_corr_ini")
        with col_cfg2:
            dia_fin_c = st.number_input("Día fin", 1, 31, 15, key="alm_corr_fin")
        with col_cfg3:
            meses_atras_c = st.number_input("Meses atrás", 1, 24, 6, key="alm_corr_meses")
        
        if st.button("Generar Almanaque Suma Corridos", key="btn_alm_corr"):
            with st.spinner("Generando almanaque..."):
                resultado = generar_almanaque_sumas(df, 'Suma_Corridos', dia_ini_c, dia_fin_c, meses_atras_c,
                                                    tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if not resultado['success']:
                st.warning(resultado.get('mensaje', 'Sin datos'))
            else:
                st.info(f"📅 Períodos analizados: {', '.join(resultado['nombres_bloques'])}")
                
                if resultado['sumas_persistentes']:
                    st.success(f"🔥 **Sumas Persistentes:** {resultado['sumas_persistentes']}")
                else:
                    st.info("No hay sumas que aparezcan en todos los períodos")
                
                st.subheader("📊 Ranking de Sumas")
                df_rank = resultado['df_rank'].copy()
                df_rank = df_rank[df_rank['Frecuencia'] > 0]
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
                
                if not resultado['df_historial_actual'].empty:
                    st.subheader("📋 Historial Período Actual")
                    st.markdown(f"**{resultado['estado_periodo']}**")
                    st.dataframe(resultado['df_historial_actual'], use_container_width=True, hide_index=True)
                
                if not resultado['df_faltantes'].empty:
                    st.subheader("⏳ Sumas Calientes Faltantes")
                    st.dataframe(resultado['df_faltantes'], use_container_width=True, hide_index=True)

    # === TAB 4: ALMANAQUE SUMA FIJO ===
    with tab4:
        st.header("📅 Almanaque Suma Fijo")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            dia_ini_f = st.number_input("Día inicio", 1, 31, 1, key="alm_fijo_ini")
        with col_cfg2:
            dia_fin_f = st.number_input("Día fin", 1, 31, 15, key="alm_fijo_fin")
        with col_cfg3:
            meses_atras_f = st.number_input("Meses atrás", 1, 24, 6, key="alm_fijo_meses")
        
        if st.button("Generar Almanaque Suma Fijo", key="btn_alm_fijo"):
            with st.spinner("Generando almanaque..."):
                resultado = generar_almanaque_sumas(df, 'Suma_Fijo', dia_ini_f, dia_fin_f, meses_atras_f,
                                                    tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if not resultado['success']:
                st.warning(resultado.get('mensaje', 'Sin datos'))
            else:
                st.info(f"📅 Períodos analizados: {', '.join(resultado['nombres_bloques'])}")
                
                if resultado['sumas_persistentes']:
                    st.success(f"🔥 **Sumas Persistentes:** {resultado['sumas_persistentes']}")
                else:
                    st.info("No hay sumas que aparezcan en todos los períodos")
                
                st.subheader("📊 Ranking de Sumas")
                df_rank = resultado['df_rank'].copy()
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
                
                if not resultado['df_historial_actual'].empty:
                    st.subheader("📋 Historial Período Actual")
                    st.markdown(f"**{resultado['estado_periodo']}**")
                    st.dataframe(resultado['df_historial_actual'], use_container_width=True, hide_index=True)

    # === TAB 5: BUSCAR SUMA ===
    with tab5:
        st.header("🔍 Buscar Suma Específica")
        
        col_tipo, col_valor = st.columns(2)
        with col_tipo:
            tipo_busqueda = st.selectbox("Tipo de suma", ['Suma_Fijo', 'Suma_Total', 'Suma_Corridos'],
                                         format_func=lambda x: {'Suma_Fijo': 'Suma Fijo (0-18)', 
                                                               'Suma_Total': 'Suma Total (0-54)',
                                                               'Suma_Corridos': 'Suma Corridos (0-36)'}[x])
        with col_valor:
            if tipo_busqueda == 'Suma_Fijo':
                suma_val = st.number_input("Valor de suma", 0, 18, 9)
            elif tipo_busqueda == 'Suma_Total':
                suma_val = st.number_input("Valor de suma", 0, 54, 27)
            else:
                suma_val = st.number_input("Valor de suma", 0, 36, 18)
        
        if st.button("Buscar Suma", key="btn_buscar"):
            # Mostrar números que componen esta suma
            st.subheader(f"🔢 Números de 2 dígitos que suman {suma_val}")
            nums = obtener_numeros_que_suman(suma_val)
            st.markdown(f"**{', '.join(nums)}**")
            
            # Buscar historial
            df_resultado = buscar_suma_detalle(df, tipo_busqueda, suma_val,
                                               tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if df_resultado is None or df_resultado.empty:
                st.warning(f"No se encontraron resultados para suma = {suma_val}")
            else:
                st.subheader(f"📋 Historial de Apariciones de Suma {suma_val}")
                st.info(f"Total de apariciones: {len(df_resultado)}")
                st.dataframe(df_resultado, use_container_width=True, hide_index=True)

    # === INFO ADICIONAL ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Información")
    st.sidebar.markdown(f"**Datos cargados:** {len(df)} registros")
    if not df.empty:
        st.sidebar.markdown(f"**Desde:** {df['Fecha'].min().strftime('%d/%m/%Y')}")
        st.sidebar.markdown(f"**Hasta:** {df['Fecha'].max().strftime('%d/%m/%Y')}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*SumaDigitos v3.0*")


# Ejecutar si es el archivo principal
if __name__ == "__main__":
    main()