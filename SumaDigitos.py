# -*- coding: utf-8 -*-
"""
SumaDigitos - Análisis de Sumas
Hoja: Geotodo (M=Mañana, T=Tarde, N=Noche)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import calendar
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

if __name__ == "__main__":
    st.set_page_config(page_title="SumaDigitos", page_icon="🔢", layout="wide")

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

SUMA_NUMEROS = {}
for i in range(100):
    suma = int(f"{i:02d}"[0]) + int(f"{i:02d}"[1])
    if suma not in SUMA_NUMEROS:
        SUMA_NUMEROS[suma] = []
    SUMA_NUMEROS[suma].append(i)

# Números que componen cada suma de corridos (0-36)
SUMA_CORRIDOS_NUMEROS = {}
for c1 in range(100):
    for c2 in range(100):
        s1 = int(f"{c1:02d}"[0]) + int(f"{c1:02d}"[1])
        s2 = int(f"{c2:02d}"[0]) + int(f"{c2:02d}"[1])
        suma_total = s1 + s2
        if suma_total not in SUMA_CORRIDOS_NUMEROS:
            SUMA_CORRIDOS_NUMEROS[suma_total] = []
        if (c1, c2) not in SUMA_CORRIDOS_NUMEROS[suma_total]:
            SUMA_CORRIDOS_NUMEROS[suma_total].append((c1, c2))

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

def suma_digitos(n):
    try:
        s = f"{int(n):02d}"
        return int(s[0]) + int(s[1])
    except:
        return None

def calcular_estadisticas_suma(df, col_suma, suma, fecha_max):
    """Calcula estadísticas para una suma específica"""
    df_s = df[df[col_suma] == suma].copy()
    
    if len(df_s) == 0:
        return {
            'frecuencia': 0,
            'promedio_dias': 0,
            'ausencia_maxima': 0,
            'dias_sin_aparecer': 0,
            'ultima_fecha': None
        }
    
    fechas = df_s['Fecha_Parsed'].dropna().sort_values().tolist()
    
    if not fechas:
        return {
            'frecuencia': 0,
            'promedio_dias': 0,
            'ausencia_maxima': 0,
            'dias_sin_aparecer': 0,
            'ultima_fecha': None
        }
    
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
        'ultima_fecha': ultima_fecha
    }

def analizar_pronostico_tarde_noche_manana(df):
    """Analiza: Si en Tarde y Noche sale suma X, qué sale en Mañana del día siguiente"""
    patrones = {}
    conteo_total = {}
    
    for fecha, grupo in df.groupby('Fecha_Parsed'):
        if pd.isna(fecha):
            continue
        
        sumas = {}
        for _, row in grupo.iterrows():
            ses = str(row['Sesion']).lower() if pd.notna(row['Sesion']) else ''
            suma = int(row['Suma_Fijo']) if pd.notna(row['Suma_Fijo']) else None
            if suma is not None:
                if ses in ['tarde', 't']:
                    sumas['tarde'] = suma
                elif ses in ['noche', 'n']:
                    sumas['noche'] = suma
                elif ses in ['mañana', 'manana', 'm']:
                    sumas['mañana'] = suma
        
        fecha_siguiente = fecha + pd.Timedelta(days=1)
        df_siguiente = df[df['Fecha_Parsed'] == fecha_siguiente]
        
        if 'tarde' in sumas and 'noche' in sumas and len(df_siguiente) > 0:
            suma_tarde = sumas['tarde']
            suma_noche = sumas['noche']
            key = (suma_tarde, suma_noche)
            
            if key not in patrones:
                patrones[key] = {}
                conteo_total[key] = 0
            
            for _, row_sig in df_siguiente.iterrows():
                ses_sig = str(row_sig['Sesion']).lower() if pd.notna(row_sig['Sesion']) else ''
                if ses_sig in ['mañana', 'manana', 'm']:
                    suma_manana = int(row_sig['Suma_Fijo']) if pd.notna(row_sig['Suma_Fijo']) else None
                    if suma_manana is not None:
                        if suma_manana not in patrones[key]:
                            patrones[key][suma_manana] = 0
                        patrones[key][suma_manana] += 1
                        conteo_total[key] += 1
    
    return patrones, conteo_total

def analizar_almanaque(df, dia_inicio, dia_fin, cantidad_meses, col_suma):
    """Analiza sumas por meses"""
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
                'fecha_ini': fecha_ini_mes, 
                'fecha_fin': fecha_fin_mes,
                'nombre_mes': fecha_ini_mes.strftime('%B %Y')
            })
        except ValueError:
            continue
    
    resultados_meses = {}
    sumas_por_mes = {}
    
    for mes_info in meses_analizar:
        df_mes = df[(df['Fecha_Parsed'] >= mes_info['fecha_ini']) & (df['Fecha_Parsed'] <= mes_info['fecha_fin'])].copy()
        sumas_mes = df_mes[col_suma].value_counts().to_dict() if col_suma in df_mes.columns else {}
        sumas_por_mes[mes_info['nombre_mes']] = sumas_mes
        resultados_meses[mes_info['nombre_mes']] = {
            'total_sorteos': len(df_mes),
            'sumas': sumas_mes
        }
    
    sumas_persistentes = set()
    if sumas_por_mes:
        conjuntos = [set(s.keys()) for s in sumas_por_mes.values() if s]
        if conjuntos:
            sumas_persistentes = set.intersection(*conjuntos)
    
    return {
        'meses_analizados': meses_analizar,
        'resultados_por_mes': resultados_meses,
        'sumas_persistentes': sumas_persistentes,
        'sumas_por_mes': sumas_por_mes
    }

def main():
    st.title("🔢 SumaDigitos - Análisis de Sumas")
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
    
    # Procesar datos
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df['Sesion'] = df[COL_SESION].apply(normalizar_sesion)
    fecha_max = df['Fecha_Parsed'].max()
    
    # Calcular sumas
    df['Suma_Fijo'] = df[COL_FIJO].apply(suma_digitos)
    df['Suma_Corr1'] = df[COL_CORR1].apply(suma_digitos)
    df['Suma_Corr2'] = df[COL_CORR2].apply(suma_digitos)
    df['Suma_Corridos'] = df.apply(lambda x: 
        (x['Suma_Corr1'] if pd.notna(x['Suma_Corr1']) else 0) + 
        (x['Suma_Corr2'] if pd.notna(x['Suma_Corr2']) else 0), axis=1)
    df['Suma_Total'] = df.apply(lambda x: 
        (x['Suma_Fijo'] if pd.notna(x['Suma_Fijo']) else 0) + 
        (x['Suma_Corr1'] if pd.notna(x['Suma_Corr1']) else 0) + 
        (x['Suma_Corr2'] if pd.notna(x['Suma_Corr2']) else 0), axis=1)
    
    # Orden para historial: Noche=1, Tarde=2, Mañana=3
    orden_sesion = {'Noche': 1, 'Tarde': 2, 'Mañana': 3}
    df['Orden'] = df['Sesion'].map(orden_sesion).fillna(99)
    
    # Pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Suma Fijo", 
        "🎯 Suma Corridos", 
        "🔥 Suma Total",
        "🔮 Pronóstico T+N→M",
        "📜 Historial"
    ])
    
    # === TAB 1: SUMA FIJO ===
    with tab1:
        st.header("📊 Suma del Fijo (00-99)")
        
        # Selector de suma
        suma_sel = st.selectbox("Selecciona una suma:", list(range(19)), key="sel_fijo")
        
        # Mostrar números que componen la suma
        st.subheader(f"📋 Números que componen la Suma {suma_sel}")
        nums = SUMA_NUMEROS.get(suma_sel, [])
        nums_str = ", ".join([f"{n:02d}" for n in nums])
        st.write(f"**{len(nums)} números:** {nums_str}")
        
        # Estadísticas
        stats = calcular_estadisticas_suma(df, 'Suma_Fijo', suma_sel, fecha_max)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Frecuencia", stats['frecuencia'])
        with col2:
            st.metric("Promedio Días", stats['promedio_dias'])
        with col3:
            st.metric("Ausencia Máx", f"{stats['ausencia_maxima']} días")
        with col4:
            st.metric("Días Sin Aparecer", stats['dias_sin_aparecer'])
        with col5:
            fecha_str = stats['ultima_fecha'].strftime('%d/%m/%Y') if stats['ultima_fecha'] else '-'
            st.metric("Última Fecha", fecha_str)
        
        st.markdown("---")
        
        # Tabla resumen
        st.subheader("📋 Resumen de todas las sumas")
        data = []
        for s in range(19):
            st_s = calcular_estadisticas_suma(df, 'Suma_Fijo', s, fecha_max)
            data.append({
                'Suma': s,
                'Números': f"{len(SUMA_NUMEROS.get(s, []))} nums",
                'Frecuencia': st_s['frecuencia'],
                'Promedio': st_s['promedio_dias'],
                'Ausencia Máx': st_s['ausencia_maxima'],
                'Días Sin Aparecer': st_s['dias_sin_aparecer'],
                'Última Fecha': st_s['ultima_fecha'].strftime('%d/%m/%Y') if st_s['ultima_fecha'] else '-'
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🏆 Números Más Salidores")
        top = df[COL_FIJO].value_counts().head(15)
        st.dataframe(pd.DataFrame({'Número': [f"{n:02d}" for n in top.index], 'Frecuencia': top.values}), use_container_width=True, hide_index=True)
    
    # === TAB 2: SUMA CORRIDOS (ALMANAQUE) ===
    with tab2:
        st.header("🎯 Suma de los 2 Corridos")
        st.markdown("*Suma: (Dígitos Corrido1) + (Dígitos Corrido2)*")
        
        # Selector de suma
        max_suma_corr = int(df['Suma_Corridos'].max()) if len(df) > 0 else 36
        suma_sel = st.selectbox("Selecciona una suma:", list(range(max_suma_corr + 1)), key="sel_corr")
        
        # Estadísticas
        stats = calcular_estadisticas_suma(df, 'Suma_Corridos', suma_sel, fecha_max)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Frecuencia", stats['frecuencia'])
        with col2:
            st.metric("Promedio Días", stats['promedio_dias'])
        with col3:
            st.metric("Ausencia Máx", f"{stats['ausencia_maxima']} días")
        with col4:
            st.metric("Días Sin Aparecer", stats['dias_sin_aparecer'])
        with col5:
            fecha_str = stats['ultima_fecha'].strftime('%d/%m/%Y') if stats['ultima_fecha'] else '-'
            st.metric("Última Fecha", fecha_str)
        
        st.markdown("---")
        
        # Almanaque
        st.subheader("📅 Almanaque")
        col1, col2, col3 = st.columns(3)
        with col1:
            dia_inicio = st.number_input("Día inicial:", 1, 31, 1, key="dia_corr")
        with col2:
            dia_fin = st.number_input("Día final:", 1, 31, 15, key="fin_corr")
        with col3:
            cantidad_meses = st.slider("Meses:", 1, 12, 3, key="meses_corr")
        
        if st.button("🔍 Analizar Almanaque Corridos", type="primary"):
            almanaque = analizar_almanaque(df, dia_inicio, dia_fin, cantidad_meses, 'Suma_Corridos')
            
            for mes, datos in almanaque['resultados_por_mes'].items():
                with st.expander(f"📆 {mes} - {datos['total_sorteos']} sorteos"):
                    if datos['sumas']:
                        df_mes = pd.DataFrame([
                            {'Suma': s, 'Frecuencia': f}
                            for s, f in sorted(datos['sumas'].items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(df_mes, use_container_width=True, hide_index=True)
            
            if almanaque['sumas_persistentes']:
                st.success(f"✅ Sumas persistentes: {sorted(almanaque['sumas_persistentes'])}")
    
    # === TAB 3: SUMA TOTAL (ALMANAQUE) ===
    with tab3:
        st.header("🔥 Suma Total (Fijo + Corrido1 + Corrido2)")
        st.markdown("*Suma de las 3 sumas individuales*")
        
        # Selector de suma
        max_suma_total = int(df['Suma_Total'].max()) if len(df) > 0 else 54
        suma_sel = st.selectbox("Selecciona una suma:", sorted(df['Suma_Total'].dropna().unique().astype(int)), key="sel_total")
        
        # Estadísticas
        stats = calcular_estadisticas_suma(df, 'Suma_Total', suma_sel, fecha_max)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Frecuencia", stats['frecuencia'])
        with col2:
            st.metric("Promedio Días", stats['promedio_dias'])
        with col3:
            st.metric("Ausencia Máx", f"{stats['ausencia_maxima']} días")
        with col4:
            st.metric("Días Sin Aparecer", stats['dias_sin_aparecer'])
        with col5:
            fecha_str = stats['ultima_fecha'].strftime('%d/%m/%Y') if stats['ultima_fecha'] else '-'
            st.metric("Última Fecha", fecha_str)
        
        st.markdown("---")
        
        # Almanaque
        st.subheader("📅 Almanaque")
        col1, col2, col3 = st.columns(3)
        with col1:
            dia_inicio = st.number_input("Día inicial:", 1, 31, 1, key="dia_total")
        with col2:
            dia_fin = st.number_input("Día final:", 1, 31, 15, key="fin_total")
        with col3:
            cantidad_meses = st.slider("Meses:", 1, 12, 3, key="meses_total")
        
        if st.button("🔍 Analizar Almanaque Total", type="primary"):
            almanaque = analizar_almanaque(df, dia_inicio, dia_fin, cantidad_meses, 'Suma_Total')
            
            for mes, datos in almanaque['resultados_por_mes'].items():
                with st.expander(f"📆 {mes} - {datos['total_sorteos']} sorteos"):
                    if datos['sumas']:
                        df_mes = pd.DataFrame([
                            {'Suma': s, 'Frecuencia': f}
                            for s, f in sorted(datos['sumas'].items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(df_mes, use_container_width=True, hide_index=True)
            
            if almanaque['sumas_persistentes']:
                st.success(f"✅ Sumas persistentes: {sorted(almanaque['sumas_persistentes'])}")
    
    # === TAB 4: PRONÓSTICO TARDE+NOCHE → MAÑANA ===
    with tab4:
        st.header("🔮 Pronóstico: Tarde + Noche → Mañana")
        st.markdown("*Si en la Tarde sale X y en la Noche sale Y, qué suele salir en la Mañana del día siguiente*")
        
        col1, col2 = st.columns(2)
        with col1:
            suma_tarde = st.selectbox("Suma en Tarde:", list(range(19)), key="p_tarde")
        with col2:
            suma_noche = st.selectbox("Suma en Noche:", list(range(19)), key="p_noche")
        
        patrones, conteo = analizar_pronostico_tarde_noche_manana(df)
        key = (suma_tarde, suma_noche)
        
        if key in patrones and patrones[key]:
            total = conteo[key]
            st.info(f"📊 Se encontraron **{total}** patrones con Tarde={suma_tarde} y Noche={suma_noche}")
            
            df_p = pd.DataFrame([
                {'Suma Mañana': s, 'Veces': v, 'Porcentaje': f"{v/total*100:.1f}%"}
                for s, v in sorted(patrones[key].items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_p, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos históricos para esta combinación")
    
    # === TAB 5: HISTORIAL ===
    with tab5:
        st.header("📜 Historial")
        st.markdown("**Orden:** Fecha descendente. Dentro de cada día: Noche → Tarde → Mañana")
        
        tipo_hist = st.selectbox("Ver suma de:", ['Fijo', 'Corridos', 'Total'], key="tipo_hist")
        cantidad = st.number_input("Registros:", 10, 200, 50)
        
        col_suma = {'Fijo': 'Suma_Fijo', 'Corridos': 'Suma_Corridos', 'Total': 'Suma_Total'}[tipo_hist]
        
        df_h = df.sort_values(['Fecha_Parsed', 'Orden'], ascending=[False, True]).head(cantidad).copy()
        df_h['Fecha'] = df_h['Fecha_Parsed'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '-')
        df_h['Fijo'] = df_h[COL_FIJO].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else '-')
        
        st.dataframe(df_h[['Fecha', 'Sesion', 'Fijo', col_suma]].reset_index(drop=True), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()