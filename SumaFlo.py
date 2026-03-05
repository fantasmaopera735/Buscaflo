# -*- coding: utf-8 -*-
"""
SumaFlo - Análisis de Sumas del Fijo
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
import os

if __name__ == "__main__":
    st.set_page_config(page_title="SumaFlo", page_icon="🔢", layout="wide")

GS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'
GS_SHEET = 'Sorteos'
COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

MAPEO = {'t': 'Tarde', 'tarde': 'Tarde', 'n': 'Noche', 'noche': 'Noche'}

SUMA_NUMEROS = {}
for i in range(100):
    suma = int(f"{i:02d}"[0]) + int(f"{i:02d}"[1])
    if suma not in SUMA_NUMEROS:
        SUMA_NUMEROS[suma] = []
    SUMA_NUMEROS[suma].append(i)

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

def analizar_almanaque(df, dia_inicio, dia_fin, cantidad_meses, tipo_suma='Fijo'):
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
    
    col_suma = f'Suma_{tipo_suma}'
    
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
    st.title("🔢 SumaFlo - Análisis de Sumas")
    st.markdown("**Hoja: Sorteos** | Sesiones: Tarde y Noche")
    
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
    df['Suma_Total'] = df.apply(lambda x: (x['Suma_Fijo'] if pd.notna(x['Suma_Fijo']) else 0) + 
                                           (x['Suma_Corr1'] if pd.notna(x['Suma_Corr1']) else 0) + 
                                           (x['Suma_Corr2'] if pd.notna(x['Suma_Corr2']) else 0), axis=1)
    df['Suma_Corridos'] = df.apply(lambda x: (x['Suma_Corr1'] if pd.notna(x['Suma_Corr1']) else 0) + 
                                            (x['Suma_Corr2'] if pd.notna(x['Suma_Corr2']) else 0), axis=1)
    
    # Orden para historial: Noche=1, Tarde=2
    orden_sesion = {'Noche': 1, 'Tarde': 2}
    df['Orden'] = df['Sesion'].map(orden_sesion).fillna(99)
    
    # Pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Suma Fijo", 
        "🎯 Suma Corridos", 
        "🔥 Suma Total (3)",
        "🔮 Pronóstico T→N",
        "📅 Almanaque",
        "📜 Historial"
    ])
    
    # === TAB 1: SUMA FIJO ===
    with tab1:
        st.header("📊 Suma del Fijo")
        
        data = []
        for s in range(19):
            df_s = df[df['Suma_Fijo'] == s]
            freq = len(df_s)
            ultima = df_s['Fecha_Parsed'].max() if freq > 0 else None
            dias = (fecha_max - ultima).days if ultima else 0
            data.append({'Suma': s, 'Frecuencia': freq, 'Días sin aparecer': dias})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        
        st.subheader("🏆 Más Salidores")
        top = df[COL_FIJO].value_counts().head(15)
        st.dataframe(pd.DataFrame({'Número': [f"{n:02d}" for n in top.index], 'Frecuencia': top.values}), use_container_width=True, hide_index=True)
    
    # === TAB 2: SUMA CORRIDOS ===
    with tab2:
        st.header("🎯 Suma de los 2 Corridos")
        st.markdown("*Suma: Primer_Corrido + Segundo_Corrido*")
        
        data = []
        for s in range(19):
            df_s = df[df['Suma_Corridos'] == s]
            freq = len(df_s)
            ultima = df_s['Fecha_Parsed'].max() if freq > 0 else None
            dias = (fecha_max - ultima).days if ultima else 0
            data.append({'Suma': s, 'Frecuencia': freq, 'Días sin aparecer': dias})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    # === TAB 3: SUMA TOTAL ===
    with tab3:
        st.header("🔥 Suma Total (Fijo + Corrido1 + Corrido2)")
        
        data = []
        for s in sorted(df['Suma_Total'].unique()):
            df_s = df[df['Suma_Total'] == s]
            freq = len(df_s)
            ultima = df_s['Fecha_Parsed'].max() if freq > 0 else None
            dias = (fecha_max - ultima).days if ultima else 0
            data.append({'Suma Total': s, 'Frecuencia': freq, 'Días sin aparecer': dias})
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    # === TAB 4: PRONÓSTICO TARDE → NOCHE ===
    with tab4:
        st.header("🔮 Pronóstico: Tarde → Noche")
        st.markdown("*Si en la Tarde sale X, qué suele salir en la Noche*")
        
        suma_tarde = st.selectbox("Suma en Tarde:", list(range(19)), key="p_tarde")
        
        patrones = defaultdict(lambda: defaultdict(int))
        conteo = defaultdict(int)
        
        for fecha, g in df.groupby('Fecha_Parsed'):
            if pd.isna(fecha):
                continue
            sumas = {}
            for _, row in g.iterrows():
                ses = str(row['Sesion']).lower() if pd.notna(row['Sesion']) else ''
                if ses in ['tarde', 't']:
                    sumas['tarde'] = int(row['Suma_Fijo']) if pd.notna(row['Suma_Fijo']) else None
                elif ses in ['noche', 'n']:
                    sumas['noche'] = int(row['Suma_Fijo']) if pd.notna(row['Suma_Fijo']) else None
            
            if sumas.get('tarde') == suma_tarde and 'noche' in sumas:
                patrones[sumas['noche']] += 1
                conteo[suma_tarde] += 1
        
        if patrones:
            total = sum(patrones.values())
            st.info(f"📊 **{total}** patrones encontrados con Tarde={suma_tarde}")
            
            df_p = pd.DataFrame([
                {'Suma Noche': s, 'Veces': v, 'Porcentaje': f"{v/total*100:.1f}%"}
                for s, v in sorted(patrones.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_p, use_container_width=True, hide_index=True)
        else:
            st.warning("No hay datos para esta combinación")
    
    # === TAB 5: ALMANAQUE ===
    with tab5:
        st.header("📅 Almanaque")
        
        tipo = st.selectbox("Tipo de suma:", ['Fijo', 'Corridos', 'Total'])
        col1, col2, col3 = st.columns(3)
        with col1:
            dia_inicio = st.number_input("Día inicial:", 1, 31, 1)
        with col2:
            dia_fin = st.number_input("Día final:", 1, 31, 15)
        with col3:
            cantidad_meses = st.slider("Meses:", 1, 12, 3)
        
        if st.button("🔍 Analizar", type="primary"):
            almanaque = analizar_almanaque(df, dia_inicio, dia_fin, cantidad_meses, tipo)
            
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
    
    # === TAB 6: HISTORIAL ===
    with tab6:
        st.header("📜 Historial")
        st.markdown("**Orden:** Fecha descendente. Dentro de cada día: Noche → Tarde")
        
        tipo_hist = st.selectbox("Ver suma de:", ['Fijo', 'Corridos', 'Total'], key="tipo_hist")
        cantidad = st.number_input("Registros:", 10, 200, 50)
        
        # Ordenar: fecha descendente, dentro de cada día: Noche(1) → Tarde(2)
        df_h = df.sort_values(['Fecha_Parsed', 'Orden'], ascending=[False, True]).head(cantidad)
        
        col_suma = f'Suma_{tipo_hist}' if tipo_hist != 'Total' else 'Suma_Total'
        col_suma = 'Suma_Corridos' if tipo_hist == 'Corridos' else col_suma
        
        df_h = df_h.copy()
        df_h['Fecha'] = df_h['Fecha_Parsed'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '-')
        df_h['Fijo'] = df_h[COL_FIJO].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else '-')
        
        st.dataframe(df_h[['Fecha', 'Sesion', 'Fijo', col_suma]].reset_index(drop=True), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()