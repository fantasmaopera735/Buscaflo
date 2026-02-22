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
    page_icon="🎰",
    layout="wide"
)

st.title("🎰 Flotodo - Suite Ultimate")

@st.cache_resource
def cargar_datos_flotodo(_ruta_csv):
    try:
        if not os.path.exists(_ruta_csv):
            st.error(f"No se encontro el archivo {_ruta_csv}")
            st.stop()
        
        with open(_ruta_csv, 'r', encoding='utf-8-sig') as f:
            primera_linea = f.readline()
        
        separador = ',' if ',' in primera_linea and ';' not in primera_linea else ';'
        
        df = pd.read_csv(_ruta_csv, sep=separador, encoding='utf-8-sig', dtype=str)
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        
        col_fecha = col_tipo = col_centena = col_fijo = col_1er = col_2do = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            col_clean = col_lower.replace('/', '').replace('_', '').replace('-', '')
            
            if 'fecha' in col_lower: col_fecha = col
            if col_tipo is None:
                if 'tarde' in col_lower or 'noche' in col_lower: col_tipo = col
                elif 'tipo' in col_lower or 'sorteo' in col_lower: col_tipo = col
                elif 'tardenoche' in col_clean: col_tipo = col
            if col_centena is None and ('centena' in col_lower or 'cent' in col_lower): col_centena = col
            if col_fijo is None and col_lower == 'fijo': col_fijo = col
            if col_1er is None and ('1er' in col_lower or 'primer' in col_lower): col_1er = col
            if col_2do is None and ('2do' in col_lower or 'segundo' in col_lower): col_2do = col
        
        num_cols = len(df.columns)
        if col_fecha is None and num_cols >= 1: col_fecha = df.columns[0]
        if col_tipo is None and num_cols >= 2: col_tipo = df.columns[1]
        
        if num_cols >= 6:
            if col_centena is None: col_centena = df.columns[2]
            if col_fijo is None: col_fijo = df.columns[3]
            if col_1er is None: col_1er = df.columns[4]
            if col_2do is None: col_2do = df.columns[5]
        elif num_cols >= 5:
            if col_fijo is None: col_fijo = df.columns[2]
            if col_1er is None: col_1er = df.columns[3]
            if col_2do is None: col_2do = df.columns[4]
        
        mapeo = {}
        if col_fecha: mapeo[col_fecha] = 'Fecha'
        if col_tipo: mapeo[col_tipo] = 'Tipo_Sorteo'
        if col_centena: mapeo[col_centena] = 'Centena'
        if col_fijo: mapeo[col_fijo] = 'Fijo'
        if col_1er: mapeo[col_1er] = 'Primer_Corrido'
        if col_2do: mapeo[col_2do] = 'Segundo_Corrido'
        
        df = df.rename(columns=mapeo)
        if 'Centena' not in df.columns: df['Centena'] = '0'
        
        def convertir_fecha(valor):
            if pd.isna(valor) or str(valor).strip() == '': return pd.NaT
            valor_str = str(valor).strip()
            formatos = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y', '%Y/%m/%d']
            for fmt in formatos:
                try: return pd.to_datetime(valor_str, format=fmt)
                except: continue
            try: return pd.to_datetime(valor_str, dayfirst=True)
            except: return pd.NaT
        
        df['Fecha'] = df['Fecha'].apply(convertir_fecha)
        df = df.dropna(subset=['Fecha'])
        
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].apply(lambda x: 
                'T' if x in ['T', 'TARDE', 'TARDE/'] else
                'N' if x in ['N', 'NOCHE', '/NOCHE', 'NOCHE/'] else
                'T' if 'TARDE' in x else 'N' if 'NOCHE' in x else 'OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        if 'Fijo' in df.columns:
            df_fijos = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
            df_fijos = df_fijos.rename(columns={'Fijo': 'Numero'})
            df_fijos['Numero'] = pd.to_numeric(df_fijos['Numero'], errors='coerce')
            df_fijos = df_fijos.dropna(subset=['Numero'])
            df_fijos['Numero'] = df_fijos['Numero'].astype(int)
        else:
            st.error("No se encontro la columna Fijo")
            st.stop()
        
        df_fijos = df_fijos.sort_values(by='Fecha').reset_index(drop=True)
        return df_fijos, df
        
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

def analizar_estabilidad_numeros(df_fijos, dias_analisis=365):
    fecha_limite = datetime.now() - timedelta(days=dias_analisis)
    df_historico = df_fijos[df_fijos['Fecha'] >= fecha_limite].copy()
    if df_historico.empty: return None
    
    estabilidad_data = []
    hoy = datetime.now()
    
    for num in range(100):
        df_num = df_historico[df_historico['Numero'] == num].sort_values('Fecha')
        
        if len(df_num) < 2:
            gap_actual = (hoy - df_num['Fecha'].max()).days if not df_num.empty else dias_analisis
            avg_gap = 999
            estado = "SIN DATOS"
        else:
            fechas = df_num['Fecha'].tolist()
            gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
            avg_gap = np.mean(gaps) if gaps else 0
            gap_actual = (hoy - fechas[-1]).days
            
            if gap_actual == 0:
                estado = "🔥 EN RACHA"
            elif gap_actual <= avg_gap:
                estado = "✅ NORMAL"
            elif gap_actual <= avg_gap * 2.0:
                estado = "⏳ VENCIDO"
            else:
                estado = "🔴 MUY VENCIDO"

        estabilidad_data.append({
            'Número': f"{num:02d}", 'Gap Actual': gap_actual,
            'Gap Promedio': round(avg_gap, 1), 'Estado': estado
        })
    
    return pd.DataFrame(estabilidad_data)

# --- FUNCIÓN FINAL: FALTANTES DEL MES (Corregida) ---
def analizar_faltantes_mes(df_fijos, mes, anio, sorteos_freq):
    hoy = datetime.now()
    
    # 1. Determinar números salidos en el mes
    fecha_inicio_mes = datetime(anio, mes, 1)
    last_day = calendar.monthrange(anio, mes)[1]
    fecha_fin_mes = datetime(anio, mes, last_day)
    
    if mes == hoy.month and anio == hoy.year:
        fecha_fin_mes = hoy
    
    df_mes = df_fijos[(df_fijos['Fecha'] >= fecha_inicio_mes) & (df_fijos['Fecha'] <= fecha_fin_mes)]
    salidos = set(df_mes['Numero'].unique())
    faltantes = sorted(list(set(range(100)) - salidos))
    
    if not faltantes:
        return pd.DataFrame(), "Todos los números salieron.", pd.DataFrame()
    
    # 2. Calcular Estabilidad (Último año)
    df_estabilidad = analizar_estabilidad_numeros(df_fijos, 365)
    est_map = {}
    if df_estabilidad is not None:
        for _, row in df_estabilidad.iterrows():
            est_map[row['Número']] = {'Gap': row['Gap Actual'], 'Estado': row['Estado']}

    # 3. Calcular Frecuencia (Basado en la cantidad de sorteos indicados)
    # Tomamos los últimos N sorteos del dataframe
    df_reciente = df_fijos.tail(sorteos_freq)
    conteo = df_reciente['Numero'].value_counts()
    
    # Definimos "Alta Frecuencia" como el Top 25 de esos sorteos
    top_frecuencia = conteo.head(25).index.tolist()
    
    # 4. Construir Lista
    resultados = []
    for num in faltantes:
        est_data = est_map.get(f"{num:02d}", {'Gap': 999, 'Estado': 'SIN DATOS'})
        es_vencido = ("VENCIDO" in est_data['Estado'])
        
        es_favorito = (num in top_frecuencia)
        freq_val = conteo.get(num, 0)
        
        # Lógica OR
        if es_vencido or es_favorito:
            prioridad = "🔴 ALTA"
            razones = []
            if es_vencido: razones.append("Atrasado")
            if es_favorito: razones.append("Favorito")
            razon = " + ".join(razones)
        else:
            prioridad = "⚪ BAJA"
            razon = "Sin condiciones"
        
        resultados.append({
            'Número': f"{num:02d}",
            'Prioridad': prioridad,
            'Razón': razon,
            'Veces Salidas': freq_val, # Título corregido dinámicamente en la UI
            'Estado Estabilidad': est_data['Estado'],
            'Gap Actual': est_data['Gap']
        })
    
    df_res = pd.DataFrame(resultados)
    ord_map = {"🔴 ALTA": 0, "⚪ BAJA": 1}
    df_res['ord'] = df_res['Prioridad'].map(ord_map)
    df_res = df_res.sort_values(['ord', 'Veces Salidas'], ascending=[True, False]).drop('ord', axis=1)
    
    return df_res, None, df_mes

def eliminar_ultimo_sorteo(ruta_csv):
    try:
        with open(ruta_csv, 'r', encoding='utf-8-sig') as f:
            lineas = f.readlines()
        if len(lineas) > 1:
            with open(ruta_csv, 'w', encoding='utf-8-sig') as f:
                f.writelines(lineas[:-1])
            return True, lineas[-1].strip()
        return False, "No hay registros"
    except Exception as e:
        return False, str(e)

# --- MAIN ---
df_fijos, df_completo = cargar_datos_flotodo(RUTA_CSV)

# SIDEBAR
st.sidebar.header("📋 Últimos Sorteos")

df_ultima_tarde = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].tail(1)
if not df_ultima_tarde.empty:
    fecha_t = df_ultima_tarde['Fecha'].values[0]
    num_t = int(df_ultima_tarde['Numero'].values[0])
    st.sidebar.metric("🌞 Último Tarde", f"{num_t:02d}", delta=pd.Timestamp(fecha_t).strftime('%d/%m'))

df_ultima_noche = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].tail(1)
if not df_ultima_noche.empty:
    fecha_n = df_ultima_noche['Fecha'].values[0]
    num_n = int(df_ultima_noche['Numero'].values[0])
    st.sidebar.metric("🌙 Último Noche", f"{num_n:02d}", delta=pd.Timestamp(fecha_n).strftime('%d/%m'))

with st.sidebar.expander("📝 Agregar Sorteo"):
    f = st.date_input("Fecha:", datetime.now().date(), format="DD/MM/YYYY", label_visibility="collapsed")
    s = st.radio("Sesión:", ["Tarde (T)", "Noche (N)"], horizontal=True, label_visibility="collapsed")
    cent = st.number_input("Centena:", 0, 9, 0)
    c1, c2 = st.columns(2)
    with c1: fj = st.number_input("Fijo", 0, 99, 0, format="%02d")
    with c2: c1v = st.number_input("1er Corrido", 0, 99, 0, format="%02d")
    p2 = st.number_input("2do Corrido", 0, 99, 0, format="%02d")
    
    if st.button("💾 Guardar", type="primary"):
        cd = s.split('(')[1].replace(')', '')
        try:
            with open(RUTA_CSV, 'r', encoding='utf-8-sig') as file:
                primera = file.readline()
            sep = ',' if ',' in primera and ';' not in primera else ';'
            with open(RUTA_CSV, 'a', encoding='utf-8-sig') as file:
                file.write(f"{f.strftime('%d-%m-%Y')}{sep}{cd}{sep}{cent}{sep}{fj:02d}{sep}{c1v:02d}{sep}{p2:02d}\n")
            st.success("✅ Guardado")
            time.sleep(1)
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

with st.sidebar.expander("🗑️ Eliminar Sorteo"):
    if st.button("❌ Eliminar Último", type="secondary"):
        exito, mensaje = eliminar_ultimo_sorteo(RUTA_CSV)
        if exito:
            st.success(f"✅ Eliminado")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"❌ {mensaje}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Recargar"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.subheader("🎲 Modo")
modo = st.sidebar.radio("Filtro:", ["General", "Tarde", "Noche"])

if modo == "Tarde":
    dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].copy()
elif modo == "Noche":
    dfa = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].copy()
else:
    dfa = df_fijos.copy()

if dfa.empty:
    st.warning(f"⚠️ No hay datos para: {modo}")
    st.stop()

# PESTAÑAS
tabs = st.tabs(["🗓️ Faltantes del Mes", "🔄 Transferencia", "🔢 Dígito Faltante", "🔍 Patrones", "📅 Almanaque", "📉 Estabilidad"])

# PESTAÑA 0: FALTANTES DEL MES
with tabs[0]:
    st.subheader("🗓️ Análisis de Faltantes del Mes")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    meses_nombres = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                     7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    
    with col_f1:
        mes_sel = st.selectbox("Mes a Analizar:", list(meses_nombres.values()), index=datetime.now().month - 1)
        mes_num = [k for k, v in meses_nombres.items() if v == mes_sel][0]
    
    with col_f2:
        anio_sel = st.number_input("Año:", min_value=2020, max_value=datetime.now().year, value=datetime.now().year)
    
    with col_f3:
        # Slider con valor por defecto 1000
        cant_sorteos = st.slider("Sorteos para Frecuencia:", 100, 5000, 1000, step=100, 
                                 help="Busca los favoritos en los últimos X sorteos.")

    if st.button("🔍 Analizar Faltantes", type="primary"):
        with st.spinner("Calculando..."):
            df_faltantes_res, error_msg, df_salidos_mes = analizar_faltantes_mes(dfa, mes_num, anio_sel, cant_sorteos)
        
        if error_msg:
            st.info(error_msg)
        elif not df_faltantes_res.empty:
            # 1. Resumen de Faltantes
            total_faltantes = len(df_faltantes_res)
            alta = df_faltantes_res[df_faltantes_res['Prioridad'] == '🔴 ALTA']
            
            st.markdown(f"### ⏳ Faltan por salir: {total_faltantes} números")
            st.markdown(f"#### 🔴 Prioridad Alta (Apostar): {len(alta)} números")
            st.write(" ".join([f"`{n}`" for n in alta['Número'].tolist()]))

            # 2. Tabla Principal (Título dinámico)
            st.markdown("---")
            st.markdown("#### 📊 Detalle de Faltantes")
            # Renombramos la columna para que coincida con el slider
            df_show = df_faltantes_res.rename(columns={'Veces Salidas': f'Frec. ({cant_sorteos} sort.)'})
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            # 3. Tabla de Aciertos Recientes (NUEVA)
            st.markdown("---")
            st.markdown("#### 📝 Historial de Aciertos Recientes (Verificación)")
            st.info("Ordenado del más reciente al más antiguo. Noche (N) tiene prioridad sobre Tarde (T) en el mismo día.")
            
            # Lógica de ordenamiento: Primero Fecha, luego Tipo (N antes que T)
            # Creamos columna auxiliar para ordenar
            df_historial = df_fijos.tail(20).copy()
            # Mapeamos N=0, T=1 para que N salga primero
            orden_tipo = {'N': 0, 'T': 1, 'OTRO': 2}
            df_historial['orden_tipo'] = df_historial['Tipo_Sorteo'].map(orden_tipo)
            
            # Ordenamos: Fecha Descendente, Tipo (N antes que T)
            df_historial = df_historial.sort_values(by=['Fecha', 'orden_tipo'], ascending=[False, True])
            
            # Verificamos si salió en el mes analizado
            df_historial['¿Salió en el Mes?'] = df_historial['Numero'].apply(lambda x: "✅ SÍ" if x in df_salidos_mes['Numero'].values else "")
            
            # Formateamos fecha para mostrar
            df_historial['Fecha Str'] = df_historial['Fecha'].dt.strftime('%d/%m/%Y')
            
            st.dataframe(
                df_historial[['Fecha Str', 'Tipo_Sorteo', 'Numero', '¿Salió en el Mes?']].head(10),
                column_config={
                    "Fecha Str": "Fecha",
                    "Tipo_Sorteo": "Sorteo",
                    "Numero": st.column_config.NumberColumn("Número", format="%02d"),
                    "¿Salió en el Mes?": "Estado"
                },
                hide_index=True
            )

# PESTAÑAS RESTANTES
with tabs[1]: st.subheader("🔄 Transferencia")
with tabs[2]: st.subheader("🔢 Dígito Faltante")
with tabs[3]: st.subheader(f"🔍 Patrones")
with tabs[4]: st.subheader("📅 Almanaque")
with tabs[5]:
    st.subheader("📉 Estabilidad General")
    if st.button("Calcular", key="est_btn"):
        df_est = analizar_estabilidad_numeros(dfa, 365)
        if df_est is not None:
            st.dataframe(df_est.sort_values('Gap Actual', ascending=False).head(30), hide_index=True)

st.markdown("---")
st.caption("Flotodo Suite Ultimate v2.5 | Frecuencia Dinámica y Orden Corregido")
