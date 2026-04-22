# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from collections import Counter

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RUTA_CSV = 'Flotodo.csv'

st.set_page_config(
    page_title="Flotodo - Análisis de Dígitos P75",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🌺 Flotodo - Análisis de Dígitos P75 (Decena/Unidad)")
st.markdown("Motor predictivo basado en comportamiento individual de dígitos + Percentil 75")

# =============================================================================
# ESTADO DE SESIÓN
# =============================================================================
if 'df_digitos_stats' not in st.session_state:
    st.session_state.df_digitos_stats = None
if 'df_full_cache' not in st.session_state:
    st.session_state.df_full_cache = None
if 'last_session_filter' not in st.session_state:
    st.session_state.last_session_filter = None

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def inicializar_archivo(ruta, columnas):
    if not os.path.exists(ruta):
        try:
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(";".join(columnas) + "\n")
            return True
        except Exception as e:
            st.error(f"❌ Error inicializando {ruta}: {e}")
            return False
    return True

def parse_fecha_safe(fecha_str):
    if pd.isna(fecha_str) or str(fecha_str).strip() == '': return None
    fecha_str = str(fecha_str).strip()
    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
        try: return pd.to_datetime(fecha_str, format=fmt, dayfirst=True)
        except: continue
    return pd.to_datetime(fecha_str, dayfirst=True, errors='coerce')

def calcular_estado_actual(gap, limite_p75):
    if pd.isna(limite_p75) or limite_p75 == 0: return "Normal"
    if gap > limite_p75: return "Muy Vencido"
    elif gap > (limite_p75 * 0.66): return "Vencido"
    else: return "Normal"

# =============================================================================
# CARGA DE DATOS (FLOTDO: SOLO T Y N)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_flotodo(_ruta_csv):
    if not os.path.exists(_ruta_csv):
        inicializar_archivo(_ruta_csv, ["Fecha", "Tipo_Sorteo", "Fijo"])
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Fijo"]), pd.DataFrame()
    
    try:
        with open(_ruta_csv, 'r', encoding='latin-1') as f:
            primera = f.readline()
            sep = ';' if ';' in primera else (',' if ',' in primera else '\t')
        df = pd.read_csv(_ruta_csv, sep=sep, encoding='latin-1', header=0, dtype=str, on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Error leyendo CSV: {e}")
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Fijo"]), pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    if 'Fecha' not in df.columns or 'Fijo' not in df.columns:
        st.error("❌ El CSV debe contener columnas 'Fecha' y 'Fijo'")
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Fijo"]), pd.DataFrame()
        
    df['Fecha'] = df['Fecha'].apply(parse_fecha_safe)
    df = df.dropna(subset=['Fecha']).copy()
    df['Fijo'] = pd.to_numeric(df['Fijo'], errors='coerce').fillna(0).astype(int)
    
    # 🧹 LIMPIEZA AGRESIVA (Elimina espacios ocultos y normaliza)
    if 'Tipo_Sorteo' in df.columns:
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
        
        # ✅ MAPEO CORREGIDO (SIN ESPACIOS EXTRAS)
        mapeo = {'TARDE': 'T', 'AFTERNOON': 'T', 'T': 'T', 'NOCHE': 'N', 'NIGHT': 'N', 'N': 'N'}
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map(mapeo)
        
        # Filtramos estrictamente
        df = df[df['Tipo_Sorteo'].isin(['T', 'N'])].copy()
        
        # 📊 DEBUG: Te muestra en consola cuántos cargó de cada uno
        counts = df['Tipo_Sorteo'].value_counts().to_dict()
        st.sidebar.caption(f"📊 Cargados: Tarde={counts.get('T',0)} | Noche={counts.get('N',0)}")
    else:
        df['Tipo_Sorteo'] = 'T'
        
    df = df.sort_values('Fecha').reset_index(drop=True)
    return df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy(), df.copy()
# =============================================================================
# MOTOR DE ANÁLISIS DE DÍGITOS (P75 + ESTABILIDAD)
# =============================================================================
def analizar_estadisticas_digitos(df_fijos, fecha_ref):
    resultados = []
    for tipo_pos in ['Decena', 'Unidad']:
        for digito in range(10):
            if tipo_pos == 'Decena':
                fechas = df_fijos[(df_fijos['Fijo'] // 10) == digito]['Fecha'].sort_values()
            else:
                fechas = df_fijos[(df_fijos['Fijo'] % 10) == digito]['Fecha'].sort_values()
                
            if len(fechas) < 2:
                resultados.append({
                    'Dígito': digito, 'Tipo': tipo_pos, 'Frecuencia': len(fechas),
                    'Veces Normal': 0, 'Veces Vencido': 0, 'Veces Muy Vencido': 0,
                    'Estado Actual': 'Normal', 'Estabilidad': 100.0, 'P75 (Límite)': 0,
                    'Alerta': '-', 'Gap Actual': (fecha_ref - fechas.iloc[-1]).days if not fechas.empty else 999,
                    'Estado Última Salida': 'Normal', 'Estabilidad Última Salida': 100.0, 'Exceso Última Salida': 0,
                    'Dist_Normal': 33.3, 'Dist_Vencido': 33.3, 'Dist_MuyVencido': 33.3,
                    'Estado_Comun': 'Ninguno', 'Porc_Comun': 0
                })
                continue
                
            fechas_list = fechas.tolist()
            gaps = [(fechas_list[i] - fechas_list[i-1]).days for i in range(1, len(fechas_list))]
            
            limite_p75 = int(np.percentile(gaps, 75)) if len(gaps) >= 4 else (int(np.median(gaps) * 2) if gaps else 0)
            gap_actual = (fecha_ref - fechas_list[-1]).days
            estado_actual = calcular_estado_actual(gap_actual, limite_p75)
            
            estados_hist = []
            estado_ult_salida = 'Normal'
            estab_ult_salida = 100.0
            exceso_ult_salida = 0
            
            for i, g in enumerate(gaps):
                gaps_prev = gaps[:i]
                lim_prev = int(np.percentile(gaps_prev, 75)) if len(gaps_prev) >= 4 else (int(np.median(gaps_prev) * 2) if gaps_prev else 0)
                est = calcular_estado_actual(g, lim_prev)
                estados_hist.append(est)
                
                if i == len(gaps) - 1:
                    estado_ult_salida = est
                    if estado_ult_salida == 'Muy Vencido' and lim_prev > 0:
                        exceso_ult_salida = int(g - lim_prev)
                    
                    ests_prev = [calcular_estado_actual(g_p, lim_prev) for g_p in gaps_prev] if gaps_prev else []
                    mv_prev = ests_prev.count('Muy Vencido')
                    estab_ult_salida = ((len(ests_prev) - mv_prev) / len(ests_prev) * 100) if ests_prev else 100.0
            
            total_hist = len(estados_hist)
            count_n = estados_hist.count('Normal')
            count_v = estados_hist.count('Vencido')
            count_mv = estados_hist.count('Muy Vencido')
            
            estabilidad = ((total_hist - count_mv) / total_hist * 100) if total_hist > 0 else 0
            alerta = '⚠️ RECUPERAR' if (estabilidad > 60 and estado_actual in ['Vencido', 'Muy Vencido']) else '-'
            
            d_n = (count_n / total_hist * 100) if total_hist > 0 else 0
            d_v = (count_v / total_hist * 100) if total_hist > 0 else 0
            d_mv = (count_mv / total_hist * 100) if total_hist > 0 else 0
            
            estado_comun = 'Ninguno'
            porc_comun = 0
            max_dist = max(d_n, d_v, d_mv)
            if max_dist >= 60:
                estado_comun = 'Normal' if max_dist == d_n else ('Vencido' if max_dist == d_v else 'Muy Vencido')
                porc_comun = max_dist
                
            resultados.append({
                'Dígito': digito, 'Tipo': tipo_pos, 'Frecuencia': total_hist + 1,
                'Veces Normal': count_n, 'Veces Vencido': count_v, 'Veces Muy Vencido': count_mv,
                'Estado Actual': estado_actual, 'Estabilidad': round(estabilidad, 1), 'P75 (Límite)': limite_p75,
                'Alerta': alerta, 'Gap Actual': gap_actual,
                'Estado Última Salida': estado_ult_salida, 'Estabilidad Última Salida': round(estab_ult_salida, 1), 'Exceso Última Salida': exceso_ult_salida,
                'Dist_Normal': round(d_n, 1), 'Dist_Vencido': round(d_v, 1), 'Dist_MuyVencido': round(d_mv, 1),
                'Estado_Comun': estado_comun, 'Porc_Comun': porc_comun
            })
            
    return pd.DataFrame(resultados)

# =============================================================================
# COMPONENTES DE UI
# =============================================================================
def mostrar_ultimos_resultados_sidebar(df_full):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Últimos Resultados por Sesión")
    if df_full.empty:
        st.sidebar.info("ℹ️ Sin datos")
        return
    
    for letra, nombre in [('T', '☀️ Tarde'), ('N', '🌙 Noche')]:
        df_sesion = df_full[df_full['Tipo_Sorteo'] == letra]
        if not df_sesion.empty:
            ultimo = df_sesion.sort_values('Fecha', ascending=False).iloc[0]
            num = int(ultimo['Fijo'])
            fecha = ultimo['Fecha'].strftime('%d/%m/%Y') if pd.notna(ultimo['Fecha']) else 'N/A'
            st.sidebar.markdown(f"**{nombre}**")
            st.sidebar.markdown(f"📅 {fecha}")
            st.sidebar.markdown(f"🔢 `{num:02d}`")
            st.sidebar.markdown(f"🔟 Dec: `{num // 10}` | 1️⃣ Uni: `{num % 10}`")
            st.sidebar.markdown("---")

def mostrar_tabla_comportamiento(df_stats):
    st.subheader("📊 Comportamiento Histórico de Dígitos")
    filas = []
    for _, row in df_stats.iterrows():
        rec = f"✅ Jugar en {row['Estado_Comun']}" if row['Estado_Comun'] != 'Ninguno' else "⚠️ Sin patrón claro"
        filas.append({
            'Dígito': row['Dígito'], 'Tipo': row['Tipo'],
            'Normal': f"{row['Dist_Normal']:.1f}%", 'Vencido': f"{row['Dist_Vencido']:.1f}%", 'Muy Vencido': f"{row['Dist_MuyVencido']:.1f}%",
            'Estado Común': row['Estado_Comun'], 'Recomendación': rec
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

def mostrar_tablas_separadas(df_stats):
    st.markdown("---")
    st.subheader("📈 Estadística de Dígitos (Separada por Posición)")
    
    df_dec = df_stats[df_stats['Tipo'] == 'Decena'].sort_values('Dígito').reset_index(drop=True)
    df_uni = df_stats[df_stats['Tipo'] == 'Unidad'].sort_values('Dígito').reset_index(drop=True)
    
    cols = ['Dígito', 'Frecuencia', 'Veces Normal', 'Veces Vencido', 'Veces Muy Vencido', 
            'Estado Actual', 'Estabilidad', 'P75 (Límite)', 'Alerta', 'Gap Actual', 
            'Estado Última Salida', 'Estabilidad Última Salida', 'Exceso Última Salida']
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔟 Decenas")
        st.dataframe(df_dec[cols], hide_index=True, use_container_width=True)
    with c2:
        st.markdown("#### 1️⃣ Unidades")
        st.dataframe(df_uni[cols], hide_index=True, use_container_width=True)

def mostrar_alertas_detalladas(df_stats):
    df_alertas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR'].copy()
    if df_alertas.empty:
        st.info("✅ No hay alertas activas para dígitos.")
        return

    st.markdown("---")
    st.subheader("🚨 Detalle de Alertas Activas (Dígitos)")
    for _, row in df_alertas.iterrows():
        d, tipo = row['Dígito'], row['Tipo']
        estado = row['Estado Actual']
        gap = row['Gap Actual']
        limite = row['P75 (Límite)']
        
        time_str = ""
        if estado == "Normal":
            falta = int(limite - gap) if limite > gap else 0
            time_str = f"🟢 Faltan ~{falta} días para Vencido"
        elif estado == "Vencido":
            falta_mv = int(limite - gap) if limite > gap else 0
            exceso = int(gap - (limite * 0.66))
            time_str = f"🟠 Exceso: {exceso} días | Faltan {falta_mv} para Muy Vencido"
        elif estado == "Muy Vencido":
            exceso = int(gap - limite) if limite > 0 else gap
            time_str = f"🔴 +{exceso} días sobre límite P75"
            
        dist_str = f"Normal: {row['Dist_Normal']:.1f}% | Vencido: {row['Dist_Vencido']:.1f}% | Muy Vencido: {row['Dist_MuyVencido']:.1f}%"
        
        with st.container(border=True):
            icon = "🔟" if tipo == "Decena" else "1️⃣"
            st.markdown(f"**{icon} Dígito `{d}` ({tipo}) | Estado: `{estado}` | ⏳ {time_str}**")
            st.markdown(f"📉 **Estabilidad Actual:** `{row['Estabilidad']}%` | 🔔 **Alerta:** ⚠️ RECUPERAR")
            st.markdown(f"📊 **Datos:** Gap Actual: **{gap} días** | Límite P75: **{limite} días**")
            st.markdown(f"📈 **Distribución Histórica:** {dist_str}")
            
            if row['Estado_Comun'] != 'Ninguno':
                st.success(f"✅ **Patrón Claro:** Suele salir en estado `{row['Estado_Comun']}` ({row['Porc_Comun']:.1f}% histórico)")
            else:
                st.warning("⚠️ Sin patrón dominante histórico (<60%)")
                
            st.info("🔙 **Historia de la última salida de este dígito:**")
            st.markdown(f"- Estado en ese momento: **{row['Estado Última Salida']}**")
            st.markdown(f"- Estabilidad previa: **{row['Estabilidad Última Salida']}%**")
            if row['Exceso Última Salida'] > 0:
                st.markdown(f"- Días en exceso: **+{row['Exceso Última Salida']} días**")
            st.markdown("---")
def ordenador_numeros(df_stats):
    st.markdown("---")
    st.subheader("🔢 Ordenador de Números por Dígito")
    st.caption("Ingresa números separados por comas o espacios. La app los ordenará según la presión estadística de sus dígitos.")
    
    df_stats_activo = df_stats if 'df_stats' in locals() else st.session_state.get('df_digitos_stats')
    
    if df_stats_activo is not None and not df_stats_activo.empty:
        entrada = st.text_area("Números:", height=80, placeholder="12, 45, 08, 99...", key="input_nums")
        
        if st.button("🔄 Ordenar por Algoritmo", key="btn_ordenar", use_container_width=True):
            if not entrada.strip():
                st.warning("⚠️ Ingresa al menos un número.")
                return
                
            nums = []
            for n in entrada.replace(',', ' ').replace('\n', ' ').split():
                n = n.strip()
                if n.isdigit() and 0 <= int(n) <= 99:
                    nums.append(int(n))
            nums = sorted(list(set(nums)))
            
            if not nums:
                st.error("❌ No se encontraron números válidos (00-99).")
                return
                
            resultados = []
            for num in nums:
                dec, uni = num // 10, num % 10
                row_dec = df_stats_activo[(df_stats_activo['Dígito'] == dec) & (df_stats_activo['Tipo'] == 'Decena')]
                row_uni = df_stats_activo[(df_stats_activo['Dígito'] == uni) & (df_stats_activo['Tipo'] == 'Unidad')]
                
                if row_dec.empty or row_uni.empty: continue
                r_dec, r_uni = row_dec.iloc[0], row_uni.iloc[0]
                
                score = 0
                for r in [r_dec, r_uni]:
                    if r['Porc_Comun'] >= 60: score += 100
                    elif r['Porc_Comun'] >= 50: score += 70
                    else: score += 40
                    score += min(max((r['Gap Actual'] - r['P75 (Límite)']), 0) * 3, 30)
                    score += r['Estabilidad'] * 0.25
                    if r['Alerta'] == '⚠️ RECUPERAR': score += 20
                score = round(score, 1)
                
                resultados.append({
                    'Número': f"{num:02d}", 'Score': score,
                    'Estado Dec': f"{r_dec['Dígito']} ({r_dec['Estado Actual']})",
                    'Estado Uni': f"{r_uni['Dígito']} ({r_uni['Estado Actual']})"
                })
                
            resultados.sort(key=lambda x: x['Score'], reverse=True)
            st.success(f"✅ Ordenados {len(resultados)} números.")
            st.dataframe(pd.DataFrame(resultados), hide_index=True, use_container_width=True)
            
            if len(resultados) >= 2:
                st.info(f"🎯 **Recomendación:** Juega `{resultados[0]['Número']}` y `{resultados[1]['Número']}`")
    else:
        st.info("ℹ️ Ejecuta el análisis primero para activar el ordenador.")

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.sidebar.header("⚙️ Configuración Flotodo")
    
    # 🔄 BOTÓN DE FORZAR RECARGA
    if st.sidebar.button("🔄 Forzar Recarga / Actualizar CSV", type="primary", use_container_width=True):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    modo_sesion = st.sidebar.radio("Sesión de Análisis:", ["General", "Tarde", "Noche"], key="radio_sesion")
    
    # ✅ DETECTAR CAMBIO DE SESIÓN Y LIMPIAR CACHE
    if modo_sesion != st.session_state.get('last_session_filter'):
        st.session_state.df_digitos_stats = None
        st.session_state.df_analisis_cache = None
        st.session_state.last_session_filter = modo_sesion
    
    # 1. Cargamos TODOS los datos primero
    df_todo, df_full = cargar_datos_flotodo(RUTA_CSV)
    st.session_state.df_full_cache = df_full
    
    # 2. Sidebar: Muestra últimos resultados de T y N SIEMPRE
    mostrar_ultimos_resultados_sidebar(df_full) 
    
    if df_todo.empty:
        st.warning("⚠️ Archivo vacío o sin datos válidos. Agrega sorteos a `Flotodo.csv`.")
        return

        # 3. Filtrado estricto por sesión seleccionada (SOLO T y N)
    if modo_sesion == "Tarde":
        df_analisis = df_todo[df_todo['Tipo_Sorteo'] == 'T'].copy()
    elif modo_sesion == "Noche":
        df_analisis = df_todo[df_todo['Tipo_Sorteo'] == 'N'].copy()
    else: # General
        df_analisis = df_todo.copy()
        
    # 🔍 DEBUG VISUAL
    st.info(f"📊 Analizando **{len(df_analisis)}** sorteos para la sesión: **{modo_sesion}** (T={len(df_todo[df_todo['Tipo_Sorteo']=='T'])}, N={len(df_todo[df_todo['Tipo_Sorteo']=='N'])})")

    # 4. Botón de ejecución
    if st.button("🚀 Ejecutar Análisis de Dígitos", type="primary", key="btn_analisis"):
        if len(df_analisis) == 0:
            st.error(f"❌ No hay datos para '{modo_sesion}'. Verifica que el CSV tenga 'T' o 'N'.")
            return

        with st.spinner("Calculando Gaps, P75 y Estados por dígito..."):
            df_stats = analizar_estadisticas_digitos(df_analisis, fecha_ref)
            
            # 🧠 Guardar en memoria para persistencia
            st.session_state.df_digitos_stats = df_stats
            st.session_state.df_analisis_cache = df_analisis
            
            # Renderizado con tablas separadas
            mostrar_tabla_comportamiento(df_stats)
            mostrar_tablas_separadas(df_stats)
            mostrar_alertas_detalladas(df_stats)
            ordenador_numeros(df_stats)

    # 5. Recuperación de estado (evita cierre al interactuar)
    elif st.session_state.df_digitos_stats is not None:
        df_stats = st.session_state.df_digitos_stats
        df_analisis = st.session_state.get('df_analisis_cache', df_todo)
        
        mostrar_tabla_comportamiento(df_stats)
        mostrar_tablas_separadas(df_stats)
        mostrar_alertas_detalladas(df_stats)
        ordenador_numeros(df_stats)
    else:
        st.info("👈 Presiona 'Ejecutar Análisis de Dígitos' para comenzar.")

if __name__ == "__main__":
    main()