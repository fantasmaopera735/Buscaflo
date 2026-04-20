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
RUTA_CSV = 'Geotodo.csv'
RUTA_CACHE = 'cache_sumas_georgia.csv'

st.set_page_config(
    page_title="Georgia - Análisis por Sumas",
    page_icon="🍑",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🍑 Georgia - Análisis por Sumas de Dígitos (0-18)")
st.markdown("Motor predictivo basado en comportamiento histórico de sumas + Percentil 75 (P75)")

# =============================================================================
# ESTADO DE SESIÓN
# =============================================================================
if 'df_stats_sumas' not in st.session_state:
    st.session_state.df_stats_sumas = None
if 'distribuciones_sumas' not in st.session_state:
    st.session_state.distribuciones_sumas = None
if 'df_full_cache' not in st.session_state:
    st.session_state.df_full_cache = None

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

def obtener_suma(num):
    return (num // 10) + (num % 10)

def obtener_combinaciones_suma(s):
    return [f"{d}{u}" for d in range(10) for u in range(10) if d + u == s]

def calcular_estado_actual(gap, limite_p75):
    if pd.isna(limite_p75) or limite_p75 == 0: return "Normal"
    if gap > limite_p75: return "Muy Vencido"
    elif gap > (limite_p75 * 0.66): return "Vencido"
    else: return "Normal"

# =============================================================================
# CARGA DE DATOS (GEORGIA - Geotodo.csv)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_geotodo(_ruta_csv):
    if not os.path.exists(_ruta_csv):
        inicializar_archivo(_ruta_csv, ["Fecha", "Tipo_Sorteo", "Fijo"])
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Fijo", "Suma"]), pd.DataFrame()
    
    try:
        with open(_ruta_csv, 'r', encoding='latin-1') as f:
            primera = f.readline()
            sep = ';' if ';' in primera else (',' if ',' in primera else '\t')
        df = pd.read_csv(_ruta_csv, sep=sep, encoding='latin-1', header=0, dtype=str, on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Error leyendo CSV: {e}")
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Fijo", "Suma"]), pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    if 'Fecha' not in df.columns or 'Fijo' not in df.columns:
        st.error("❌ El CSV debe contener columnas 'Fecha' y 'Fijo'")
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Fijo", "Suma"]), pd.DataFrame()
        
    df['Fecha'] = df['Fecha'].apply(parse_fecha_safe)
    df = df.dropna(subset=['Fecha']).copy()
    df['Fijo'] = pd.to_numeric(df['Fijo'], errors='coerce').fillna(0).astype(int)
    df['Suma'] = df['Fijo'].apply(obtener_suma)
    
    # 🧹 Limpieza estricta de sesiones para Georgia
    if 'Tipo_Sorteo' in df.columns:
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.upper().str.strip()
        mapeo = {
            'MAÑANA': 'M', 'MANANA': 'M', 'MORNING': 'M', 'M': 'M',
            'TARDE': 'T', 'AFTERNOON': 'T', 'T': 'T',
            'NOCHE': 'N', 'NIGHT': 'N', 'N': 'N'
        }
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map(mapeo)
        df = df[df['Tipo_Sorteo'].isin(['M', 'T', 'N'])].copy()
    else:
        df['Tipo_Sorteo'] = 'T'
        
    df = df.sort_values('Fecha').reset_index(drop=True)
    return df[['Fecha', 'Tipo_Sorteo', 'Fijo', 'Suma']].copy(), df.copy()

# =============================================================================
# MOTOR DE ANÁLISIS DE SUMAS (P75 + ESTABILIDAD)
# =============================================================================
def pre_calcular_distribuciones_sumas(df_historial):
    distribuciones = {}
    for suma in range(19):
        fechas = df_historial[df_historial['Suma'] == suma]['Fecha'].sort_values().tolist()
        if len(fechas) < 2:
            distribuciones[suma] = {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3}
            continue
            
        estados_historicos = []
        for i in range(1, len(fechas)):
            gaps_prev = [(fechas[j] - fechas[j-1]).days for j in range(1, i)]
            if not gaps_prev:
                limite = 0
            elif len(gaps_prev) >= 4:
                limite = int(np.percentile(gaps_prev, 75))
            else:
                limite = int(np.median(gaps_prev) * 2)
            
            gap_actual = (fechas[i] - fechas[i-1]).days
            estados_historicos.append(calcular_estado_actual(gap_actual, limite))
            
        if estados_historicos:
            contador = Counter(estados_historicos)
            total = len(estados_historicos)
            distribucion = {e: (c/total*100) for e, c in contador.items()}
            estado_comun = max(distribucion, key=distribucion.get)
            distribuciones[suma] = {
                **distribucion, 
                'Estado_Comun': estado_comun if distribucion[estado_comun] >= 60 else 'Ninguno', 
                'porcentaje': distribucion.get(estado_comun, 0)
            }
        else:
            distribuciones[suma] = {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Ninguno', 'porcentaje': 0}
    return distribuciones

def analizar_estadisticas_sumas(df_fijos, fecha_ref, distribuciones_cache=None):
    if distribuciones_cache is None:
        distribuciones_cache = pre_calcular_distribuciones_sumas(df_fijos)
        
    resultados = []
    for suma in range(19):
        fechas = df_fijos[df_fijos['Suma'] == suma]['Fecha'].sort_values().tolist()
        if not fechas: continue
            
        gaps = [(fechas[i] - fechas[i-1]).days for i in range(1, len(fechas))]
        limite_p75 = int(np.percentile(gaps, 75)) if len(gaps) >= 4 else (int(np.median(gaps) * 2) if gaps else 0)
        gap_actual = (fecha_ref - fechas[-1]).days
        estado_actual = calcular_estado_actual(gap_actual, limite_p75)
        
        estados_hist = [calcular_estado_actual(g, limite_p75) for g in gaps]
        total_hist = len(estados_hist)
        count_n = estados_hist.count('Normal')
        count_v = estados_hist.count('Vencido')
        count_mv = estados_hist.count('Muy Vencido')
        
        estabilidad = ((total_hist - count_mv) / total_hist * 100) if total_hist > 0 else 0
        alerta = '⚠️ RECUPERAR' if (estabilidad > 60 and estado_actual in ['Vencido', 'Muy Vencido']) else '-'
        
        exceso_ult = 0
        estado_ult = 'Normal'
        if len(gaps) >= 1:
            estado_ult = estados_hist[-1] if estados_hist else 'Normal'
            if estado_ult == 'Muy Vencido' and limite_p75 > 0:
                exceso_ult = gaps[-1] - limite_p75
                
        dist = distribuciones_cache.get(suma, {})
        resultados.append({
            'Suma': suma,
            'Frecuencia': total_hist + 1,
            'Veces Normal': count_n,
            'Veces Vencido': count_v,
            'Veces Muy Vencido': count_mv,
            'Estado Actual': estado_actual,
            'Estabilidad': round(estabilidad, 1),
            'Tiempo Limite (P75)': limite_p75,
            'Alerta': alerta,
            'Gap Actual': gap_actual,
            'Estado Ultima Salida': estado_ult,
            'Exceso Ultima Salida': exceso_ult,
            'Estado_Comun': dist.get('Estado_Comun', 'Ninguno'),
            'Porc_Comun': dist.get('porcentaje', 0)
        })
    return pd.DataFrame(resultados), distribuciones_cache

# =============================================================================
# COMPONENTES DE UI
# =============================================================================
def mostrar_tabla_comportamiento(distribuciones):
    st.subheader("📊 Comportamiento Histórico de Sumas")
    filas = []
    for s in sorted(distribuciones.keys()):
        d = distribuciones[s]
        rec = f"✅ Jugar en {d['Estado_Comun']}" if d['Estado_Comun'] != 'Ninguno' else "⚠️ Sin patrón claro"
        filas.append({
            'Suma': s,
            'Normal': f"{d.get('Normal', 0):.1f}%",
            'Vencido': f"{d.get('Vencido', 0):.1f}%",
            'Muy Vencido': f"{d.get('Muy Vencido', 0):.1f}%",
            'Estado Común': d['Estado_Comun'],
            'Recomendación': rec
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

def mostrar_estadistica_sumas(df_stats):
    st.markdown("---")
    st.subheader("📈 Estadística de Sumas (Completa)")
    if df_stats.empty:
        st.info("ℹ️ Sin datos estadísticos para esta selección.")
        return
    cols = ['Suma', 'Frecuencia', 'Veces Normal', 'Veces Vencido', 'Veces Muy Vencido', 
            'Estado Actual', 'Estabilidad', 'Tiempo Limite (P75)', 'Alerta', 'Gap Actual', 'Exceso Ultima Salida']
    st.dataframe(df_stats[cols].sort_values('Frecuencia', ascending=False), hide_index=True, use_container_width=True)

def mostrar_historial_sumas(df_data):
    st.markdown("---")
    st.subheader("📜 Historial de Salidas (Reciente a Antiguo)")
    if df_data.empty:
        st.info("ℹ️ Sin historial disponible.")
        return
    df_hist = df_data[['Fecha', 'Tipo_Sorteo', 'Fijo', 'Suma']].copy()
    df_hist = df_hist.sort_values('Fecha', ascending=False).reset_index(drop=True)
    df_hist['Tipo_Sorteo'] = df_hist['Tipo_Sorteo'].map({'M': 'Mañana', 'T': 'Tarde', 'N': 'Noche'})
    st.dataframe(df_hist.head(30), hide_index=True, use_container_width=True)

def mostrar_señales_juego(df_stats):
    if df_stats.empty: return
    df_senales = df_stats[
        (df_stats['Alerta'] == '⚠️ RECUPERAR') &
        (df_stats['Estabilidad'] >= 60) &
        (df_stats['Estado Actual'].isin(['Vencido', 'Muy Vencido']))
    ].copy()
    
    if not df_senales.empty:
        st.markdown("---")
        st.subheader("🎯 Señales de Juego (Auto-Detectadas)")
        df_senales['Exceso Días'] = (df_senales['Gap Actual'] - df_senales['Tiempo Limite (P75)']).astype(int)
        df_senales['Prioridad'] = df_senales['Exceso Días'].apply(lambda x: "🔴 Crítica" if x > 5 else ("🟠 Alta" if x > 2 else "🟡 Media"))
        
        df_con = df_senales[df_senales['Porc_Comun'] >= 60]
        df_sin = df_senales[df_senales['Porc_Comun'] < 60]
        
        if not df_con.empty:
            st.success(f"🔥 **{len(df_con)} suma(s) con VENTAJA ESTADÍSTICA (Patrón ≥60%)**")
            st.dataframe(df_con[['Suma', 'Estado Actual', 'Estado_Comun', 'Porc_Comun', 'Exceso Días', 'Prioridad']], hide_index=True, use_container_width=True)
        if not df_sin.empty:
            st.info(f"🟡 **{len(df_sin)} suma(s) con timing favorable pero SIN patrón claro (<60%)**")
            st.caption("⚠️ Estos perfiles tienen alta estabilidad pero sin Estado Común ≥60%. Usar solo como respaldo con gestión de riesgo.")
            st.dataframe(df_sin[['Suma', 'Estado Actual', 'Estado_Comun', 'Porc_Comun', 'Exceso Días', 'Prioridad']], hide_index=True, use_container_width=True)

def mostrar_alertas_detalladas(df_stats, distribuciones):
    if df_stats.empty: return
    df_alertas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR'].copy()
    if df_alertas.empty:
        st.info("✅ No hay alertas activas en este momento.")
        return

    st.markdown("---")
    st.subheader("🚨 Detalle de Alertas Activas")
    for _, row in df_alertas.iterrows():
        s = row['Suma']
        gap = row['Gap Actual']
        limite = row['Tiempo Limite (P75)']
        estado = row['Estado Actual']
        
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
            
        dist = distribuciones.get(s, {})
        dist_str = " | ".join([f"{k}: {v:.1f}%" for k, v in dist.items() if k in ['Normal', 'Vencido', 'Muy Vencido']])
        combinaciones = obtener_combinaciones_suma(s)
        
        with st.container(border=True):
            st.markdown(f"**🔢 Suma `{s:02d}` | Estado: `{estado}` | ⏳ {time_str}**")
            st.markdown(f"📉 **Estabilidad Actual:** `{row['Estabilidad']}%` | 🔔 **Alerta:** ⚠️ RECUPERAR")
            st.markdown(f"📊 **Datos:** Gap Actual: **{gap} días** | Límite P75: **{limite} días**")
            st.markdown(f"📈 **Distribución Histórica:** {dist_str}")
            
            st.markdown(f"🔗 **Combinaciones de la Suma {s}:** `{' | '.join(combinaciones)}`")
            
            if row['Estado_Comun'] != 'Ninguno':
                st.success(f"✅ **Patrón Claro:** Suele salir en estado `{row['Estado_Comun']}` ({row['Porc_Comun']:.1f}% histórico)")
            else:
                st.warning("⚠️ Sin patrón dominante histórico (<60%)")
            
            st.info("🔙 **Historia de la última salida de esta suma:**")
            st.markdown(f"- Estado en ese momento: **{row['Estado Ultima Salida']}**")
            st.markdown(f"- Estabilidad previa: **{row['Estabilidad']}%**")
            if row['Exceso Ultima Salida'] > 0:
                st.markdown(f"- Días en exceso: **+{row['Exceso Ultima Salida']} días**")
            st.markdown("---")

def mostrar_ultimos_resultados_sidebar(df_full):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Últimos Resultados + Suma")
    if df_full.empty:
        st.sidebar.info("ℹ️ Sin datos")
        return
    
    for letra, nombre in [('M', '🌅 Mañana'), ('T', '☀️ Tarde'), ('N', '🌙 Noche')]:
        df_sesion = df_full[df_full['Tipo_Sorteo'] == letra]
        if not df_sesion.empty:
            ultimo = df_sesion.sort_values('Fecha', ascending=False).iloc[0]
            num = int(ultimo['Fijo'])
            suma = obtener_suma(num)
            fecha = ultimo['Fecha'].strftime('%d/%m/%Y') if pd.notna(ultimo['Fecha']) else 'N/A'
            st.sidebar.markdown(f"**{nombre}**")
            st.sidebar.markdown(f"📅 {fecha}")
            st.sidebar.markdown(f"🔢 `{num:02d}` → ➕ `{suma}`")
            st.sidebar.markdown("---")

def ordenador_numeros(df_stats, distribuciones):
    st.markdown("---")
    st.subheader("🔢 Ordenador de Números por Suma")
    st.caption("Ingresa números separados por comas o espacios. La app los ordenará según la presión estadística de su suma.")
    
    df_stats_activo = df_stats if 'df_stats' in locals() else st.session_state.get('df_stats_sumas')
    distribuciones_activo = distribuciones if 'distribuciones' in locals() else st.session_state.get('distribuciones_sumas')
    
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
                s = obtener_suma(num)
                row = df_stats_activo[df_stats_activo['Suma'] == s]
                if row.empty: continue
                row = row.iloc[0]
                
                score = 0
                if row['Porc_Comun'] >= 60: score += 100
                elif row['Porc_Comun'] >= 50: score += 70
                else: score += 40
                
                score += min(max((row['Gap Actual'] - row['Tiempo Limite (P75)']), 0) * 3, 30)
                score += row['Estabilidad'] * 0.5
                if row['Alerta'] == '⚠️ RECUPERAR': score += 20
                
                resultados.append({
                    'Número': f"{num:02d}",
                    'Suma': s,
                    'Score': round(score, 1),
                    'Estado Suma': row['Estado Actual'],
                    'Estado Común': row['Estado_Comun'],
                    'Prioridad': "🔴 Crítica" if (row['Gap Actual'] - row['Tiempo Limite (P75)']) > 5 else "🟠 Alta"
                })
                
            resultados.sort(key=lambda x: x['Score'], reverse=True)
            st.success(f"✅ Ordenados {len(resultados)} números.")
            st.dataframe(pd.DataFrame(resultados), hide_index=True, use_container_width=True)
            
            if len(resultados) >= 2:
                st.info(f"🎯 **Recomendación:** Juega `{resultados[0]['Número']}` y `{resultados[1]['Número']}`")
    else:
        st.info("ℹ️ Ejecuta el análisis primero para activar el ordenador de números.")
# =============================================================================
# MAIN
# =============================================================================
def main():
    st.sidebar.header("⚙️ Configuración Georgia")
    
    # 🔄 BOTÓN DE FORZAR RECARGA
    if st.sidebar.button("🔄 Forzar Recarga / Actualizar CSV", type="primary", use_container_width=True):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    modo_sesion = st.sidebar.radio("Sesión de Análisis:", ["General", "Mañana", "Tarde", "Noche"], key="radio_sesion")
    
    # 1. Cargamos TODOS los datos primero (df_full contiene M, T y N)
    df_todo, df_full = cargar_datos_geotodo(RUTA_CSV)
    st.session_state.df_full_cache = df_full
    
    # 2. Mostramos la barra lateral con los últimos resultados de TODAS las sesiones
    mostrar_ultimos_resultados_sidebar(df_full) 
    
    if df_todo.empty:
        st.warning("⚠️ Archivo vacío o sin datos válidos. Agrega sorteos a `Geotodo.csv`.")
        return

    # 3. Filtrado estricto según la selección del usuario
    if modo_sesion == "Mañana":
        df_analisis = df_todo[df_todo['Tipo_Sorteo'] == 'M'].copy()
    elif modo_sesion == "Tarde":
        df_analisis = df_todo[df_todo['Tipo_Sorteo'] == 'T'].copy()
    elif modo_sesion == "Noche":
        df_analisis = df_todo[df_todo['Tipo_Sorteo'] == 'N'].copy()
    else:
        df_analisis = df_todo.copy()
        
    fecha_ref = datetime.now()
    st.info(f"📊 Analizando **{len(df_analisis)}** sorteos para la sesión: **{modo_sesion}**")

    # 4. Botón de ejecución
    if st.button("🚀 Ejecutar Análisis de Sumas", type="primary", key="btn_analisis"):
        if len(df_analisis) == 0:
            st.error(f"❌ No hay datos para '{modo_sesion}'. Verifica que el CSV tenga sesiones válidas (M/T/N).")
            return

        with st.spinner("Calculando Gaps, P75 y Estados..."):
            # 🔥 IMPORTANTE: Pasamos 'df_analisis' (el filtrado) al motor de estadísticas
            df_stats, distribuciones = analizar_estadisticas_sumas(df_analisis, fecha_ref)
            
            # 🧠 Guardar en memoria para persistencia (evita que se cierre al ordenar números)
            st.session_state.df_stats_sumas = df_stats
            st.session_state.distribuciones_sumas = distribuciones
            st.session_state.df_analisis_cache = df_analisis
            
            # Renderizado de tablas
            mostrar_tabla_comportamiento(distribuciones)
            mostrar_estadistica_sumas(df_stats)
            mostrar_historial_sumas(df_analisis)
            mostrar_señales_juego(df_stats)
            mostrar_alertas_detalladas(df_stats, distribuciones)
            ordenador_numeros(df_stats, distribuciones)

    # 5. Recuperación de estado (si la app se reinicia tras usar el ordenador de números)
    elif st.session_state.df_stats_sumas is not None:
        df_stats = st.session_state.df_stats_sumas
        distribuciones = st.session_state.distribuciones_sumas
        df_analisis = st.session_state.get('df_analisis_cache', df_todo)
        
        mostrar_tabla_comportamiento(distribuciones)
        mostrar_estadistica_sumas(df_stats)
        mostrar_historial_sumas(df_analisis)
        mostrar_señales_juego(df_stats)
        mostrar_alertas_detalladas(df_stats, distribuciones)
        ordenador_numeros(df_stats, distribuciones)
    else:
        st.info("👈 Presiona 'Ejecutar Análisis de Sumas' para comenzar.")

if __name__ == "__main__":
    main()