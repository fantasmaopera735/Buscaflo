import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Flotodo", page_icon="🌺", layout="wide")
st.title("🌺 Análisis Inteligente - Florida (Tarde / Noche)")

ARCHIVO_CSV = "Flotodo.csv"

# ==========================================
# 📂 CARGA ROBUSTA & LIMPIEZA
# ==========================================
@st.cache_data(ttl=300)
def cargar_flotodo(ruta):
    try:
        # Leer con ; y codificación segura
        df = pd.read_csv(ruta, sep=';', encoding='utf-8-sig')
        
        # Normalizar nombres de columnas según tu CSV real
        # Tu CSV: Fecha;Tarde/Noche;Centena;Fijo;1er Corrido;2do Corrido
        col_map = {
            'Fecha': 'Fecha',
            'Tarde/Noche': 'Tipo_Sorteo',
            'Fijo': 'Fijo'
        }
        # Detectar columnas aunque tengan espacios o mayúsculas distintas
        for orig, std in col_map.items():
            match = [c for c in df.columns if orig.lower() in c.lower().replace(' ', '')]
            if match: df = df.rename(columns={match[0]: std})
            
        # Conservar solo las necesarias
        if not all(c in df.columns for c in ['Fecha', 'Tipo_Sorteo', 'Fijo']):
            return None, "❌ El CSV no tiene columnas Fecha, Tarde/Noche o Fijo."
            
        df = df[['Fecha', 'Tipo_Sorteo', 'Fijo']].copy()
        
        # Convertir y limpiar
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df['Fijo'] = pd.to_numeric(df['Fijo'], errors='coerce') % 100
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
        
        # Filtrar solo T y N válidos
        df = df[(df['Tipo_Sorteo'].isin(['T', 'N'])) & (~df['Fecha'].isna()) & (~df['Fijo'].isna())]
        df = df.sort_values('Fecha').reset_index(drop=True)
        
        return df, None
    except Exception as e:
        return None, f"❌ Error leyendo CSV: {e}"

df, error = cargar_flotodo(ARCHIVO_CSV)
if error:
    st.error(error)
    st.stop()

# ==========================================
# 📋 SIDEBAR: ÚLTIMOS RESULTADOS
# ==========================================
st.sidebar.subheader("📋 Últimos Resultados en CSV")
if not df.empty:
    df_t = df[df['Tipo_Sorteo'] == 'T'].tail(1)
    df_n = df[df['Tipo_Sorteo'] == 'N'].tail(1)
    
    if not df_t.empty:
        st.sidebar.markdown(f"**☀️ Tarde:** `{int(df_t['Fijo'].iloc[0]):02d}` ({df_t['Fecha'].iloc[0].strftime('%d/%m')})")
    else:
        st.sidebar.info("☀️ Tarde: Sin datos")
        
    if not df_n.empty:
        st.sidebar.markdown(f"**🌙 Noche:** `{int(df_n['Fijo'].iloc[0]):02d}` ({df_n['Fecha'].iloc[0].strftime('%d/%m')})")
    else:
        st.sidebar.info("🌙 Noche: Sin datos")
else:
    st.sidebar.warning("⚠️ CSV vacío")

# ==========================================
# 🎛️ FILTRO POR SESIÓN
# ==========================================
modo = st.sidebar.radio("🔍 Filtro de análisis:", ["General (T+N)", "Tarde (T)", "Noche (N)"], index=0)

if modo == "Tarde (T)":
    dfa = df[df['Tipo_Sorteo'] == 'T'].copy()
elif modo == "Noche (N)":
    dfa = df[df['Tipo_Sorteo'] == 'N'].copy()
else:
    dfa = df.copy()

if dfa.empty:
    st.warning(f"⚠️ No hay datos para la sesión: {modo}")
    st.stop()

# ==========================================
# 🧠 LÓGICA INTELIGENTE DE PRÓXIMO SORTEO
# ==========================================
max_fecha = df['Fecha'].max()
ultimos_dia = df[df['Fecha'] == max_fecha]
ultima_sesion = ultimos_dia['Tipo_Sorteo'].iloc[-1] if not ultimos_dia.empty else None

if ultima_sesion == 'T':
    proxima_fecha = max_fecha          # Misma fecha
    proxima_sesion = 'N'
    mensaje = f"Próximo sorteo: **Noche del {proxima_fecha.strftime('%d/%m/%Y')}**"
else:
    proxima_fecha = max_fecha + timedelta(days=1)  # Siguiente día
    proxima_sesion = 'T'
    mensaje = f"Próximo sorteo: **Tarde del {proxima_fecha.strftime('%d/%m/%Y')}**"

st.success(f"✅ Última fecha registrada: `{max_fecha.strftime('%d/%m/%Y')}` | Sesión: **{modo}**")
st.info(f"🎯 {mensaje}")

# ==========================================
# 📊 CÁLCULO DE PUNTUACIÓN (Gap, Tendencia, Frecuencia)
# ==========================================
dfa['Decena'] = (dfa['Fijo'] // 10).astype(int)
dfa['Terminacion'] = (dfa['Fijo'] % 10).astype(int)
dfa['DiaSemana'] = dfa['Fecha'].dt.day_name()

resultados = []
dia_objetivo_nombre = proxima_fecha.day_name()

for n in range(0, 100):
    dec, ter = n // 10, n % 10
    
    # 1. Gap desde última aparición en la sesión analizada
    apariciones = dfa[dfa['Fijo'] == n]['Fecha']
    gap = (max_fecha - apariciones.max()).days if not apariciones.empty else 999
    
    # 2. Tendencia escalonada (últimas 3 decenas)
    ultimas_dec = dfa.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1: tendencia = 15
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1: tendencia = 15
            
    # 3. Frecuencia en este día de la semana (últimos 6 meses)
    hace_180 = max_fecha - timedelta(days=180)
    freq_dia = len(dfa[(dfa['Fecha'] >= hace_180) & (dfa['DiaSemana'] == dia_objetivo_nombre) & (dfa['Fijo'] == n)])
    
    # 4. Fórmula de Puntuación
    pts = 0
    if 20 <= gap <= 45: pts += 20
    elif gap > 45: pts += 10
    pts += tendencia
    if freq_dia >= 3: pts += 12
    if 8 <= (dec + ter) <= 14: pts += 5
    
    resultados.append({
        'Numero': n, 'Decena': dec, 'Terminacion': ter, 
        'Gap': gap, 'Tendencia': tendencia, 'Freq_Dia': freq_dia, 'Puntaje': pts
    })

df_scores = pd.DataFrame(resultados)
df_scores['Estado'] = df_scores['Puntaje'].apply(
    lambda x: '🔴 JUGAR' if x >= 50 else ('🟡 OBSERVAR' if x >= 35 else '⚪ ESPERAR')
)
df_scores = df_scores.sort_values('Puntaje', ascending=False).reset_index(drop=True)

# ==========================================
# 💾 RENDERIZADO
# ==========================================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🏆 Top 10 Proyección")
    st.dataframe(df_scores.head(10), use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Resumen")
    st.metric("🔴 JUGAR", len(df_scores[df_scores['Estado']=='🔴 JUGAR']))
    st.metric("🟡 OBSERVAR", len(df_scores[df_scores['Estado']=='🟡 OBSERVAR']))
    st.metric("📅 Próximo", f"{dia_objetivo_nombre} {proxima_fecha.strftime('%d/%m')}")

if st.button("📥 Descargar Excel con análisis", use_container_width=True):
    df_scores.to_excel('Prediccion_Flotodo.xlsx', index=False)
    st.success("✅ `Prediccion_Flotodo.xlsx` generado en la carpeta raíz.")