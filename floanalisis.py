import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Flotodo", page_icon="🌺", layout="wide")
st.title("🌺 Análisis Inteligente - Florida (Tarde / Noche)")

# ==========================================
# 📂 CARGA AUTOMÁTICA DEL ARCHIVO
# ==========================================
ARCHIVO_CSV = "Flotodo.csv"

try:
    # header=None + names=[] evita KeyError cuando el CSV no tiene fila de títulos
    df_raw = pd.read_csv(
        ARCHIVO_CSV, 
        sep=';', 
        encoding='utf-8-sig',
        header=None,
        names=['Fecha', 'Tipo_Sorteo', 'Centena', 'Fijo', '1er_Corrido', '2do_Corrido'],
        dayfirst=True
    )
except FileNotFoundError:
    st.error(f"❌ No se encontró `{ARCHIVO_CSV}` en el repositorio.")
    st.info("💡 Asegúrate de que el archivo esté en la raíz del repo y se llame exactamente `Flotodo.csv`")
    st.stop()
except Exception as e:
    st.error(f"❌ Error leyendo el CSV: {e}")
    st.stop()

# ==========================================
# 🧹 LIMPIEZA Y VALIDACIÓN
# ==========================================
df_raw['Tipo_Sorteo'] = df_raw['Tipo_Sorteo'].astype(str).str.strip().str.upper()
df_valido = df_raw[df_raw['Tipo_Sorteo'].isin(['T', 'N'])].copy()

if df_valido.empty:
    st.error("⚠️ No hay sorteos de Tarde (T) o Noche (N) en el archivo.")
    st.stop()

df_valido['Fijo'] = pd.to_numeric(df_valido['Fijo'], errors='coerce') % 100
df_valido = df_valido.dropna(subset=['Fecha', 'Fijo']).sort_values('Fecha').reset_index(drop=True)

st.sidebar.success(f"✅ `{ARCHIVO_CSV}` cargado: `{len(df_valido)}` sorteos válidos")

# ==========================================
# 🎛️ FILTRO POR SESIÓN
# ==========================================
modo = st.sidebar.radio("🔍 Filtro de sesión:", ["General (T+N)", "Tarde (T)", "Noche (N)"], index=0)

if modo == "Tarde (T)":
    dfa = df_valido[df_valido['Tipo_Sorteo'] == 'T'].copy()
elif modo == "Noche (N)":
    dfa = df_valido[df_valido['Tipo_Sorteo'] == 'N'].copy()
else:
    dfa = df_valido.copy()

if dfa.empty:
    st.warning(f"⚠️ No hay datos disponibles para: {modo}")
    st.stop()

# ==========================================
# 🧠 CÁLCULO INTELIGENTE
# ==========================================
fecha_base = dfa['Fecha'].max()
dfa['Decena'] = (dfa['Fijo'] // 10).astype(int)
dfa['Terminacion'] = (dfa['Fijo'] % 10).astype(int)
dfa['DiaSemana'] = dfa['Fecha'].dt.day_name()

dia_objetivo = (fecha_base + timedelta(days=1)).day_name()

resultados = []
for n in range(0, 100):
    dec, ter = n // 10, n % 10
    
    # 1. Separación (Gap) desde última aparición
    apariciones = dfa[dfa['Fijo'] == n]['Fecha']
    gap = (fecha_base - apariciones.max()).days if not apariciones.empty else 999
    
    # 2. Tendencia escalonada (últimas 3 decenas)
    ultimas_dec = dfa.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1: tendencia = 15
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1: tendencia = 15
            
    # 3. Frecuencia en el día objetivo (últimos 6 meses)
    hace_180 = fecha_base - timedelta(days=180)
    freq_dia = len(dfa[(dfa['Fecha'] >= hace_180) & (dfa['DiaSemana'] == dia_objetivo) & (dfa['Fijo'] == n)])
    
    # 4. Fórmula de Puntuación
    pts = 0
    if 20 <= gap <= 45: pts += 20      # Zona dulce
    elif gap > 45: pts += 10           # Presión alta
    pts += tendencia                   # Tendencia
    if freq_dia >= 3: pts += 12        # Día fuerte
    if 8 <= (dec + ter) <= 14: pts += 5 # Suma frecuente
    
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
st.success(f"📅 Última fecha: `{fecha_base.strftime('%d/%m/%Y')}` | Sesión: **{modo}**")
st.info(f"🎯 Próximo sorteo: **{(fecha_base + timedelta(days=1)).strftime('%A %d/%m/%Y')}**")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🏆 Top 10 Proyección")
    st.dataframe(df_scores.head(10), use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Resumen")
    st.metric("🔴 JUGAR", len(df_scores[df_scores['Estado']=='🔴 JUGAR']))
    st.metric("🟡 OBSERVAR", len(df_scores[df_scores['Estado']=='🟡 OBSERVAR']))
    st.metric("📅 Próximo", (fecha_base + timedelta(days=1)).strftime('%A %d/%m'))

if st.button("📥 Descargar Excel con análisis", use_container_width=True):
    df_scores.to_excel('Prediccion_Flotodo.xlsx', index=False)
    st.success("✅ `Prediccion_Flotodo.xlsx` generado en la carpeta raíz.")