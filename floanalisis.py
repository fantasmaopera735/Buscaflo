import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Flotodo", page_icon="🌺", layout="wide")
st.title("🌺 Análisis Inteligente - Florida (Tarde / Noche)")

# ==========================================
# 📂 CARGA ROBUSTA (Inspirada en tu Flodigito.py)
# ==========================================
ARCHIVO_CSV = "Flotodo.csv"

@st.cache_data(ttl=300)
def cargar_flotodo_robusto(ruta):
    if not os.path.exists(ruta):
        return None, f"❌ No se encontró `{ruta}` en la carpeta."

    try:
        # 1. Detectar separador leyendo la primera línea
        with open(ruta, 'r', encoding='latin-1') as f:
            primera = f.readline()
        sep = ';' if ';' in primera else (',' if ',' in primera else '\t')

        # 2. Cargar como string para evitar errores de parsing
        df = pd.read_csv(ruta, sep=sep, encoding='latin-1', header=0, dtype=str, on_bad_lines='skip')
        
        # 3. Limpiar nombres de columnas (espacios, BOM, etc.)
        df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
        
    except Exception as e:
        return None, f"❌ Error leyendo CSV: {e}"

    # 4. Mapeo flexible de columnas
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace(' ', '')
        if 'fecha' in cl: col_map[col] = 'Fecha'
        elif cl in ['fijo', 'numero', 'resultado', 'fijo2']: col_map[col] = 'Fijo'
        elif cl in ['tipo', 'tipo_sorteo', 'sesion', 'sorteo', 'tardenoche']: col_map[col] = 'Tipo_Sorteo'
    df = df.rename(columns=col_map)

    if 'Fecha' not in df.columns or 'Fijo' not in df.columns:
        return None, "❌ El CSV debe contener columnas 'Fecha' y 'Fijo'"

    # 5. Limpieza y conversión
    df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
    df['Fijo'] = pd.to_numeric(df['Fijo'], errors='coerce') % 100

    # 6. Normalizar Tipo_Sorteo
    if 'Tipo_Sorteo' in df.columns:
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].astype(str).str.strip().str.upper()
        mapeo = {'TARDE': 'T', 'AFTERNOON': 'T', 'T': 'T', 'NOCHE': 'N', 'NIGHT': 'N', 'N': 'N', 'MAÑANA': 'M', 'M': 'M'}
        df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map(mapeo).fillna(df['Tipo_Sorteo'])
    else:
        df['Tipo_Sorteo'] = 'T'

    # 7. Filtrar solo T y N (como solicitaste)
    df_valido = df[df['Tipo_Sorteo'].isin(['T', 'N'])].copy()
    df_valido = df_valido.dropna(subset=['Fecha', 'Fijo']).sort_values('Fecha').reset_index(drop=True)

    return df_valido, None

# ==========================================
# 🚀 EJECUCIÓN
# ==========================================
df, error = cargar_flotodo_robusto(ARCHIVO_CSV)
if error:
    st.error(error)
    st.stop()

st.sidebar.success(f"✅ `{ARCHIVO_CSV}` cargado: `{len(df)}` sorteos válidos (T/N)")

# Filtro por sesión
modo = st.sidebar.radio("🔍 Filtro de sesión:", ["General (T+N)", "Tarde (T)", "Noche (N)"], index=0)
if modo == "Tarde (T)": dfa = df[df['Tipo_Sorteo'] == 'T'].copy()
elif modo == "Noche (N)": dfa = df[df['Tipo_Sorteo'] == 'N'].copy()
else: dfa = df.copy()

if dfa.empty:
    st.warning(f"⚠️ No hay datos para: {modo}")
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
    
    # 1. Gap
    apariciones = dfa[dfa['Fijo'] == n]['Fecha']
    gap = (fecha_base - apariciones.max()).days if not apariciones.empty else 999
    
    # 2. Tendencia escalonada
    ultimas_dec = dfa.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1: tendencia = 15
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1: tendencia = 15
            
    # 3. Frecuencia en día objetivo (6 meses)
    hace_180 = fecha_base - timedelta(days=180)
    freq_dia = len(dfa[(dfa['Fecha'] >= hace_180) & (dfa['DiaSemana'] == dia_objetivo) & (dfa['Fijo'] == n)])
    
    # 4. Puntuación
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
    st.success("✅ `Prediccion_Flotodo.xlsx` generado.")