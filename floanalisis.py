import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Flotodo (T/N)", page_icon="🌺", layout="wide")
st.title("🌺 Análisis Inteligente - Florida (Tarde / Noche)")

# ==========================================
# 📂 CARGA DE DATOS (Flotodo.csv)
# ==========================================
st.sidebar.header("📁 Carga de Datos")
RUTA_CSV = st.sidebar.text_input("Nombre del archivo CSV:", value="Flotodo.csv")

df = pd.DataFrame()
try:
    # Carga con separador ; y detección automática de fechas DD/MM/AAAA
    df_raw = pd.read_csv(RUTA_CSV, sep=';', parse_dates=['Fecha'], dayfirst=True)
    
    # Limpieza y estandarización de columnas
    df_raw = df_raw.dropna(subset=['Fecha'])
    df_raw['Tipo_Sorteo'] = df_raw['Tipo_Sorteo'].astype(str).str.strip().str.upper()
    
    # Validación: solo permitir T y N
    sesiones_validas = df_raw['Tipo_Sorteo'].isin(['T', 'N'])
    df_valido = df_raw[sesiones_validas].copy()
    df_invalido = df_raw[~sesiones_validas]
    
    if not df_invalido.empty:
        st.sidebar.warning(f"⚠️ Se ignoraron {len(df_invalido)} filas con sesiones distintas a T/N")
        
    # Extraer Fijo como número de 2 dígitos
    df_valido['Fijo'] = pd.to_numeric(df_valido['Fijo'], errors='coerce') % 100
    df = df_valido.dropna(subset=['Fijo']).sort_values('Fecha').reset_index(drop=True)
    
    st.sidebar.success(f"✅ {len(df)} sorteos cargados (T/N)")
    
except Exception as e:
    st.sidebar.error(f"❌ Error cargando CSV: {e}\nVerifica que '{RUTA_CSV}' esté en la misma carpeta.")
    st.stop()

if df.empty:
    st.warning("No hay datos válidos después de filtrar. Revisa el archivo CSV.")
    st.stop()

# ==========================================
# 🎛️ FILTRO POR SESIÓN
# ==========================================
modo = st.sidebar.radio("🔍 Filtro de sesión:", ["General (T+N)", "Tarde (T)", "Noche (N)"], index=0)

if modo == "Tarde (T)":
    dfa = df[df['Tipo_Sorteo'] == 'T'].copy()
elif modo == "Noche (N)":
    dfa = df[df['Tipo_Sorteo'] == 'N'].copy()
else:
    dfa = df.copy()

if dfa.empty:
    st.warning(f"⚠️ No hay datos disponibles para: {modo}")
    st.stop()

# ==========================================
# 📊 ANÁLISIS INTELIGENTE (Gap + Frecuencia + Puntuación)
# ==========================================
fecha_base = dfa['Fecha'].max()
dfa['Decena'] = (dfa['Fijo'] // 10).astype(int)
dfa['Terminacion'] = (dfa['Fijo'] % 10).astype(int)
dfa['DiaSemana'] = dfa['Fecha'].dt.day_name()

resultados = []
for n in range(0, 100):
    dec, ter = n // 10, n % 10
    
    # 1. Separación (Gap) desde última aparición en la sesión seleccionada
    apariciones = dfa[dfa['Fijo'] == n]['Fecha']
    gap = (fecha_base - apariciones.max()).days if not apariciones.empty else 999
    
    # 2. Tendencia escalonada (últimas 3 decenas)
    ultimas_dec = dfa.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1: tendencia = 15
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1: tendencia = 15
            
    # 3. Frecuencia histórica en este día de la semana (últimos 6 meses)
    hace_180 = fecha_base - timedelta(days=180)
    freq_dia = len(dfa[(dfa['Fecha'] >= hace_180) & (dfa['DiaSemana'] == datetime.now().day_name()) & (dfa['Fijo'] == n)])
    
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
# 💾 RENDERIZADO EN STREAMLIT
# ==========================================
st.success(f"📅 Última fecha analizada: `{fecha_base.strftime('%d/%m/%Y')}` | Sesión: **{modo}**")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🏆 Top 10 Proyección")
    st.dataframe(df_scores.head(10), use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Resumen")
    st.metric("🔴 JUGAR", len(df_scores[df_scores['Estado']=='🔴 JUGAR']))
    st.metric("🟡 OBSERVAR", len(df_scores[df_scores['Estado']=='🟡 OBSERVAR']))
    st.metric("📅 Próximo", (fecha_base + timedelta(days=1)).strftime('%A %d/%m'))

st.divider()
if st.button("📥 Descargar Excel con análisis completo", use_container_width=True):
    df_scores.to_excel('Prediccion_Flotodo.xlsx', index=False)
    st.success("✅ `Prediccion_Flotodo.xlsx` generado.")