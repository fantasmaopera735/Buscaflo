import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Inteligente FL Tarde", layout="wide")
st.title("🧠 Análisis Inteligente - Florida Tarde")

# ==========================================
# 📂 CONFIGURACIÓN
# ==========================================
ARCHIVO_ENTRADA = 'david esgrima.xlsx'
HOJA_OBJETIVO = 'TARDE 1 (1)'

# ==========================================
# 🔍 1. CARGA SEGURA CON DEPURACIÓN
# ==========================================
@st.cache_data
def cargar_datos():
    try:
        df_raw = pd.read_excel(ARCHIVO_ENTRADA, sheet_name=HOJA_OBJETIVO, header=None)
        st.success(f"✅ Archivo cargado. Forma: {df_raw.shape}")
        return df_raw
    except Exception as e:
        st.error(f"❌ No se pudo leer el Excel: {e}")
        return None

df_raw = cargar_datos()
if df_raw is None:
    st.stop()

# ==========================================
# 🧩 2. PARSER ROBUSTO (Busca fecha + número en cualquier celda)
# ==========================================
datos_limpios = []
rows, cols = df_raw.shape

# Recorremos todo el grid buscando pares fecha-número
for r in range(rows):
    for c in range(cols):
        val = str(df_raw.iloc[r, c]).strip()
        # 1. Buscar fecha DD/MM/AAAA o DD-MM-AAAA
        fecha_match = re.search(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})', val)
        if fecha_match:
            fecha_str = fecha_match.group(1).replace('-', '/').replace('.', '/')
            try:
                fecha = pd.to_datetime(fecha_str, format='%d/%m/%Y')
            except:
                try:
                    fecha = pd.to_datetime(fecha_str, format='%d/%m/%y')
                except:
                    continue

            # 2. Buscar número en celdas cercanas (misma fila, siguiente columna o fila siguiente)
            candidatos = []
            if c + 1 < cols: candidatos.append(str(df_raw.iloc[r, c+1]).strip())
            if r + 1 < rows: candidatos.append(str(df_raw.iloc[r+1, c]).strip())
            
            for cand in candidatos:
                cand = cand.upper().replace('O', '0').replace(' ', '').replace('ERROR:#N/A', '')
                if cand.isdigit() and len(cand) >= 2:
                    num = int(cand[-2:]) # Tu regla: coger últimos 2 dígitos
                    datos_limpios.append({'Fecha': fecha, 'Numero': num})
                    break # Ya encontramos el número para esta fecha, pasar a la siguiente

df = pd.DataFrame(datos_limpios).drop_duplicates(subset='Fecha').sort_values('Fecha').reset_index(drop=True)

if df.empty:
    st.warning("⚠️ No se encontraron pares fecha-número válidos. Revisa la estructura del Excel.")
    with st.expander("👁️ Ver primeras celdas del Excel para depurar"):
        st.dataframe(df_raw.head(5), use_container_width=True)
    st.stop()

st.success(f"📊 {len(df)} sorteos válidos detectados.")

# ==========================================
# 📈 3. CÁLCULO DE MÉTRICAS Y PROYECCIÓN
# ==========================================
df['Decena'] = (df['Numero'] // 10).astype(int)
df['Terminacion'] = (df['Numero'] % 10).astype(int)
df['Suma'] = df['Numero'].apply(lambda x: (x//10) + (x%10))
df['DiaSemana'] = df['Fecha'].dt.day_name()

fecha_base = df['Fecha'].max()
dia_objetivo = fecha_base + timedelta(days=1)
nombre_dia = dia_objetivo.day_name()

st.info(f"📅 Última fecha en datos: `{fecha_base.strftime('%d/%m/%Y')}` → Proyectando para: **{dia_objetivo.strftime('%d/%m/%Y')} ({nombre_dia})**")

# ==========================================
# 🧠 4. SISTEMA DE PUNTOS (TU METODOLOGÍA)
# ==========================================
resultados = []
for n in range(0, 100):
    dec = n // 10
    ter = n % 10
    
    apariciones = df[df['Numero'] == n]['Fecha']
    gap = (fecha_base - apariciones.max()).days if not apariciones.empty else 999
    
    # Tendencia escalonada simple
    ultimas_dec = df.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1:
            tendencia = 15
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1:
            tendencia = 15
            
    # Frecuencia en este día (últimos 6 meses)
    hace_180 = fecha_base - timedelta(days=180)
    freq_dia = len(df[(df['Fecha'] >= hace_180) & (df['DiaSemana'] == nombre_dia) & (df['Numero'] == n)])
    
    # Fórmula
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
# 💾 5. RENDERIZADO EN STREAMLIT
# ==========================================
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🏆 Top 10 Proyección")
    st.dataframe(df_scores.head(10), use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Resumen")
    st.metric("🔴 JUGAR", len(df_scores[df_scores['Estado']=='🔴 JUGAR']))
    st.metric("🟡 OBSERVAR", len(df_scores[df_scores['Estado']=='🟡 OBSERVAR']))
    st.metric("📅 Proyección", f"{dia_objetivo.strftime('%A %d/%m')}")

st.divider()
if st.button("📥 Descargar Excel con análisis completo"):
    df_scores.to_excel('Prediccion_Inteligente_FL_Tarde.xlsx', index=False)
    st.success("✅ `Prediccion_Inteligente_FL_Tarde.xlsx` generado en la carpeta raíz.")