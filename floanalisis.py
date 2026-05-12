import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Análisis Inteligente FL Tarde", layout="wide")
st.title("🧠 Análisis Inteligente - Florida Tarde")

# ==========================================
# 🔍 CARGA INTELIGENTE (Busca SOLO celdas ROJAS)
# ==========================================
ARCHIVO_ENTRADA = 'david esgrima.xlsx'
HOJA_OBJETIVO = 'TARDE 1 (1)'

@st.cache_data(ttl=3600)
def cargar_datos_desde_excel():
    try:
        wb = load_workbook(ARCHIVO_ENTRADA, data_only=True)
        # Si la hoja no existe, toma la primera
        ws = wb[HOJA_OBJETIVO] if HOJA_OBJETIVO in wb.sheetnames else wb.active
        
        fecha_regex = re.compile(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})')
        resultados = []
        
        # Escanea toda la hoja buscando celdas con relleno rojo y fuente blanca
        for row in ws.iter_rows():
            for cell in row:
                # Verificar color de relleno (rojo)
                fill_rgb = getattr(cell.fill.start_color, 'rgb', None)
                is_red = False
                if fill_rgb:
                    rgb_str = str(fill_rgb).upper()
                    # OpenPyXL guarda rojo como FFFF0000 o similar
                    if 'FF0000' in rgb_str or rgb_str.endswith('FF0000'):
                        is_red = True
                        
                # Verificar color de fuente (blanca)
                font_rgb = getattr(cell.font.color, 'rgb', None)
                is_white = False
                if font_rgb:
                    font_str = str(font_rgb).upper()
                    if 'FFFFFF' in font_str or font_str.endswith('FFFFFF'):
                        is_white = True
                
                # Si cumple el formato rojo/blanco, extrae fecha
                if is_red and is_white:
                    val = str(cell.value).strip()
                    match = fecha_regex.search(val)
                    if match:
                        fecha_str = match.group(1).replace('.', '/')
                        try:
                            fecha = pd.to_datetime(fecha_str, dayfirst=True)
                            
                            # Buscar el número correspondiente (generalmente está abajo o a la derecha)
                            num_val = None
                            for offset_row, offset_col in [(1, 0), (0, 1), (1, 1)]:
                                c = ws.cell(row=cell.row + offset_row, column=cell.column + offset_col)
                                if c.value:
                                    clean = str(c.value).upper().replace('O', '0').replace(' ', '').replace('HOY', '')
                                    if clean.isdigit() and len(clean) >= 2:
                                        num_val = int(clean[-2:])
                                        break
                                        
                            if num_val is not None:
                                resultados.append({'Fecha': fecha, 'Numero': num_val})
                        except: pass
                        
        df = pd.DataFrame(resultados).drop_duplicates(subset='Fecha').sort_values('Fecha').reset_index(drop=True)
        wb.close()
        
        if df.empty:
            return None, "⚠️ No se encontraron celdas con formato rojo/blanco. Verifica el Excel."
            
        return df, f"✅ {len(df)} sorteos cargados desde celdas resaltadas."
        
    except Exception as e:
        return None, f"❌ Error leyendo Excel: {e}"

df_fijos, msg = cargar_datos_desde_excel()
if df_fijos is None:
    st.error(msg)
    st.stop()
st.success(msg)

# ==========================================
# 📊 DETECCIÓN DE FECHA BASE Y PROYECCIÓN
# ==========================================
fecha_base = df_fijos['Fecha'].max()
dia_objetivo = fecha_base + timedelta(days=1)
nombre_dia = dia_objetivo.day_name()

st.info(f"📅 Última fecha válida detectada: `{fecha_base.strftime('%d/%m/%Y')}`")
st.info(f"🎯 Próximo sorteo proyectado: **{dia_objetivo.strftime('%d/%m/%Y')} ({nombre_dia})**")

# ==========================================
# 🧠 LÓGICA DE ANÁLISIS (Tu metodología)
# ==========================================
df_fijos['Decena'] = (df_fijos['Numero'] // 10).astype(int)
df_fijos['Terminacion'] = (df_fijos['Numero'] % 10).astype(int)
df_fijos['Suma'] = df_fijos['Numero'].apply(lambda x: (x//10) + (x%10))
df_fijos['DiaSemana'] = df_fijos['Fecha'].dt.day_name()

resultados = []
for n in range(0, 100):
    dec = n // 10
    ter = n % 10
    
    # 1. Separación (Gap) desde última aparición
    apariciones = df_fijos[df_fijos['Numero'] == n]['Fecha']
    gap = (fecha_base - apariciones.max()).days if not apariciones.empty else 999
    
    # 2. Tendencia escalonada (últimas 3 decenas)
    ultimas_dec = df_fijos.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1: tendencia = 15
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1: tendencia = 15
            
    # 3. Frecuencia histórica en este día (últimos 6 meses)
    hace_180 = fecha_base - timedelta(days=180)
    freq_dia = len(df_fijos[(df_fijos['Fecha'] >= hace_180) & (df_fijos['DiaSemana'] == nombre_dia) & (df_fijos['Numero'] == n)])
    
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
# 💾 RENDERIZADO EN STREAMLIT
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
    st.success("✅ `Prediccion_Inteligente_FL_Tarde.xlsx` generado.")