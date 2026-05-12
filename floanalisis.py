import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 📂 CONFIGURACIÓN AUTOMÁTICA
# ==========================================
ARCHIVO_ENTRADA = 'david esgrima.xlsx'
HOJA_OBJETIVO = 'TARDE 1 (1)'  # Si cambia, el script usa la primera hoja por defecto
ARCHIVO_SALIDA = 'Prediccion_Inteligente_FL_Tarde.xlsx'

# ==========================================
# 🔍 1. LECTURA Y PARSING INTELIGENTE
# ==========================================
print("⏳ Leyendo archivo Excel y detectando estructura...")
try:
    df_raw = pd.read_excel(ARCHIVO_ENTRADA, sheet_name=HOJA_OBJETIVO, header=None)
except Exception:
    print(f"⚠️ Hoja '{HOJA_OBJETIVO}' no encontrada. Usando la primera hoja disponible.")
    df_raw = pd.read_excel(ARCHIVO_ENTRADA, header=None)

# Parsear formato horizontal (Fechas arriba, Números abajo)
datos_limpios = []
rows, cols = df_raw.shape

for r in range(rows - 1):
    for c in range(cols):
        celda = str(df_raw.iloc[r, c]).strip()
        # Buscar patrón de fecha (DD/MM/YYYY o DD-MM-YYYY)
        fecha_match = re.search(r'(\d{2}[/\-\.]\d{2}[/\-\.]\d{2,4})', celda)
        if fecha_match:
            fecha_str = fecha_match.group(1).replace('-', '/').replace('.', '/')
            try:
                fecha = pd.to_datetime(fecha_str, format='%d/%m/%Y')
            except:
                try:
                    fecha = pd.to_datetime(fecha_str, format='%d/%m/%y')
                except:
                    continue

            # Buscar número en la celda de abajo
            num_raw = str(df_raw.iloc[r+1, c]).strip().upper().replace('O', '0').replace(' ', '')
            if num_raw.isdigit() and len(num_raw) >= 2:
                # Tu regla: si es 428, cogemos 28
                numero = int(num_raw[-2:]) 
                datos_limpios.append({'Fecha': fecha, 'Numero': numero})

df = pd.DataFrame(datos_limpios).drop_duplicates(subset='Fecha').sort_values('Fecha').reset_index(drop=True)

if df.empty:
    print("❌ No se encontraron fechas/números válidos en el archivo.")
    exit()

print(f"✅ Datos cargados: {len(df)} sorteos analizados.")

# ==========================================
# 📊 2. MÉTRICAS BASE
# ==========================================
df['Decena'] = (df['Numero'] // 10).astype(int)
df['Terminacion'] = (df['Numero'] % 10).astype(int)
df['Suma'] = df['Numero'].apply(lambda x: (x//10) + (x%10))
df['DiaSemana'] = df['Fecha'].dt.day_name()

# Detectar fecha más reciente y calcular día siguiente
fecha_base = df['Fecha'].max()
dia_objetivo = fecha_base + timedelta(days=1)
nombre_dia = dia_objetivo.day_name()

print(f"📅 Fecha base detectada: {fecha_base.strftime('%d/%m/%Y')}")
print(f"🎯 Proyección para: {dia_objetivo.strftime('%d/%m/%Y')} ({nombre_dia})")

# ==========================================
# 🧠 3. ANÁLISIS INTELIGENTE (TU METODOLOGÍA)
# ==========================================
resultados = []
for n in range(0, 100):
    dec = n // 10
    ter = n % 10
    
    # 1. Separación (Gap) desde última aparición
    apariciones = df[df['Numero'] == n]['Fecha']
    gap = (fecha_base - apariciones.max()).days if not apariciones.empty else 999
    
    # 2. Tendencia de Decena (Ascendente/Descendente en últimos 3 sorteos)
    ultimas_dec = df.tail(3)['Decena'].tolist()
    tendencia = 0
    if len(ultimas_dec) == 3:
        if ultimas_dec[0] < ultimas_dec[1] < ultimas_dec[2] and dec == ultimas_dec[2] + 1:
            tendencia = 15  # Escalón ascendente
        elif ultimas_dec[0] > ultimas_dec[1] > ultimas_dec[2] and dec == ultimas_dec[2] - 1:
            tendencia = 15  # Escalón descendente
            
    # 3. Frecuencia histórica en este día de la semana (últimos 6 meses)
    hace_180_dias = fecha_base - timedelta(days=180)
    hist_dia = df[(df['Fecha'] >= hace_180_dias) & (df['DiaSemana'] == nombre_dia)]['Numero']
    freq_dia = list(hist_dia).count(n)
    
    # 4. Fórmula de Puntuación (Ajustable)
    pts = 0
    if 20 <= gap <= 45: pts += 20      # Zona dulce de separación
    elif gap > 45: pts += 10           # Presión alta
    pts += tendencia                   # Tendencia de decena
    if freq_dia >= 3: pts += 12        # Día fuerte
    if 8 <= (dec + ter) <= 14: pts += 5 # Suma en rango frecuente
    
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
# 💾 4. EXPORTACIÓN Y REPORTE
# ==========================================
with pd.ExcelWriter(ARCHIVO_SALIDA, engine='openpyxl') as writer:
    # Hoja 1: Top 30 Proyección
    df_scores.head(30).to_excel(writer, sheet_name='Top_30_Proyeccion', index=False)
    
    # Hoja 2: Configuración detectada
    meta = pd.DataFrame([
        ['Archivo Fuente', ARCHIVO_ENTRADA],
        ['Fecha Base', fecha_base.strftime('%d/%m/%Y')],
        ['Día Proyección', dia_objetivo.strftime('%d/%m/%Y')],
        ['Nombre Día', nombre_dia],
        ['Registros Analizados', len(df)],
        ['Criterio Zona Dulce', 'Gap 20-45 días'],
        ['Criterio Tendencia', 'Escalón Decena ±1'],
        ['Criterio Día Fuerte', '≥3 apariciones últimos 180d']
    ], columns=['Parametro', 'Valor'])
    meta.to_excel(writer, sheet_name='Configuracion', index=False)

print("\n" + "="*40)
print("🎉 ANÁLISIS COMPLETADO CON ÉXITO")
print("="*40)
print(f"📁 Archivo generado: {ARCHIVO_SALIDA}")
print("\n🔥 TOP 5 RECOMENDADOS PARA MAÑANA:")
print(df_scores.head(5)[['Numero', 'Decena', 'Terminacion', 'Gap', 'Tendencia', 'Freq_Dia', 'Puntaje', 'Estado']].to_string(index=False))
print("\n💡 Nota: Abre el Excel para ver el desglose completo y ajustar pesos si es necesario.")