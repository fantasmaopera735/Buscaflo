# -*- coding: utf-8 -*-
"""
SumaFlo - Análisis de Sumas para Flotodo (Sorteos: Tarde y Noche)
=================================================================
Análisis de sumas de dígitos (SIEMPRE rango 0-18):
- Suma Fijo: suma de dígitos del Fijo
- Suma Corridos: qué sumas aparecen entre los 2 corridos
- Suma Total (Día): qué sumas aparecen entre los 3 números

ORDEN DE SESIONES (de más reciente a menos reciente): N → T
"""

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter

# --- CONFIGURACIÓN GOOGLE SHEETS ---
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
SPREADSHEET_ID = "1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw"

@st.cache_resource
def get_gsheet_client():
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPE)
        return gspread.authorize(creds)
    except:
        import os
        creds_path = 'credentials.json'
        if os.path.exists(creds_path):
            creds = Credentials.from_service_account_file(creds_path, scopes=SCOPE)
            return gspread.authorize(creds)
        raise Exception("No se encontraron credenciales")

def get_worksheet(sheet_name):
    client = get_gsheet_client()
    return client.open_by_key(SPREADSHEET_ID).worksheet(sheet_name)

# --- FUNCIONES DE SUMA ---
def suma_digitos(numero):
    """Calcula la suma de los dígitos de un número de 2 dígitos (00-99) -> Resultado: 0-18"""
    try:
        n = int(float(numero))
        return (n // 10) + (n % 10)
    except:
        return 0

def obtener_numeros_que_suman(suma_objetivo):
    """Devuelve todos los números de 2 dígitos que suman el valor objetivo (0-18)"""
    numeros = []
    for n in range(100):
        if suma_digitos(n) == suma_objetivo:
            numeros.append(f"{n:02d}")
    return numeros

# --- CARGA DE DATOS ---
@st.cache_data(ttl=60)
def cargar_datos():
    try:
        wks = get_worksheet("Sorteos")
        data = wks.get_all_records()
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
        
        def convertir_fecha(valor):
            if pd.isna(valor) or str(valor).strip() == '':
                return pd.NaT
            valor_str = str(valor).strip()
            formatos = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y']
            for fmt in formatos:
                try:
                    return pd.to_datetime(valor_str, format=fmt)
                except:
                    continue
            try:
                return pd.to_datetime(valor_str, dayfirst=True)
            except:
                return pd.NaT
        
        df['Fecha'] = df['Fecha'].apply(convertir_fecha)
        df = df.dropna(subset=['Fecha'])
        
        if 'Tipo_Sorteo' in df.columns or 'Tarde/Noche' in df.columns:
            col_tipo = 'Tipo_Sorteo' if 'Tipo_Sorteo' in df.columns else 'Tarde/Noche'
            df['Tipo_Sorteo'] = df[col_tipo].astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].apply(lambda x: 
                'T' if x in ['T', 'TARDE', 'TARDE/'] else
                'N' if x in ['N', 'NOCHE', '/NOCHE', 'NOCHE/'] else
                'T' if 'TARDE' in x else 'N' if 'NOCHE' in x else 'OTRO')
        else:
            df['Tipo_Sorteo'] = 'OTRO'
        
        col_fijo = 'Fijo'
        col_corr1 = None
        col_corr2 = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if '1er' in col_lower or 'primer' in col_lower:
                col_corr1 = col
            elif '2do' in col_lower or 'segundo' in col_lower:
                col_corr2 = col
        
        if col_corr1 is None and len(df.columns) >= 5:
            col_corr1 = df.columns[4]
        if col_corr2 is None and len(df.columns) >= 6:
            col_corr2 = df.columns[5]
        
        df['Suma_Fijo'] = df[col_fijo].apply(suma_digitos)
        if col_corr1:
            df['Suma_Corr1'] = df[col_corr1].apply(suma_digitos)
        else:
            df['Suma_Corr1'] = 0
        if col_corr2:
            df['Suma_Corr2'] = df[col_corr2].apply(suma_digitos)
        else:
            df['Suma_Corr2'] = 0
        
        df['Fijo_Num'] = df[col_fijo].apply(lambda x: f"{int(float(x)):02d}" if pd.notna(x) else "00")
        if col_corr1:
            df['Corr1_Num'] = df[col_corr1].apply(lambda x: f"{int(float(x)):02d}" if pd.notna(x) else "00")
        else:
            df['Corr1_Num'] = "00"
        if col_corr2:
            df['Corr2_Num'] = df[col_corr2].apply(lambda x: f"{int(float(x)):02d}" if pd.notna(x) else "00")
        else:
            df['Corr2_Num'] = "00"
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return pd.DataFrame()

# --- FUNCIONES DE ESTADÍSTICAS ---
def calcular_estadisticas_suma(df, tipo_sesion=None):
    """Calcula estadísticas para cada suma (0-18)."""
    if tipo_sesion and tipo_sesion != 'Todas':
        df = df[df['Tipo_Sorteo'] == tipo_sesion].copy()
    
    resultados = {}
    hoy = datetime.now().date()
    rango_sumas = range(19)
    
    # SUMA FIJO
    stats_fijo = []
    for suma in rango_sumas:
        df_suma = df[df['Suma_Fijo'] == suma].sort_values('Fecha')
        fechas = df_suma['Fecha'].dt.date.tolist()
        
        if len(fechas) == 0:
            stats_fijo.append({'Suma': suma, 'Frecuencia': 0, 'Ausencia_Maxima': 999,
                'Promedio_Dias': 0, 'Dias_Sin_Aparecer': 999, 'Ultima_Fecha': None})
        else:
            frecuencia = len(fechas)
            ultima_fecha = fechas[-1]
            dias_sin_aparecer = (hoy - ultima_fecha).days
            if len(fechas) >= 2:
                gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
                ausencia_maxima = max(gaps)
                promedio_dias = round(np.mean(gaps), 1)
            else:
                ausencia_maxima = dias_sin_aparecer
                promedio_dias = dias_sin_aparecer
            stats_fijo.append({'Suma': suma, 'Frecuencia': frecuencia, 'Ausencia_Maxima': ausencia_maxima,
                'Promedio_Dias': promedio_dias, 'Dias_Sin_Aparecer': dias_sin_aparecer, 'Ultima_Fecha': ultima_fecha})
    resultados['Suma_Fijo'] = pd.DataFrame(stats_fijo)
    
    # SUMA CORRIDOS
    stats_corr = []
    for suma in rango_sumas:
        df_suma = df[(df['Suma_Corr1'] == suma) | (df['Suma_Corr2'] == suma)].sort_values('Fecha')
        fechas = df_suma['Fecha'].dt.date.tolist()
        
        if len(fechas) == 0:
            stats_corr.append({'Suma': suma, 'Frecuencia': 0, 'Ausencia_Maxima': 999,
                'Promedio_Dias': 0, 'Dias_Sin_Aparecer': 999, 'Ultima_Fecha': None})
        else:
            frecuencia = len(fechas)
            ultima_fecha = fechas[-1]
            dias_sin_aparecer = (hoy - ultima_fecha).days
            if len(fechas) >= 2:
                gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
                ausencia_maxima = max(gaps)
                promedio_dias = round(np.mean(gaps), 1)
            else:
                ausencia_maxima = dias_sin_aparecer
                promedio_dias = dias_sin_aparecer
            stats_corr.append({'Suma': suma, 'Frecuencia': frecuencia, 'Ausencia_Maxima': ausencia_maxima,
                'Promedio_Dias': promedio_dias, 'Dias_Sin_Aparecer': dias_sin_aparecer, 'Ultima_Fecha': ultima_fecha})
    resultados['Suma_Corridos'] = pd.DataFrame(stats_corr)
    
    # SUMA TOTAL
    stats_total = []
    for suma in rango_sumas:
        df_suma = df[(df['Suma_Fijo'] == suma) | (df['Suma_Corr1'] == suma) | (df['Suma_Corr2'] == suma)].sort_values('Fecha')
        fechas = df_suma['Fecha'].dt.date.tolist()
        
        if len(fechas) == 0:
            stats_total.append({'Suma': suma, 'Frecuencia': 0, 'Ausencia_Maxima': 999,
                'Promedio_Dias': 0, 'Dias_Sin_Aparecer': 999, 'Ultima_Fecha': None})
        else:
            frecuencia = len(fechas)
            ultima_fecha = fechas[-1]
            dias_sin_aparecer = (hoy - ultima_fecha).days
            if len(fechas) >= 2:
                gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
                ausencia_maxima = max(gaps)
                promedio_dias = round(np.mean(gaps), 1)
            else:
                ausencia_maxima = dias_sin_aparecer
                promedio_dias = dias_sin_aparecer
            stats_total.append({'Suma': suma, 'Frecuencia': frecuencia, 'Ausencia_Maxima': ausencia_maxima,
                'Promedio_Dias': promedio_dias, 'Dias_Sin_Aparecer': dias_sin_aparecer, 'Ultima_Fecha': ultima_fecha})
    resultados['Suma_Total'] = pd.DataFrame(stats_total)
    
    return resultados

def generar_almanaque_sumas(df, tipo_analisis, dia_inicio, dia_fin, meses_atras, tipo_sesion=None):
    """Genera almanaque de SUMAS (0-18) para el período especificado."""
    if tipo_sesion and tipo_sesion != 'Todas':
        df = df[df['Tipo_Sorteo'] == tipo_sesion].copy()
    
    fecha_hoy = datetime.now()
    mes_actual = fecha_hoy.month
    anio_actual = fecha_hoy.year
    
    bloques_validos = []
    nombres_bloques = []
    
    for offset in range(1, meses_atras + 1):
        f_obj = fecha_hoy - relativedelta(months=offset)
        if f_obj.month == mes_actual and f_obj.year == anio_actual:
            continue
        
        try:
            last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
            f_i = datetime(f_obj.year, f_obj.month, min(dia_inicio, last_day))
            f_f = datetime(f_obj.year, f_obj.month, min(dia_fin, last_day))
            if f_i > f_f:
                continue
            df_b = df[(df['Fecha'] >= f_i) & (df['Fecha'] <= f_f)]
            if not df_b.empty:
                bloques_validos.append(df_b)
                nombres_bloques.append(f"{f_i.strftime('%d/%m')}-{f_f.strftime('%d/%m')}")
        except:
            continue
    
    if not bloques_validos:
        for offset in range(1, meses_atras + 1):
            f_obj = fecha_hoy - relativedelta(months=offset)
            if f_obj.month == mes_actual and f_obj.year == anio_actual:
                continue
            try:
                f_i = datetime(f_obj.year, f_obj.month, 1)
                last_day = calendar.monthrange(f_obj.year, f_obj.month)[1]
                f_f = datetime(f_obj.year, f_obj.month, last_day)
                df_b = df[(df['Fecha'] >= f_i) & (df['Fecha'] <= f_f)]
                if not df_b.empty:
                    bloques_validos.append(df_b)
                    nombres_bloques.append(f"{f_obj.strftime('%b')} (Todo)")
            except:
                continue
    
    if not bloques_validos:
        return {'success': False, 'mensaje': 'Sin datos históricos'}
    
    df_total = pd.concat(bloques_validos)
    rango_sumas = range(19)
    
    conteo_sumas = {s: 0 for s in rango_sumas}
    
    for _, row in df_total.iterrows():
        if tipo_analisis == 'Fijo':
            conteo_sumas[row['Suma_Fijo']] += 1
        elif tipo_analisis == 'Corridos':
            conteo_sumas[row['Suma_Corr1']] += 1
            conteo_sumas[row['Suma_Corr2']] += 1
        else:
            conteo_sumas[row['Suma_Fijo']] += 1
            conteo_sumas[row['Suma_Corr1']] += 1
            conteo_sumas[row['Suma_Corr2']] += 1
    
    ranking = [{'Suma': s, 'Frecuencia': conteo_sumas[s]} for s in rango_sumas]
    df_rank = pd.DataFrame(ranking).sort_values('Frecuencia', ascending=False)
    
    def clasificar_sumas(df_rank):
        df_t = df_rank.copy()
        total = len(df_t)
        tercio = total // 3
        conds = [df_t.index < tercio, df_t.index < 2 * tercio]
        vals = ['🔥 Caliente', '🟡 Tibio']
        df_t['Estado'] = np.select(conds, vals, default='🧊 Frío')
        return df_t
    
    df_rank = clasificar_sumas(df_rank)
    
    sumas_persistentes = []
    for suma in rango_sumas:
        aparece_en_todos = True
        for bloque in bloques_validos:
            if tipo_analisis == 'Fijo':
                if suma not in bloque['Suma_Fijo'].values:
                    aparece_en_todos = False
                    break
            elif tipo_analisis == 'Corridos':
                if suma not in bloque['Suma_Corr1'].values and suma not in bloque['Suma_Corr2'].values:
                    aparece_en_todos = False
                    break
            else:
                if suma not in bloque['Suma_Fijo'].values and suma not in bloque['Suma_Corr1'].values and suma not in bloque['Suma_Corr2'].values:
                    aparece_en_todos = False
                    break
        if aparece_en_todos:
            sumas_persistentes.append(suma)
    
    sumas_calientes = df_rank[df_rank['Estado'] == '🔥 Caliente']['Suma'].tolist()
    
    hoy = datetime.now()
    estado_periodo = ""
    df_historial_actual = pd.DataFrame()
    
    try:
        fin_mes = calendar.monthrange(hoy.year, hoy.month)[1]
        fecha_ini = datetime(hoy.year, hoy.month, min(dia_inicio, fin_mes))
        fecha_fin = datetime(hoy.year, hoy.month, min(dia_fin, fin_mes))
        
        if hoy.date() < fecha_ini.date():
            estado_periodo = f"⚪ Período inicia el {fecha_ini.strftime('%d/%m')}"
        else:
            fecha_fin_real = min(hoy, fecha_fin)
            df_eval = df[(df['Fecha'] >= fecha_ini) & (df['Fecha'] <= fecha_fin_real)]
            
            if not df_eval.empty:
                historial = []
                for _, row in df_eval.iterrows():
                    if tipo_analisis == 'Fijo':
                        sumas_row = [row['Suma_Fijo']]
                    elif tipo_analisis == 'Corridos':
                        sumas_row = [row['Suma_Corr1'], row['Suma_Corr2']]
                    else:
                        sumas_row = [row['Suma_Fijo'], row['Suma_Corr1'], row['Suma_Corr2']]
                    
                    for suma_val in sumas_row:
                        estado = df_rank[df_rank['Suma'] == suma_val]['Estado'].values
                        estado_val = estado[0] if len(estado) > 0 else '❓'
                        es_persistente = suma_val in sumas_persistentes
                        
                        historial.append({
                            'Fecha': row['Fecha'], 'Sesión': row['Tipo_Sorteo'], 'Suma': suma_val,
                            'Estado': estado_val, 'Persistente': '✅ SÍ' if es_persistente else '❌ NO',
                            'Fijo': row.get('Fijo_Num', '00'), 'Corr1': row.get('Corr1_Num', '00'), 'Corr2': row.get('Corr2_Num', '00')
                        })
                
                df_historial_actual = pd.DataFrame(historial)
                # ORDEN: N=0 (más reciente), T=1 (menos reciente) - CORREGIDO
                orden_sesion = {'N': 0, 'T': 1}
                df_historial_actual['orden'] = df_historial_actual['Sesión'].map(orden_sesion).fillna(2)
                # Ordenar: Fecha descendente (más reciente primero), luego por orden de sesión
                df_historial_actual = df_historial_actual.sort_values(['Fecha', 'orden'], ascending=[False, True])
                df_historial_actual = df_historial_actual.drop(columns=['orden'])
            
            estado_periodo = f"🟢 Período activo (hasta {fecha_fin_real.strftime('%d/%m')})"
    except Exception as e:
        estado_periodo = f"Error: {str(e)}"
    
    df_faltantes = pd.DataFrame()
    if not df_historial_actual.empty and sumas_calientes:
        salidas = set(df_historial_actual['Suma'].unique())
        faltantes = [s for s in sumas_calientes if s not in salidas]
        if faltantes:
            df_faltantes = pd.DataFrame([{'Suma Faltante': s, 'Estado': '⏳ Pendiente'} for s in sorted(faltantes)])
    
    return {
        'success': True, 'df_rank': df_rank, 'sumas_persistentes': sumas_persistentes,
        'sumas_calientes': sumas_calientes, 'nombres_bloques': nombres_bloques,
        'df_historial_actual': df_historial_actual, 'df_faltantes': df_faltantes, 'estado_periodo': estado_periodo
    }

def buscar_suma_detalle(df, suma_buscar, tipo_sesion=None):
    """Busca una suma específica y muestra todas las veces que apareció."""
    if tipo_sesion and tipo_sesion != 'Todas':
        df = df[df['Tipo_Sorteo'] == tipo_sesion].copy()
    
    df_filtrado = df[
        (df['Suma_Fijo'] == suma_buscar) | 
        (df['Suma_Corr1'] == suma_buscar) | 
        (df['Suma_Corr2'] == suma_buscar)
    ].sort_values('Fecha', ascending=False)
    
    if df_filtrado.empty:
        return None
    
    resultados = []
    for _, row in df_filtrado.iterrows():
        aparece_en = []
        if row['Suma_Fijo'] == suma_buscar:
            aparece_en.append('Fijo')
        if row['Suma_Corr1'] == suma_buscar:
            aparece_en.append('Corr1')
        if row['Suma_Corr2'] == suma_buscar:
            aparece_en.append('Corr2')
        
        resultados.append({
            'Fecha': row['Fecha'].strftime('%d/%m/%Y'), 'Sesión': row['Tipo_Sorteo'],
            'Fijo': f"{row['Fijo_Num']} (Σ{row['Suma_Fijo']})",
            'Corr1': f"{row['Corr1_Num']} (Σ{row['Suma_Corr1']})",
            'Corr2': f"{row['Corr2_Num']} (Σ{row['Suma_Corr2']})",
            'Aparece en': ', '.join(aparece_en)
        })
    
    df_resultados = pd.DataFrame(resultados)
    # ORDEN: N=0 (más reciente), T=1 (menos reciente) - CORREGIDO
    orden_sesion = {'N': 0, 'T': 1}
    df_resultados['orden'] = df_resultados['Sesión'].map(orden_sesion).fillna(2)
    df_resultados = df_resultados.sort_values(['Fecha', 'orden'], ascending=[False, True])
    df_resultados = df_resultados.drop(columns=['orden'])
    
    return df_resultados


# --- FUNCIONES DE COMBINACIONES/TRANSICIONES PARA 2 SESIONES ---
def preparar_datos_transiciones(df):
    """Prepara los datos para análisis de transiciones entre sesiones."""
    df['Fecha_Str'] = df['Fecha'].dt.strftime('%Y-%m-%d')
    
    datos = {}
    for _, row in df.iterrows():
        fecha = row['Fecha_Str']
        sesion = row['Tipo_Sorteo']
        if fecha not in datos:
            datos[fecha] = {}
        datos[fecha][sesion] = {
            'Suma_Fijo': row['Suma_Fijo'],
            'Suma_Corr1': row['Suma_Corr1'],
            'Suma_Corr2': row['Suma_Corr2']
        }
    
    return datos

def obtener_pronostico_tarde_noche(datos, suma_tarde, tipo_suma='Suma_Fijo'):
    """Dado una suma de la tarde, devuelve qué puede salir en la noche."""
    resultados = Counter()
    
    for fecha, sesiones in datos.items():
        if 'T' in sesiones and 'N' in sesiones:
            if sesiones['T'][tipo_suma] == suma_tarde:
                resultados[sesiones['N'][tipo_suma]] += 1
    
    return resultados.most_common(10)

def obtener_pronostico_dia_siguiente(datos, suma_tarde, suma_noche, tipo_suma='Suma_Fijo'):
    """Dado suma tarde y noche, devuelve qué puede salir en la tarde siguiente."""
    resultados = Counter()
    fechas_ordenadas = sorted(datos.keys())
    
    for i in range(len(fechas_ordenadas) - 1):
        fecha_actual = fechas_ordenadas[i]
        fecha_siguiente = fechas_ordenadas[i + 1]
        
        if all(s in datos[fecha_actual] for s in ['T', 'N']) and 'T' in datos[fecha_siguiente]:
            if (datos[fecha_actual]['T'][tipo_suma] == suma_tarde and 
                datos[fecha_actual]['N'][tipo_suma] == suma_noche):
                resultados[datos[fecha_siguiente]['T'][tipo_suma]] += 1
    
    return resultados.most_common(10)

def analizar_transicion_tarde_noche(datos, tipo_suma='Suma_Fijo'):
    """Analiza todas las transiciones Tarde → Noche."""
    transiciones = Counter()
    
    for fecha, sesiones in datos.items():
        if 'T' in sesiones and 'N' in sesiones:
            suma_tarde = sesiones['T'][tipo_suma]
            suma_noche = sesiones['N'][tipo_suma]
            transiciones[(suma_tarde, suma_noche)] += 1
    
    return transiciones


def obtener_sumas_totales_sesion(sesiones, sesion):
    """Obtiene todas las sumas (Fijo + Corr1 + Corr2) de una sesión."""
    if sesion not in sesiones:
        return []
    return [
        sesiones[sesion]['Suma_Fijo'],
        sesiones[sesion]['Suma_Corr1'],
        sesiones[sesion]['Suma_Corr2']
    ]

def obtener_pronostico_tarde_noche_total(datos, suma_tarde):
    """Dado una suma de la tarde, devuelve qué sumas totales pueden aparecer en la noche."""
    resultados = Counter()
    
    for fecha, sesiones in datos.items():
        if 'T' in sesiones and 'N' in sesiones:
            sumas_tarde = obtener_sumas_totales_sesion(sesiones, 'T')
            if suma_tarde in sumas_tarde:
                for s in obtener_sumas_totales_sesion(sesiones, 'N'):
                    resultados[s] += 1
    
    return resultados.most_common(10)

def obtener_pronostico_dia_siguiente_total(datos, suma_tarde, suma_noche):
    """Dado suma tarde y noche (totales), devuelve qué puede salir en la tarde siguiente."""
    resultados = Counter()
    fechas_ordenadas = sorted(datos.keys())
    
    for i in range(len(fechas_ordenadas) - 1):
        fecha_actual = fechas_ordenadas[i]
        fecha_siguiente = fechas_ordenadas[i + 1]
        
        if all(s in datos[fecha_actual] for s in ['T', 'N']) and 'T' in datos[fecha_siguiente]:
            sumas_t = obtener_sumas_totales_sesion(datos[fecha_actual], 'T')
            sumas_n = obtener_sumas_totales_sesion(datos[fecha_actual], 'N')
            
            if suma_tarde in sumas_t and suma_noche in sumas_n:
                for s in obtener_sumas_totales_sesion(datos[fecha_siguiente], 'T'):
                    resultados[s] += 1
    
    return resultados.most_common(10)


# --- FUNCIÓN PRINCIPAL ---
def main():
    """Función principal que ejecuta la aplicación."""
    st.set_page_config(page_title="SumaFlo - Análisis de Sumas", page_icon="🔢", layout="wide")

    st.title("🔢 SumaFlo - Análisis de Sumas (Tarde y Noche)")
    st.markdown("""
    **Suma de dígitos:** Para un número de 2 dígitos, se suman sus componentes (rango: 0-18).
    - Ejemplo: 69 → 6+9 = 15 | 00 → 0+0 = 0 | 99 → 9+9 = 18

    **Tipos de análisis (TODOS con rango 0-18):**
    - **Suma Fijo:** Análisis de la suma de dígitos del número Fijo
    - **Suma Corridos:** Qué sumas aparecen entre los 2 corridos
    - **Suma Total/Día:** Qué sumas aparecen entre los 3 números (Fijo + Corr1 + Corr2)

    **Sesiones:** T = Tarde | N = Noche
    
    **Orden en historial:** N → T (de más reciente a menos reciente)
    """)

    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = cargar_datos()

    if df.empty:
        st.error("No se pudieron cargar los datos")
        st.stop()

    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    tipo_sesion = st.sidebar.selectbox("Sesión", ['Todas', 'T', 'N'], 
                                       format_func=lambda x: 'Todas' if x == 'Todas' else ('Tarde' if x == 'T' else 'Noche'))

    # Pestañas principales
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Tabla de Estadísticas", "🔗 Combinaciones/Pronóstico", "📅 Almanaque Suma Total", "📅 Almanaque Suma Corridos", "📅 Almanaque Suma Fijo", "🔍 Buscar Suma"])

    # === TAB 1: TABLA DE ESTADÍSTICAS ===
    with tab1:
        st.header("📊 Tabla Completa de Estadísticas (Rango: 0-18)")
        
        with st.spinner("Calculando estadísticas..."):
            stats = calcular_estadisticas_suma(df, tipo_sesion if tipo_sesion != 'Todas' else None)
        
        st.subheader("➕ Suma Total/Día (entre Fijo + Corrido1 + Corrido2)")
        st.markdown("*Qué sumas (0-18) aparecen más frecuentemente entre los 3 números del sorteo*")
        df_total = stats['Suma_Total'].copy()
        df_total['Ultima_Fecha'] = df_total['Ultima_Fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if x else 'N/A')
        df_total = df_total.rename(columns={
            'Suma': 'Suma', 'Frecuencia': 'Frecuencia', 'Ausencia_Maxima': 'Ausencia Máx',
            'Promedio_Dias': 'Promedio Días', 'Dias_Sin_Aparecer': 'Días Sin Aparecer', 'Ultima_Fecha': 'Última Fecha'
        })
        df_total = df_total.sort_values('Frecuencia', ascending=False)
        st.dataframe(df_total, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🎲 Suma Corridos (entre Corrido1 + Corrido2)")
        st.markdown("*Qué sumas (0-18) aparecen más frecuentemente entre los 2 corridos*")
        df_corr = stats['Suma_Corridos'].copy()
        df_corr['Ultima_Fecha'] = df_corr['Ultima_Fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if x else 'N/A')
        df_corr = df_corr.rename(columns={
            'Suma': 'Suma', 'Frecuencia': 'Frecuencia', 'Ausencia_Maxima': 'Ausencia Máx',
            'Promedio_Dias': 'Promedio Días', 'Dias_Sin_Aparecer': 'Días Sin Aparecer', 'Ultima_Fecha': 'Última Fecha'
        })
        df_corr = df_corr.sort_values('Frecuencia', ascending=False)
        st.dataframe(df_corr, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🔢 Suma Fijo (solo del número Fijo)")
        st.markdown("*Análisis de la suma de dígitos del Fijo únicamente*")
        df_fijo = stats['Suma_Fijo'].copy()
        df_fijo['Ultima_Fecha'] = df_fijo['Ultima_Fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if x else 'N/A')
        df_fijo = df_fijo.rename(columns={
            'Suma': 'Suma', 'Frecuencia': 'Frecuencia', 'Ausencia_Maxima': 'Ausencia Máx',
            'Promedio_Dias': 'Promedio Días', 'Dias_Sin_Aparecer': 'Días Sin Aparecer', 'Ultima_Fecha': 'Última Fecha'
        })
        df_fijo = df_fijo.sort_values('Frecuencia', ascending=False)
        st.dataframe(df_fijo, use_container_width=True, hide_index=True)

    # === TAB 2: COMBINACIONES/PRONÓSTICO ===
    with tab2:
        st.header("🔗 Combinaciones / Pronóstico por Transiciones")
        st.markdown("""
        **Análisis de transiciones entre sesiones (Tarde y Noche):**
        - Si en la Tarde sale suma X → ¿Qué puede salir en la Noche?
        - Si Tarde=X y Noche=Y → ¿Qué puede salir en la Tarde siguiente?
        """)
        
        datos = preparar_datos_transiciones(df)
        
        tipo_suma_sel = st.selectbox(
            "Tipo de suma a analizar:",
            ['Suma_Total', 'Suma_Fijo', 'Suma_Corr1', 'Suma_Corr2'],
            format_func=lambda x: {'Suma_Total': 'Suma Total/Día (Fijo+CorridOS)', 'Suma_Fijo': 'Suma Fijo', 'Suma_Corr1': 'Suma Corrido 1', 'Suma_Corr2': 'Suma Corrido 2'}[x],
            key="comb_tipo_suma"
        )
        
        st.markdown("---")
        
        # === SECCIÓN 1: TARDE → NOCHE ===
        st.subheader("☀️ Si en la Tarde sale suma X → ¿Qué puede salir en la Noche?")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            suma_tarde_input = st.selectbox("Suma de la Tarde:", list(range(19)), key="comb_tarde")
        
        if st.button("Analizar Tarde → Noche", key="btn_comb_tn"):
            if tipo_suma_sel == 'Suma_Total':
                pronostico = obtener_pronostico_tarde_noche_total(datos, suma_tarde_input)
            else:
                pronostico = obtain_pronostico_tarde_noche(datos, suma_tarde_input, tipo_suma_sel)
            
            if pronostico:
                df_pron = pd.DataFrame([
                    {'Suma Noche': s, 'Frecuencia': f} for s, f in pronostico
                ])
                st.success(f"Si Tarde = Σ{suma_tarde_input}, lo más probable para Noche:")
                st.dataframe(df_pron, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No hay datos históricos con Tarde = Σ{suma_tarde_input}")
        
        st.markdown("---")
        
        # === SECCIÓN 2: TARDE + NOCHE → TARDE SIGUIENTE ===
        st.subheader("📅 Si Tarde=X y Noche=Y → ¿Qué puede salir en la Tarde siguiente?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            suma_t2 = st.selectbox("Suma Tarde:", list(range(19)), key="comb_t2")
        with col2:
            suma_n2 = st.selectbox("Suma Noche:", list(range(19)), key="comb_n2")
        
        if st.button("Analizar Tarde+Noche → Tarde Siguiente", key="btn_comb_dia"):
            if tipo_suma_sel == 'Suma_Total':
                pronostico = obtener_pronostico_dia_siguiente_total(datos, suma_t2, suma_n2)
            else:
                pronostico = obtener_pronostico_dia_siguiente(datos, suma_t2, suma_n2, tipo_suma_sel)
            
            if pronostico:
                df_pron = pd.DataFrame([
                    {'Suma Tarde Sig.': s, 'Frecuencia': f} for s, f in pronostico
                ])
                st.success(f"Si Tarde=Σ{suma_t2} y Noche=Σ{suma_n2}, lo más probable para la Tarde siguiente:")
                st.dataframe(df_pron, use_container_width=True, hide_index=True)
            else:
                st.warning(f"No hay datos históricos con esa combinación exacta")
        
        st.markdown("---")
        
        # === ESTADÍSTICAS DE TRANSICIONES ===
        st.subheader("📊 Estadísticas Generales de Transiciones")
        
        if tipo_suma_sel != 'Suma_Total':
            trans_tn = analizar_transicion_tarde_noche(datos, tipo_suma_sel)
            top_tn = trans_tn.most_common(15)
            
            if top_tn:
                st.markdown("**Top 15 transiciones Tarde → Noche:**")
                df_trans_tn = pd.DataFrame([
                    {'Tarde': f"Σ{t}", 'Noche': f"Σ{n}", 'Frecuencia': f} 
                    for (t, n), f in top_tn
                ])
                st.dataframe(df_trans_tn, use_container_width=True, hide_index=True)
        else:
            st.info("Las estadísticas de transiciones no están disponibles para Suma Total. Seleccione un tipo de suma específico (Fijo, Corrido 1 o Corrido 2).")

    # === TAB 3: ALMANAQUE SUMA TOTAL ===
    with tab3:
        st.header("📅 Almanaque Suma Total/Día (entre Fijo + Corrido1 + Corrido2)")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            dia_ini = st.number_input("Día inicio", 1, 31, 1, key="alm_total_ini")
        with col_cfg2:
            dia_fin = st.number_input("Día fin", 1, 31, 15, key="alm_total_fin")
        with col_cfg3:
            meses_atras = st.number_input("Meses atrás", 1, 24, 6, key="alm_total_meses")
        
        if st.button("Generar Almanaque", key="btn_alm_total"):
            with st.spinner("Generando almanaque..."):
                resultado = generar_almanaque_sumas(df, 'Total', dia_ini, dia_fin, meses_atras,
                                                    tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if not resultado['success']:
                st.warning(resultado.get('mensaje', 'Sin datos'))
            else:
                st.info(f"📅 Períodos analizados: {', '.join(resultado['nombres_bloques'])}")
                
                if resultado['sumas_persistentes']:
                    st.success(f"🔥 **Sumas Persistentes:** {resultado['sumas_persistentes']}")
                else:
                    st.info("No hay sumas que aparezcan en todos los períodos")
                
                st.subheader("📊 Ranking de Sumas (0-18)")
                df_rank = resultado['df_rank'].copy()
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
                
                if not resultado['df_historial_actual'].empty:
                    st.subheader("📋 Historial Período Actual (Orden: N → T)")
                    st.markdown(f"**{resultado['estado_periodo']}**")
                    st.dataframe(resultado['df_historial_actual'], use_container_width=True, hide_index=True)
                
                if not resultado['df_faltantes'].empty:
                    st.subheader("⏳ Sumas Calientes Faltantes")
                    st.dataframe(resultado['df_faltantes'], use_container_width=True, hide_index=True)

    # === TAB 4: ALMANAQUE SUMA CORRIDOS ===
    with tab4:
        st.header("📅 Almanaque Suma Corridos (entre Corrido1 + Corrido2)")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            dia_ini_c = st.number_input("Día inicio", 1, 31, 1, key="alm_corr_ini")
        with col_cfg2:
            dia_fin_c = st.number_input("Día fin", 1, 31, 15, key="alm_corr_fin")
        with col_cfg3:
            meses_atras_c = st.number_input("Meses atrás", 1, 24, 6, key="alm_corr_meses")
        
        if st.button("Generar Almanaque", key="btn_alm_corr"):
            with st.spinner("Generando almanaque..."):
                resultado = generar_almanaque_sumas(df, 'Corridos', dia_ini_c, dia_fin_c, meses_atras_c,
                                                    tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if not resultado['success']:
                st.warning(resultado.get('mensaje', 'Sin datos'))
            else:
                st.info(f"📅 Períodos analizados: {', '.join(resultado['nombres_bloques'])}")
                
                if resultado['sumas_persistentes']:
                    st.success(f"🔥 **Sumas Persistentes:** {resultado['sumas_persistentes']}")
                else:
                    st.info("No hay sumas que aparezcan en todos los períodos")
                
                st.subheader("📊 Ranking de Sumas (0-18)")
                df_rank = resultado['df_rank'].copy()
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
                
                if not resultado['df_historial_actual'].empty:
                    st.subheader("📋 Historial Período Actual (Orden: N → T)")
                    st.markdown(f"**{resultado['estado_periodo']}**")
                    st.dataframe(resultado['df_historial_actual'], use_container_width=True, hide_index=True)
                
                if not resultado['df_faltantes'].empty:
                    st.subheader("⏳ Sumas Calientes Faltantes")
                    st.dataframe(resultado['df_faltantes'], use_container_width=True, hide_index=True)

    # === TAB 5: ALMANAQUE SUMA FIJO ===
    with tab5:
        st.header("📅 Almanaque Suma Fijo")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            dia_ini_f = st.number_input("Día inicio", 1, 31, 1, key="alm_fijo_ini")
        with col_cfg2:
            dia_fin_f = st.number_input("Día fin", 1, 31, 15, key="alm_fijo_fin")
        with col_cfg3:
            meses_atras_f = st.number_input("Meses atrás", 1, 24, 6, key="alm_fijo_meses")
        
        if st.button("Generar Almanaque", key="btn_alm_fijo"):
            with st.spinner("Generando almanaque..."):
                resultado = generar_almanaque_sumas(df, 'Fijo', dia_ini_f, dia_fin_f, meses_atras_f,
                                                    tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if not resultado['success']:
                st.warning(resultado.get('mensaje', 'Sin datos'))
            else:
                st.info(f"📅 Períodos analizados: {', '.join(resultado['nombres_bloques'])}")
                
                if resultado['sumas_persistentes']:
                    st.success(f"🔥 **Sumas Persistentes:** {resultado['sumas_persistentes']}")
                else:
                    st.info("No hay sumas que aparezcan en todos los períodos")
                
                st.subheader("📊 Ranking de Sumas (0-18)")
                df_rank = resultado['df_rank'].copy()
                st.dataframe(df_rank, use_container_width=True, hide_index=True)
                
                if not resultado['df_historial_actual'].empty:
                    st.subheader("📋 Historial Período Actual (Orden: N → T)")
                    st.markdown(f"**{resultado['estado_periodo']}**")
                    st.dataframe(resultado['df_historial_actual'], use_container_width=True, hide_index=True)
                
                if not resultado['df_faltantes'].empty:
                    st.subheader("⏳ Sumas Calientes Faltantes")
                    st.dataframe(resultado['df_faltantes'], use_container_width=True, hide_index=True)

    # === TAB 6: BUSCAR SUMA ===
    with tab6:
        st.header("🔍 Buscar Suma Específica (0-18)")
        
        suma_val = st.number_input("Valor de suma a buscar", 0, 18, 9)
        
        if st.button("Buscar Suma", key="btn_buscar"):
            st.subheader(f"🔢 Números de 2 dígitos que suman {suma_val}")
            nums = obtener_numeros_que_saman(suma_val)
            st.markdown(f"**{', '.join(nums)}**")
            
            df_resultado = buscar_suma_detalle(df, suma_val, tipo_sesion if tipo_sesion != 'Todas' else None)
            
            if df_resultado is None or df_resultado.empty:
                st.warning(f"No se encontraron resultados para suma = {suma_val}")
            else:
                st.subheader(f"📋 Historial de Apariciones (Orden: N → T)")
                st.info(f"Total de apariciones: {len(df_resultado)}")
                st.dataframe(df_resultado, use_container_width=True, hide_index=True)

    # === INFO ADICIONAL ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Información")
    st.sidebar.markdown(f"**Datos cargados:** {len(df)} registros")
    if not df.empty:
        st.sidebar.markdown(f"**Desde:** {df['Fecha'].min().strftime('%d/%m/%Y')}")
        st.sidebar.markdown(f"**Hasta:** {df['Fecha'].max().strftime('%d/%m/%Y')}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*SumaFlo v5.1*")


if __name__ == "__main__":
    main()