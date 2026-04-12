# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import os
import time
import calendar
from collections import defaultdict, Counter
import unicodedata

# =============================================================================
# CONFIGURACION
# =============================================================================
RUTA_CSV = 'Geotodo.csv'
RUTA_CACHE = 'cache_perfiles_georgia.csv'
RUTA_BACKTEST = 'backtest_detalle_georgia.csv'
RUTA_HISTORICO = 'historico_predicciones_georgia.csv'

st.set_page_config(
    page_title="Georgia - Análisis de Sorteos",
    page_icon="🍑",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🍑 Georgia - Análisis de Sorteos")
st.markdown("Motor con Backtest Real + Dashboard + Faltantes del Mes + TOP40 Inteligente")

# =============================================================================
# ESTADO DE SESIÓN
# =============================================================================
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []
if 'invalid_dates_df' not in st.session_state:
    st.session_state.invalid_dates_df = None

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

def get_file_signature(filepath):
    try:
        if not os.path.exists(filepath):
            return None
        stat = os.stat(filepath)
        return f"{stat.st_size}_{int(stat.st_mtime)}"
    except:
        return None

def normalizar_numero(valor):
    try:
        if pd.isna(valor):
            return None
        return int(float(str(valor).strip()))
    except:
        return None

def numero_en_lista(numero, lista):
    try:
        num_norm = normalizar_numero(numero)
        if num_norm is None:
            return False
        for item in lista:
            if normalizar_numero(item) == num_norm:
                return True
        return False
    except:
        return False

def parse_fecha_safe(fecha_str):
    if pd.isna(fecha_str) or str(fecha_str).strip() == '':
        return None
    fecha_str = str(fecha_str).strip()
    formatos = ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d', '%d/%m/%y']
    for fmt in formatos:
        try:
            return pd.to_datetime(fecha_str, format=fmt, dayfirst=True)
        except:
            continue
    try:
        parsed = pd.to_datetime(fecha_str, dayfirst=True, errors='coerce')
        if pd.notna(parsed):
            return parsed
    except:
        pass
    return None

# =============================================================================
# DETECTAR MES ANTERIOR COMPLETO
# =============================================================================
def obtener_mes_anterior_completo(df_fijos):
    try:
        if df_fijos.empty:
            return None, None
        fecha_max = df_fijos['Fecha'].max()
        if fecha_max.month == 1:
            mes_anterior = 12
            anio_anterior = fecha_max.year - 1
        else:
            mes_anterior = fecha_max.month - 1
            anio_anterior = fecha_max.year
        return mes_anterior, anio_anterior
    except:
        return None, None

# =============================================================================
# CARGA DE DATOS
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_geotodo(_ruta_csv, _file_signature):
    try:
        if not os.path.exists(_ruta_csv):
            inicializar_archivo(_ruta_csv, ["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"])
            return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None
        try:
            with open(_ruta_csv, 'r', encoding='latin-1') as f:
                primera_linea = f.readline()
            separador = ';' if ';' in primera_linea else (',' if ',' in primera_linea else '\t')
            df = pd.read_csv(_ruta_csv, sep=separador, encoding='latin-1', header=0,
                           on_bad_lines='skip', dtype=str, skipinitialspace=True)
        except pd.errors.EmptyDataError:
            st.warning("⚠️ El archivo CSV está vacío")
            return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None
        except Exception as e:
            st.error(f"❌ Error al leer CSV: {e}")
            return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None
        
        if df.empty or len(df.columns) == 0:
            st.warning("⚠️ El archivo CSV no tiene datos válidos")
            return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None
        
        df.columns = [str(c).strip() for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            c = str(col).strip().upper()
            if 'FECHA' in c:
                rename_map[col] = 'Fecha'
            elif 'TARDE' in c or 'NOCHE' in c or 'SESION' in c or 'TIPO' in c:
                rename_map[col] = 'Tipo_Sorteo'
            elif 'CENTENA' in c or 'CEN' in c:
                rename_map[col] = 'Centena'
            elif 'FIJO' in c and 'CORRIDO' not in c:
                rename_map[col] = 'Fijo'
            elif '1ER' in c or 'PRIMER' in c or 'PRIMERO' in c or 'C1' in c:
                rename_map[col] = 'Primer_Corrido'
            elif '2DO' in c or 'SEGUNDO' in c or 'SEGUND' in c or 'C2' in c:
                rename_map[col] = 'Segundo_Corrido'
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        
        cols_requeridas = ['Fecha', 'Tipo_Sorteo', 'Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido']
        cols_faltantes = [c for c in cols_requeridas if c not in df.columns]
        if cols_faltantes:
            for col in cols_faltantes:
                df[col] = ''
        
        invalid_dates_list = []
        fechas_procesadas = []
        for idx, row in df.iterrows():
            fecha_original = str(row.get('Fecha', '')).strip() if pd.notna(row.get('Fecha')) else ''
            fecha_parseada = parse_fecha_safe(fecha_original)
            if fecha_parseada is None and fecha_original != '':
                invalid_dates_list.append({
                     'Numero_Fila': idx, 'Fecha_Original': fecha_original,
                    'Problema': 'Fecha no reconocida'
                })
                fechas_procesadas.append(None)
            else:
                fechas_procesadas.append(fecha_parseada)
        
        df['Fecha'] = fechas_procesadas
        
        if invalid_dates_list:
            st.session_state.invalid_dates_df = pd.DataFrame(invalid_dates_list)
            df = df[df['Fecha'].notna()].copy()
        
        if df.empty:
            st.warning("⚠️ No hay datos válidos después de procesar fechas")
            return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None
        
        if 'Tipo_Sorteo' in df.columns:
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].fillna('').astype(str).str.strip().str.upper()
            df['Tipo_Sorteo'] = df['Tipo_Sorteo'].map({
                'MAÑANA': 'M', 'MANANA': 'M', 'MORNING': 'M', 'M': 'M',
                'TARDE': 'T', 'AFTERNOON': 'T', 'T': 'T', 'TARDE/NOCHE': 'T',
                'NOCHE': 'N', 'NIGHT': 'N', 'N': 'N'
            }).fillna('OTRO')
            df = df[df['Tipo_Sorteo'].isin(['M', 'T', 'N'])].copy()
        else:
            st.error("❌ Columna Tipo_Sorteo no encontrada")
            return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None
        
        for col in ['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido']:
            if col not in df.columns:
                df[col] = '0'
            df[col] = df[col].fillna('0').astype(str).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df_long = df.melt(id_vars=['Fecha', 'Tipo_Sorteo'],
                          value_vars=['Centena', 'Fijo', 'Primer_Corrido', 'Segundo_Corrido'],
                         var_name='Posicion', value_name='Numero')
        
        pos_map = {
            'Centena': 'Centena',
            'Fijo': 'Fijo',
            'Primer_Corrido': '1er Corrido',
            'Segundo_Corrido': '2do Corrido'
        }
        df_long['Posicion'] = df_long['Posicion'].map(pos_map)
        
        df_historial = df_long.dropna(subset=['Numero']).copy()
        df_historial['Numero'] = df_historial['Numero'].astype(int)
        
        draw_order_map = {'M': 0, 'T': 1, 'N': 2}
        df_historial['draw_order'] = df_historial['Tipo_Sorteo'].map(draw_order_map)
        df_historial['sort_key'] = df_historial['Fecha'] + pd.to_timedelta(df_historial['draw_order'], unit='h')
        df_historial = df_historial.sort_values(by='sort_key').reset_index(drop=True)
        df_historial.drop(columns=['draw_order', 'sort_key'], inplace=True)
        
        if len(df_historial) > 1000:
            df_historial = df_historial.tail(1000).reset_index(drop=True)
        
        return df_historial, None

    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(columns=["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"]), None

# =============================================================================
# ANALIZAR ESTABILIDAD DE NÚMEROS (SOLO FIJO)
# =============================================================================
def analizar_estabilidad_numeros(df_fijos, dias_analisis=365):
    try:
        fecha_limite = datetime.now() - timedelta(days=dias_analisis)
        df_historico = df_fijos[(df_fijos['Fecha'] >= fecha_limite) & (df_fijos['Posicion'] == 'Fijo')].copy()
        if df_historico.empty:
            return None
        
        estabilidad_data = []
        hoy = datetime.now()
        
        for num in range(100):
            df_num = df_historico[df_historico['Numero'] == num].sort_values('Fecha')
            if len(df_num) < 2:
                max_gap = 9999
                avg_gap = 9999
                std_gap = 0
                gap_actual = (hoy - df_num['Fecha'].max()).days if not df_num.empty else dias_analisis
                estado = "SIN DATOS"
                ultima_fecha = df_num['Fecha'].max() if not df_num.empty else None
            else:
                fechas = df_num['Fecha'].tolist()
                gaps = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas)-1)]
                max_gap = max(gaps) if gaps else 9999
                avg_gap = np.mean(gaps) if gaps else 0
                std_gap = np.std(gaps) if gaps else 0
                ultima_salida = fechas[-1]
                gap_actual = (hoy - ultima_salida).days
                if gap_actual > max_gap:
                    max_gap = gap_actual
                
                if gap_actual == 0:
                    estado = "🔥 EN RACHA"
                elif gap_actual <= avg_gap:
                    estado = "✅ NORMAL"
                elif gap_actual <= avg_gap * 2.0:
                    estado = "⏳ VENCIDO"
                else:
                    estado = "🔴 MUY VENCIDO"
                ultima_fecha = ultima_salida
            
            estabilidad_data.append({
                'Número': f"{num:02d}",
                'Gap Actual': gap_actual,
                'Gap Máximo (Días)': max_gap,
                'Gap Promedio': round(avg_gap, 1),
                'Desviación (Irregularidad)': round(std_gap, 1),
                'Estado': estado,
                'Última Salida': ultima_fecha.strftime('%d/%m/%Y') if ultima_fecha else "N/A"
            })
        
        df_est = pd.DataFrame(estabilidad_data)
        df_est = df_est.sort_values(by=['Gap Máximo (Días)', 'Desviación (Irregularidad)'], ascending=[True, True]).reset_index(drop=True)
        return df_est
    except Exception as e:
        st.error(f"❌ Error en analizar_estabilidad_numeros: {e}")
        return None

# =============================================================================
# ANALIZAR FALTANTES DEL MES (SOLO POSICION FIJO)
# =============================================================================
def analizar_faltantes_mes(df_fijos, mes, anio, sorteos_freq=1000):
    try:
        df_fijos_only = df_fijos[df_fijos['Posicion'] == 'Fijo'].copy()
        hoy = datetime.now()
        fecha_inicio_mes = datetime(anio, mes, 1)
        last_day = calendar.monthrange(anio, mes)[1]
        fecha_fin_mes = datetime(anio, mes, last_day)
        
        if mes == hoy.month and anio == hoy.year:
            fecha_fin_mes = hoy
        
        df_mes = df_fijos_only[(df_fijos_only['Fecha'] >= fecha_inicio_mes) &
                               (df_fijos_only['Fecha'] <= fecha_fin_mes)]
        
        if df_mes.empty:
            return pd.DataFrame(), "No hay datos para este mes", pd.DataFrame()
        
        salidos = set(df_mes['Numero'].unique())
        faltantes = sorted(list(set(range(100)) - salidos))
        
        if not faltantes:
            return pd.DataFrame(), "✅ Todos los números salieron este mes", pd.DataFrame()
        
        df_estabilidad = analizar_estabilidad_numeros(df_fijos_only, 365)
        est_map = {}
        if df_estabilidad is not None:
            for _, row in df_estabilidad.iterrows():
                est_map[row['Número']] = {'Gap': row['Gap Actual'], 'Estado': row['Estado']}
        
        df_reciente = df_fijos_only.tail(sorteos_freq)
        conteo = df_reciente['Numero'].value_counts()
        top_frecuencia = conteo.head(25).index.tolist()
        
        resultados = []
        for num in faltantes:
            est_data = est_map.get(f"{num:02d}", {'Gap': 999, 'Estado': 'SIN DATOS'})
            es_vencido = ("VENCIDO" in est_data['Estado'])
            es_favorito = (num in top_frecuencia)
            freq_val = conteo.get(num, 0)
            
            puntaje = 0
            condiciones = []
            
            if es_vencido:
                puntaje += 30
                condiciones.append("Atrasado")
            if es_favorito:
                puntaje += 30
                condiciones.append("Favorito")
            
            if not es_vencido and not es_favorito:
                puntaje = 50
                condiciones.append("Sin Condiciones")
            
            if puntaje >= 60:
                prioridad = "🔴 ALTA"
            elif puntaje >= 40:
                prioridad = "🟡 MEDIA"
            else:
                prioridad = "⚪ BAJA"
            
            razon = " + ".join(condiciones)
            resultados.append({
                'Número': f"{num:02d}",
                'Prioridad': prioridad,
                'Puntaje': puntaje,
                'Razón': razon,
                'Veces Salidas': freq_val,
                'Estado Estabilidad': est_data['Estado'],
                'Gap Actual': est_data['Gap']
            })
        
        df_res = pd.DataFrame(resultados)
        df_res = df_res.sort_values(['Puntaje', 'Veces Salidas'], ascending=[False, False]).reset_index(drop=True)
        return df_res, None, df_mes
    except Exception as e:
        st.error(f"❌ Error en analizar_faltantes_mes: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), f"Error: {str(e)}", pd.DataFrame()

# =============================================================================
# OBTENER FALTANTES DEL MES ANTERIOR COMPLETO (PARA TOP40)
# =============================================================================
def obtener_faltantes_mes_anterior(df_fijos):
    try:
        mes_anterior, anio_anterior = obtener_mes_anterior_completo(df_fijos)
        if mes_anterior is None:
            return []
        
        df_fijos_only = df_fijos[df_fijos['Posicion'] == 'Fijo'].copy()
        
        fecha_inicio_mes = datetime(anio_anterior, mes_anterior, 1)
        last_day = calendar.monthrange(anio_anterior, mes_anterior)[1]
        fecha_fin_mes = datetime(anio_anterior, mes_anterior, last_day)
        
        df_mes = df_fijos_only[(df_fijos_only['Fecha'] >= fecha_inicio_mes) &
                               (df_fijos_only['Fecha'] <= fecha_fin_mes)]
        
        if df_mes.empty:
            return []
        
        salidos = set(df_mes['Numero'].unique())
        faltantes = list(set(range(100)) - salidos)
        return faltantes
    except:
        return []

def calcular_estado_actual(gap, limite_dinamico):
    if pd.isna(limite_dinamico) or limite_dinamico == 0 or limite_dinamico is None:
        return "Normal"
    if gap > limite_dinamico:
        return "Muy Vencido"
    elif gap > (limite_dinamico * 0.66):
        return "Vencido"
    else:
        return "Normal"

def obtener_df_temperatura(contador):
    if not contador:
        return pd.DataFrame({'Dígito': range(10), 'Frecuencia': 0, 'Temperatura': '🟡 Tibio'})
    df = pd.DataFrame.from_dict(contador, orient='index', columns=['Frecuencia'])
    df = df.reset_index().rename(columns={'index': 'Dígito'})
    df = df.sort_values('Frecuencia', ascending=False).reset_index(drop=True)
    todos_digitos = pd.DataFrame({'Dígito': range(10)})
    df = todos_digitos.merge(df, on='Dígito', how='left').fillna({'Frecuencia': 0})
    df['Temperatura'] = '🟡 Tibio'
    if len(df) >= 3:
        df.loc[0:2, 'Temperatura'] = '🔥 Caliente'
    if len(df) >= 7:
        df.loc[6:9, 'Temperatura'] = '🧊 Frío'
    if len(df) >= 3:
        df.loc[3:5, 'Temperatura'] = '🟡 Tibio'
    return df

def analizar_oportunidad_por_digito(df_historial, fecha_referencia):
    if df_historial.empty:
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}
    df_base_fijos = df_historial[df_historial['Posicion'] == 'Fijo'].copy()
    contador_decenas = Counter()
    contador_unidades = Counter()

    for num in df_base_fijos['Numero']:
        contador_decenas[num // 10] += 1
        contador_unidades[num % 10] += 1

    df_temp_dec = obtener_df_temperatura(contador_decenas)
    df_temp_uni = obtener_df_temperatura(contador_unidades)

    mapa_temp_dec = pd.Series(df_temp_dec.Temperatura.values, index=df_temp_dec.Dígito).to_dict()
    mapa_temp_uni = pd.Series(df_temp_uni.Temperatura.values, index=df_temp_uni.Dígito).to_dict()

    df_hist_estado = df_base_fijos[df_base_fijos['Fecha'] <= fecha_referencia].copy()
    res_dec, res_uni = [], []
    gaps_dec = {}
    gaps_uni = {}
    freq_dec = {}
    freq_uni = {}

    for i in range(10):
        fechas_d = df_hist_estado[df_hist_estado['Numero'] // 10 == i]['Fecha'].sort_values()
        gap_d, prom_d = 0, 0
        if not fechas_d.empty:
            gaps = fechas_d.diff().dt.days.dropna()
            prom_d = gaps.median() if len(gaps) > 0 else 0
            gap_d = (fecha_referencia - fechas_d.max()).days
            gaps_dec[i] = gap_d
        else:
            gaps_dec[i] = 999
        
        ed = calcular_estado_actual(gap_d, prom_d)
        
        fechas_u = df_hist_estado[df_hist_estado['Numero'] % 10 == i]['Fecha'].sort_values()
        gap_u, prom_u = 0, 0
        if not fechas_u.empty:
            gaps = fechas_u.diff().dt.days.dropna()
            prom_u = gaps.median() if len(gaps) > 0 else 0
            gap_u = (fecha_referencia - fechas_u.max()).days
            gaps_uni[i] = gap_u
        else:
            gaps_uni[i] = 999
        
        eu = calcular_estado_actual(gap_u, prom_u)
        freq_dec[i] = contador_decenas.get(i, 0)
        freq_uni[i] = contador_unidades.get(i, 0)
        
        p_base_d = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}.get(ed, 0)
        p_base_u = {'Muy Vencido': 100, 'Vencido': 50, 'Normal': 0}.get(eu, 0)
        
        res_dec.append({
            'Dígito': i,
            'Temperatura': mapa_temp_dec.get(i, '🟡 Tibio'),
             'Estado': ed,
            'Punt. Base': p_base_d,
            'Última Salida': fechas_d.max().strftime('%d/%m') if not fechas_d.empty else '-',
            'Frecuencia': freq_dec[i]
        })
        res_uni.append({
            'Dígito': i,
            'Temperatura': mapa_temp_uni.get(i, '🟡 Tibio'),
            'Estado': eu,
            'Punt. Base': p_base_u,
            'Última Salida': fechas_u.max().strftime('%d/%m') if not fechas_u.empty else '-',
            'Frecuencia': freq_uni[i]
        })

    return pd.DataFrame(res_dec), pd.DataFrame(res_uni), gaps_dec, gaps_uni, freq_dec, freq_uni

def obtener_historial_perfiles_cacheado(df_full, ruta_cache=None):
    if df_full.empty:
        return pd.DataFrame()
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if df_fijos.empty:
        return pd.DataFrame()

    if len(df_fijos) > 1000:
        df_fijos = df_fijos.tail(1000).reset_index(drop=True)

    sort_val_map = {'M': 0, 'T': 1, 'N': 2}
    df_fijos['sort_val'] = df_fijos['Tipo_Sorteo'].map(sort_val_map)
    df_fijos = df_fijos.sort_values(by=['Fecha', 'sort_val']).reset_index(drop=True)
    df_fijos.drop(columns=['sort_val'], inplace=True, errors='ignore')

    df_fijos['ID_Sorteo'] = df_fijos['Fecha'].dt.strftime('%Y-%m-%d') + "_" + df_fijos['Tipo_Sorteo']
    df_fijos = df_fijos.drop_duplicates(subset=['ID_Sorteo'], keep='last').reset_index(drop=True)

    df_cache = pd.DataFrame()
    use_file = ruta_cache and os.path.exists(ruta_cache)

    if use_file:
        try:
            df_cache = pd.read_csv(ruta_cache, parse_dates=['Fecha'], encoding='latin-1')
        except:
            df_cache = pd.DataFrame()

    ids_en_cache = set()
    if not df_cache.empty and 'Fecha' in df_cache.columns and 'Sorteo' in df_cache.columns:
        sorteo_map_inv = {'Mañana': 'M', 'Tarde': 'T', 'Noche': 'N', 'M': 'M', 'T': 'T', 'N': 'N'}
        df_cache['ID_Sorteo'] = df_cache['Fecha'].astype(str).str[:10] + "_" + df_cache['Sorteo'].astype(str).map(sorteo_map_inv)
        ids_en_cache = set(df_cache['ID_Sorteo'].dropna())

    df_nuevos = df_fijos[~df_fijos['ID_Sorteo'].isin(ids_en_cache)].copy()

    if df_nuevos.empty:
        cols_to_drop = [c for c in ['ID_Sorteo'] if c in df_cache.columns]
        if cols_to_drop:
            df_cache = df_cache.drop(columns=cols_to_drop)
        return df_cache

    df_nuevos = df_nuevos.sort_values(by=['Fecha', 'Tipo_Sorteo'])
    hist_decenas = defaultdict(list)
    hist_unidades = defaultdict(list)

    if not df_cache.empty and 'Fecha' in df_cache.columns:
        sort_val_inv = {'Mañana': 0, 'Tarde': 1, 'Noche': 2, 'M': 0, 'T': 1, 'N': 2}
        df_cache['sort_val'] = df_cache['Sorteo'].astype(str).map(sort_val_inv).fillna(0)
        df_cache_sorted = df_cache.sort_values(by=['Fecha', 'sort_val'])
        for _, row in df_cache_sorted.iterrows():
            try:
                num = int(row['Numero'])
                fecha = pd.to_datetime(row['Fecha'])
                hist_decenas[num // 10].append(fecha)
                hist_unidades[num % 10].append(fecha)
            except:
                continue

    nuevos_registros = []
    for idx, row in df_nuevos.iterrows():
        try:
            fecha_actual = pd.to_datetime(row['Fecha'])
            num_actual = int(row['Numero'])
            tipo_actual = str(row['Tipo_Sorteo']).upper()
            dec = num_actual // 10
            uni = num_actual % 10
            
            fechas_dec_ant = [f for f in hist_decenas[dec] if f < fecha_actual]
            if fechas_dec_ant:
                last_dec = max(fechas_dec_ant)
                gap_dec = (fecha_actual - last_dec).days
                sorted_fds = sorted(fechas_dec_ant)
                gaps_d = [(sorted_fds[i] - sorted_fds[i-1]).days for i in range(1, len(sorted_fds))]
                med_d = np.median(gaps_d) if gaps_d else 0
                estado_dec = calcular_estado_actual(gap_dec, med_d)
            else:
                estado_dec = "Normal"
            
            fechas_uni_ant = [f for f in hist_unidades[uni] if f < fecha_actual]
            if fechas_uni_ant:
                last_uni = max(fechas_uni_ant)
                gap_uni = (fecha_actual - last_uni).days
                sorted_fus = sorted(fechas_uni_ant)
                gaps_u = [(sorted_fus[i] - sorted_fus[i-1]).days for i in range(1, len(sorted_fus))]
                med_u = np.median(gaps_u) if gaps_u else 0
                estado_uni = calcular_estado_actual(gap_uni, med_u)
            else:
                estado_uni = "Normal"
            
            perfil = f"{estado_dec}-{estado_uni}"
            nombre_sorteo = {'M': 'Mañana', 'T': 'Tarde', 'N': 'Noche'}.get(tipo_actual, 'Otro')
            
            nuevos_registros.append({
                'Fecha': fecha_actual,
                'Sorteo': nombre_sorteo,
                'Numero': num_actual,
                'Perfil': perfil
            })
            
            hist_decenas[dec].append(fecha_actual)
            hist_unidades[uni].append(fecha_actual)
        except Exception as e:
            continue

    if nuevos_registros:
        df_nuevos_cache = pd.DataFrame(nuevos_registros)
        if not df_cache.empty:
            cols_to_drop = [c for c in ['ID_Sorteo', 'sort_val'] if c in df_cache.columns]
            df_final = pd.concat([df_cache.drop(columns=cols_to_drop, errors='ignore'), df_nuevos_cache], ignore_index=True)
        else:
            df_final = df_nuevos_cache
        
        if ruta_cache:
            try:
                df_final.to_csv(ruta_cache, index=False, encoding='latin-1')
            except:
                pass
        return df_final
    else:
        cols_to_drop = [c for c in ['ID_Sorteo'] if c in df_cache.columns]
        if cols_to_drop:
            df_cache = df_cache.drop(columns=cols_to_drop)
        return df_cache

def calcular_estabilidad_historica_digitos(df_full):
    if df_full.empty:
        return pd.DataFrame()
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if len(df_fijos) > 1000:
        df_fijos = df_fijos.tail(1000)

    resultados = []
    for i in range(10):
        fechas_d = df_fijos[df_fijos['Numero'] // 10 == i]['Fecha'].sort_values()
        if len(fechas_d) > 1:
            gaps = fechas_d.diff().dt.days.dropna()
            if len(gaps) > 0:
                med = gaps.median()
                excesos = sum(g > (med * 1.5) for g in gaps)
                estabilidad = 100 - (excesos / len(gaps) * 100)
            else:
                estabilidad = 50
        else:
            estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Decena', 'EstabilidadHist': round(estabilidad, 1)})
        
        fechas_u = df_fijos[df_fijos['Numero'] % 10 == i]['Fecha'].sort_values()
        if len(fechas_u) > 1:
            gaps = fechas_u.diff().dt.days.dropna()
            if len(gaps) > 0:
                med = gaps.median()
                excesos = sum(g > (med * 1.5) for g in gaps)
                estabilidad = 100 - (excesos / len(gaps) * 100)
            else:
                estabilidad = 50
        else:
            estabilidad = 50
        resultados.append({'Digito': i, 'Tipo': 'Unidad', 'EstabilidadHist': round(estabilidad, 1)})

    return pd.DataFrame(resultados)

def pre_calcular_distribuciones_perfiles(df_historial_perfiles):
    if df_historial_perfiles.empty:
        return {}
    distribuciones = {}
    perfiles_unicos = df_historial_perfiles['Perfil'].unique()

    for perfil in perfiles_unicos:
        df_perfil = df_historial_perfiles[df_historial_perfiles['Perfil'] == perfil].copy()
        if df_perfil.empty:
            distribuciones[perfil] = {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3}
            continue
        
        estados_calculados = []
        fechas = df_perfil['Fecha'].sort_values().tolist()
        
        for i, fecha_actual in enumerate(fechas):
            if i == 0:
                estados_calculados.append('Normal')
                continue
            
            fechas_previas = fechas[:i]
            gaps = []
            for j in range(1, len(fechas_previas)):
                gap = (fechas_previas[j] - fechas_previas[j-1]).days
                if gap >= 0:
                    gaps.append(gap)
            
            if len(gaps) == 0:
                estados_calculados.append('Normal')
                continue
            
            if len(gaps) >= 4:
                limite_dinamico = int(np.percentile(gaps, 75))
            else:
                limite_dinamico = int(np.median(gaps) * 2)
            
            gap_actual = (fecha_actual - fechas_previas[-1]).days
            estado = calcular_estado_actual(gap_actual, limite_dinamico)
            estados_calculados.append(estado)
        
        if len(estados_calculados) > 0:
            contador = Counter(estados_calculados)
            total = len(estados_calculados)
            distribucion = {estado: (count / total * 100) for estado, count in contador.items()}
            estado_comun = max(distribucion.keys(), key=lambda k: distribucion[k])
            
            if distribucion[estado_comun] < 60:
                estado_comun = 'Ninguno'
            
            distribuciones[perfil] = {
                'Normal': distribucion.get('Normal', 0),
                'Vencido': distribucion.get('Vencido', 0),
                'Muy Vencido': distribucion.get('Muy Vencido', 0),
                'Estado_Comun': estado_comun,
                'porcentaje': distribucion.get(estado_comun, 0)
            }
        else:
            distribuciones[perfil] = {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3}

    return distribuciones

def calcular_p75_perfil(df_historial_perfiles, perfil_objetivo):
    if df_historial_perfiles.empty:
        return 0
    df_perfil = df_historial_perfiles[df_historial_perfiles['Perfil'] == perfil_objetivo].copy()
    if len(df_perfil) < 2:
        return 0

    fechas = df_perfil['Fecha'].sort_values()
    gaps = fechas.diff().dt.days.dropna()
    return int(np.percentile(gaps, 75)) if len(gaps) >= 4 else int(gaps.median() * 2) if len(gaps) > 0 else 0

def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia, distribuciones_cache=None):
    if df_historial_perfiles.empty:
        return pd.DataFrame(), Counter(), None
    if distribuciones_cache is None:
        distribuciones_cache = pre_calcular_distribuciones_perfiles(df_historial_perfiles)

    historial_fechas_perfiles = defaultdict(list)
    ultimo_suceso_perfil = {}
    transiciones = Counter()
    ultimo_perfil_global = None

    sort_val = {'Mañana': 0, 'Tarde': 1, 'Noche': 2, 'M': 0, 'T': 1, 'N': 2}
    df_historial_perfiles = df_historial_perfiles.copy()
    df_historial_perfiles['sort_val'] = df_historial_perfiles['Sorteo'].astype(str).map(sort_val).fillna(0)
    df_historial_perfiles = df_historial_perfiles.sort_values(by=['Fecha', 'sort_val'])

    for _, row in df_historial_perfiles.iterrows():
        perfil = str(row['Perfil'])
        fecha = pd.to_datetime(row['Fecha'])
        numero = int(row['Numero'])
        historial_fechas_perfiles[perfil].append(fecha)
        ultimo_suceso_perfil[perfil] = row
        
        if ultimo_perfil_global:
            transiciones[(ultimo_perfil_global, perfil)] += 1
        ultimo_perfil_global = perfil

    total_salidas_perfil = Counter()
    for (origen, destino), count in transiciones.items():
        total_salidas_perfil[origen] += count

    analisis_perfiles = []
    for perfil, fechas in historial_fechas_perfiles.items():
        fechas_ordenadas = sorted(fechas)
        ultima_fecha = fechas_ordenadas[-1]
        gaps = []
        
        for k in range(1, len(fechas_ordenadas)):
            diff = (fechas_ordenadas[k] - fechas_ordenadas[k-1]).days
            if diff >= 0:
                gaps.append(diff)
        
        mediana_gap_actual = np.median(gaps) if gaps else 0
        gap_actual = (fecha_referencia - ultima_fecha).days
        
        if len(gaps) >= 4:
            limite_dinamico = int(np.percentile(gaps, 75))
        elif len(gaps) > 0:
            limite_dinamico = int(mediana_gap_actual * 2)
        else:
            limite_dinamico = 0
        
        estado_actual = calcular_estado_actual(gap_actual, limite_dinamico)
        estados_historicos = [calcular_estado_actual(g, limite_dinamico) for g in gaps] if gaps else []
        total_hist = len(estados_historicos)
        
        count_normal = estados_historicos.count('Normal')
        count_vencido = estados_historicos.count('Vencido')
        count_muy_vencido = estados_historicos.count('Muy Vencido')
        muy_vencidos_count = count_muy_vencido
        estabilidad_actual = ((total_hist - muy_vencidos_count) / total_hist * 100) if total_hist > 0 else 0
        
        alerta_recuperacion = False
        if estabilidad_actual > 60 and estado_actual in ['Vencido', 'Muy Vencido']:
            alerta_recuperacion = True
        
        tiempo_limite = limite_dinamico
        repeticiones = transiciones.get((perfil, perfil), 0)
        total_salidas = total_salidas_perfil.get(perfil, 0)
        prob_repeticion = (repeticiones / total_salidas * 100) if total_salidas > 0 else 0
        semana_activa = "Sí" if estado_actual in ['Vencido', 'Muy Vencido'] else "No"
        
        last_row = ultimo_suceso_perfil[perfil]
        estado_ultima_salida = "Normal"
        estabilidad_ultima_salida = 100.0
        exceso_ultima_salida = 0
        
        if len(gaps) >= 1:
            gap_ultima_espera = gaps[-1]
            if len(gaps) > 1:
                gaps_prev = gaps[:-1]
                if len(gaps_prev) >= 4:
                    lim_prev = int(np.percentile(gaps_prev, 75))
                else:
                    lim_prev = int(np.median(gaps_prev) * 2) if gaps_prev else 0
                estado_ultima_salida = calcular_estado_actual(gap_ultima_espera, lim_prev)
                
                if estado_ultima_salida == "Muy Vencido" and lim_prev > 0:
                    exceso_ultima_salida = int(gap_ultima_espera - lim_prev)
                
                ests_prev = [calcular_estado_actual(g, lim_prev) for g in gaps_prev] if gaps_prev else []
                mv_prev = ests_prev.count('Muy Vencido')
                estabilidad_ultima_salida = ((len(ests_prev) - mv_prev) / len(ests_prev) * 100) if ests_prev else 100.0
            else:
                estado_ultima_salida = "Normal"
                estabilidad_ultima_salida = 100.0
        else:
            estado_ultima_salida = "Normal"
            estabilidad_ultima_salida = 100.0
        
        distribucion_estados = distribuciones_cache.get(perfil, {'Normal': 33.3, 'Vencido': 33.3, 'Muy Vencido': 33.3, 'Estado_Comun': 'Normal', 'porcentaje': 33.3})
        estado_ultima_salida_str = str(estado_ultima_salida)
        porc_ultima_salida = distribucion_estados.get(estado_ultima_salida_str, 33.3)
        fue_atipica = porc_ultima_salida < 20
        
        p75_perfil = calcular_p75_perfil(df_historial_perfiles, perfil)
        dias_desde_ultima_salida = (fecha_referencia - ultima_fecha).days if pd.notna(ultima_fecha) else 999
        dentro_enfriamiento = dias_desde_ultima_salida < p75_perfil if p75_perfil > 0 else False
        es_estado_comun = distribucion_estados.get(estado_actual, 0) > 60
        
        analisis_perfiles.append({
            'Perfil': perfil,
            'Frecuencia': total_hist + 1,
            'Última Fecha': ultima_fecha,
            'Gap Actual': gap_actual,
            'Mediana Gap': int(mediana_gap_actual),
            'Estado Actual': estado_actual,
            'Estabilidad': round(estabilidad_actual, 1),
            'Tiempo Limite': tiempo_limite,
            'Alerta': '⚠️ RECUPERAR' if alerta_recuperacion else '-',
            'Prob Repeticiones %': round(prob_repeticion, 1),
            'Semana Activa': semana_activa,
            'Último Numero': last_row['Numero'],
            'Último Sorteo': last_row['Sorteo'],
            'Veces Normal': count_normal,
            'Veces Vencido': count_vencido,
            'Veces Muy Vencido': count_muy_vencido,
            'Estado Ultima Salida': estado_ultima_salida,
            'Estabilidad Ultima Salida': round(estabilidad_ultima_salida, 1),
            'Exceso Ultima Salida': exceso_ultima_salida,
            'Distribucion_Estados': distribucion_estados,
            'Fue_Atipica': fue_atipica,
            'P75_Perfil': p75_perfil,
            'Dias_Desde_Ultima': dias_desde_ultima_salida,
            'Dentro_Enfriamiento': dentro_enfriamiento,
            'Es_Estado_Comun': es_estado_comun,
            'Porc_Ultima_Salida': porc_ultima_salida
        })

    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil_global

# =============================================================================
# TOP40 CON FALTANTES DEL MES ANTERIOR (+50 PTS BONUS)
# =============================================================================
def obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni, faltantes_mes_anterior=None, debug_mode=False, debug_logs=None):
    if df_stats.empty:
        return list(range(10, 50))
    map_est_dec = {}
    map_est_uni = {}
    if not estabilidad_digitos.empty:
        dec_df = estabilidad_digitos[estabilidad_digitos['Tipo']=='Decena']
        uni_df = estabilidad_digitos[estabilidad_digitos['Tipo']=='Unidad']
        if not dec_df.empty:
            map_est_dec = dec_df.set_index('Digito')['EstabilidadHist'].to_dict()
        if not uni_df.empty:
            map_est_uni = uni_df.set_index('Digito')['EstabilidadHist'].to_dict()

    scores = []
    for _, row in df_stats.iterrows():
        p = str(row['Perfil'])
        score = 0
        estado = str(row['Estado Actual'])
        
        if row['Alerta'] == '⚠️ RECUPERAR':
            score += 150
        elif estado == 'Vencido':
            score += 70
        elif estado == 'Normal':
            score += 50
        elif estado == 'Muy Vencido':
            score += 100
        
        score += row['Estabilidad'] * 0.5
        
        if ultimo_perfil:
            trans_count = transizioni.get((str(ultimo_perfil), p), 0)
            score += trans_count * 10
        
        if row.get('Fue_Atipica', False):
            score -= 50
        if row.get('Dentro_Enfriamiento', False):
            score -= 30
        if row.get('Es_Estado_Comun', False):
            score += 30
        
        scores.append({'Perfil': p, 'Score': int(score), 'Estado': estado})

    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_7 = df_scores.head(7)

    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}

    df_hist_nums = {}
    if not df_historial_perfiles.empty and 'Numero' in df_historial_perfiles.columns:
        df_hist_nums = df_historial_perfiles.groupby('Numero')['Fecha'].max().to_dict()

    candidatos_totales = []
    map_temp_dec = df_oport_dec.set_index('Dígito')['Temperatura'].to_dict() if not df_oport_dec.empty else {}
    map_temp_uni = df_oport_uni.set_index('Dígito')['Temperatura'].to_dict() if not df_oport_uni.empty else {}
    temp_val = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}

    faltantes_set = set(faltantes_mes_anterior) if faltantes_mes_anterior else set()

    for _, row in top_7.iterrows():
        perfil = str(row['Perfil'])
        partes = perfil.split('-')
        if len(partes) != 2:
            continue
        
        ed_req, eu_req = partes[0], partes[1]
        decenas_estado = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
        unidades_estado = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
        
        if not decenas_estado:
            decenas_estado = list(range(10))
        if not unidades_estado: 
            unidades_estado = list(range(10))
        
        total_combinaciones = len(decenas_estado) * len(unidades_estado)
        porc_perfil = distribuciones_perfiles.get(perfil, {}).get('porcentaje', 0)
        
        for d in decenas_estado:
            for u in unidades_estado:
                num = int(f"{d}{u}")
                last_seen = df_hist_nums.get(num, pd.Timestamp('2000-01-01'))
                gap_n = (fecha_ref - last_seen).days if isinstance(last_seen, pd.Timestamp) else 999
                
                score_base = int(row['Score'])
                temp_d = temp_val.get(map_temp_dec.get(d, '🟡 Tibio'), 2)
                temp_u = temp_val.get(map_temp_uni.get(u, '🟡 Tibio'), 2)
                est_d = map_est_dec.get(d, 50)
                est_u = map_est_uni.get(u, 50)
                gap_bonus = min(gap_n / 10, 20)
                
                bonus_salidor = 0
                gap_d = gaps_dec.get(d, 0)
                gap_u = gaps_uni.get(u, 0)
                
                if map_estado_dec.get(d) in ['Vencido', 'Muy Vencido'] and freq_dec.get(d, 0) >= 8:
                    bonus_salidor += min(gap_d / 5, 20)
                if map_estado_dec.get(d) == 'Muy Vencido':
                    bonus_salidor += 10
                if map_estado_uni.get(u) in ['Vencido', 'Muy Vencido'] and freq_uni.get(u, 0) >= 8:
                    bonus_salidor += min(gap_u / 5, 20)
                if map_estado_uni.get(u) == 'Muy Vencido':
                    bonus_salidor += 10
                if (map_estado_dec.get(d) in ['Vencido', 'Muy Vencido'] and freq_dec.get(d, 0) >= 8) and \
                   (map_estado_uni.get(u) in ['Vencido', 'Muy Vencido'] and freq_uni.get(u, 0) >= 8):
                    bonus_salidor += 25
                
                bonus_escasez = 0
                if porc_perfil >= 40:
                    bonus_escasez += 15
                if total_combinaciones < 20:
                    bonus_escasez += 20
                if total_combinaciones < 10:
                    bonus_escasez += 10
                if porc_perfil >= 40:
                    if freq_uni.get(u, 0) >= 8 and gap_u > 15:
                        bonus_escasez += 25
                    if freq_dec.get(d, 0) >= 8 and gap_d > 15:
                        bonus_escasez += 25
                if porc_perfil >= 40 and total_combinaciones < 20 and \
                   (freq_uni.get(u, 0) >= 8 and gap_u > 15):
                    bonus_escasez += 30
                
                bonus_estabilidad = 0
                if map_estado_dec.get(d) == 'Normal' and freq_dec.get(d, 0) < 8:
                    bonus_estabilidad += 5
                if map_estado_uni.get(u) == 'Normal' and freq_uni.get(u, 0) < 8:
                    bonus_estabilidad += 5
                
                bonus_historial = 0
                if num in df_hist_nums:
                    freq_num = sum(1 for n in df_hist_nums.keys() if n == num)
                    if freq_num >= 3:
                        bonus_historial += 10
                    if gap_n > 20:
                        bonus_historial += min(gap_n / 10, 15)
                
                bonus_faltantes = 0
                es_faltante = num in faltantes_set
                if es_faltante:
                    bonus_faltantes = 50
                
                temp_score = temp_d + temp_u + (est_d + est_u) / 20 + gap_bonus + bonus_salidor + bonus_escasez + bonus_estabilidad + bonus_historial + bonus_faltantes
                
                candidatos_totales.append({
                    'Numero': num,
                    'Perfil': perfil,
                    'Score': score_base,
                    'Gap_Num': gap_n,
                    'Temp_Score': temp_score,
                    'Bonus_Salidor': bonus_salidor,
                    'Bonus_Escasez': bonus_escasez,
                    'Bonus_Faltantes': bonus_faltantes,
                    'Es_Faltante': es_faltante
                })

    if not candidatos_totales:
        return list(range(0, 40))

    df_cands = pd.DataFrame(candidatos_totales)
    max_por_perfil = 20
    df_cands = df_cands.drop_duplicates(subset=['Numero'], keep='first')
    df_cands = df_cands.sort_values(['Temp_Score', 'Score', 'Gap_Num', 'Numero'], ascending=[False, False, False, True])

    candidatos_finales = []
    perfil_counter = Counter()
    for _, row in df_cands.iterrows():
        perfil = row['Perfil'] 
        if perfil_counter[perfil] < max_por_perfil:
            candidatos_finales.append(row['Numero'])
            perfil_counter[perfil] += 1
        
        if len(candidatos_finales) >= 40:
            break

    if len(candidatos_finales) < 40:
        for num in range(0, 100):
            if num not in candidatos_finales:
                candidatos_finales.append(num)
            if len(candidatos_finales) >= 40:
                break

    resultado = list(dict.fromkeys(candidatos_finales))[:40]
    if debug_mode and debug_logs is not None:
        debug_logs.append(f"📋 Top 40: {resultado}")
    return resultado

# =============================================================================
# GENERAR SUGERENCIA CON ALERTAS DETALLADAS COMPLETAS
# =============================================================================
def generar_sugerencia_fusionada(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni, faltantes_mes_anterior=None):
    st.subheader("🤖 Sugerencia Inteligente Fusionada")
    with st.expander("📖 ¿Cómo funciona la lógica?", expanded=False):
        st.markdown("""
        **Estados (P75):** Normal | Vencido | Muy Vencido
        **TOP 7:** Usa 7 de 9 perfiles (78% cobertura)
        **TOP 40:** Genera 40 números
        **🆕 Faltantes del Mes Anterior:** Reciben +50 pts bonus
        **Bonus Salidor en Deuda:** Dígitos frecuentes con gap alto
        **Bonus Escasez:** Perfiles frecuentes con pocas combinaciones
        """)

    st.markdown("### 🚨 Detalle de Alertas Activas")
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}

    alertas_activas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR'].copy() if not df_stats.empty else pd.DataFrame()

    if not alertas_activas.empty:
        for _, row_alert in alertas_activas.iterrows():
            perfil_name = str(row_alert['Perfil'])
            partes = perfil_name.split('-')
            if len(partes) != 2:
                continue
            
            ed_req, eu_req = partes[0], partes[1]
            decenas_cumplen = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
            unidades_cumplen = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
            nums_alerta = [f"{int(f'{d}{u}'):02d}" for d in decenas_cumplen for u in unidades_cumplen]
            
            gap = row_alert['Gap Actual']
            med = row_alert['Mediana Gap']
            estado = row_alert['Estado Actual']
            tiempo_limite = row_alert['Tiempo Limite']
            ult_fecha = row_alert['Última Fecha']
            ult_estado = row_alert['Estado Ultima Salida']
            ult_estabilidad = row_alert['Estabilidad Ultima Salida']
            ult_exceso = row_alert['Exceso Ultima Salida']
            
            time_str = ""
            if estado == "Normal":
                falta = int(med - gap) if med > gap else 0
                time_str = f"🟢 Faltan ~{falta} días para Vencido"
            elif estado == "Vencido":
                falta_mv = int(tiempo_limite - gap) if tiempo_limite > gap else 0
                exceso = int(gap - med)
                time_str = f"🟠 Exceso: {exceso} días | Faltan {falta_mv} para Muy Vencido"
            elif estado == "Muy Vencido":
                exceso = int(gap - tiempo_limite) if tiempo_limite > 0 else gap
                time_str = f"🔴 +{exceso} días sobre límite P75"
            
            fue_atipica = row_alert.get('Fue_Atipica', False)
            dentro_enfriamiento = row_alert.get('Dentro_Enfriamiento', False)
            es_comun = row_alert.get('Es_Estado_Comun', False)
            porc_ultima = row_alert.get('Porc_Ultima_Salida', 0)
            distribucion = row_alert.get('Distribucion_Estados', {})
            
            with st.container(border=True):
                st.markdown(f"**Perfil Alertado: `{perfil_name}`**")
                st.markdown(f"📍 **Estado Actual:** `{estado}` | ⏳ {time_str}")
                st.markdown(f"📉 **Estabilidad Actual:** `{row_alert['Estabilidad']}%`")
                st.markdown(f"📉 **Estabilidad Última Salida:** `{ult_estabilidad}%`")
                st.markdown(f"📊 **Datos:** Gap: **{gap} días** | Límite P75: **{tiempo_limite} días** | Mediana: **{med} días**")
                
                if distribucion:
                    dist_str = " | ".join([f"{k}: {v:.1f}%" for k, v in distribucion.items() if isinstance(v, (int, float))])
                    st.markdown(f"📈 **Distribución Histórica:** {dist_str}")
                
                if fue_atipica:
                    st.warning(f"⚠️ **Estado Atípico Detectado:** La última salida fue en estado '{ult_estado}' (solo {porc_ultima:.1f}% del histórico)")
                
                if dentro_enfriamiento:
                    st.warning(f"⏳ **Dentro de Enfriamiento:** Faltan {row_alert['P75_Perfil'] - row_alert['Dias_Desde_Ultima']} días")
                
                if es_comun:
                    st.success(f"✅ **Estado Común:** Este perfil suele salir en estado '{estado}'")
                
                st.markdown(f"🔢 **Decenas ({ed_req}):** `{decenas_cumplen}` | **Unidades ({eu_req}):** `{unidades_cumplen}`")
                
                if nums_alerta:
                    with st.expander(f"🔢 Ver {len(nums_alerta)} números de esta alerta"):
                        chunks = [nums_alerta[x:x+10] for x in range(0, len(nums_alerta), 10)]
                        for chunk in chunks:
                            st.write(f"`{' '.join(chunk)}`")
                
                st.info("🔙 **Historia de la última salida de este perfil:**")
                try:
                    fecha_fmt = ult_fecha.strftime('%d/%m/%Y') if pd.notna(ult_fecha) else 'N/A'
                    st.markdown(f"- Fecha: **{fecha_fmt}**")
                except:
                    st.markdown(f"- Fecha: **{ult_fecha}**")
                st.markdown(f"- Estado en ese momento: **{ult_estado}**")
                st.markdown(f"- Estabilidad previa: **{ult_estabilidad}%**")
                if ult_estado == "Muy Vencido" and ult_exceso > 0:
                    st.markdown(f"- Días en exceso: **{ult_exceso} días**")
                
                st.markdown("---")
    else:
        st.info("✅ No hay alertas activas.")

    st.markdown("### 🎲 Top 40 Números Sugeridos")
    lista_nums = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni, faltantes_mes_anterior)

    if not lista_nums:
        st.warning("⚠️ No se generaron candidatos.")
        return

    def get_state_color_hex(state):
        return {'Normal': '#22c55e', 'Vencido': '#f59e0b', 'Muy Vencido': '#ef4444'}.get(str(state), '#94a3b8')

    def shorten_state(text): 
        return {"Muy Vencido": "M.Vencido", "Vencido": "Vencido", "Normal": "Normal"}.get(str(text), str(text))

    cols = st.columns(8)
    for idx, num in enumerate(lista_nums):
        try:
            es_faltante = num in (faltantes_mes_anterior if faltantes_mes_anterior else [])
            borde = "2px solid #ff6b6b" if es_faltante else "1px solid #444"
            
            d_int, u_int = int(num // 10), int(num % 10)
            ed, eu = map_estado_dec.get(d_int, "?"), map_estado_uni.get(u_int, "?")
            
            cols[idx % 8].markdown(f"""
             <div style="background-color:#1e1e1e; padding:12px; border-radius:10px; text-align:center; border:{borde};">
             <h3 style="margin:0; color:#ff6b6b; font-size:1.4em;">{num:02d}{'🔥' if es_faltante else ''}</h3>
             <div style="font-size:0.8em; color:{get_state_color_hex(str(ed))};">{shorten_state(str(ed))}</div>
             <div style="font-size:0.8em; color:{get_state_color_hex(str(eu))};">{shorten_state(str(eu))}</div>
             </div>
             """, unsafe_allow_html=True)
        except:
            continue

    return lista_nums

# =============================================================================
# GUARDAR PREDICCION EN HISTORICO
# =============================================================================
def guardar_prediccion_en_historico(fecha, sorteo, top40, perfil_top, score_promedio, numero_real=None):
    archivo = RUTA_HISTORICO
    if numero_real is not None:
        estuvo_en_top40 = numero_real in top40
        acierto = '✅' if estuvo_en_top40 else '❌'
        posicion = str(top40.index(numero_real) + 1) if estuvo_en_top40 else 'N/A'
    else:
        acierto = 'PENDIENTE'
        posicion = 'PENDIENTE'

    sorteo_nombre = 'Mañana' if str(sorteo).upper() == 'M' else ('Tarde' if str(sorteo).upper() == 'T' else ('Noche' if str(sorteo).upper() == 'N' else str(sorteo)))

    if hasattr(fecha, 'strftime'):
        fecha_str = fecha.strftime('%Y-%m-%d')
    else:
        fecha_str = str(fecha)[:10]

    if os.path.exists(archivo):
        try:
            df_existente = pd.read_csv(archivo, encoding='utf-8')
            if not df_existente.empty:
                mask_pendiente = (
                    (df_existente['Fecha'].astype(str).str[:10] == fecha_str) &
                    (df_existente['Sorteo'].astype(str) == sorteo_nombre) &
                    (df_existente['Acierto'].astype(str) == 'PENDIENTE')
                )
                if mask_pendiente.any():
                    return False, "duplicado"
        except:
            pass

    existe = os.path.exists(archivo)
    try:
        with open(archivo, 'a', encoding='utf-8', newline='') as f:
            if not existe:
                f.write('Fecha,Sorteo,Top40,Perfil_Top,Score_Promedio,Numero_Real,Acierto,Posicion\n')
            linea = f'{fecha_str},{sorteo_nombre},"{top40}",{perfil_top},{score_promedio},{numero_real if numero_real else "PENDIENTE"},{acierto},{posicion}\n'
            f.write(linea)
            f.flush()
        return True, "creado"
    except Exception as e:
        st.error(f"❌ Error guardando: {e}")
        return False, "error"

def leer_historico_predicciones():
    if not os.path.exists(RUTA_HISTORICO):
        return pd.DataFrame()
    try:
        df = pd.read_csv(RUTA_HISTORICO, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"❌ Error leyendo histórico: {e}")
        return pd.DataFrame()

def actualizar_resultados_pendientes(numero_real, fecha, sorteo):
    if not os.path.exists(RUTA_HISTORICO):
        return 0
    try:
        df = pd.read_csv(RUTA_HISTORICO, encoding='utf-8')
        if df.empty:
            return 0
        
        sorteo_nombre = 'Mañana' if str(sorteo).upper() == 'M' else ('Tarde' if str(sorteo).upper() == 'T' else ('Noche' if str(sorteo).upper() == 'N' else str(sorteo)))
        
        if hasattr(fecha, 'strftime'):
            fecha_str = fecha.strftime('%Y-%m-%d')
        else:
            fecha_str = str(fecha)[:10]
        
        mask_pendiente = (
            (df['Fecha'].astype(str).str[:10] == fecha_str) &
            (df['Sorteo'].astype(str) == sorteo_nombre) &
            (df['Acierto'].astype(str) == 'PENDIENTE')
        )
        
        if mask_pendiente.any():
            actualizados = 0
            for idx in df[mask_pendiente].index:
                try:
                    top40_str = str(df.loc[idx, 'Top40']).strip('[]').strip('"')
                    top40 = [int(x.strip()) for x in top40_str.split(',') if x.strip()]
                    estuvo_en_top40 = numero_real in top40
                    df.loc[idx, 'Numero_Real'] = str(numero_real)
                    df.loc[idx, 'Acierto'] = '✅' if estuvo_en_top40 else '❌'
                    df.loc[idx, 'Posicion'] = str(top40.index(numero_real) + 1) if estuvo_en_top40 else 'N/A'
                    actualizados += 1
                except:
                    pass
            
            with open(RUTA_HISTORICO, 'w', encoding='utf-8', newline='') as f:
                f.write('Fecha,Sorteo,Top40,Perfil_Top,Score_Promedio,Numero_Real,Acierto,Posicion\n')
                for _, row in df.iterrows():
                    linea = f'{row["Fecha"]},{row["Sorteo"]},"{row["Top40"]}",{row["Perfil_Top"]},{row["Score_Promedio"]},{row["Numero_Real"]},{row["Acierto"]},{row["Posicion"]}\n'
                    f.write(linea)
            return actualizados
    except Exception as e:
        st.error(f"❌ Error actualizando: {e}")
        return 0

# =============================================================================
# DASHBOARD EFECTIVIDAD
# =============================================================================
def mostrar_dashboard_efectividad():
    st.header("📊 Dashboard de Efectividad")
    st.markdown("Análisis automático de predicciones anteriores.")
    df = leer_historico_predicciones()
    with st.expander("🐛 DEBUG: Ver contenido del histórico", expanded=True):
        st.write(f"**📁 Archivo:** {RUTA_HISTORICO}")
        st.write(f"**✅ Archivo existe:** {os.path.exists(RUTA_HISTORICO)}")
        if os.path.exists(RUTA_HISTORICO):
            st.write(f"**📏 Tamaño:** {os.path.getsize(RUTA_HISTORICO)} bytes")
            st.write(f"**📊 Total filas:** {len(df)}")
            if len(df) > 0:
                st.write(f"**📈 Con resultado:** {len(df[df['Acierto'] != 'PENDIENTE'])}")
                st.write(f"**⏳ Pendientes:** {len(df[df['Acierto'] == 'PENDIENTE'])}")
                st.write(f"**📋 Columnas:** {list(df.columns)}")

    if df.empty:
        st.info("ℹ️ No hay predicciones guardadas aún. Ejecuta análisis primero.")
        return

    df_completas = df[df['Acierto'] != 'PENDIENTE'].copy()
    if len(df_completas) == 0:
        st.warning("⚠️ Aún no hay predicciones con resultado. Agrega el sorteo real después de generar una predicción.")
        return

    col1, col2, col3 = st.columns(3)
    total = len(df_completas)
    aciertos = len(df_completas[df_completas['Acierto'] == '✅'])
    efectividad = (aciertos / total * 100) if total > 0 else 0

    col1.metric("📊 Total Predicciones", total)
    col2.metric("✅ Aciertos", aciertos)
    col3.metric("📈 Efectividad General", f"{efectividad:.1f}%")

    st.markdown("---")
    st.subheader("📈 Efectividad por Perfil")
    if 'Perfil_Top' in df_completas.columns and df_completas['Perfil_Top'].notna().any():
        perfil_stats = df_completas.groupby('Perfil_Top').apply(
            lambda x: pd.Series({
                'Total': len(x),
                'Aciertos': len(x[x['Acierto'] == '✅']),
                'Efectividad': round(len(x[x['Acierto'] == '✅']) / len(x) * 100, 1) if len(x) > 0 else 0
            })
        ).reset_index()
        if not perfil_stats.empty and len(perfil_stats) > 0:
            perfil_stats = perfil_stats.sort_values('Efectividad', ascending=False)
            for _, row in perfil_stats.iterrows():
                color = "🟢" if row['Efectividad'] >= 40 else ("🟡" if row['Efectividad'] >= 25 else "🔴")
                st.markdown(f"{color} **{row['Perfil_Top']}**: {row['Efectividad']}% ({row['Aciertos']} de {row['Total']})")

    st.markdown("---")
    st.subheader("📈 Efectividad por Rango de Score")
    if 'Score_Promedio' in df_completas.columns and df_completas['Score_Promedio'].notna().any():
        score_dict = {}
        for label in ['<150', '150-200', '200-250', '>250']:
            if label == '<150':
                mask = df_completas['Score_Promedio'] < 150
            elif label == '150-200':
                mask = (df_completas['Score_Promedio'] >= 150) & (df_completas['Score_Promedio'] < 200)
            elif label == '200-250':
                mask = (df_completas['Score_Promedio'] >= 200) & (df_completas['Score_Promedio'] < 250)
            else:
                mask = df_completas['Score_Promedio'] >= 250
            total_en_rango = len(df_completas[mask])
            aciertos_en_rango = len(df_completas[mask & (df_completas['Acierto'] == '✅')])
            efectividad_en_rango = round(aciertos_en_rango / total_en_rango * 100, 1) if total_en_rango > 0 else 0
            score_dict[label] = {
                'Total': total_en_rango,
                'Aciertos': aciertos_en_rango,
                'Efectividad': efectividad_en_rango
            }
        for label, datos in score_dict.items():
            if datos['Total'] > 0:
                color = "🟢" if datos['Efectividad'] >= 40 else ("🟡" if datos['Efectividad'] >= 25 else "🔴")
                st.markdown(f"{color} **Score {label}**: {datos['Efectividad']}% ({datos['Aciertos']} de {datos['Total']})")

    with st.expander("📋 Ver Histórico Completo"):
        st.dataframe(df.sort_values('Fecha', ascending=False).head(50), hide_index=True, use_container_width=True)

def ejecutar_backtest(df_full, sorteos_objetivo, nombre_sesion, debug_mode=False):
    if df_full.empty:
        st.warning(f"⚠️ No hay datos para backtest de {nombre_sesion}")
        return pd.DataFrame(), 0, 0, []
    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if df_fijos.empty:
        return pd.DataFrame(), 0, 0, []

    if len(df_fijos) > 1000:
        df_fijos = df_fijos.tail(1000).reset_index(drop=True)

    df_fijos['sort_val'] = df_fijos['Tipo_Sorteo'].map({'M': 0, 'T': 1, 'N': 2})
    df_fijos = df_fijos.sort_values(by=['Fecha', 'sort_val'], ascending=[False, False]).reset_index(drop=True)

    sorteos_a_procesar = min(sorteos_objetivo, len(df_fijos))
    if sorteos_a_procesar < 1:
        return pd.DataFrame(), 0, 0, []

    resultados, aciertos, total_sorteos = [], 0, 0
    debug_logs = [] if debug_mode else None
    df_cache_full = obtener_historial_perfiles_cacheado(df_full)
    distribuciones_cache = pre_calcular_distribuciones_perfiles(df_cache_full) if not df_cache_full.empty else {}
    estabilidad_digitos_global = calcular_estabilidad_historica_digitos(df_full)

    progress_bar = st.progress(0)
    start_time = time.time()

    for i in range(sorteos_a_procesar):
        sorteo_actual = df_fijos.iloc[i]
        fecha_ref, tipo_sorteo = sorteo_actual['Fecha'], sorteo_actual['Tipo_Sorteo']
        resultado_real = int(sorteo_actual['Numero'])
        total_sorteos += 1
        
        df_historial = df_full[df_full['Fecha'] < fecha_ref].copy()
        if len(df_historial) > 1000:
            df_historial = df_historial.tail(1000)
        
        if df_historial.empty:
            resultados.append({
                'Fecha': fecha_ref.strftime('%d/%m/%Y'),
                'Sorteo': tipo_sorteo,
                'Real': f"{resultado_real:02d}",
                'En Top40': '⚠️'
            })
            progress_bar.progress((i + 1) / sorteos_a_procesar)
            continue
        
        df_oport_dec, df_oport_uni, gaps_dec, gaps_uni, freq_dec, freq_uni = analizar_oportunidad_por_digito(df_historial, fecha_ref)
        df_historial_perfiles = df_cache_full[df_cache_full['Fecha'] < fecha_ref].copy() if not df_cache_full.empty else pd.DataFrame()
        
        if df_historial_perfiles.empty:
            df_stats, transizioni, ultimo_perfil = pd.DataFrame(), Counter(), None
        else:
            df_stats, transizioni, ultimo_perfil = analizar_estadisticas_perfiles(df_historial_perfiles, fecha_ref, distribuciones_cache=distribuciones_cache)
        
        faltantes_mes = obtener_faltantes_mes_anterior(df_historial)
        prediccion = obtener_prediccion_numeros_lista(
            df_stats, transizioni, ultimo_perfil,
            df_oport_dec, df_oport_uni,
            df_historial_perfiles, fecha_ref,
            estabilidad_digitos_global,
            gaps_dec, gaps_uni,
            distribuciones_cache,
            freq_dec, freq_uni,
            faltantes_mes
        )
        
        es_acierto = numero_en_lista(resultado_real, prediccion)
        if es_acierto:
            aciertos += 1
        
        if debug_mode:
            debug_logs.append({
                'Fecha': fecha_ref.strftime('%d/%m/%Y'),
                'Sorteo': tipo_sorteo,
                'Real': resultado_real,
                'En Top40': '✅' if es_acierto else '❌' 
            })
        
        resultados.append({
            'Fecha': fecha_ref.strftime('%d/%m/%Y'),
            'Sorteo': tipo_sorteo,
            'Real': f"{resultado_real:02d}",
            'En Top40': '✅' if es_acierto else '❌'
        })
        
        elapsed = time.time() - start_time
        progress_bar.progress((i + 1) / sorteos_a_procesar)
        if elapsed > 120:
            st.warning("⚠️ Backtest excedió 120 segundos.")
            break

    progress_bar.empty()
    return pd.DataFrame(resultados), aciertos, total_sorteos, debug_logs if debug_mode else []

def mostrar_tabla_personalidad_perfiles(df_historial_perfiles):
    st.header("📊 Comportamiento Histórico de Perfiles")
    if df_historial_perfiles.empty:
        st.info("ℹ️ No hay datos suficientes.")
        return
    distribuciones = pre_calcular_distribuciones_perfiles(df_historial_perfiles)
    if not distribuciones:
        st.info("ℹ️ No hay perfiles.")
        return

    filas_tabla = []
    for perfil, dist in distribuciones.items():
        estado_comun = dist.get('Estado_Comun', 'Ninguno')
        recomendacion = "✅ Jugar en " + estado_comun if estado_comun in ['Normal', 'Vencido', 'Muy Vencido'] else "⚠️ Sin patrón"
        filas_tabla.append({
            'Perfil': perfil,
            'Normal': f"{dist.get('Normal', 0):.1f}%",
            'Vencido': f"{dist.get('Vencido', 0):.1f}%",
            'Muy Vencido': f"{dist.get('Muy Vencido', 0):.1f}%",
            'Estado Común': estado_comun,
            'Recomendación': recomendacion
        })

    df_tabla = pd.DataFrame(filas_tabla)
    st.dataframe(df_tabla, hide_index=True, use_container_width=True)
    st.caption(f"📊 {len(filas_tabla)} perfiles | últimos 1000 sorteos")

# =============================================================================
# MOSTRAR ÚLTIMOS RESULTADOS EN SIDEBAR
# =============================================================================
def mostrar_ultimos_resultados_sidebar(df_full):
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Últimos Resultados")
    if df_full.empty:
        st.sidebar.info("ℹ️ Sin datos")
        return

    df_fijos = df_full[df_full['Posicion'] == 'Fijo'].copy()
    if df_fijos.empty:
        st.sidebar.info("ℹ️ Sin sorteos")
        return

    for sesion, nombre in [('M', '🌅 Mañana'), ('T', '☀️ Tarde'), ('N', '🌙 Noche')]:
        df_sesion = df_fijos[df_fijos['Tipo_Sorteo'] == sesion].sort_values('Fecha', ascending=False)
        
        if not df_sesion.empty:
            ultimo = df_sesion.iloc[0]
            with st.sidebar.container(border=True):
                st.sidebar.markdown(f"**{nombre}**")
                st.sidebar.markdown(f"📅 {ultimo['Fecha'].strftime('%d/%m/%Y') if pd.notna(ultimo['Fecha']) else 'N/A'}")
                st.sidebar.markdown(f"🔢 **Fijo:** `{int(ultimo['Numero']):02d}`")
                
                fecha_sorteo = ultimo['Fecha']
                df_sorteo_completo = df_full[(df_full['Fecha'] == fecha_sorteo) & (df_full['Tipo_Sorteo'] == sesion)]
                
                corridos = {}
                for _, row in df_sorteo_completo.iterrows():
                    if row['Posicion'] == '1er Corrido':
                        corridos['C1'] = int(row['Numero'])
                    elif row['Posicion'] == '2do Corrido':
                        corridos['C2'] = int(row['Numero'])
                
                if 'C1' in corridos:
                    st.sidebar.markdown(f"🎯 **C1:** `{corridos['C1']:02d}`")
                if 'C2' in corridos:
                    st.sidebar.markdown(f"🎯 **C2:** `{corridos['C2']:02d}`")

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.sidebar.header("⚙️ Panel de Control")
    st.session_state.debug_mode = st.sidebar.checkbox("🐛 Modo Debug", value=st.session_state.debug_mode)

    if st.sidebar.button("🔄 Forzar Recarga", type="primary", key="btn_forzar_recarga"):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            if key not in ['debug_mode']:
                del st.session_state[key]
        if os.path.exists(RUTA_CACHE):
            try:
                os.remove(RUTA_CACHE)
            except:
                pass
        time.sleep(0.5)
        st.rerun()

    file_signature = None
    if os.path.exists(RUTA_CSV):
        try:
            file_signature = get_file_signature(RUTA_CSV)
            file_size = os.path.getsize(RUTA_CSV)
            file_mod = datetime.fromtimestamp(os.path.getmtime(RUTA_CSV)).strftime('%d/%m %H:%M')
            st.sidebar.caption(f"📄 {RUTA_CSV}\n📏 {file_size} bytes\n🕐 {file_mod}")
        except Exception as e:
            st.sidebar.error(f"Error leyendo archivo: {e}")
            file_signature = None
    else:
        st.sidebar.warning(f"⚠️ {RUTA_CSV} no encontrado")
        inicializar_archivo(RUTA_CSV, ["Fecha", "Tipo_Sorteo", "Centena", "Fijo", "Primer_Corrido", "Segundo_Corrido"])
        st.info(f"✅ Archivo creado. Agrega tu primer sorteo.")
        st.stop()

    df_full, invalid_dates_df = cargar_datos_geotodo(RUTA_CSV, file_signature)
    st.session_state.invalid_dates_df = invalid_dates_df

    if st.session_state.invalid_dates_df is not None and len(st.session_state.invalid_dates_df) > 0:
        st.warning(f"⚠️ {len(st.session_state.invalid_dates_df)} filas con fechas inválidas")
        st.dataframe(st.session_state.invalid_dates_df.head(10), hide_index=True)

    st.markdown("---")

    with st.sidebar.expander("📝 Agregar Sorteo", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            f_nueva = st.date_input("Fecha", datetime.now().date(), key="date_input_nuevo")
        with col_f2:
            ses = st.radio("Sesión", ["Mañana", "Tarde", "Noche"], horizontal=True, key="sesion_radio_nuevo")
        
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            cent = st.number_input("Centena", 0, 999, 0, key="inp_cent_nuevo")
            fij = st.number_input("Fijo", 0, 99, 0, key="inp_fijo_nuevo")
        with col_n2:
            c1 = st.number_input("C1", 0, 99, 0, key="inp_c1_nuevo")
            c2 = st.number_input("C2", 0, 99, 0, key="inp_c2_nuevo")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("💾 Guardar", type="primary", use_container_width=True, key="btn_guardar_nuevo"):
                s_code = {"Mañana": "M", "Tarde": "T", "Noche": "N"}[ses]
                line = f"{f_nueva.strftime('%d/%m/%Y')};{s_code};{int(cent)};{int(fij)};{int(c1)};{int(c2)}\n"
                try:
                    with open(RUTA_CSV, 'a', encoding='latin-1') as f:
                        f.write(line)
                    st.success("✅ Guardado!")
                    numero_fijo = int(fij)
                    actualizados = actualizar_resultados_pendientes(numero_fijo, f_nueva, s_code)
                    if actualizados > 0:
                        st.success(f"✅ {actualizados} predicción(es) actualizada(s)")
                    st.cache_data.clear()
                    st.session_state.invalid_dates_df = None
                    if os.path.exists(RUTA_CACHE):
                        os.remove(RUTA_CACHE)
                    time.sleep(1)
                    st.rerun()
                except Exception as err:
                    st.error(f"❌ Error: {err}")
        
        with col_btn2:
            if st.button("⏪ Deshacer", use_container_width=True, key="btn_deshacer_nuevo"):
                try:
                    with open(RUTA_CSV, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        lines.pop()
                        with open(RUTA_CSV, 'w', encoding='latin-1') as f:
                            f.writelines(lines)
                        st.warning("🗑️ Eliminado")
                        st.cache_data.clear()
                        st.session_state.invalid_dates_df = None
                        if os.path.exists(RUTA_CACHE):
                            os.remove(RUTA_CACHE)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("⚠️ Vacío")
                except Exception as err:
                    st.error(f"❌ Error: {err}")

    mostrar_ultimos_resultados_sidebar(df_full)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Configuración de Análisis")
    modo_sorteo = st.sidebar.radio("Análisis:", ["General", "Mañana", "Tarde", "Noche"], key="modo_sorteo_radio")
    modo_fecha = st.sidebar.radio("Fecha Ref:", ["Auto (Último Dato)", "Personalizado"], key="modo_fecha_radio")

    fecha_ref = pd.to_datetime(datetime.now())
    target_sesion = "Tarde"

    if modo_fecha == "Personalizado":
        fecha_ref = st.sidebar.date_input("Fecha:", datetime.now().date(), key="fecha_ref_input")
        fecha_ref = pd.to_datetime(fecha_ref)
        sesion_estado = st.sidebar.radio("Estado:", ["Antes de Mañana", "Después de Mañana", "Después de Tarde"], horizontal=False, key="sesion_estado_radio")
        if sesion_estado == "Antes de Mañana":
            target_sesion = "Mañana"
        elif sesion_estado == "Después de Mañana":
            target_sesion = "Tarde"
        else:
            target_sesion = "Noche"

    if st.sidebar.button("🔄 Recargar", key="btn_recargar_sidebar"):
        st.cache_data.clear()
        if os.path.exists(RUTA_CACHE):
            os.remove(RUTA_CACHE)
        st.rerun()

    if modo_sorteo == "Mañana":
        df_analisis = df_full[df_full['Tipo_Sorteo'] == 'M'].copy()
        nombre_sesion_backtest = "Mañana"
    elif modo_sorteo == "Tarde":
        df_analisis = df_full[df_full['Tipo_Sorteo'] == 'T'].copy()
        nombre_sesion_backtest = "Tarde"
    elif modo_sorteo == "Noche":
        df_analisis = df_full[df_full['Tipo_Sorteo'] == 'N'].copy()
        nombre_sesion_backtest = "Noche"
    else:
        df_analisis = df_full.copy()
        nombre_sesion_backtest = "General (Todas)"

    if df_analisis.empty and modo_sorteo != "General":
        st.warning(f"⚠️ No hay datos para {modo_sorteo}. Cambia a 'General' o agrega sorteos.")

    if df_full.empty:
        st.warning("⚠️ No hay datos en el archivo CSV.")
        st.stop()

    tabs = st.tabs([
        "📊 Análisis Principal",
        "🗓️ Faltantes del Mes",
        "📊 Dashboard",
        "🧪 Backtest"
    ])

    with tabs[0]:
        df_historial_perfiles_full = obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE)
        
        # 🔍 CORRECCIÓN APLICADA: Filtro por sesión seleccionada
        if modo_sorteo == "Mañana":
            df_hist_perfiles = df_historial_perfiles_full[df_historial_perfiles_full['Sorteo'] == 'Mañana'].copy()
        elif modo_sorteo == "Tarde":
            df_hist_perfiles = df_historial_perfiles_full[df_historial_perfiles_full['Sorteo'] == 'Tarde'].copy()
        elif modo_sorteo == "Noche":
            df_hist_perfiles = df_historial_perfiles_full[df_historial_perfiles_full['Sorteo'] == 'Noche'].copy()
        else:
            df_hist_perfiles = df_historial_perfiles_full.copy()

        st.markdown("---")
        mostrar_tabla_personalidad_perfiles(df_hist_perfiles)
        
        st.markdown("---")
        st.header("📜 Historial de Combinaciones y Estados")
        if not df_hist_perfiles.empty:
            df_hist_view = df_hist_perfiles.copy()
            df_hist_view['ID_Unico'] = df_hist_view['Fecha'].astype(str).str[:10] + "_" + df_hist_view['Sorteo'].astype(str)
            df_hist_view = df_hist_view.drop_duplicates(subset=['ID_Unico'], keep='last').reset_index(drop=True)
            df_hist_view.drop(columns=['ID_Unico'], inplace=True)
            df_hist_view['Decena'] = df_hist_view['Numero'] // 10
            df_hist_view['Unidad'] = df_hist_view['Numero'] % 10
            sort_map = {'Mañana': 0, 'Tarde': 1, 'Noche': 2, 'M': 0, 'T': 1, 'N': 2}
            df_hist_view['sort_key'] = df_hist_view['Sorteo'].astype(str).map(sort_map).fillna(0)
            df_hist_view = df_hist_view.sort_values(by=['Fecha', 'sort_key'], ascending=[False, False])
            df_hist_view['Fecha'] = pd.to_datetime(df_hist_view['Fecha']).dt.strftime('%d/%m/%Y')
            df_hist_view = df_hist_view.rename(columns={'Perfil': 'Estado Salida'})
            cols_display = ['Fecha', 'Sorteo', 'Numero', 'Decena', 'Unidad', 'Estado Salida']
            st.dataframe(df_hist_view[cols_display].head(30), hide_index=True, use_container_width=True)
            st.caption(f"✅ {len(df_hist_view)} sorteos únicos (mostrando los 30 más recientes)")
        else:
            st.warning("No se pudo generar el historial de perfiles.")
        
        st.markdown("---")
        df_oport_dec, df_oport_uni, gaps_dec, gaps_uni, freq_dec, freq_uni = analizar_oportunidad_por_digito(df_analisis, fecha_ref)
        
        st.header(f"🎯 Estado de Dígitos ({target_sesion})")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔟 Decenas")
            st.dataframe(df_oport_dec.sort_values('Punt. Base', ascending=False), hide_index=True)
        with col2:
            st.subheader("1️⃣ Unidades")
            st.dataframe(df_oport_uni.sort_values('Punt. Base', ascending=False), hide_index=True)
        
        st.markdown("---")
        st.header("📅 Análisis de Perfiles (Motor Mejorado)")
        
        if st.button("🚀 Ejecutar Análisis", type="primary", key="btn_ejecutar_analisis_principal"):
            with st.spinner("Analizando..."):
                if not df_hist_perfiles.empty:
                    distribuciones_cache = pre_calcular_distribuciones_perfiles(df_hist_perfiles)
                    df_stats, transizioni, ultimo_perfil = analizar_estadisticas_perfiles(df_hist_perfiles, fecha_ref, distribuciones_cache=distribuciones_cache)
                    estabilidad_digitos = calcular_estabilidad_historica_digitos(df_analisis)
                    
                    faltantes_mes_anterior = obtener_faltantes_mes_anterior(df_analisis)
                    mes_ant, anio_ant = obtener_mes_anterior_completo(df_analisis)
                    if mes_ant and anio_ant:
                        meses_nombres = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
                        st.info(f"📅 TOP40 usa faltantes de **{meses_nombres[mes_ant]} {anio_ant}** ({len(faltantes_mes_anterior)} números)")
                    
                    top40 = generar_sugerencia_fusionada(
                        df_stats, transizioni, ultimo_perfil,
                        df_oport_dec, df_oport_uni,
                        df_hist_perfiles, fecha_ref,
                        estabilidad_digitos, gaps_dec, gaps_uni,
                        distribuciones_cache, freq_dec, freq_uni,
                        faltantes_mes_anterior
                    )
                    
                    if top40 and len(top40) > 0:
                        scores = []
                        for _, row in df_stats.iterrows():
                            p = str(row['Perfil'])
                            score = 0
                            estado = str(row['Estado Actual'])
                            if row['Alerta'] == '⚠️ RECUPERAR':
                                score += 150
                            elif estado == 'Vencido':
                                score += 70
                            elif estado == 'Normal':
                                score += 50
                            elif estado == 'Muy Vencido':
                                score += 100
                            scores.append({'Perfil': p, 'Score': int(score)})
                        
                        df_scores = pd.DataFrame(scores)
                        if not df_scores.empty:
                            df_scores = df_scores.sort_values('Score', ascending=False)
                            perfil_top = df_scores.iloc[0]['Perfil']
                            score_prom = df_scores['Score'].mean()
                            
                            guardado, estado = guardar_prediccion_en_historico(fecha_ref, target_sesion, top40, perfil_top, score_prom)
                            if guardado:
                                st.success(f"📝 Predicción guardada - Perfil: **{perfil_top}** | Score: **{score_prom:.1f}**")
                            elif estado == "duplicado":
                                st.warning("⚠️ Ya existe predicción pendiente para esta fecha/sorteo")
                            else:
                                st.error(f"❌ Error guardando: {estado}")
                        
                        st.markdown("---")
                        st.subheader("📊 Estadística de Perfiles (Completa)")
                        cols_tabla = ['Perfil', 'Frecuencia', 'Veces Normal', 'Veces Vencido', 'Veces Muy Vencido', 'Estado Actual', 'Estabilidad', 'Tiempo Limite', 'Alerta', 'Estado Ultima Salida', 'Estabilidad Ultima Salida', 'Exceso Ultima Salida']
                        df_display = df_stats[cols_tabla].copy()
                        st.dataframe(df_display.sort_values('Frecuencia', ascending=False), hide_index=True, use_container_width=True)
                        # 🎲 ORDENADOR DE NÚMEROS PERSONALIZADO
                        st.markdown("---")
                        st.subheader("🔢 Ordenador de Números Personalizado")
                        st.caption("Pega o escribe tus números separados por comas, espacios o saltos de línea (ej: 12, 34, 05, 88)")
                        
                        numeros_input = st.text_area(
                            "Ingresa tus números aquí:",
                            height=80,
                            placeholder="12, 34, 05, 88, 91...",
                            key="input_numeros_ordenar"
                        )
                        
                        if st.button("🔄 Ordenar por Prioridad del Algoritmo", key="btn_ordenar_manual"):
                            if numeros_input.strip():
                                raw_nums = numeros_input.replace(',', ' ').replace(';', ' ').replace('\n', ' ').split()
                                nums_validos = []
                                for n in raw_nums:
                                    n = n.strip()
                                    if n.isdigit() and 0 <= int(n) <= 99:
                                        nums_validos.append(int(n))
                                nums_validos = sorted(list(set(nums_validos)))
                                
                                if not nums_validos:
                                    st.warning("⚠️ No se encontraron números válidos (00-99).")
                                else:
                                    distribuciones = pre_calcular_distribuciones_perfiles(df_hist_perfiles)
                                    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
                                    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}
                                    
                                    def get_estado_comun(perfil):
                                        dist = distribuciones.get(perfil, {})
                                        estados = {k: v for k, v in dist.items() if k in ['Normal', 'Vencido', 'Muy Vencido']}
                                        if not estados: return 'Ninguno', 0
                                        max_estado = max(estados, key=estados.get)
                                        return max_estado, estados[max_estado]
                                    
                                    perfil_metrics = {}
                                    for p in df_stats['Perfil'].unique():
                                        ec, pc = get_estado_comun(p)
                                        row = df_stats[df_stats['Perfil'] == p].iloc[0]
                                        perfil_metrics[p] = {
                                            'Estado_Comun': ec, 'Porc_Comun': pc,
                                            'Estabilidad': row.get('Estabilidad', 0),
                                            'Alerta': row.get('Alerta', '-'),
                                            'Exceso_Dias': row.get('Exceso Días', 0) if 'Exceso Días' in df_stats.columns else 0,
                                            'Prioridad_Senal': row.get('Prioridad', '🟡 Media') if 'Prioridad' in df_stats.columns else '🟡 Media'
                                        }
                                    
                                    def calcular_score_manual(num):
                                        dec, uni = num // 10, num % 10
                                        ed, eu = map_estado_dec.get(dec, 'Normal'), map_estado_uni.get(uni, 'Normal')
                                        perfil = f"{ed}-{eu}"
                                        m = perfil_metrics.get(perfil, {'Porc_Comun': 0, 'Estabilidad': 50, 'Alerta': '-', 'Exceso_Dias': 0, 'Prioridad_Senal': '🟡 Media'})
                                        score = 0
                                        if m['Porc_Comun'] >= 60: score += 100
                                        elif m['Porc_Comun'] >= 50: score += 70
                                        else: score += 40
                                        score += m.get('Estabilidad', 50) * 0.5
                                        if m['Alerta'] == '⚠️ RECUPERAR': score += 20
                                        score += min(m.get('Exceso_Dias', 0) * 2, 25)
                                        score += {'🔴 Crítica': 15, '🟠 Alta': 10, '🟡 Media': 5}.get(m['Prioridad_Senal'], 0)
                                        return score, perfil, m['Estado_Comun'], m['Porc_Comun']
                                    
                                    resultados = []
                                    for num in nums_validos:
                                        score, perfil, ec, pc = calcular_score_manual(num)
                                        resultados.append({'Número': f"{num:02d}", 'Score': score, 'Perfil': perfil, 'Estado_Común': ec, '% Común': f"{pc:.1f}%"})
                                    
                                    resultados.sort(key=lambda x: x['Score'], reverse=True)
                                    st.success(f"✅ Se ordenaron **{len(resultados)} números** según la prioridad del algoritmo.")
                                    st.dataframe(pd.DataFrame(resultados), hide_index=True, use_container_width=True)
                                    
                                    if len(resultados) > 0:
                                        top_recom = pd.DataFrame(resultados).head(2)['Número'].tolist()
                                        st.info(f"🎯 **Recomendación para jugar:** {', '.join(top_recom)}")
                            else:
                                st.warning("⚠️ Ingresa al menos un número para ordenar.")
                                                                                    # 🎯 SEÑALES DE JUEGO AUTOMÁTICAS (CON CRUCE DE PATRÓN + PRIORIDAD VISUAL)
                    if 'df_stats' in locals() and not df_stats.empty:
                        # Obtener distribuciones para cruzar con Estado Común
                        distribuciones = pre_calcular_distribuciones_perfiles(df_hist_perfiles)
                        
                        df_senales = df_stats[
                            (df_stats['Alerta'] == '⚠️ RECUPERAR') &
                            (df_stats['Estabilidad'] >= 60) &
                            (df_stats['Estado Actual'].isin(['Vencido', 'Muy Vencido']))
                        ].copy()
                        
                        if not df_senales.empty:
                            # Agregar columna de Estado Común y validar ≥60%
                            def get_estado_comun(perfil):
                                dist = distribuciones.get(perfil, {})
                                estados = {k: v for k, v in dist.items() if k in ['Normal', 'Vencido', 'Muy Vencido']}
                                if not estados:
                                    return 'Ninguno', 0
                                max_estado = max(estados, key=estados.get)
                                return max_estado, estados[max_estado]
                            
                            df_senales[['Estado_Comun', 'Porc_Comun']] = df_senales['Perfil'].apply(
                                lambda p: pd.Series(get_estado_comun(p))
                            )
                            
                            # Calcular Exceso Días y Prioridad para TODOS los perfiles
                            df_senales['Exceso Días'] = (df_senales['Gap Actual'] - df_senales['Tiempo Limite']).astype(int)
                            df_senales['Prioridad'] = df_senales['Exceso Días'].apply(
                                lambda x: "🔴 Crítica" if x > 5 else ("🟠 Alta" if x > 2 else "🟡 Media")
                            )
                            
                            # Separar en dos grupos
                            df_con_patron = df_senales[df_senales['Porc_Comun'] >= 60].copy()
                            df_sin_patron = df_senales[df_senales['Porc_Comun'] < 60].copy()
                            
                            st.markdown("---")
                            st.subheader("🎯 Señales de Juego (Cruzadas con Patrón)")
                            
                            if not df_con_patron.empty:
                                st.success(f"🔥 **{len(df_con_patron)} perfil(es) con VENTAJA ESTADÍSTICA**")
                                cols = ['Perfil', 'Estado Actual', 'Estado_Comun', 'Porc_Comun', 'Exceso Días', 'Prioridad']
                                st.dataframe(df_con_patron[cols].sort_values('Porc_Comun', ascending=False), hide_index=True, use_container_width=True)
                            
                            if not df_sin_patron.empty:
                                st.info(f"🟡 **{len(df_sin_patron)} perfil(es) con timing favorable pero SIN patrón claro**")
                                st.caption("⚠️ Estos perfiles tienen alta estabilidad pero sin Estado Común ≥60%. Usar solo como respaldo con gestión de riesgo.")
                                cols = ['Perfil', 'Estado Actual', 'Estado_Comun', 'Porc_Comun', 'Exceso Días', 'Prioridad']
                                st.dataframe(df_sin_patron[cols], hide_index=True, use_container_width=True)
                            
                            st.caption("💡 Cruza los perfiles con patrón ≥60% con la sección *🚨 Detalle de Alertas Activas* para ver los números exactos.")

                        # 📱 ICONOS DE EXPORTACIÓN OPTIMIZADOS PARA MÓVIL
                        st.markdown("---")
                        st.subheader("📥 Descargar Resultados")
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            top40_df = pd.DataFrame({'🔢 Número': [f"{n:02d}" for n in top40], '📊 Orden': range(1, len(top40)+1)})
                            csv_top40 = top40_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📱 Guardar TOP 40",
                                data=csv_top40,
                                file_name=f"TOP40_{modo_sorteo}_{fecha_ref.strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        with col_dl2:
                            csv_stats = df_display.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📊 Guardar Estadísticas",
                                data=csv_stats,
                                file_name=f"Stats_{modo_sorteo}_{fecha_ref.strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    else:
                        st.warning("⚠️ No se generaron números")
                else:
                    st.error("❌ No hay historial suficiente. Agrega más datos. ")

    with tabs[1]:
        st.header("🗓️ Faltantes del Mes")
        st.info("🎯 Algoritmo Geotodo: Sin Condiciones = 50 pts (50% peso)")
        st.warning("⚠️ **Nota:** Si seleccionas un mes, se muestran los faltantes del mes **ANTERIOR** al seleccionado")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        meses_nombres = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
        
        with col_f1:
            mes_sel = st.selectbox("Mes:", list(meses_nombres.values()), index=datetime.now().month - 1, key="mes_sel_faltantes")
            mes_num = [k for k, v in meses_nombres.items() if v == mes_sel][0]
        
        with col_f2:
            anio_sel = st.number_input("Año:", min_value=2020, max_value=datetime.now().year, value=datetime.now().year, key="anio_sel_faltantes")
        
        with col_f3:
            cant_sorteos = st.slider("Sorteos para frecuencia:", 100, 5000, 1000, step=100, key="cant_sorteos_faltantes")
        
        if st.button("🔍 Analizar Faltantes", type="primary", key="btn_analizar_faltantes_tab"):
            with st.spinner("Calculando..."):
                try:
                    if mes_num == 1:
                        mes_anterior_sel = 12
                        anio_anterior_sel = anio_sel - 1
                    else:
                        mes_anterior_sel = mes_num - 1
                        anio_anterior_sel = anio_sel
                    
                    st.info(f"📅 Mostrando faltantes de **{meses_nombres[mes_anterior_sel]} {anio_anterior_sel}** (mes anterior al seleccionado)")
                    
                    df_faltantes_res, error_msg, df_salidos_mes = analizar_faltantes_mes(df_full, mes_anterior_sel, anio_anterior_sel, cant_sorteos)
                    
                    if error_msg:
                        st.info(error_msg)
                    elif not df_faltantes_res.empty:
                        total_faltantes = len(df_faltantes_res)
                        alta = df_faltantes_res[df_faltantes_res['Prioridad'] == '🔴 ALTA']
                        media = df_faltantes_res[df_faltantes_res['Prioridad'] == '🟡 MEDIA']
                        baja = df_faltantes_res[df_faltantes_res['Prioridad'] == '⚪ BAJA']
                        
                        st.markdown(f"### ⏳ Faltan: **{total_faltantes}** números de 100")
                        st.progress(total_faltantes / 100)
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        col_stat1.metric("🔴 Prioridad Alta", len(alta))
                        col_stat2.metric("🟡 Prioridad Media", len(media))
                        col_stat3.metric("⚪ Prioridad Baja", len(baja))
                        
                        st.markdown("---")
                        
                        if len(alta) > 0:
                            st.markdown("#### 🔴 Números de Prioridad Alta")
                            st.markdown(f"**{len(alta)} números** con mayor probabilidad de salir:")
                            alta_nums = "  ".join([f"`{n}`" for n in alta['Número'].tolist()])
                            st.markdown(alta_nums)
                            st.markdown("---")
                        
                        if len(media) > 0:
                            st.markdown("#### 🟡 Números de Prioridad Media")
                            media_nums = "  ".join([f"`{n}`" for n in media['Número'].tolist()])
                            st.markdown(media_nums)
                            st.markdown("---")
                        
                        if len(baja) > 0:
                            st.markdown("#### ⚪ Números de Prioridad Baja")
                            baja_nums = "  ".join([f"`{n}`" for n in baja['Número'].tolist()])
                            st.markdown(baja_nums)
                            st.markdown("---")
                        
                        st.markdown("### 📊 Tabla Completa de Números Faltantes")
                        st.info(f"📋 Mostrando **{total_faltantes} números** con todas sus características")
                        
                        df_display = df_faltantes_res.copy()
                        df_display = df_display.rename(columns={
                            'Número': '🔢 Número',
                            'Prioridad': '🚨 Prioridad',
                            'Puntaje': '⭐ Puntaje',
                            'Razón': '📝 Razón',
                            'Veces Salidas': '📊 Frecuencia',
                            'Estado Estabilidad': '📈 Estabilidad',
                            'Gap Actual': '⏳ Gap (Días)'
                        })
                        
                        st.dataframe(
                            df_display,
                            use_container_width=True, 
                            hide_index=True
                        )
                        
                        with st.expander("📖 ¿Qué significa cada columna?"):
                            st.markdown("""
                            | Columna | Descripción |
                            |---------|-------------|
                            | **🔢 Número** | El número faltante (00-99) |
                            | **🚨 Prioridad** | 🔴 ALTA (60+ pts), 🟡 MEDIA (40-59 pts), ⚪ BAJA (<40 pts) |
                            | **⭐ Puntaje** | Suma de puntos según condiciones |
                            | **📝 Razón** | Por qué tiene ese puntaje (Favorito, Atrasado, Sin Condiciones) |
                            | **📊 Frecuencia** | Veces que salió en los últimos X sorteos |
                            | **📈 Estabilidad** | Estado según gap actual (Normal, Vencido, Muy Vencido) |
                            | **⏳ Gap** | Días transcurridos desde su última salida |
                            
                            **Sistema de Puntaje:**
                            - ✅ **Atrasado (Vencido):** +30 puntos
                            - ✅ **Favorito (Top 25 frecuencia):** +30 puntos
                            - ✅ **Sin Condiciones:** +50 puntos (peso neutral)
                            """)
                        
                        csv = df_faltantes_res.to_csv(index=False, encoding='utf-8').encode('utf-8')
                        st.download_button(
                            label="📥 Descargar tabla completa (CSV)",
                            data=csv,
                            file_name=f"faltantes_{meses_nombres[mes_anterior_sel]}_{anio_anterior_sel}.csv",
                            mime="text/csv",
                            key="btn_descargar_faltantes"
                        )
                        
                    else:
                        st.warning("⚠️ Sin resultados")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tabs[2]:
        mostrar_dashboard_efectividad()

    with tabs[3]:
        st.header("🧪 Backtesting (Real)")
        st.info(f"📌 **Sesión para backtest:** {nombre_sesion_backtest}")
        sorteos_back = st.slider("Número de sorteos", 3, 15, 7, key="slider_backtest")
        
        if st.button("▶️ Iniciar Backtest", key="btn_iniciar_backtest"):
            if df_analisis.empty:
                st.error(f"❌ No hay datos para backtest de {nombre_sesion_backtest}")
            else:
                with st.spinner(f"🔄 Backtest para {nombre_sesion_backtest} ({sorteos_back} sorteos)..."):
                    df_res, aciertos, total, debug_logs = ejecutar_backtest(df_analisis, sorteos_back, nombre_sesion_backtest, debug_mode=st.session_state.debug_mode)
                    
                    if total > 0:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Sorteos", total)
                        col2.metric("Aciertos", aciertos)
                        col3.metric("Efectividad", f"{round((aciertos/total)*100, 1)}%")
                        
                        if aciertos > 0:
                            st.success(f"✨ {aciertos} de {total} aciertos en {nombre_sesion_backtest}")
                        else:
                            st.warning(f"⚠️ 0 aciertos en {nombre_sesion_backtest}")
                        
                        if st.session_state.debug_mode and debug_logs:
                            st.markdown(f"### 🔍 Logging ({nombre_sesion_backtest})")
                            st.dataframe(pd.DataFrame(debug_logs), hide_index=True)
                        
                        with st.expander("📋 Ver resultados"):
                            st.dataframe(df_res, hide_index=True)

if __name__ == "__main__":
    main()