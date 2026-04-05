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

# ============================================================================
# CONFIGURACION
# ============================================================================
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

# ============================================================================
# ESTADO DE SESIÓN
# ============================================================================
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'debug_logs' not in st.session_state:
    st.session_state.debug_logs = []
if 'invalid_dates_df' not in st.session_state:
    st.session_state.invalid_dates_df = None

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================
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

# ============================================================================
# DETECTAR MES ANTERIOR COMPLETO
# ============================================================================
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

# ============================================================================
# CARGA DE DATOS
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def cargar_datos_geotodo(_ruta_csv, _file_signature):
    try:
        if not os.path.exists(_ruta_csv):
            inicializar_archivo(_ruta_csv, ["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"])
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        
        try:
            with open(_ruta_csv, 'r', encoding='latin-1') as f:
                primera_linea = f.readline()
            separador = ';' if ';' in primera_linea else (',' if ',' in primera_linea else '\t')
            df = pd.read_csv(_ruta_csv, sep=separador, encoding='latin-1', header=0,
                           on_bad_lines='skip', dtype=str, skipinitialspace=True)
        except pd.errors.EmptyDataError:
            st.warning("⚠️ El archivo CSV está vacío")
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        except Exception as e:
            st.error(f"❌ Error al leer CSV: {e}")
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        
        if df.empty or len(df.columns) == 0:
            st.warning("⚠️ El archivo CSV no tiene datos válidos")
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        
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
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        
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
            return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None
        
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
        return pd.DataFrame(columns=["Fecha","Tipo_Sorteo","Centena","Fijo","Primer_Corrido","Segundo_Corrido"]), None

# ============================================================================
# ANALIZAR ESTABILIDAD DE NÚMEROS (SOLO FIJO)
# ============================================================================
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

# ============================================================================
# ANALIZAR FALTANTES DEL MES (SOLO POSICION FIJO)
# ============================================================================
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
                condizioni.append("Favorito")
            
            if not es_vencido and not es_favorito:
                puntaje = 50
                condizioni.append("Sin Condiciones")
            
            if puntaje >= 60:
                prioridad = "🔴 ALTA"
            elif puntaje >= 40:
                prioridad = "🟡 MEDIA"
            else:
                prioridad = "⚪ BAJA"
            
            razon = " + ".join(condiciones)
            risultati.append({
                'Número': f"{num:02d}",
                'Prioridad': prioridad,
                'Puntaje': puntaje,
                'Razón': razon,
                'Veces Salidas': freq_val,
                'Estado Estabilità': est_data['Estado'],
                'Gap Attuale': est_data['Gap']
            })
        
        df_res = pd.DataFrame(risultati)
        df_res = df_res.sort_values(['Puntaje', 'Veces Salidas'], ascending=[False, False]).reset_index(drop=True)
        return df_res, None, df_mes
    except Exception as e:
        st.error(f"❌ Error in analizzare_faltanti_mese: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), f"Errore: {str(e)}", pd.DataFrame()

# ============================================================================
# OTTENERE FALTANTI DEL MESE PRECEDENTE COMPLETO (PER TOP40)
# ============================================================================
def ottenere_faltanti_mese_precedente(df_fijos):
    try:
        mese_precedente, anno_precedente = ottenere_mese_precedente_completo(df_fijos)
        
        if mese_precedente is None:
            return []
        
        df_fijos_only = df_fijos[df_fijos['Posizione'] == 'Fisso'].copia()
        
        data_inizio_mese = datetime(anno_precedente, mese_precedente, 1)
        ultimo_giorno = calendar.mese_range(anno_precedente, mese_precedente)[1]
        data_fine_mese = datetime(anno_precedente, mese_precedente, ultimo_giorno)
        
        df_mese = df_fijos_solo[(df_fijos_solo['Data'] >= data_inizio_mese) &
                               (df_fijos_solo['Data'] <= data_fine_mese)]
        
        if df_mese.vuoto:
            return []
        
        usciti = set(df_mese['Numero'].unico())
        mancanti = list(set(range(100)) - usciti)
        return mancanti
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

def analizar_estadisticas_perfiles(df_historial_perfiles, fecha_referencia, distribuciones_cache=None, session_filter=None):
    if session_filter and session_filter != "General":
        df_historial_perfiles = df_historial_perfiles[df_historial_perfiles['Sorteo'] == session_filter].copy()
    
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
                if len(gaps_prev) >= 4: lim_prev = int(np.percentile(gaps_prev, 75))
                else: lim_prev = int(np.median(gaps_prev) * 2) if gaps_prev else 0
                estado_ultima_salida = calcular_estado_actual(gap_ultima_espera, lim_prev)
                if estado_ultima_salida == "Muy Vencido" and lim_prev > 0: exceso_ultima_salida = int(gap_ultima_espera - lim_prev)
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
            'Perfil': perfil, 'Frecuencia': total_hist + 1, 'Última Fecha': ultima_fecha,
            'Gap Actual': gap_actual, 'Mediana Gap': int(mediana_gap_actual),
            'Estado Actual': estado_actual, 'Estabilidad': round(estabilidad_actual, 1),
            'Tiempo Limite': tiempo_limite,
            'Alerta': '⚠️ RECUPERAR' if alerta_recuperacion else '-',
            'Prob Repeticiones %': round(prob_repeticion, 1), 'Semana Activa': semana_activa,
            'Último Numero': last_row['Numero'], 'Último Sorteo': last_row['Sorteo'],
            'Veces Normal': count_normal, 'Veces Vencido': count_vencido, 'Veces Muy Vencido': count_muy_vencido,
            'Estado Ultima Salida': estado_ultima_salida,
            'Estabilidad Ultima Salida': round(estabilidad_ultima_salida, 1),
            'Exceso Ultima Salida': exceso_ultima_salida, 'Distribucion_Estados': distribucion_estados,
            'Fue_Atipica': fue_atipica, 'P75_Perfil': p75_perfil, 'Dias_Desde_Ultima': dias_desde_ultima_salida,
            'Dentro_Enfriamiento': dentro_enfriamiento, 'Es_Estado_Comun': es_estado_comun, 'Porc_Ultima_Salida': porc_ultima_salida
        })
    
    df_stats = pd.DataFrame(analisis_perfiles)
    return df_stats, transiciones, ultimo_perfil_global
# ============================================================================
# TOP40 CON FALTANTES DEL MES ANTERIOR (+50 PTS BONUS)
# ============================================================================
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
        if row['Alerta'] == '⚠️ RECUPERAR': score += 150
        elif estado == 'Vencido': score += 70
        elif estado == 'Normal': score += 50
        elif estado == 'Muy Vencido': score += 100
        score += row['Estabilidad'] * 0.5
        if ultimo_perfil:
            trans_count = transizioni.get((str(ultimo_perfil), p), 0)
            score += trans_count * 10
        if row.get('Fue_Atipica', False): score -= 50
        if row.get('Dentro_Enfriamiento', False): score -= 30
        if row.get('Es_Estado_Comun', False): score += 30
        scores.append({'Perfil': p, 'Score': int(score), 'Estado': estado})
    
    df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
    top_7 = df_scores.head(7)
    
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}
    df_hist_nums = df_historial_perfiles.groupby('Numero')['Fecha'].max().to_dict() if not df_historial_perfiles.empty else {}
    map_temp_dec = df_oport_dec.set_index('Dígito')['Temperatura'].to_dict() if not df_oport_dec.empty else {}
    map_temp_uni = df_oport_uni.set_index('Dígito')['Temperatura'].to_dict() if not df_oport_uni.empty else {}
    temp_val = {'🔥 Caliente': 3, '🟡 Tibio': 2, '🧊 Frío': 1}
    faltantes_set = set(faltantes_mes_anterior) if faltantes_mes_anterior else set()
    
    candidatos_totales = []
    for _, row in top_7.iterrows():
        perfil = str(row['Perfil'])
        partes = perfil.split('-')
        if len(partes) != 2: continue
        ed_req, eu_req = partes[0], partes[1]
        decenas_estado = [d for d in range(10) if map_estado_dec.get(d) == ed_req]
        unidades_estado = [u for u in range(10) if map_estado_uni.get(u) == eu_req]
        if not decenas_estado: decenas_estado = list(range(10))
        if not unidades_estado: unidades_estado = list(range(10))
        
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
                if map_estado_dec.get(d) in ['Vencido', 'Muy Vencido'] and freq_dec.get(d, 0) >= 8: bonus_salidor += min(gap_d / 5, 20)
                if map_estado_dec.get(d) == 'Muy Vencido': bonus_salidor += 10
                if map_estado_uni.get(u) in ['Vencido', 'Muy Vencido'] and freq_uni.get(u, 0) >= 8: bonus_salidor += min(gap_u / 5, 20)
                if map_estado_uni.get(u) == 'Muy Vencido': bonus_salidor += 10
                if (map_estado_dec.get(d) in ['Vencido', 'Muy Vencido'] and freq_dec.get(d, 0) >= 8) and (map_estado_uni.get(u) in ['Vencido', 'Muy Vencido'] and freq_uni.get(u, 0) >= 8): bonus_salidor += 25
                
                bonus_escasez = 0
                porc_perfil = distribuciones_perfiles.get(perfil, {}).get('porcentaje', 0)
                total_combinaciones = len(decenas_estado) * len(unidades_estado)
                if porc_perfil >= 40: bonus_escasez += 15
                if total_combinaciones < 20: bonus_escasez += 20
                if total_combinaciones < 10: bonus_escasez += 10
                
                bonus_faltantes = 50 if num in faltantes_set else 0
                temp_score = temp_d + temp_u + (est_d + est_u) / 20 + gap_bonus + bonus_salidor + bonus_escasez + bonus_faltantes
                candidatos_totales.append({'Numero': num, 'Perfil': perfil, 'Score': score_base, 'Temp_Score': temp_score})
    
    if not candidatos_totales: return list(range(0, 40))
    df_cands = pd.DataFrame(candidatos_totales)
    df_cands = df_cands.drop_duplicates(subset=['Numero'], keep='first')
    df_cands = df_cands.sort_values(['Temp_Score', 'Score'], ascending=[False, False])
    return df_cands.head(30)['Numero'].tolist()
def generar_sugerencia_fusionada(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni, faltantes_mes_anterior=None):
    st.subheader("🤖 Sugerencia Inteligente Fusionada")
    st.markdown("### 🚨 Detalle de Alertas Activas")
    map_estado_dec = df_oport_dec.set_index('Dígito')['Estado'].to_dict() if not df_oport_dec.empty else {}
    map_estado_uni = df_oport_uni.set_index('Dígito')['Estado'].to_dict() if not df_oport_uni.empty else {}
    
    alertas_activas = df_stats[df_stats['Alerta'] == '⚠️ RECUPERAR'].copy() if not df_stats.empty else pd.DataFrame()
    
    if not alertas_activas.empty:
        for _, row_alert in alertas_activas.iterrows():
            perfil_name = str(row_alert['Perfil'])
            partes = perfil_name.split('-')
            if len(partes) != 2: continue
            
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
            
            time_str = ""
            if estado == "Normal":
                falta = int(med - gap) if med > gap else 0
                time_str = f"🟢 Faltan ~{falta} días"
            elif estado == "Vencido":
                exceso = int(gap - med)
                time_str = f"🟠 Exceso: {exceso} días"
            elif estado == "Muy Vencido":
                exceso = int(gap - tiempo_limite) if tiempo_limite > 0 else gap
                time_str = f"🔴 +{exceso} días sobre límite"
            
            with st.container():
                st.markdown(f"**Perfil: `{perfil_name}`** | Estado: `{estado}` | {time_str}")
                st.markdown(f"Estabilidad: `{row_alert['Estabilidad']}%`")
                if nums_alerta:
                    st.success(f"({len(nums_alerta)} nums): `{' '.join(nums_alerta[:20])}`")
                st.markdown("---")
    else:
        st.info("No hay alertas activas.")

    st.markdown("### 🎲 Top 30 Números")
    lista_nums = obtener_prediccion_numeros_lista(df_stats, transizioni, ultimo_perfil, df_oport_dec, df_oport_uni, df_historial_perfiles, fecha_ref, estabilidad_digitos, gaps_dec, gaps_uni, distribuciones_perfiles, freq_dec, freq_uni, faltantes_mes_anterior)
    
    if lista_nums:
        cols = st.columns(6)
        for idx, num in enumerate(lista_nums):
            d_int = int(num // 10)
            u_int = int(num % 10)
            ed = map_estado_dec.get(d_int, "?")
            eu = map_estado_uni.get(u_int, "?")
            cols[idx % 6].markdown(f"<div style='background-color:#000;padding:5px;border-radius:5px;text-align:center;border:1px solid #333'><h4 style='margin:0;color:#0F0'>{num:02d}</h4><small style='color:#AAA'>{ed}-{eu}</small></div>", unsafe_allow_html=True)
def comparar_alertas_general_vs_sesion(df_full, fecha_ref, target_sesion, ruta_cache):
    st.subheader("⚖️ Comparador: General vs Sesión")
    
    # 1. General
    df_backtest_gen = df_full[df_full['Fecha'] < fecha_ref].copy()
    df_oport_dec_gen, df_oport_uni_gen, _, _, _, _ = analizar_oportunidad_por_digito(df_backtest_gen, fecha_ref)
    df_hist_perf_gen = obtener_historial_perfiles_cacheado(df_backtest_gen, ruta_cache)
    df_stats_gen, _, _ = analizar_estadisticas_perfiles(df_hist_perf_gen, fecha_ref, session_filter="General")
    
    alertas_gen = df_stats_gen[df_stats_gen['Alerta'] == '⚠️ RECUPERAR'].copy()
    map_dec_gen = df_oport_dec_gen.set_index('Dígito')['Estado'].to_dict()
    map_uni_gen = df_oport_uni_gen.set_index('Dígito')['Estado'].to_dict()
    
    nums_gen = set()
    if not alertas_gen.empty:
        for _, r in alertas_gen.iterrows():
            p = r['Perfil'].split('-')
            if len(p)==2:
                decs = [d for d in range(10) if map_dec_gen.get(d) == p[0]]
                unis = [u for u in range(10) if map_uni_gen.get(u) == p[1]]
                for d in decs:
                    for u in unis: nums_gen.add(int(f"{d}{u}"))

    # 2. Sesión
    orden = {'Mañana': 0, 'Tarde': 1, 'Noche': 2}
    target_val = orden[target_sesion]
    
    if target_val == 0: df_backtest_ses = df_full[df_full['Fecha'] < fecha_ref].copy()
    else: df_backtest_ses = df_full[(df_full['Fecha'] < fecha_ref) | ((df_full['Fecha'] == fecha_ref) & (df_full['Tipo_Sorteo'].isin(['M', 'T'][:target_val])))].copy()
        
    df_oport_dec_ses, df_oport_uni_ses, _, _, _, _ = analizar_oportunidad_por_digito(df_backtest_ses, fecha_ref)
    df_hist_perf_ses = obtener_historial_perfiles_cacheado(df_backtest_ses, ruta_cache)
    df_stats_ses, _, _ = analizar_estadisticas_perfiles(df_hist_perf_ses, fecha_ref, session_filter=target_sesion)
    
    alertas_ses = df_stats_ses[df_stats_ses['Alerta'] == '⚠️ RECUPERAR'].copy()
    map_dec_ses = df_oport_dec_ses.set_index('Dígito')['Estado'].to_dict()
    map_uni_ses = df_oport_uni_ses.set_index('Dígito')['Estado'].to_dict()
    
    nums_ses = set()
    if not alertas_ses.empty:
        for _, r in alertas_ses.iterrows():
            p = r['Perfil'].split('-')
            if len(p)==2:
                decs = [d for d in range(10) if map_dec_ses.get(d) == p[0]]
                unis = [u for u in range(10) if map_uni_ses.get(u) == p[1]]
                for d in decs:
                    for u in unis: nums_ses.add(int(f"{d}{u}"))

    total_nums = nums_gen.union(nums_ses)
    if not total_nums:
        st.info("No hay alertas.")
        return

    res = []
    for n in sorted(list(total_nums)):
        in_g = n in nums_gen
        in_s = n in nums_ses
        tipo = "🔥 COINCIDENCIA" if (in_g and in_s) else ("👀 OPORTUNIDAD OCULTA" if in_s else "📊 SOLO GENERAL")
        res.append({'Numero': f"{n:02d}", 'Tipo': tipo, 'En General': "✅" if in_g else "", 'En Sesión': "✅" if in_s else ""})
    
    df_res = pd.DataFrame(res)
    st.dataframe(df_res, hide_index=True, use_container_width=True)

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.sidebar.header("⚙️ Opciones")
    df_full, _ = cargar_datos_geotodo(RUTA_CSV, get_file_signature(RUTA_CSV))
    
    # Sidebar: Agregar Sorteo
    with st.sidebar.expander("📝 Agregar", True):
        f_n = st.date_input("Fecha", datetime.now().date())
        ses = st.radio("Sesión", ["Mañana", "Tarde", "Noche"], horizontal=True)
        c = st.number_input("Cen", 0, 999, 0)
        f = st.number_input("Fijo", 0, 99, 0)
        c1 = st.number_input("C1", 0, 99, 0)
        c2 = st.number_input("C2", 0, 99, 0)
        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Guardar"):
                code = {"Mañana":"M", "Tarde":"T", "Noche":"N"}[ses]
                line = f"{f_n.strftime('%d/%m/%Y')};{code};{c};{f};{c1};{c2}\n"
                try:
                    with open(RUTA_CSV, 'a', encoding='latin-1') as file: file.write(line)
                    st.success("OK")
                    st.cache_data.clear()
                    time.sleep(0.5)
                    st.rerun()
                except Exception as err: st.error(f"Err: {err}")
        with b2:
            if st.button("⏪ Deshacer"):
                try:
                    if os.path.exists(RUTA_CSV):
                        with open(RUTA_CSV, 'r', encoding='latin-1') as file: lines = file.readlines()
                        if len(lines) > 1:
                            lines.pop()
                            with open(RUTA_CSV, 'w', encoding='latin-1') as file: file.writelines(lines)
                            st.warning("Borrado")
                            st.cache_data.clear()
                            time.sleep(0.5)
                            st.rerun()
                except Exception as err: st.error(f"Err: {err}")

    st.sidebar.markdown("---")
    
    # Lógica de Fecha y Sesión
    fecha_ref = pd.Timestamp.now(tz=None).normalize()
    target = "Mañana"
    
    if not df_full.empty:
        ult = df_full[df_full['Posicion']=='Fijo'].iloc[-1]
        u_t = ult['Tipo_Sorteo']
        u_f = ult['Fecha']
        if u_t == 'M': target = "Tarde"
        elif u_t == 'T': target = "Noche"
        else: target = "Mañana"; fecha_ref = u_f + timedelta(days=1)
    
    modo = st.sidebar.radio("Análisis:", ["General", "Mañana", "Tarde", "Noche"])
    
    if st.sidebar.button("🔄"): st.rerun()
    
    # Filtrar DF
    if modo == "Mañana": df_a = df_full[df_full['Tipo_Sorteo']=='M'].copy()
    elif modo == "Tarde": df_a = df_full[df_full['Tipo_Sorteo']=='T'].copy()
    elif modo == "Noche": df_a = df_full[df_full['Tipo_Sorteo']=='N'].copy()
    else: df_a = df_full.copy()
    
    if df_a.empty: st.stop()
    
    ord_s = {'Mañana':0, 'Tarde':1, 'Noche':2}
    t_val = ord_s[target]
    
    if t_val == 0: df_b = df_a[df_a['Fecha'] < fecha_ref].copy()
    elif t_val == 1: df_b = df_a[(df_a['Fecha'] < fecha_ref) | ((df_a['Fecha']==fecha_ref) & (df_a['Tipo_Sorteo']=='M'))].copy()
    else: df_b = df_a[(df_a['Fecha'] < fecha_ref) | ((df_a['Fecha']==fecha_ref) & (df_a['Tipo_Sorteo'].isin(['M','T'])))].copy()
    
    st.header(f"🎯 Estado ({target})")
    dec, uni, g_d, g_u, f_d, f_u = analizar_oportunidad_por_digito(df_b, fecha_ref)
    c1, c2 = st.columns(2)
    c1.dataframe(dec.sort_values('Punt. Base', ascending=False), hide_index=True)
    c2.dataframe(uni.sort_values('Punt. Base', ascending=False), hide_index=True)
    st.markdown("---")
    if st.button("🚀 Analizar Perfiles", type="primary"):
        with st.spinner("Calculando..."):
            df_h = obtener_historial_perfiles_cacheado(df_full, RUTA_CACHE)
            if not df_h.empty:
                if t_val == 0: df_hp = df_h[df_h['Fecha'] < fecha_ref].copy()
                elif t_val == 1: df_hp = df_h[(df_h['Fecha'] < fecha_ref) | ((df_h['Fecha']==fecha_ref) & (df_h['Sorteo']=='Mañana'))].copy()
                else: df_hp = df_h[(df_h['Fecha'] < fecha_ref) | ((df_h['Fecha']==fecha_ref) & (df_h['Sorteo'].isin(['Mañana','Tarde'])))].copy()
                
                if not df_hp.empty:
                    dist = pre_calcular_distribuciones_perfiles(df_hp)
                    # IMPORTANTE: Pasamos session_filter=modo
                    df_s, tr, up = analizar_estadisticas_perfiles(df_hp, fecha_ref, dist, session_filter=modo)
                    est = calcular_estabilidad_historica_digitos(df_b)
                    falt = obtener_faltantes_mes_anterior(df_full)
                    generar_sugerencia_fusionada(df_s, tr, up, dec, uni, df_hp, fecha_ref, est, g_d, g_u, dist, f_d, f_u, falt)
                    
                    st.subheader("📊 Estadísticas")
                    st.dataframe(df_s[['Perfil','Estado Actual','Estabilidad','Alerta']].sort_values('Estabilidad', ascending=False), hide_index=True)

    st.markdown("---")
    if st.button("⚖️ Comparar Alertas"):
        comparar_alertas_general_vs_sesion(df_full, fecha_ref, modo, RUTA_CACHE)

if __name__ == "__main__":
    main()