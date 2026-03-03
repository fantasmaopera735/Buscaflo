# =============================================================================
# 🎯 PaleGeo v3.3 - Análisis Completo de Pales y Combinaciones
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os
import calendar

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(page_title="PaleGeo Pro", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #fff; text-align: center; margin-bottom: 2rem;}
    .pale-card {background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; border: 1px solid rgba(255,255,255,0.1);}
    .normal {color: #22c55e; font-weight: bold;}
    .vencido {color: #f59e0b; font-weight: bold;}
    .muy-vencido {color: #ef4444; font-weight: bold;}
    .caliente {background: #ef4444; color: white; padding: 0.2rem 0.5rem; border-radius: 5px;}
    .tibio {background: #f59e0b; color: white; padding: 0.2rem 0.5rem; border-radius: 5px;}
    .frio {background: #3b82f6; color: white; padding: 0.2rem 0.5rem; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

GRUPOS = {
    'CERRADOS': {'digitos': ['0','6','8','9'], 'color': '#ef4444', 'icono': '🔒', 'numeros': 16},
    'ABIERTOS': {'digitos': ['2','3','5'], 'color': '#22c55e', 'icono': '🔓', 'numeros': 9},
    'RECTOS': {'digitos': ['1','4','7'], 'color': '#3b82f6', 'icono': '📏', 'numeros': 9}
}

GOOGLE_SHEETS_ID = "1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw"
COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

# =============================================================================
# PARSEO DE FECHAS
# =============================================================================
def parsear_fecha(fecha_str):
    if not fecha_str or pd.isna(fecha_str):
        return None
    
    fecha_str = str(fecha_str).strip()
    
    if isinstance(fecha_str, datetime):
        return fecha_str
    
    formatos = ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y']
    
    for fmt in formatos:
        try:
            fecha = datetime.strptime(fecha_str, fmt)
            if fecha.year <= datetime.now().year + 1:
                return fecha
        except ValueError:
            continue
    
    return None

# =============================================================================
# CLASIFICACIÓN
# =============================================================================
def clasificar_numero(numero):
    if pd.isna(numero): return None
    try: num_str = str(int(float(numero))).zfill(2)
    except: return None
    if len(num_str) < 2: return None
    d1, d2 = num_str[0], num_str[1]
    for grupo, data in GRUPOS.items():
        if d1 in data['digitos'] and d2 in data['digitos']:
            return grupo
    return None

def detectar_pales_sesion(fila):
    pales = []
    numeros = [
        ('Fijo', fila.get(COL_FIJO)),
        ('1er Corrido', fila.get(COL_CORR1)),
        ('2do Corrido', fila.get(COL_CORR2))
    ]
    
    por_grupo = defaultdict(list)
    for pos, num in numeros:
        if pd.notna(num):
            g = clasificar_numero(num)
            if g:
                num_str = str(int(float(num))).zfill(2)
                por_grupo[g].append({'posicion': pos, 'numero': num_str})
    
    for grupo, items in por_grupo.items():
        if len(items) >= 2:
            pales.append({
                'fecha_str': fila.get(COL_FECHA, ''),
                'sesion': fila.get(COL_SESION, ''),
                'grupo': grupo,
                'numeros': [i['numero'] for i in items],
                'posiciones': [i['posicion'] for i in items],
                'combinacion': '-'.join(sorted([i['numero'] for i in items]))
            })
    return pales

# =============================================================================
# ANÁLISIS DE COMBINACIONES
# =============================================================================
def analizar_combinaciones(df):
    """Analiza todas las combinaciones de pales con estadísticas completas"""
    
    # Recopilar todas las apariciones de cada combinación
    combinaciones_data = defaultdict(list)
    
    # También guardar todas las fechas del dataset para calcular promedio por día
    todas_fechas = []
    
    for _, fila in df.iterrows():
        fecha_dt = parsear_fecha(fila.get(COL_FECHA))
        if fecha_dt:
            todas_fechas.append(fecha_dt)
        
        pales = detectar_pales_sesion(fila)
        for p in pales:
            key = f"{p['grupo']}|{p['combinacion']}"
            combinaciones_data[key].append({
                'fecha_dt': fecha_dt,
                'fecha_str': p['fecha_str'],
                'sesion': p['sesion'],
                'numeros': p['numeros'],
                'grupo': p['grupo']
            })
    
    # Calcular estadísticas para cada combinación
    resultados = []
    hoy = datetime.now()
    
    # Fecha más antigua y más reciente del dataset
    if todas_fechas:
        fecha_min = min(todas_fechas)
        fecha_max = max(todas_fechas)
        dias_totales = (fecha_max - fecha_min).days + 1
    else:
        dias_totales = 1
    
    for key, apariciones in combinaciones_data.items():
        grupo, combinacion = key.split('|')
        
        # Ordenar por fecha
        apariciones.sort(key=lambda x: x['fecha_dt'] if x['fecha_dt'] else datetime.min)
        
        # Fechas de aparición
        fechas = [a['fecha_dt'] for a in apariciones if a['fecha_dt']]
        
        # Calcular gaps (días entre apariciones)
        gaps = []
        for i in range(1, len(fechas)):
            gap = (fechas[i] - fechas[i-1]).days
            gaps.append(gap)
        
        # Última aparición
        ultima_fecha = fechas[-1] if fechas else None
        ultima_fecha_str = ultima_fecha.strftime('%d-%m-%Y') if ultima_fecha else 'N/A'
        
        # Días sin aparecer (ausencia actual)
        dias_sin_aparecer = (hoy - ultima_fecha).days if ultima_fecha else 0
        
        # Frecuencia
        frecuencia = len(apariciones)
        
        # Promedio por día (cada cuántos días aparece en promedio)
        if dias_totales > 0 and frecuencia > 1:
            promedio_por_dia = dias_totales / frecuencia
        else:
            promedio_por_dia = dias_totales if frecuencia > 0 else 0
        
        # Promedio de gaps
        promedio_gaps = round(np.mean(gaps), 1) if gaps else 0
        
        # Ausencia máxima (gap máximo histórico)
        ausencia_maxima = max(gaps) if gaps else 0
        
        # Determinar estado
        if dias_sin_aparecer <= promedio_gaps:
            estado = 'NORMAL'
            estado_class = 'normal'
        elif dias_sin_aparecer > promedio_gaps * 1.5:
            estado = 'MUY VENCIDO'
            estado_class = 'muy-vencido'
        else:
            estado = 'VENCIDO'
            estado_class = 'vencido'
        
        # Determinar temperatura
        if dias_sin_aparecer <= promedio_gaps * 0.5:
            temperatura = '🔥 CALIENTE'
            temp_class = 'caliente'
        elif dias_sin_aparecer <= promedio_gaps:
            temperatura = '🌡️ TIBIO'
            temp_class = 'tibio'
        else:
            temperatura = '❄️ FRÍO'
            temp_class = 'frio'
        
        resultados.append({
            'grupo': grupo,
            'combinacion': combinacion,
            'numeros': apariciones[-1]['numeros'] if apariciones else [],
            'frecuencia': frecuencia,
            'promedio_dias': round(promedio_por_dia, 1),
            'promedio_gaps': promedio_gaps,
            'ausencia_maxima': ausencia_maxima,
            'dias_sin_aparecer': dias_sin_aparecer,
            'ultima_fecha': ultima_fecha_str,
            'ultima_fecha_dt': ultima_fecha,
            'estado': estado,
            'estado_class': estado_class,
            'temperatura': temperatura,
            'temp_class': temp_class,
            'color': GRUPOS[grupo]['color'],
            'icono': GRUPOS[grupo]['icono']
        })
    
    # Ordenar por frecuencia descendente
    return sorted(resultados, key=lambda x: (-x['frecuencia'], x['dias_sin_aparecer']))

# =============================================================================
# ANÁLISIS DETALLADO POR GRUPO
# =============================================================================
def analizar_grupo_detallado(df, grupo):
    pales_grupo = []
    fechas_con_pale = []
    
    for _, fila in df.iterrows():
        fecha_dt = parsear_fecha(fila.get(COL_FECHA))
        pales = detectar_pales_sesion(fila)
        
        for p in pales:
            p['fecha_dt'] = fecha_dt
            if p['grupo'] == grupo:
                pales_grupo.append(p)
                if fecha_dt:
                    fechas_con_pale.append(fecha_dt)
    
    pales_grupo.sort(key=lambda x: x['fecha_dt'] if x['fecha_dt'] else datetime.min, reverse=True)
    
    fechas_unicas_pale = sorted(set(fechas_con_pale))
    
    gaps = []
    for i in range(1, len(fechas_unicas_pale)):
        gap = (fechas_unicas_pale[i] - fechas_unicas_pale[i-1]).days
        gaps.append(gap)
    
    gap_actual = 0
    ultima_fecha = None
    ultima_fecha_str = ''
    
    if fechas_unicas_pale:
        ultima_fecha = fechas_unicas_pale[-1]
        ultima_fecha_str = ultima_fecha.strftime('%d-%m-%Y')
        gap_actual = (datetime.now() - ultima_fecha).days
    
    por_sesion = {'M': {'total': 0, 'pales': 0}, 
                  'T': {'total': 0, 'pales': 0}, 
                  'N': {'total': 0, 'pales': 0}}
    
    for _, fila in df.iterrows():
        sesion = str(fila.get(COL_SESION, '')).strip().upper()
        if sesion in por_sesion:
            por_sesion[sesion]['total'] += 1
            pales = detectar_pales_sesion(fila)
            for p in pales:
                if p['grupo'] == grupo:
                    por_sesion[sesion]['pales'] += 1
    
    sesiones_con_pale = len(set(f"{p['fecha_str']}-{p['sesion']}" for p in pales_grupo))
    total_sesiones = len(df)
    
    return {
        'grupo': grupo,
        'total_sesiones': total_sesiones,
        'sesiones_con_pale': sesiones_con_pale,
        'porcentaje_total': round(sesiones_con_pale/total_sesiones*100, 2) if total_sesiones > 0 else 0,
        'total_pales': len(pales_grupo),
        'ultima_fecha': ultima_fecha_str,
        'gap_actual': gap_actual,
        'gap_promedio': round(np.mean(gaps), 1) if gaps else 0,
        'gap_maximo': max(gaps) if gaps else 0,
        'gap_minimo': min(gaps) if gaps else 0,
        'por_sesion': por_sesion,
        'pales_detalle': pales_grupo,
        'fechas_pales': fechas_unicas_pale
    }

# =============================================================================
# ANÁLISIS SEMANAL
# =============================================================================
def analisis_semanal(df, semanas=12):
    hoy = datetime.now()
    resultados = []
    
    for i in range(semanas):
        dias_desde_domingo = (hoy.weekday() + 1) % 7
        domingo_actual = hoy - timedelta(days=dias_desde_domingo)
        domingo_semana = domingo_actual - timedelta(weeks=i)
        sabado_semana = domingo_semana + timedelta(days=6)
        
        if i == 0:
            sabado_semana = hoy
        
        fecha_inicio = domingo_semana.strftime('%d/%m')
        fecha_fin = sabado_semana.strftime('%d/%m')
        semana_label = f"{fecha_inicio} - {fecha_fin}"
        
        pales_semana = {g: 0 for g in GRUPOS}
        sesiones_semana = 0
        
        for _, fila in df.iterrows():
            fecha_dt = parsear_fecha(fila.get(COL_FECHA))
            if fecha_dt and domingo_semana <= fecha_dt <= sabado_semana:
                sesiones_semana += 1
                pales = detectar_pales_sesion(fila)
                for p in pales:
                    pales_semana[p['grupo']] += 1
        
        resultados.append({
            'semana': semana_label,
            'sesiones': sesiones_semana,
            'cerrados': pales_semana['CERRADOS'],
            'abiertos': pales_semana['ABIERTOS'],
            'rectos': pales_semana['RECTOS'],
            'total_pales': sum(pales_semana.values())
        })
    
    return resultados

# =============================================================================
# BACKTEST MENSUAL
# =============================================================================
def backtest_mensual(df, meses=6):
    resultados = []
    hoy = datetime.now()
    
    for i in range(meses):
        mes = hoy.month - i
        anio = hoy.year
        
        while mes <= 0:
            mes += 12
            anio -= 1
        
        fecha_ini = datetime(anio, mes, 1)
        if mes == 12:
            fecha_fin = datetime(anio + 1, 1, 1) - timedelta(days=1)
        else:
            fecha_fin = datetime(anio, mes + 1, 1) - timedelta(days=1)
        
        if i == 0:
            fecha_fin = hoy
        
        mes_nombre = calendar.month_name[mes].capitalize() + ' ' + str(anio)
        
        pales_mes = {g: 0 for g in GRUPOS}
        sesiones_mes = 0
        
        for _, fila in df.iterrows():
            fecha_dt = parsear_fecha(fila.get(COL_FECHA))
            if fecha_dt and fecha_ini <= fecha_dt <= fecha_fin:
                sesiones_mes += 1
                pales = detectar_pales_sesion(fila)
                for p in pales:
                    pales_mes[p['grupo']] += 1
        
        resultados.append({
            'mes': mes_nombre,
            'sesiones': sesiones_mes,
            'cerrados': pales_mes['CERRADOS'],
            'abiertos': pales_mes['ABIERTOS'],
            'rectos': pales_mes['RECTOS'],
            'total_pales': sum(pales_mes.values())
        })
    
    return resultados

# =============================================================================
# ALERTAS
# =============================================================================
def generar_alertas(analisis):
    alertas = []
    
    for grupo, data in analisis.items():
        if data['gap_actual'] > data['gap_promedio'] * 1.5 and data['gap_promedio'] > 0:
            alertas.append({
                'tipo': 'PROLONGADO',
                'grupo': grupo,
                'mensaje': f"⚠️ {grupo}: Sin pale hace {data['gap_actual']} días (promedio: {data['gap_promedio']})",
                'severidad': 'ALTA' if data['gap_actual'] > data['gap_maximo'] * 0.8 else 'MEDIA'
            })
        
        if data['gap_actual'] >= data['gap_maximo'] and data['gap_maximo'] > 0:
            alertas.append({
                'tipo': 'RÉCORD',
                'grupo': grupo,
                'mensaje': f"🔴 {grupo}: ¡RÉCORD! {data['gap_actual']} días sin pale",
                'severidad': 'ALTA'
            })
    
    return sorted(alertas, key=lambda x: 0 if x['severidad']=='ALTA' else 1)

# =============================================================================
# CONEXIÓN
# =============================================================================
@st.cache_resource
def conectar_gs():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_file = next((n for n in ['credentials.json', 'credenciales.json'] if os.path.exists(n)), None)
        if not creds_file: return None, "No hay credentials.json"
        cred = Credentials.from_service_account_file(creds_file, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
        return gspread.authorize(cred), None
    except Exception as e: return None, str(e)

@st.cache_data(ttl=300)
def cargar_gs(_gc, sid, hoja):
    try: 
        return pd.DataFrame(_gc.open_by_key(sid).worksheet(hoja).get_all_records()), None
    except Exception as e: 
        return None, str(e)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.markdown("<h1 class='main-header'>🎯 PaleGeo Pro v3.3</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#94a3b8'>Análisis Completo de Pales y Combinaciones</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Grupos")
    for grupo, data in GRUPOS.items():
        st.sidebar.markdown(f"""
        <div style='background:{data['color']}20;padding:0.75rem;border-radius:8px;border-left:3px solid {data['color']};margin-bottom:0.5rem'>
            <b style='color:{data['color']}'>{data['icono']} {grupo}</b><br>
            <span style='color:#94a3b8;font-size:0.8rem'>Dígitos: {', '.join(data['digitos'])}</span><br>
            <span style='color:#64748b;font-size:0.75rem'>{data['numeros']} números</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Cargar datos
    gc, err = conectar_gs()
    df = None
    
    if err:
        st.warning(f"⚠️ {err}")
        archivo = st.file_uploader("📁 Carga archivo", type=['csv','xlsx','xls'])
        if archivo:
            try: df = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo, sheet_name='Geotodo')
            except Exception as e: st.error(f"Error: {e}"); return
    else:
        with st.spinner("📊 Cargando Google Sheets..."):
            df, err2 = cargar_gs(gc, GOOGLE_SHEETS_ID, "Geotodo")
        if err2: st.error(f"❌ {err2}"); return
        st.success(f"✅ {len(df)} registros")
    
    if df is None or df.empty: return
    
    df.columns = [str(c).strip() for c in df.columns]
    
    faltan = [c for c in [COL_FECHA, COL_SESION, COL_FIJO, COL_CORR1, COL_CORR2] if c not in df.columns]
    if faltan: st.error(f"❌ Faltan: {faltan}"); return
    
    with st.spinner("🔄 Analizando..."):
        analisis = {g: analizar_grupo_detallado(df, g) for g in GRUPOS}
        backtest = backtest_mensual(df)
        semanal = analisis_semanal(df)
        alertas = generar_alertas(analisis)
        combinaciones = analizar_combinaciones(df)
    
    # =========================================================================
    # RESUMEN GENERAL
    # =========================================================================
    st.header("📊 Resumen General")
    
    resumen_data = []
    for grupo, data in analisis.items():
        g = GRUPOS[grupo]
        resumen_data.append({
            'Grupo': f"{g['icono']} {grupo}",
            'Sesiones con Pale': data['sesiones_con_pale'],
            'Total': data['total_sesiones'],
            '%': f"{data['porcentaje_total']}%",
            'Última Fecha': data['ultima_fecha'],
            'Gap Actual': f"{data['gap_actual']} días",
            'Gap Promedio': f"{data['gap_promedio']} días",
            'Gap Máximo': f"{data['gap_maximo']} días"
        })
    
    st.dataframe(pd.DataFrame(resumen_data), use_container_width=True, hide_index=True)
    
    # =========================================================================
    # ALERTAS
    # =========================================================================
    if alertas:
        st.header("🚨 Alertas")
        for a in alertas:
            color = '#ef4444' if a['severidad'] == 'ALTA' else '#f59e0b'
            st.markdown(f"""
            <div style='background:{color}20;padding:1rem;border-radius:8px;border-left:4px solid {color};margin-bottom:0.5rem'>
                <b style='color:{color}'>[{a['tipo']}]</b> {a['mensaje']}
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # COMBINACIONES MÁS SALIDORAS (NUEVO)
    # =========================================================================
    st.header("🎯 Combinaciones Más Salidoras")
    st.markdown("""
    <p style='color:#94a3b8;font-size:0.85rem'>
        <b>Estado:</b> 
        <span class='normal'>NORMAL</span> = ausencia ≤ promedio | 
        <span class='vencido'>VENCIDO</span> = ausencia > promedio | 
        <span class='muy-vencido'>MUY VENCIDO</span> = ausencia > 1.5×promedio
    </p>
    """, unsafe_allow_html=True)
    
    # Filtros
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filtro_grupo = st.selectbox("Grupo", ['Todos'] + list(GRUPOS.keys()))
    with col_f2:
        filtro_estado = st.selectbox("Estado", ['Todos', 'NORMAL', 'VENCIDO', 'MUY VENCIDO'])
    with col_f3:
        filtro_temp = st.selectbox("Temperatura", ['Todos', '🔥 CALIENTE', '🌡️ TIBIO', '❄️ FRÍO'])
    
    # Aplicar filtros
    comb_filtradas = combinaciones.copy()
    if filtro_grupo != 'Todos':
        comb_filtradas = [c for c in comb_filtradas if c['grupo'] == filtro_grupo]
    if filtro_estado != 'Todos':
        comb_filtradas = [c for c in comb_filtradas if c['estado'] == filtro_estado]
    if filtro_temp != 'Todos':
        comb_filtradas = [c for c in comb_filtradas if c['temperatura'] == filtro_temp]
    
    # Mostrar tabla de combinaciones
    comb_df_data = []
    for c in comb_filtradas[:50]:  # Top 50
        comb_df_data.append({
            'Grupo': f"{c['icono']} {c['grupo']}",
            'Combinación': c['combinacion'],
            'Frecuencia': c['frecuencia'],
            'Prom.Días': c['promedio_gaps'],
            'Ausencia Máx': c['ausencia_maxima'],
            'Sin Aparecer': c['dias_sin_aparecer'],
            'Última Fecha': c['ultima_fecha'],
            'Temperatura': c['temperatura'],
            'Estado': c['estado']
        })
    
    if comb_df_data:
        df_comb = pd.DataFrame(comb_df_data)
        st.dataframe(df_comb, use_container_width=True, hide_index=True)
    else:
        st.info("No hay combinaciones con los filtros seleccionados")
    
    # =========================================================================
    # ANÁLISIS SEMANAL
    # =========================================================================
    st.header("📅 Análisis Semanal (Dom-Sáb)")
    
    semanal_df = pd.DataFrame([{
        'Semana': s['semana'],
        'Sesiones': s['sesiones'],
        '🔒': s['cerrados'],
        '🔓': s['abiertos'],
        '📏': s['rectos'],
        'Total': s['total_pales']
    } for s in semanal])
    
    st.dataframe(semanal_df, use_container_width=True, hide_index=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        prom_cerr = np.mean([s['cerrados'] for s in semanal if s['sesiones'] > 0])
        st.metric("🔒 Cerrados/Semana", f"{prom_cerr:.1f}")
    with col2:
        prom_abier = np.mean([s['abiertos'] for s in semanal if s['sesiones'] > 0])
        st.metric("🔓 Abiertos/Semana", f"{prom_abier:.1f}")
    with col3:
        prom_rect = np.mean([s['rectos'] for s in semanal if s['sesiones'] > 0])
        st.metric("📏 Rectos/Semana", f"{prom_rect:.1f}")
    
    # =========================================================================
    # BACKTEST MENSUAL
    # =========================================================================
    st.header("📆 Backtest Mensual")
    
    bt_df = pd.DataFrame([{
        'Mes': b['mes'],
        'Sesiones': b['sesiones'],
        '🔒 Cerrados': b['cerrados'],
        '🔓 Abiertos': b['abiertos'],
        '📏 Rectos': b['rectos'],
        'Total': b['total_pales']
    } for b in backtest])
    
    st.dataframe(bt_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # DETALLE POR GRUPO
    # =========================================================================
    st.header("📈 Detalle por Grupo")
    
    for grupo, data in analisis.items():
        g = GRUPOS[grupo]
        
        with st.expander(f"{g['icono']} {grupo}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Pales", data['total_pales'])
                st.metric("Sesiones con Pale", data['sesiones_con_pale'])
            
            with col2:
                st.metric("% Aparición", f"{data['porcentaje_total']}%")
                st.metric("Última Fecha", data['ultima_fecha'])
            
            with col3:
                st.metric("Gap Actual", f"{data['gap_actual']} días")
                st.metric("Gap Promedio", f"{data['gap_promedio']} días")
            
            with col4:
                st.metric("Gap Máximo", f"{data['gap_maximo']} días")
                st.metric("Gap Mínimo", f"{data['gap_minimo']} días")
            
            st.subheader("Por Sesión")
            sesion_df = pd.DataFrame([
                {'Sesión': s, 'Total': d['total'], 'Pales': d['pales'],
                 '%': f"{round(d['pales']/d['total']*100,1) if d['total']>0 else 0}%"}
                for s, d in data['por_sesion'].items()
            ])
            st.dataframe(sesion_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # ÚLTIMOS PALES
    # =========================================================================
    st.header("📜 Últimos Pales Detectados")
    
    todos_pales = []
    for grupo, data in analisis.items():
        for p in data['pales_detalle']:
            p['color'] = GRUPOS[grupo]['color']
            p['icono'] = GRUPOS[grupo]['icono']
            todos_pales.append(p)
    
    todos_pales.sort(key=lambda x: x['fecha_dt'] if x['fecha_dt'] else datetime.min, reverse=True)
    
    filtro = st.selectbox("Filtrar", ['Todos'] + list(GRUPOS.keys()), key='filtro_hist')
    
    if filtro != 'Todos':
        todos_pales = [p for p in todos_pales if p['grupo'] == filtro]
    
    for p in todos_pales[:30]:
        nums = " ".join([f"<span style='background:{p['color']};color:white;padding:0.3rem 0.6rem;border-radius:5px;margin:0.1rem;font-weight:bold'>{n}</span>" for n in p['numeros']])
        fecha_display = p['fecha_dt'].strftime('%d-%m-%Y') if p['fecha_dt'] else p['fecha_str']
        sesion_display = {'M': '🌅 M', 'T': '☀️ T', 'N': '🌙 N'}.get(p['sesion'], p['sesion'])
        
        st.markdown(f"""
        <div class='pale-card'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem'>
                <span style='color:#fff;font-weight:bold'>{fecha_display}</span>
                <span style='background:{p['color']}30;color:{p['color']};padding:0.2rem 0.5rem;border-radius:10px;font-size:0.85rem'>
                    {p['icono']} {p['grupo']} | {sesion_display}
                </span>
            </div>
            <div>{nums}</div>
            <small style='color:#64748b'>Posiciones: {' + '.join(p['posiciones'])} | Comb: {p.get('combinacion', '')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#64748b'>🎯 PaleGeo Pro v3.3</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()