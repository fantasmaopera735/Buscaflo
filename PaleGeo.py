# =============================================================================
# 🎯 PaleGeo - Análisis de Pales por Grupos
# =============================================================================
# Lee datos desde Google Sheets (hoja Geotodo)
# Ejecutar con: streamlit run PaleGeo.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="PaleGeo - Análisis de Pales",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTES - GRUPOS DE DÍGITOS
# =============================================================================
GRUPOS = {
    'CERRADOS': ['0', '6', '8', '9'],  # 16 números
    'ABIERTOS': ['2', '3', '5'],       # 9 números
    'RECTOS': ['1', '4', '7']          # 9 números
}

COLORES_GRUPO = {
    'CERRADOS': '#ef4444',  # Rojo
    'ABIERTOS': '#22c55e',  # Verde
    'RECTOS': '#3b82f6'     # Azul
}

ICONOS_GRUPO = {
    'CERRADOS': '🔒',
    'ABIERTOS': '🔓',
    'RECTOS': '📏'
}

# Google Sheets ID (mismo que Geotodo)
GOOGLE_SHEETS_ID = "1mKTOJbvHhHSl4NYPgYxulhWHRvYFfkyJHxQ0PfDhYzI"

# =============================================================================
# FUNCIONES DE CLASIFICACIÓN
# =============================================================================
def clasificar_digito(digito):
    """Clasifica un dígito en su grupo correspondiente"""
    for grupo, digitos in GRUPOS.items():
        if digito in digitos:
            return grupo
    return None

def clasificar_numero(numero):
    """
    Clasifica un número completo.
    Solo pertenece a un grupo si AMBOS dígitos son del mismo grupo.
    """
    if pd.isna(numero):
        return None
    
    try:
        num_str = str(int(float(numero))).zfill(2)
    except:
        return None
    
    if len(num_str) < 2:
        return None
    
    d1, d2 = num_str[0], num_str[1]
    grupo1 = clasificar_digito(d1)
    grupo2 = clasificar_digito(d2)
    
    # Ambos dígitos deben ser del mismo grupo
    if grupo1 and grupo1 == grupo2:
        return grupo1
    
    return None  # Número mixto

# =============================================================================
# FUNCIONES DE DETECCIÓN DE PALES
# =============================================================================
def detectar_pales_sesion(fila, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2):
    """
    Detecta pales en una sesión.
    Un pale ocurre cuando 2+ números del mismo grupo aparecen juntos.
    """
    pales = []
    
    # Obtener valores de la fila
    fijo = fila.get(col_fijo)
    corr1 = fila.get(col_corr1)
    corr2 = fila.get(col_corr2)
    
    numeros = [
        ('Fijo', fijo),
        ('1er Corrido', corr1),
        ('2do Corrido', corr2)
    ]
    
    # Agrupar por grupo
    por_grupo = defaultdict(list)
    
    for pos, num in numeros:
        if pd.notna(num):
            grupo = clasificar_numero(num)
            if grupo:
                num_str = str(int(float(num))).zfill(2)
                por_grupo[grupo].append((pos, num_str))
    
    # Detectar pales (2+ números del mismo grupo)
    for grupo, items in por_grupo.items():
        if len(items) >= 2:
            pales.append({
                'fecha': fila.get(col_fecha, ''),
                'sesion': fila.get(col_sesion, ''),
                'grupo': grupo,
                'numeros': [item[1] for item in items],
                'posiciones': [item[0] for item in items]
            })
    
    return pales

# =============================================================================
# FUNCIONES DE ESTADÍSTICAS
# =============================================================================
def calcular_estadisticas_grupo(df, grupo, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2):
    """Calcula estadísticas detalladas para un grupo"""
    pales_grupo = []
    
    for idx, fila in df.iterrows():
        pales = detectar_pales_sesion(fila, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
        for p in pales:
            if p['grupo'] == grupo:
                pales_grupo.append(p)
    
    # Fechas únicas ordenadas - USAR col_fecha VARIABLE
    fechas = df[col_fecha].dropna().unique()
    fechas_unicas = sorted(fechas, key=lambda x: datetime.strptime(str(x), '%d-%m-%Y'))
    
    # Calcular brechas
    brechas = []
    ultima_aparicion = ''
    ultima_fecha_pale = None
    brecha_actual = 0
    
    for fecha in fechas_unicas:
        pales_fecha = [p for p in pales_grupo if p['fecha'] == fecha]
        if pales_fecha:
            ultima_aparicion = fecha
            if ultima_fecha_pale:
                try:
                    f1 = datetime.strptime(str(ultima_fecha_pale), '%d-%m-%Y')
                    f2 = datetime.strptime(str(fecha), '%d-%m-%Y')
                    brechas.append((f2 - f1).days)
                except:
                    pass
            ultima_fecha_pale = fecha
    
    # Brecha actual
    if ultima_fecha_pale:
        hoy = datetime.now()
        try:
            f_ultima = datetime.strptime(str(ultima_fecha_pale), '%d-%m-%Y')
            brecha_actual = (hoy - f_ultima).days
        except:
            pass
    
    # Sesiones con pale único
    sesiones_con_pale = len(set(f"{p['fecha']}-{p['sesion']}" for p in pales_grupo))
    
    return {
        'total': len(df),
        'pales': sesiones_con_pale,
        'porcentaje': round((sesiones_con_pale / len(df)) * 100, 1) if len(df) > 0 else 0,
        'ultima_aparicion': ultima_aparicion,
        'brecha_actual': brecha_actual,
        'brecha_promedio': round(np.mean(brechas), 1) if brechas else 0,
        'brecha_maxima': max(brechas) if brechas else 0
    }

def calcular_estadisticas_todos_grupos(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2):
    """Calcula estadísticas para todos los grupos"""
    stats = {}
    for grupo in GRUPOS.keys():
        stats[grupo] = calcular_estadisticas_grupo(df, grupo, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
    return stats

# =============================================================================
# ANÁLISIS POR SESIÓN
# =============================================================================
def analizar_pales_por_sesion(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2):
    """Analiza pales por tipo de sesión (M, T, N)"""
    resultados = []
    sesiones = {'M': 'Mañana', 'T': 'Tarde', 'N': 'Noche'}
    
    for codigo, nombre in sesiones.items():
        df_sesion = df[df[col_sesion] == codigo]
        
        con_pale = 0
        pales_por_grupo = {g: 0 for g in GRUPOS.keys()}
        
        for idx, fila in df_sesion.iterrows():
            pales = detectar_pales_sesion(fila, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
            if pales:
                con_pale += 1
                for p in pales:
                    pales_por_grupo[p['grupo']] += 1
        
        resultados.append({
            'sesion': nombre,
            'codigo': codigo,
            'total': len(df_sesion),
            'con_pale': con_pale,
            'porcentaje': round((con_pale / len(df_sesion)) * 100, 1) if len(df_sesion) > 0 else 0,
            'pales_por_grupo': pales_por_grupo
        })
    
    return resultados

# =============================================================================
# COMBINACIONES FRECUENTES
# =============================================================================
def detectar_combinaciones_frecuentes(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2, top_n=20):
    """Detecta las combinaciones de números más frecuentes"""
    conteo = defaultdict(lambda: {'count': 0, 'grupo': '', 'ultima_fecha': ''})
    
    for idx, fila in df.iterrows():
        pales = detectar_pales_sesion(fila, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
        for p in pales:
            comb = '-'.join(sorted(p['numeros']))
            key = f"{p['grupo']}:{comb}"
            conteo[key]['count'] += 1
            conteo[key]['grupo'] = p['grupo']
            conteo[key]['ultima_fecha'] = p['fecha']
    
    # Convertir a lista ordenada
    resultados = []
    for key, data in conteo.items():
        combinacion = key.split(':')[1]
        resultados.append({
            'combinacion': combinacion,
            'grupo': data['grupo'],
            'frecuencia': data['count'],
            'ultima_aparicion': data['ultima_fecha']
        })
    
    return sorted(resultados, key=lambda x: x['frecuencia'], reverse=True)[:top_n]

# =============================================================================
# BACKTEST
# =============================================================================
def backtest_pale(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2, meses_atras=6):
    """Realiza backtest de pales por mes"""
    try:
        from dateutil.relativedelta import relativedelta
    except:
        pass
    
    resultados = []
    hoy = datetime.now()
    
    for i in range(meses_atras):
        # Calcular rango del mes
        if i == 0:
            fecha_fin = hoy
        else:
            fecha_fin = datetime(hoy.year, hoy.month - i, 1) - timedelta(days=1)
        
        fecha_ini = datetime(fecha_fin.year, fecha_fin.month, 1)
        mes_nombre = fecha_fin.strftime('%B %Y')
        
        # Filtrar datos del mes
        pales_count = 0
        pales_por_grupo = {g: 0 for g in GRUPOS.keys()}
        
        for idx, fila in df.iterrows():
            try:
                fecha_str = str(fila[col_fecha])
                f = datetime.strptime(fecha_str, '%d-%m-%Y')
                
                if fecha_ini <= f <= fecha_fin:
                    pales = detectar_pales_sesion(fila, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
                    if pales:
                        pales_count += 1
                        for p in pales:
                            pales_por_grupo[p['grupo']] += 1
            except:
                continue
        
        resultados.append({
            'periodo': mes_nombre.capitalize(),
            'pales': pales_count,
            'pales_por_grupo': pales_por_grupo
        })
    
    return resultados

# =============================================================================
# GENERADOR DE ALERTAS
# =============================================================================
def generar_alertas_pale(stats):
    """Genera alertas basadas en las estadísticas"""
    alertas = []
    
    for grupo, data in stats.items():
        # Alerta por brecha prolongada
        if data['brecha_actual'] > data['brecha_promedio'] * 1.5 and data['brecha_actual'] > 7:
            severidad = 'alta' if data['brecha_actual'] > data['brecha_maxima'] * 0.8 else 'media'
            alertas.append({
                'tipo': 'brecha_prolongada',
                'grupo': grupo,
                'mensaje': f"{grupo}: Sin pale hace {data['brecha_actual']} días (promedio: {data['brecha_promedio']})",
                'severidad': severidad
            })
        
        # Alerta por récord de brecha
        if data['brecha_actual'] >= data['brecha_maxima'] and data['brecha_maxima'] > 0:
            alertas.append({
                'tipo': 'brecha_record',
                'grupo': grupo,
                'mensaje': f"{grupo}: ¡Récord de brecha! {data['brecha_actual']} días sin pale",
                'severidad': 'alta'
            })
    
    # Ordenar por severidad
    orden = {'alta': 0, 'media': 1, 'baja': 2}
    return sorted(alertas, key=lambda x: orden[x['severidad']])

# =============================================================================
# DETECCIÓN DE COLUMNAS
# =============================================================================
def detectar_columnas(df):
    """Detecta los nombres de columnas automáticamente"""
    columnas = {}
    
    # Obtener lista de columnas en minúsculas
    df_cols_lower = {str(c).strip().lower(): c for c in df.columns}
    
    st.write("🔍 Columnas encontradas:", list(df.columns))
    st.write("🔍 Columnas en minúscula:", list(df_cols_lower.keys()))
    
    # Detectar columna de fecha
    for posible in ['fecha', 'date', 'dia']:
        if posible in df_cols_lower:
            columnas['fecha'] = df_cols_lower[posible]
            break
    
    # Si no encontró, buscar parcialmente
    if 'fecha' not in columnas:
        for col_lower, col_original in df_cols_lower.items():
            if 'fecha' in col_lower:
                columnas['fecha'] = col_original
                break
    
    # Detectar columna de sesión
    for posible in ['tarde/noche', 'tarde_noche', 'sesion', 't/n']:
        if posible in df_cols_lower:
            columnas['sesion'] = df_cols_lower[posible]
            break
    
    if 'sesion' not in columnas:
        for col_lower, col_original in df_cols_lower.items():
            if 'tarde' in col_lower or 'noche' in col_lower or 'sesion' in col_lower:
                columnas['sesion'] = col_original
                break
    
    # Detectar columna Fijo
    for posible in ['fijo', 'f']:
        if posible in df_cols_lower:
            columnas['fijo'] = df_cols_lower[posible]
            break
    
    if 'fijo' not in columnas:
        for col_lower, col_original in df_cols_lower.items():
            if 'fijo' in col_lower:
                columnas['fijo'] = col_original
                break
    
    # Detectar columna Primer Corrido
    for posible in ['primer_corrido', '1er corrido', 'primer corrido', '1er_corrido', 'corr1']:
        if posible in df_cols_lower:
            columnas['corr1'] = df_cols_lower[posible]
            break
    
    if 'corr1' not in columnas:
        for col_lower, col_original in df_cols_lower.items():
            if 'primer' in col_lower and 'corrido' in col_lower:
                columnas['corr1'] = col_original
                break
            if '1er' in col_lower:
                columnas['corr1'] = col_original
                break
    
    # Detectar columna Segundo Corrido
    for posible in ['segundo_corrido', '2do corrido', 'segundo corrido', '2do_corrido', 'corr2']:
        if posible in df_cols_lower:
            columnas['corr2'] = df_cols_lower[posible]
            break
    
    if 'corr2' not in columnas:
        for col_lower, col_original in df_cols_lower.items():
            if 'segundo' in col_lower and 'corrido' in col_lower:
                columnas['corr2'] = col_original
                break
            if '2do' in col_lower:
                columnas['corr2'] = col_original
                break
    
    return columnas

# =============================================================================
# CONEXIÓN A GOOGLE SHEETS
# =============================================================================
@st.cache_resource
def conectar_google_sheets():
    """Conecta a Google Sheets usando credenciales"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # Buscar archivo de credenciales - AMBOS NOMBRES
        creds_path = None
        for nombre in ['credentials.json', 'credenciales.json']:
            if os.path.exists(nombre):
                creds_path = nombre
                break
        
        if not creds_path:
            return None, "No se encontró 'credentials.json' ni 'credenciales.json'"
        
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        credentials = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(credentials)
        
        return gc, None
    except Exception as e:
        return None, str(e)

def cargar_datos_google_sheets(_gc, sheet_id, hoja_nombre="Geotodo"):
    """Carga datos desde Google Sheets"""
    try:
        sh = _gc.open_by_key(sheet_id)
        worksheet = sh.worksheet(hoja_nombre)
        datos = worksheet.get_all_records()
        df = pd.DataFrame(datos)
        return df, None
    except Exception as e:
        return None, str(e)

# =============================================================================
# INTERFAZ PRINCIPAL
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: white; margin-bottom: 0.5rem;'>
            🎯 PaleGeo 🌍
        </h1>
        <p style='color: #94a3b8; font-size: 1.1rem;'>
            Análisis de Pales por Grupos
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    
    # Información de grupos en sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Grupos de Análisis")
    
    for grupo, digitos in GRUPOS.items():
        st.sidebar.markdown(f"""
        <div style='background: {COLORES_GRUPO[grupo]}20; padding: 0.75rem; 
                    border-radius: 10px; border-left: 3px solid {COLORES_GRUPO[grupo]}; margin-bottom: 0.5rem;'>
            <b style='color: {COLORES_GRUPO[grupo]}'>{ICONOS_GRUPO[grupo]} {grupo}</b><br>
            <span style='color: #94a3b8; font-size: 0.85rem;'>Dígitos: {', '.join(digitos)}</span><br>
            <span style='color: #64748b; font-size: 0.8rem;'>{'16 números' if grupo == 'CERRADOS' else '9 números'} posibles</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Conectar a Google Sheets
    gc, error_conn = conectar_google_sheets()
    
    if error_conn:
        st.error(f"❌ Error de conexión: {error_conn}")
        st.info("""
        **Para conectar con Google Sheets:**
        
        1. Asegúrate de tener `credentials.json` en la carpeta
        2. El Google Sheet debe estar compartido con el email de la cuenta de servicio
        """)
        
        # Opción alternativa: cargar archivo
        st.markdown("---")
        st.header("📁 O cargar archivo manualmente")
        
        archivo = st.file_uploader(
            "Selecciona tu archivo (CSV o Excel)",
            type=['csv', 'xlsx', 'xls']
        )
        
        if archivo is None:
            return
        
        # Cargar archivo
        try:
            if archivo.name.endswith('.csv'):
                df = pd.read_csv(archivo)
            else:
                df = pd.read_excel(archivo)
        except Exception as e:
            st.error(f"Error cargando archivo: {e}")
            return
    else:
        # Cargar desde Google Sheets
        with st.spinner("📊 Cargando datos desde Google Sheets..."):
            df, error_data = cargar_datos_google_sheets(gc, GOOGLE_SHEETS_ID, "Geotodo")
        
        if error_data:
            st.error(f"❌ Error cargando datos: {error_data}")
            return
        
        st.success(f"✅ Conectado a Google Sheets - Hoja 'Geotodo'")
    
    # Limpiar datos
    df = df.dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    
    # Detectar columnas
    columnas = detectar_columnas(df)
    
    # Mostrar columnas detectadas
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Columnas Detectadas")
    st.sidebar.json(columnas)
    
    # Verificar columnas necesarias
    columnas_requeridas = ['fecha', 'sesion', 'fijo', 'corr1', 'corr2']
    faltantes = [c for c in columnas_requeridas if c not in columnas]
    
    if faltantes:
        st.error(f"❌ Columnas no detectadas: {faltantes}")
        st.write("**Columnas disponibles:**", list(df.columns))
        st.write("**Columnas necesarias:** Fecha, Sesión (Tarde/Noche), Fijo, 1er Corrido, 2do Corrido")
        return
    
    # Asignar nombres de columnas
    col_fecha = columnas['fecha']
    col_sesion = columnas['sesion']
    col_fijo = columnas['fijo']
    col_corr1 = columnas['corr1']
    col_corr2 = columnas['corr2']
    
    st.success(f"✅ Columnas detectadas correctamente")
    st.info(f"📈 {len(df)} registros cargados")
    
    # Calcular estadísticas
    with st.spinner("🔄 Calculando análisis..."):
        stats = calcular_estadisticas_todos_grupos(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
        
        # Pales históricos
        pales_historico = []
        for idx, fila in df.iterrows():
            pales_historico.extend(detectar_pales_sesion(fila, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2))
        
        analisis_sesion = analizar_pales_por_sesion(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
        combinaciones = detectar_combinaciones_frecuentes(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
        backtest = backtest_pale(df, col_fecha, col_sesion, col_fijo, col_corr1, col_corr2)
        alertas = generar_alertas_pale(stats)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Estadísticas", "🌅 Sesiones", "📅 Backtest", 
        "🔗 Combinaciones", "🚨 Alertas", "📜 Histórico"
    ])
    
    # =========================================================================
    # TAB 1: ESTADÍSTICAS
    # =========================================================================
    with tab1:
        st.header("📊 Estadísticas por Grupo")
        
        cols = st.columns(3)
        
        for i, (grupo, data) in enumerate(stats.items()):
            with cols[i]:
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3); padding: 1.25rem; 
                            border-radius: 15px; border-left: 4px solid {COLORES_GRUPO[grupo]};'>
                    <h3 style='color: {COLORES_GRUPO[grupo]}; margin-top: 0;'>
                        {ICONOS_GRUPO[grupo]} {grupo}
                    </h3>
                """, unsafe_allow_html=True)
                
                st.metric("Total Sesiones", f"{data['total']:,}")
                st.metric("Sesiones con Pale", f"{data['pales']:,}")
                st.metric("Porcentaje", f"{data['porcentaje']}%")
                st.metric("Última Aparición", data['ultima_aparicion'] or "N/A")
                
                # Brechas con indicador
                st.markdown("**Brechas:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Actual", f"{data['brecha_actual']} días", 
                             delta=None if data['brecha_actual'] <= data['brecha_promedio'] else "⚠️ Alto")
                with col2:
                    st.metric("Promedio", f"{data['brecha_promedio']} días")
                with col3:
                    st.metric("Máxima", f"{data['brecha_maxima']} días")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: SESIONES
    # =========================================================================
    with tab2:
        st.header("🌅 Análisis por Sesión")
        
        cols = st.columns(3)
        
        iconos_sesion = {'Mañana': '🌅', 'Tarde': '☀️', 'Noche': '🌙'}
        
        for i, data in enumerate(analisis_sesion):
            with cols[i]:
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: 15px;'>
                    <h3 style='color: #f59e0b; margin-top: 0;'>
                        {iconos_sesion.get(data['sesion'], '')} {data['sesion']}
                    </h3>
                """, unsafe_allow_html=True)
                
                st.metric("Total Sorteos", f"{data['total']:,}")
                st.metric("Con Pale", f"{data['con_pale']:,}")
                st.metric("Porcentaje", f"{data['porcentaje']}%")
                
                st.markdown("**Pales por Grupo:**")
                for grupo, count in data['pales_por_grupo'].items():
                    st.markdown(f"""
                    <span style='background: {COLORES_GRUPO[grupo]}30; color: {COLORES_GRUPO[grupo]}; 
                                padding: 0.25rem 0.75rem; border-radius: 20px; margin-right: 0.5rem;'>
                        {ICONOS_GRUPO[grupo]} {count}
                    </span>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 3: BACKTEST
    # =========================================================================
    with tab3:
        st.header("📅 Backtest - Últimos 6 Meses")
        
        # Crear dataframe para mostrar
        df_backtest = pd.DataFrame([
            {
                'Período': b['periodo'],
                'Total Pales': b['pales'],
                '🔒 Cerrados': b['pales_por_grupo'].get('CERRADOS', 0),
                '🔓 Abiertos': b['pales_por_grupo'].get('ABIERTOS', 0),
                '📏 Rectos': b['pales_por_grupo'].get('RECTOS', 0)
            }
            for b in backtest
        ])
        
        st.dataframe(df_backtest, use_container_width=True, hide_index=True)
        
        # Gráfico de barras
        st.subheader("📈 Visualización")
        
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            for grupo in GRUPOS.keys():
                fig.add_trace(go.Bar(
                    name=f"{ICONOS_GRUPO[grupo]} {grupo}",
                    x=[b['periodo'] for b in backtest],
                    y=[b['pales_por_grupo'].get(grupo, 0) for b in backtest],
                    marker_color=COLORES_GRUPO[grupo]
                ))
            
            fig.update_layout(
                barmode='group',
                title='Pales por Grupo - Últimos 6 Meses',
                xaxis_title='Período',
                yaxis_title='Cantidad de Pales',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Instala plotly para ver gráficos: pip install plotly")
    
    # =========================================================================
    # TAB 4: COMBINACIONES
    # =========================================================================
    with tab4:
        st.header("🔗 Combinaciones Más Frecuentes")
        
        cols = st.columns(4)
        
        for i, comb in enumerate(combinaciones[:20]):
            with cols[i % 4]:
                st.markdown(f"""
                <div style='background: rgba(0,0,0,0.3); padding: 1rem; 
                            border-radius: 10px; border-left: 3px solid {COLORES_GRUPO[comb['grupo']]}; 
                            margin-bottom: 0.75rem;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                        <span style='background: {COLORES_GRUPO[comb['grupo']]}30; 
                                     color: {COLORES_GRUPO[comb['grupo']]}; padding: 0.25rem 0.5rem; 
                                     border-radius: 5px; font-weight: bold; font-size: 1.1rem;'>
                            {comb['combinacion']}
                        </span>
                        <span style='color: #f59e0b; font-weight: bold;'>#{i+1}</span>
                    </div>
                    <div style='font-size: 0.85rem;'>
                        <span style='color: #94a3b8;'>{ICONOS_GRUPO[comb['grupo']]} {comb['grupo']}</span><br>
                        <span style='color: #22c55e;'>{comb['frecuencia']} veces</span><br>
                        <span style='color: #64748b; font-size: 0.75rem;'>Última: {comb['ultima_aparicion']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 5: ALERTAS
    # =========================================================================
    with tab5:
        st.header("🚨 Alertas Activas")
        
        if not alertas:
            st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <div style='font-size: 4rem;'>✅</div>
                <p style='color: #22c55e; font-size: 1.2rem;'>Sin alertas activas</p>
                <p style='color: #64748b;'>Todos los grupos están dentro de los parámetros normales</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for alerta in alertas:
                color = '#ef4444' if alerta['severidad'] == 'alta' else '#f59e0b'
                icono = '🔴' if alerta['severidad'] == 'alta' else '🟡'
                
                st.markdown(f"""
                <div style='background: {color}20; padding: 1rem; border-radius: 10px; 
                            border-left: 4px solid {color}; margin-bottom: 0.75rem;'>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <span style='font-size: 2rem;'>{icono}</span>
                        <div>
                            <p style='color: white; margin: 0; font-weight: bold;'>{alerta['mensaje']}</p>
                            <p style='color: #94a3b8; margin: 0.25rem 0 0 0; font-size: 0.85rem;'>
                                Tipo: {alerta['tipo'].replace('_', ' ')} • Severidad: {alerta['severidad'].upper()}
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 6: HISTÓRICO
    # =========================================================================
    with tab6:
        st.header("📜 Últimos Pales Detectados")
        
        # Filtro por grupo
        filtro = st.selectbox(
            "Filtrar por grupo:",
            ['Todos', 'CERRADOS', 'ABIERTOS', 'RECTOS']
        )
        
        pales_filtrados = pales_historico[-100:]  # Últimos 100
        
        if filtro != 'Todos':
            pales_filtrados = [p for p in pales_filtrados if p['grupo'] == filtro]
        
        # Mostrar en grid
        cols = st.columns(3)
        
        nombres_sesion = {'M': 'Mañana', 'T': 'Tarde', 'N': 'Noche'}
        
        for i, pale in enumerate(reversed(pales_filtrados[-30:])):  # Mostrar últimos 30
            with cols[i % 3]:
                numeros_html = "".join([
                    f"<span style='background: {COLORES_GRUPO[pale['grupo']]}; color: white; padding: 0.25rem 0.5rem; border-radius: 5px; font-weight: bold; margin-right: 0.25rem;'>{n}</span>"
                    for n in pale['numeros']
                ])
                
                st.markdown(f"""
                <div style='background: {COLORES_GRUPO[pale['grupo']]}15; padding: 0.75rem; 
                            border-radius: 10px; border: 1px solid {COLORES_GRUPO[pale['grupo']]}30; margin-bottom: 0.5rem;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span style='color: #94a3b8; font-size: 0.85rem;'>{pale['fecha']}</span>
                        <span style='background: {COLORES_GRUPO[pale['grupo']]}30; color: {COLORES_GRUPO[pale['grupo']]}; 
                                      padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem;'>
                            {nombres_sesion.get(pale['sesion'], pale['sesion'])}
                        </span>
                    </div>
                    <div style='display: flex; gap: 0.25rem;'>
                        {numeros_html}
                    </div>
                    <p style='color: #64748b; margin: 0.5rem 0 0 0; font-size: 0.75rem;'>
                        {' + '.join(pale['posiciones'])}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.caption(f"Mostrando {min(30, len(pales_filtrados))} de {len(pales_filtrados)} pales")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
        🎯 PaleGeo v1.0 • Análisis de Pales por Grupos
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# EJECUTAR
# =============================================================================
if __name__ == "__main__":
    main()