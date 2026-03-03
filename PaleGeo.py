# -*- coding: utf-8 -*-
"""
PaleGeo - Análisis de Pales por Grupos
Hoja: Geotodo (M=Mañana, T=Tarde, N=Noche)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

if __name__ == "__main__":
    st.set_page_config(page_title="PaleGeo", page_icon="🎯", layout="wide")

GS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'
GS_SHEET = 'Geotodo'
COL_FECHA = 'Fecha'
COL_SESION = 'Tipo_Sorteo'
COL_FIJO = 'Fijo'
COL_CORR1 = 'Primer_Corrido'
COL_CORR2 = 'Segundo_Corrido'

# MAPEO para 3 sesiones
MAPEO = {
    't': 'Tarde', 'tarde': 'Tarde',
    'n': 'Noche', 'noche': 'Noche',
    'm': 'Mañana', 'mañana': 'Mañana', 'manana': 'Mañana'
}

GRUPOS = {
    'CERRADOS': {'digitos': {0, 6, 8, 9}, 'numeros': []},
    'ABIERTOS': {'digitos': {2, 3, 5}, 'numeros': []},
    'RECTOS': {'digitos': {1, 4, 7}, 'numeros': []}
}

for i in range(100):
    s = f"{i:02d}"
    d1, d2 = int(s[0]), int(s[1])
    for g, d in GRUPOS.items():
        if d1 in d['digitos'] and d2 in d['digitos']:
            d['numeros'].append(i)

INFO = {g: {'digitos': ','.join(map(str, d['digitos'])), 'cant': len(d['numeros'])} for g, d in GRUPOS.items()}

@st.cache_resource
def conectar():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        
        try:
            if 'gcp_service_account' in st.secrets:
                from google.oauth2.service_account import Credentials
                creds_dict = dict(st.secrets['gcp_service_account'])
                creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
                return gspread.authorize(creds)
        except:
            pass
        
        for f in ['credentials.json', 'credenciales.json']:
            if os.path.exists(f):
                creds = ServiceAccountCredentials.from_json_keyfile_name(f, scope)
                return gspread.authorize(creds)
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# SIN DECORADOR
def cargar_datos(gc, archivo_id, nombre_hoja):
    if gc:
        try:
            spreadsheet = gc.open_by_key(archivo_id)
            worksheet = spreadsheet.worksheet(nombre_hoja)
            return pd.DataFrame(worksheet.get_all_records())
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def parsear_fecha(f):
    if pd.isna(f):
        return None
    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d']:
        try:
            return datetime.strptime(str(f).strip(), fmt)
        except:
            continue
    return None

def normalizar_sesion(s):
    if pd.isna(s):
        return None
    return MAPEO.get(str(s).strip().lower(), s)

def clasificar(n):
    try:
        s = f"{int(n):02d}"
        d1, d2 = int(s[0]), int(s[1])
        for g, d in GRUPOS.items():
            if d1 in d['digitos'] and d2 in d['digitos']:
                return g
        return None
    except:
        return None

def detectar_pales(fila):
    grupos = defaultdict(list)
    for col in [COL_FIJO, COL_CORR1, COL_CORR2]:
        if col in fila.index and pd.notna(fila[col]):
            try:
                n = int(fila[col])
                g = clasificar(n)
                if g:
                    grupos[g].append(f"{n:02d}")
            except:
                pass
    return {g: nums for g, nums in grupos.items() if len(nums) >= 2}

def main():
    st.title("🎯 PaleGeo - Pales por Grupos")
    st.markdown("**Hoja: Geotodo** | Sesiones: Mañana, Tarde y Noche")
    
    # Info grupos
    with st.expander("📋 Grupos"):
        for g, i in INFO.items():
            st.write(f"**{g}**: Dígitos {i['digitos']} ({i['cant']} números)")
    
    gc = conectar()
    if not gc:
        st.error("Sin conexión")
        return
    
    df = cargar_datos(gc, GS_ID, GS_SHEET)
    if df is None or len(df) == 0:
        st.error("Sin datos")
        return
    
    st.success(f"✅ {len(df)} registros")
    
    df['Fecha_Parsed'] = df[COL_FECHA].apply(parsear_fecha)
    df['Sesion'] = df[COL_SESION].apply(normalizar_sesion)
    
    todos_pales = []
    for idx, fila in df.iterrows():
        pales = detectar_pales(fila)
        for g, nums in pales.items():
            todos_pales.append({
                'fecha': fila['Fecha_Parsed'],
                'sesion': fila['Sesion'],
                'grupo': g,
                'numeros': nums
            })
    
    tab1, tab2 = st.tabs(["📊 Resumen", "📜 Historial"])
    
    with tab1:
        for grupo in GRUPOS.keys():
            pales_g = [p for p in todos_pales if p['grupo'] == grupo]
            st.markdown(f"### {grupo}")
            st.metric("Pales", len(pales_g))
            if pales_g:
                with st.expander("Ver últimos"):
                    for p in pales_g[-5:][::-1]:
                        fecha = p['fecha'].strftime('%d/%m/%Y') if pd.notna(p['fecha']) else '-'
                        st.write(f"• {fecha} ({p['sesion']}): {', '.join(p['numeros'])}")
    
    with tab2:
        # Orden: Noche → Tarde → Mañana dentro de cada día
        orden = {'Noche': 1, 'Tarde': 2, 'Mañana': 3}
        pales_ord = sorted(todos_pales, key=lambda x: (x['fecha'], orden.get(x['sesion'], 99)), reverse=True)
        for p in pales_ord[:30]:
            fecha = p['fecha'].strftime('%d/%m/%Y') if pd.notna(p['fecha']) else '-'
            st.markdown(f"**{fecha}** ({p['sesion']}) - {p['grupo']}: {', '.join(p['numeros'])}")

if __name__ == "__main__":
    main()