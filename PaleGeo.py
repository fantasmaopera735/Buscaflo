# -*- coding: utf-8 -*-
"""
PaleGeo - Análisis de Pales por Grupos
Hoja: Geotodo (3 sesiones)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import os

if __name__ == "__main__":
    st.set_page_config(page_title="PaleGeo", page_icon="🎯", layout="wide")

GRUPOS = {
    'CERRADOS': ['0', '6', '8', '9'],
    'ABIERTOS': ['2', '3', '5'],
    'RECTOS': ['1', '4', '7']
}

GOOGLE_SHEETS_ID = "1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw"

def clasificar_numero(numero):
    if pd.isna(numero):
        return None
    try:
        num_str = str(int(float(numero))).zfill(2)
    except:
        return None
    if len(num_str) < 2:
        return None
    d1, d2 = num_str[0], num_str[1]
    for grupo, digitos in GRUPOS.items():
        if d1 in digitos and d2 in digitos:
            return grupo
    return None

def detectar_pales(fila, col_fijo, col_corr1, col_corr2):
    pales = []
    numeros = [('Fijo', fila.get(col_fijo)), ('Corr1', fila.get(col_corr1)), ('Corr2', fila.get(col_corr2))]
    por_grupo = defaultdict(list)
    for pos, num in numeros:
        if pd.notna(num):
            grupo = clasificar_numero(num)
            if grupo:
                por_grupo[grupo].append(str(int(float(num))).zfill(2))
    for grupo, nums in por_grupo.items():
        if len(nums) >= 2:
            pales.append({'grupo': grupo, 'numeros': nums})
    return pales

@st.cache_resource
def conectar_google_sheets():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        
        # Streamlit Cloud - leer secrets
        if 'gcp_service_account' in st.secrets:
            creds_dict = {
                "type": st.secrets["gcp_service_account"]["type"],
                "project_id": st.secrets["gcp_service_account"]["project_id"],
                "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
                "private_key": st.secrets["gcp_service_account"]["private_key"],
                "client_email": st.secrets["gcp_service_account"]["client_email"],
                "client_id": st.secrets["gcp_service_account"]["client_id"],
                "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
                "token_uri": st.secrets["gcp_service_account"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
                "universe_domain": st.secrets["gcp_service_account"].get("universe_domain", "googleapis.com")
            }
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            gc = gspread.authorize(credentials)
            return gc, None
        
        # Local - archivo JSON
        creds_path = None
        for nombre in ['credentials.json', 'credenciales.json']:
            if os.path.exists(nombre):
                creds_path = nombre
                break
        if creds_path:
            credentials = Credentials.from_service_account_file(creds_path, scopes=scopes)
            gc = gspread.authorize(credentials)
            return gc, None
            
        return None, "No hay credenciales configuradas"
    except Exception as e:
        return None, f"Error: {str(e)}"

def cargar_datos_google_sheets(gc, sheet_id, hoja_nombre="Geotodo"):
    try:
        sh = gc.open_by_key(sheet_id)
        worksheet = sh.worksheet(hoja_nombre)
        datos = worksheet.get_all_records()
        df = pd.DataFrame(datos)
        return df, None
    except Exception as e:
        return None, str(e)

def main():
    st.title("🎯 PaleGeo - Pales por Grupos")
    st.markdown("**Hoja: Geotodo** | Sesiones: Mañana, Tarde, Noche")
    
    with st.expander("📋 Ver Grupos"):
        for grupo, digitos in GRUPOS.items():
            st.markdown(f"**{grupo}**: Dígitos {', '.join(digitos)}")
    
    gc, error = conectar_google_sheets()
    if error:
        st.error(f"Error: {error}")
        return
    
    with st.spinner("Cargando..."):
        df, error = cargar_datos_google_sheets(gc, GOOGLE_SHEETS_ID, "Geotodo")
    
    if error:
        st.error(f"Error: {error}")
        return
    
    df = df.dropna(how='all')
    df.columns = [str(c).strip() for c in df.columns]
    
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    
    col_fecha = cols_lower.get('fecha')
    col_sesion = cols_lower.get('tipo_sorteo')
    col_fijo = cols_lower.get('fijo')
    col_corr1 = next((c for k, c in cols_lower.items() if 'primer' in k), None)
    col_corr2 = next((c for k, c in cols_lower.items() if 'segundo' in k), None)
    
    if not all([col_fecha, col_fijo]):
        st.error("Faltan columnas requeridas")
        return
    
    st.success(f"✅ {len(df)} registros")
    
    todos_pales = []
    for idx, fila in df.iterrows():
        pales = detectar_pales(fila, col_fijo, col_corr1, col_corr2)
        for p in pales:
            p['fecha'] = fila.get(col_fecha, '')
            p['sesion'] = fila.get(col_sesion, '')
        todos_pales.extend(pales)
    
    tab1, tab2 = st.tabs(["📊 Resumen", "📜 Historial"])
    
    with tab1:
        for grupo in GRUPOS.keys():
            pales_g = [p for p in todos_pales if p['grupo'] == grupo]
            st.markdown(f"### {grupo}")
            st.metric("Pales encontrados", len(pales_g))
            if pales_g:
                with st.expander(f"Ver últimos 10"):
                    for p in pales_g[-10:][::-1]:
                        st.write(f"• {p['fecha']} ({p['sesion']}): {', '.join(p['numeros'])}")
    
    with tab2:
        for p in todos_pales[-30:][::-1]:
            st.markdown(f"**{p['fecha']}** ({p['sesion']}) - {p['grupo']}: {', '.join(p['numeros'])}")

if __name__ == "__main__":
    main()