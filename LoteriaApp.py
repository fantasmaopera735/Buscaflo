# -*- coding: utf-8 -*-
"""
=============================================================================
🎰 LoteriaApp - Aplicación Principal de Análisis de Lotería
=============================================================================
Ejecutar con: streamlit run LoteriaApp.py
=============================================================================
"""

import streamlit as st
import os
import sys

# Configuración de página
st.set_page_config(
    page_title="🎰 LoteriaApp",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Información de módulos
MODULOS = {
    'PaleGeo': {
        'nombre': '🎯 PaleGeo',
        'descripcion': 'Pales por Grupos (Geotodo)',
        'detalle': '3 sesiones: Mañana, Tarde, Noche',
        'hoja': 'Geotodo'
    },
    'SumaDigitos': {
        'nombre': '🔢 SumaDigitos',
        'descripcion': 'Sumas del Fijo (Geotodo)',
        'detalle': '3 sesiones: Mañana, Tarde, Noche',
        'hoja': 'Geotodo'
    },
    'SumaFlo': {
        'nombre': '🔢 SumaFlo',
        'descripcion': 'Sumas del Fijo (Sorteos)',
        'detalle': '2 sesiones: Tarde, Noche',
        'hoja': 'Sorteos'
    },
    'PaleFlo': {
        'nombre': '🎯 PaleFlo',
        'descripcion': 'Pales por Grupos (Sorteos)',
        'detalle': '2 sesiones: Tarde, Noche',
        'hoja': 'Sorteos'
    }
}

def mostrar_menu():
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: white; font-size: 3rem;'>🎰 LoteriaApp 🎰</h1>
        <p style='color: #94a3b8; font-size: 1.2rem;'>Sistema de Análisis de Lotería Cubana</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.sidebar.header("📋 Seleccionar Módulo")
    modulo = st.sidebar.radio(
        "Elige una aplicación:",
        options=list(MODULOS.keys()),
        format_func=lambda x: MODULOS[x]['nombre'],
        label_visibility="collapsed"
    )
    
    info = MODULOS[modulo]
    st.sidebar.markdown("---")
    st.sidebar.info(f"**{info['descripcion']}**\n\n{info['detalle']}\n\n📊 Hoja: **{info['hoja']}**")
    
    if st.sidebar.button("🚀 Ejecutar Módulo", type="primary", use_container_width=True):
        st.session_state['modulo_activo'] = modulo
        st.rerun()
    
    st.markdown("### 📱 Módulos Disponibles")
    cols = st.columns(2)
    
    for i, (key, info) in enumerate(MODULOS.items()):
        with cols[i % 2]:
            color = '#3b82f6' if info['hoja'] == 'Geotodo' else '#22c55e'
            st.markdown(f"""
            <div style='background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 10px; border-left: 4px solid {color}; margin-bottom: 0.5rem;'>
                <h4 style='color: {color}; margin: 0;'>{info['nombre']}</h4>
                <p style='color: white; margin: 0.25rem 0;'>{info['descripcion']}</p>
                <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>{info['detalle']} | Hoja: {info['hoja']}</p>
            </div>
            """, unsafe_allow_html=True)

def ejecutar_modulo(modulo):
    if st.sidebar.button("⬅️ Volver al Menú", type="secondary", use_container_width=True):
        if 'modulo_activo' in st.session_state:
            del st.session_state['modulo_activo']
        st.rerun()
    
    try:
        if modulo == 'PaleGeo':
            import PaleGeo
            PaleGeo.main()
        elif modulo == 'SumaDigitos':
            import SumaDigitos
            SumaDigitos.main()
        elif modulo == 'SumaFlo':
            import SumaFlo
            SumaFlo.main()
        elif modulo == 'PaleFlo':
            import PaleFlo
            PaleFlo.main()
    except ImportError as e:
        st.error(f"❌ Error importando módulo: {e}")
        st.info("Ejecuta directamente: `streamlit run {}.py`".format(modulo))
    except Exception as e:
        st.error(f"❌ Error: {e}")

def main():
    if 'modulo_activo' in st.session_state and st.session_state['modulo_activo']:
        ejecutar_modulo(st.session_state['modulo_activo'])
    else:
        mostrar_menu()

if __name__ == "__main__":
    main()