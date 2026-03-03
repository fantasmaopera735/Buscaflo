# -*- coding: utf-8 -*-
"""
🎰 LoteriaApp - Aplicación Principal
Ejecutar: streamlit run LoteriaApp.py
"""

import streamlit as st
import sys

st.set_page_config(
    page_title="🎰 LoteriaApp",
    page_icon="🎰",
    layout="wide"
)

MODULOS = {
    'PaleGeo': {'nombre': '🎯 PaleGeo', 'desc': 'Pales (Geotodo) - 3 sesiones'},
    'SumaDigitos': {'nombre': '🔢 SumaDigitos', 'desc': 'Sumas (Geotodo) - 3 sesiones'},
    'SumaFlo': {'nombre': '🔢 SumaFlo', 'desc': 'Sumas (Sorteos) - 2 sesiones'},
    'PaleFlo': {'nombre': '🎯 PaleFlo', 'desc': 'Pales (Sorteos) - 2 sesiones'}
}

def main():
    st.title("🎰 LoteriaApp")
    st.markdown("---")
    
    st.sidebar.header("📋 Módulo")
    modulo = st.sidebar.radio("Selecciona:", list(MODULOS.keys()), format_func=lambda x: MODULOS[x]['nombre'])
    
    if st.sidebar.button("🚀 Ejecutar", type="primary"):
        st.session_state['modulo'] = modulo
        st.rerun()
    
    if 'modulo' in st.session_state:
        if st.sidebar.button("⬅️ Menú"):
            del st.session_state['modulo']
            st.rerun()
        
        mod = st.session_state['modulo']
        try:
            if mod == 'PaleGeo':
                import PaleGeo
                PaleGeo.main()
            elif mod == 'SumaDigitos':
                import SumaDigitos
                SumaDigitos.main()
            elif mod == 'SumaFlo':
                import SumaFlo
                SumaFlo.main()
            elif mod == 'PaleFlo':
                import PaleFlo
                PaleFlo.main()
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        for k, v in MODULOS.items():
            st.info(f"**{v['nombre']}** - {v['desc']}")

if __name__ == "__main__":
    main()