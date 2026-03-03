# -*- coding: utf-8 -*-
"""
=============================================================================
🎰 LoteriaApp - Aplicación Principal de Análisis de Lotería
=============================================================================
Esta es la aplicación principal que integra todos los módulos de análisis:
- PaleGeo: Análisis de Pales (hoja Geotodo - 3 sesiones)
- SumaDigitos: Análisis de Sumas (hoja Geotodo - 3 sesiones)
- SumaFlo: Análisis de Sumas (hoja Sorteos - 2 sesiones)
- PaleFlo: Análisis de Pales (hoja Sorteos - 2 sesiones)

Ejecutar con: streamlit run LoteriaApp.py
=============================================================================
"""

import streamlit as st
import os
import sys

# =============================================================================
# CONFIGURACIÓN DE PÁGINA (debe ser el primer comando de Streamlit)
# =============================================================================
st.set_page_config(
    page_title="🎰 LoteriaApp - Análisis de Lotería",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# INFORMACIÓN DE LA APP
# =============================================================================
APP_VERSION = "2.0"
GOOGLE_SHEETS_ID = '1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw'

# Descripción de cada módulo
MODULOS = {
    'PaleGeo': {
        'nombre': '🎯 PaleGeo',
        'descripcion': 'Análisis de Pales por Grupos (Hoja Geotodo)',
        'detalle': 'Analiza pales en 3 sesiones: Mañana, Tarde y Noche',
        'hoja': 'Geotodo',
        'archivo': 'PaleGeo.py'
    },
    'SumaDigitos': {
        'nombre': '🔢 SumaDigitos',
        'descripcion': 'Análisis de Sumas del Fijo (Hoja Geotodo)',
        'detalle': 'Suma de dígitos en 3 sesiones: Mañana, Tarde y Noche',
        'hoja': 'Geotodo',
        'archivo': 'SumaDigitos.py'
    },
    'SumaFlo': {
        'nombre': '🔢 SumaFlo',
        'descripcion': 'Análisis de Sumas del Fijo (Hoja Sorteos)',
        'detalle': 'Suma de dígitos en 2 sesiones: Tarde y Noche',
        'hoja': 'Sorteos',
        'archivo': 'SumaFlo.py'
    },
    'PaleFlo': {
        'nombre': '🎯 PaleFlo',
        'descripcion': 'Análisis de Pales por Grupos (Hoja Sorteos)',
        'detalle': 'Analiza pales en 2 sesiones: Tarde y Noche',
        'hoja': 'Sorteos',
        'archivo': 'PaleFlo.py'
    }
}

# =============================================================================
# FUNCIONES DE NAVEGACIÓN
# =============================================================================
def mostrar_menu_principal():
    """Muestra el menú principal de selección de módulos"""
    
    # Header principal
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='color: white; margin-bottom: 0.5rem; font-size: 3rem;'>
            🎰 LoteriaApp 🎰
        </h1>
        <p style='color: #94a3b8; font-size: 1.2rem;'>
            Sistema de Análisis de Lotería Cubana
        </p>
        <p style='color: #64748b; font-size: 0.9rem;'>
            Versión {version} | Google Sheets ID: {gs_id}
        </p>
    </div>
    """.format(version=APP_VERSION, gs_id=GOOGLE_SHEETS_ID), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Información de conexión
    st.sidebar.header("📡 Conexión")
    st.sidebar.info(f"""
    **Google Sheets ID:**  
    `{GOOGLE_SHEETS_ID}`
    
    **Hojas disponibles:**
    - Geotodo (3 sesiones)
    - Sorteos (2 sesiones)
    """)
    
    st.sidebar.markdown("---")
    
    # Selector de módulo
    st.sidebar.header("📋 Seleccionar Módulo")
    
    modulo_seleccionado = st.sidebar.radio(
        "Elige una aplicación:",
        options=list(MODULOS.keys()),
        format_func=lambda x: MODULOS[x]['nombre'],
        label_visibility="collapsed"
    )
    
    # Mostrar información del módulo seleccionado
    info = MODULOS[modulo_seleccionado]
    st.sidebar.markdown("---")
    st.sidebar.header("📖 Información")
    st.sidebar.markdown(f"""
    **{info['nombre']}**  
    
    {info['descripcion']}
    
    📄 {info['detalle']}
    
    📊 Hoja: **{info['hoja']}**
    """)
    
    # Botón para ejecutar
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Ejecutar Módulo", type="primary", use_container_width=True):
        st.session_state['modulo_activo'] = modulo_seleccionado
        st.rerun()
    
    # Mostrar tarjetas de módulos en el área principal
    st.markdown("### 📱 Módulos Disponibles")
    st.markdown("Selecciona un módulo en el panel lateral y haz clic en **Ejecutar Módulo**")
    
    cols = st.columns(2)
    
    for i, (key, info) in enumerate(MODULOS.items()):
        with cols[i % 2]:
            # Determinar color según hoja
            color = '#3b82f6' if info['hoja'] == 'Geotodo' else '#22c55e'
            icono_hoja = '🌍' if info['hoja'] == 'Geotodo' else '🎯'
            
            st.markdown(f"""
            <div style='background: rgba(0,0,0,0.3); padding: 1.5rem; 
                        border-radius: 15px; border-left: 4px solid {color}; margin-bottom: 1rem;'>
                <h3 style='color: {color}; margin-top: 0;'>
                    {info['nombre']}
                </h3>
                <p style='color: white; font-size: 1.1rem; margin: 0.5rem 0;'>
                    {info['descripcion']}
                </p>
                <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0;'>
                    {info['detalle']}
                </p>
                <p style='color: #64748b; font-size: 0.85rem; margin: 0.5rem 0;'>
                    {icono_hoja} Hoja: {info['hoja']} | 📄 {info['archivo']}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sección de ayuda
    st.markdown("---")
    st.markdown("### ❓ Cómo Usar")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.2); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='font-size: 2rem;'>1️⃣</div>
            <p style='color: white; margin: 0.5rem 0;'>Selecciona un módulo</p>
            <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>
                Usa el panel lateral izquierdo
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(34, 197, 94, 0.2); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='font-size: 2rem;'>2️⃣</div>
            <p style='color: white; margin: 0.5rem 0;'>Haz clic en Ejecutar</p>
            <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>
                Botón "🚀 Ejecutar Módulo"
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: rgba(249, 115, 22, 0.2); padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='font-size: 2rem;'>3️⃣</div>
            <p style='color: white; margin: 0.5rem 0;'>Analiza los datos</p>
            <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>
                Navega por las pestañas
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
        <p>🎰 LoteriaApp v{version} | Sistema de Análisis de Lotería</p>
        <p>Archivos: PaleGeo.py | SumaDigitos.py | SumaFlo.py | PaleFlo.py</p>
    </div>
    """.format(version=APP_VERSION), unsafe_allow_html=True)


def ejecutar_modulo(modulo_nombre):
    """Ejecuta el módulo seleccionado importando y llamando a su función main()"""
    
    # Mostrar botón de regreso en el sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Volver al Menú", type="secondary", use_container_width=True):
        if 'modulo_activo' in st.session_state:
            del st.session_state['modulo_activo']
        st.rerun()
    
    # Información del módulo actual
    info = MODULOS[modulo_nombre]
    st.sidebar.header(f"📖 {info['nombre']}")
    st.sidebar.info(f"""
    **Hoja:** {info['hoja']}
    
    {info['detalle']}
    """)
    
    try:
        # Importar el módulo dinámicamente
        # Primero intentar importar desde el mismo directorio
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        
        if directorio_actual not in sys.path:
            sys.path.insert(0, directorio_actual)
        
        # Importar el módulo
        if modulo_nombre == 'PaleGeo':
            import PaleGeo
            PaleGeo.main()
        elif modulo_nombre == 'SumaDigitos':
            import SumaDigitos
            SumaDigitos.main()
        elif modulo_nombre == 'SumaFlo':
            import SumaFlo
            SumaFlo.main()
        elif modulo_nombre == 'PaleFlo':
            import PaleFlo
            PaleFlo.main()
            
    except ImportError as e:
        st.error(f"❌ Error importando el módulo: {e}")
        st.info(f"""
        **Para que funcione correctamente:**
        
        1. Asegúrate de que los archivos estén en el mismo directorio:
           - PaleGeo.py
           - SumaDigitos.py
           - SumaFlo.py
           - PaleFlo.py
        
        2. Ejecuta cada archivo individualmente si hay problemas:
           ```
           streamlit run PaleGeo.py
           streamlit run SumaDigitos.py
           streamlit run SumaFlo.py
           streamlit run PaleFlo.py
           ```
        """)
        
        # Mostrar opción de ejecutar individualmente
        st.markdown("---")
        st.markdown("### 📁 Ejecutar Archivo Individual")
        st.code(f"streamlit run {info['archivo']}", language="bash")
        
    except Exception as e:
        st.error(f"❌ Error ejecutando el módulo: {e}")
        st.exception(e)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """Función principal que controla el flujo de la aplicación"""
    
    # Verificar si hay un módulo activo
    if 'modulo_activo' in st.session_state and st.session_state['modulo_activo']:
        # Ejecutar el módulo seleccionado
        ejecutar_modulo(st.session_state['modulo_activo'])
    else:
        # Mostrar el menú principal
        mostrar_menu_principal()


# =============================================================================
# EJECUTAR
# =============================================================================
if __name__ == "__main__":
    main()