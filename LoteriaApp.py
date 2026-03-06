# -*- coding: utf-8 -*-
"""
LoteriaApp - Aplicación Principal de Análisis de Lotería
========================================================
Menú principal que permite acceder a los diferentes módulos de análisis:
- PaleGeo: Análisis de Pales para Geotodo (Mañana, Tarde, Noche)
- PaleFlo: Análisis de Pales para Sorteos (Tarde, Noche)
- SumaDigitos: Análisis de Sumas para Geotodo (Mañana, Tarde, Noche)
- SumaFlo: Análisis de Sumas para Sorteos (Tarde, Noche)
"""

import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="LoteriaApp - Análisis de Lotería",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎰 LoteriaApp - Sistema de Análisis de Lotería")
st.markdown("""
### Bienvenido al Sistema de Análisis de Lotería

Esta aplicación permite analizar diferentes aspectos de los sorteos de lotería:

| Módulo | Hoja | Sesiones | Descripción |
|--------|------|----------|-------------|
| **PaleGeo** | Geotodo | M, T, N | Análisis de Pales (Mañana, Tarde, Noche) |
| **PaleFlo** | Sorteos | T, N | Análisis de Pales (Tarde, Noche) |
| **SumaDigitos** | Geotodo | M, T, N | Análisis de Sumas de dígitos |
| **SumaFlo** | Sorteos | T, N | Análisis de Sumas de dígitos |

---
""")

# Menú de selección
st.sidebar.title("📋 Menú Principal")
st.sidebar.markdown("Selecciona el módulo que deseas utilizar:")

modulo = st.sidebar.radio(
    "Seleccionar Módulo:",
    ["🏠 Inicio", "🎯 PaleGeo", "🎯 PaleFlo", "🔢 SumaDigitos", "🔢 SumaFlo"],
    label_visibility="collapsed"
)

# Ejecutar el módulo seleccionado
if modulo == "🏠 Inicio":
    st.markdown("""
    ## 📖 Instrucciones de Uso
    
    ### Pales (PaleGeo y PaleFlo)
    Los Pales se analizan según grupos:
    - **CERRADOS:** Dígitos 0, 6, 8, 9
    - **ABIERTOS:** Dígitos 2, 3, 5
    - **RECTOS:** Dígitos 1, 4, 7
    
    ### Sumas (SumaDigitos y SumaFlo)
    La **suma de dígitos** se calcula sumando los dos dígitos de cada número:
    - Ejemplo: 69 → 6+9 = **15**
    - Ejemplo: 00 → 0+0 = **0**
    - Ejemplo: 99 → 9+9 = **18**
    
    **IMPORTANTE:** Todas las sumas están en el rango **0-18**
    
    **Tipos de Análisis:**
    - **Suma Fijo:** Suma de dígitos del número Fijo únicamente
    - **Suma Corridos:** Qué sumas (0-18) aparecen entre los 2 corridos
    - **Suma Total/Día:** Qué sumas (0-18) aparecen entre los 3 números
    
    ---
    
    ### 📊 Características de cada módulo:
    
    #### 📊 Tabla de Estadísticas
    - Frecuencia de cada suma (0-18)
    - Ausencia máxima (días máximos sin aparecer)
    - Promedio de días entre apariciones
    - Días sin aparecer (desde la última vez)
    - Última fecha de aparición
    
    #### 📅 Almanaque
    - Análisis por períodos configurables
    - Sumas calientes, tibias y frías
    - Sumas persistentes (aparecen en todos los períodos)
    - Historial del período actual
    - Sumas calientes faltantes
    
    #### 🔍 Buscar Suma
    - Búsqueda de una suma específica (0-18)
    - Muestra los números de 2 dígitos que componen esa suma
    - Historial completo de apariciones
    - Indica en qué posición apareció (Fijo, Corr1, Corr2)
    """)
    
    st.info("👈 Selecciona un módulo en el menú lateral para comenzar.")

elif modulo == "🎯 PaleGeo":
    st.markdown("## 🎯 PaleGeo - Análisis de Pales (Geotodo)")
    st.markdown("**Hoja:** Geotodo | **Sesiones:** Mañana, Tarde, Noche")
    st.markdown("---")
    try:
        import PaleGeo
        if hasattr(PaleGeo, 'main'):
            PaleGeo.main()
        else:
            st.error("El módulo PaleGeo no tiene función main()")
    except ImportError as e:
        st.error(f"Error al cargar PaleGeo: {e}")
        st.info("Asegúrate de que el archivo PaleGeo.py esté en el mismo directorio.")
    except Exception as e:
        st.error(f"Error ejecutando PaleGeo: {e}")

elif modulo == "🎯 PaleFlo":
    st.markdown("## 🎯 PaleFlo - Análisis de Pales (Sorteos)")
    st.markdown("**Hoja:** Sorteos | **Sesiones:** Tarde, Noche")
    st.markdown("---")
    try:
        import PaleFlo
        if hasattr(PaleFlo, 'main'):
            PaleFlo.main()
        else:
            st.error("El módulo PaleFlo no tiene función main()")
    except ImportError as e:
        st.error(f"Error al cargar PaleFlo: {e}")
        st.info("Asegúrate de que el archivo PaleFlo.py esté en el mismo directorio.")
    except Exception as e:
        st.error(f"Error ejecutando PaleFlo: {e}")

elif modulo == "🔢 SumaDigitos":
    st.markdown("## 🔢 SumaDigitos - Análisis de Sumas (Geotodo)")
    st.markdown("**Hoja:** Geotodo | **Sesiones:** Mañana, Tarde, Noche")
    st.markdown("---")
    try:
        import SumaDigitos
        if hasattr(SumaDigitos, 'main'):
            SumaDigitos.main()
        else:
            st.error("El módulo SumaDigitos no tiene función main()")
    except ImportError as e:
        st.error(f"Error al cargar SumaDigitos: {e}")
        st.info("Asegúrate de que el archivo SumaDigitos.py esté en el mismo directorio.")
    except Exception as e:
        st.error(f"Error ejecutando SumaDigitos: {e}")

elif modulo == "🔢 SumaFlo":
    st.markdown("## 🔢 SumaFlo - Análisis de Sumas (Sorteos)")
    st.markdown("**Hoja:** Sorteos | **Sesiones:** Tarde, Noche")
    st.markdown("---")
    try:
        import SumaFlo
        if hasattr(SumaFlo, 'main'):
            SumaFlo.main()
        else:
            st.error("El módulo SumaFlo no tiene función main()")
    except ImportError as e:
        st.error(f"Error al cargar SumaFlo: {e}")
        st.info("Asegúrate de que el archivo SumaFlo.py esté en el mismo directorio.")
    except Exception as e:
        st.error(f"Error ejecutando SumaFlo: {e}")

# Información adicional en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Información")
st.sidebar.markdown("""
**Google Sheets ID:**  
`1ID79C3pz3w5L2oA6krl9LjYEZstPgCGLoqw3FQ1qXDw`

**Hojas disponibles:**
- Geotodo (M, T, N)
- Sorteos (T, N)

**Rango de Sumas:**
- Todas son 0-18
""")
st.sidebar.markdown("---")
st.sidebar.markdown("*LoteriaApp v4.0*")