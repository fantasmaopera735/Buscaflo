# PESTAÑA 0: FALTANTES DEL MES
with tabs[0]:
    st.subheader("🗓️ Análisis de Faltantes del Mes")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    meses_nombres = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                     7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    
    with col_f1:
        mes_sel = st.selectbox("Mes a Analizar:", list(meses_nombres.values()), index=datetime.now().month - 1)
        mes_num = [k for k, v in meses_nombres.items() if v == mes_sel][0]
    
    with col_f2:
        anio_sel = st.number_input("Año:", min_value=2020, max_value=datetime.now().year, value=datetime.now().year)
    
    with col_f3:
        cant_sorteos = st.slider("Sorteos para Frecuencia:", 100, 5000, 1000, step=100)

    if st.button("🔍 Analizar Faltantes", type="primary"):
        with st.spinner("Calculando..."):
            df_faltantes_res, error_msg, df_salidos_mes = analizar_faltantes_mes(dfa, mes_num, anio_sel, cant_sorteos)
        
        if error_msg:
            st.info(error_msg)
        elif not df_faltantes_res.empty:
            total_faltantes = len(df_faltantes_res)
            alta = df_faltantes_res[df_faltantes_res['Prioridad'] == '🔴 ALTA']
            
            st.markdown(f"### ⏳ Faltan por salir: {total_faltantes} números")
            st.markdown(f"#### 🔴 Prioridad Alta (Apostar): {len(alta)} números")
            st.write(" ".join([f"`{n}`" for n in alta['Número'].tolist()]))

            st.markdown("---")
            st.markdown("#### 📊 Detalle de Faltantes")
            df_show = df_faltantes_res.rename(columns={'Veces Salidas': f'Frec. ({cant_sorteos} sort.)'})
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 📝 Historial de Aciertos Recientes (Verificación)")
            st.info("Ordenado del más reciente al más antiguo. Noche (N) tiene prioridad sobre Tarde (T) en el mismo día.")
            
            df_historial = df_fijos.tail(20).copy()
            orden_tipo = {'N': 0, 'T': 1, 'OTRO': 2}
            df_historial['orden_tipo'] = df_historial['Tipo_Sorteo'].map(orden_tipo)
            
            df_historial = df_historial.sort_values(by=['Fecha', 'orden_tipo'], ascending=[False, True])
            df_historial['¿Salió en el Mes?'] = df_historial['Numero'].apply(lambda x: "✅ SÍ" if x in df_salidos_mes['Numero'].values else "")
            df_historial['Fecha Str'] = df_historial['Fecha'].dt.strftime('%d/%m/%Y')
            
            st.dataframe(
                df_historial[['Fecha Str', 'Tipo_Sorteo', 'Numero', '¿Salió en el Mes?']].head(10),
                column_config={
                    "Fecha Str": "Fecha",
                    "Tipo_Sorteo": "Sorteo",
                    "Numero": st.column_config.NumberColumn("Número", format="%02d"),
                    "¿Salió en el Mes?": "Estado"
                },
                hide_index=True
            )

# PESTAÑA 1: TRANSFERENCIA
with tabs[1]:
    st.subheader("🔄 Transferencia Decena → Unidad")
    st.markdown("**Analiza cuando la decena de un sorteo pasa como unidad al siguiente**")
    st.info("T→N: Decena Tarde → Unidad Noche | N→T: Decena Noche → Unidad Tarde (día siguiente)")
    
    st.markdown("### Lógica de Ciclos")
    st.markdown("""
    - **1ra vez**: El evento ocurre → Se marca el ciclo (NO se apuesta)
    - **2da vez**: Puede repetir → **ALERTA: apostar**
    - **3ra vez**: Puede repetir → **ALERTA: apostar**
    - **Si se aleja 3x del promedio**: Reiniciar ciclo, esperar 1ra vez
    - **ACELERADO**: Secuencia actual más rápida que el promedio → usar secuencia reciente
    """)
    
    dias_stats = st.slider("Días de historial:", 30, 365, 180, key="trans_stats")
    
    if st.button("Analizar Transferencias", type="primary", key="btn_trans"):
        with st.spinner("Analizando..."):
            df_stats = analizar_transferencia_flotodo(df_completo, dias_stats)
        
        for _, row in df_stats.iterrows():
            st.markdown(f"### 📊 **{row['Transferencia']}**")
            
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.metric("📅 Promedio Histórico", f"{row['Promedio_Historico']} días")
            with col_pred2:
                st.metric("⚡ Tipo Secuencia", row['Tipo_Secuencia'])
            
            if row['Alerta']:
                st.success(f"✅ **{row['Transferencia']}** - ALERTA: Puede repetir")
                st.markdown(f"📅 Último evento: {row['Ultima_Fecha']} (dígito {row['Ultimo_Digito']})")
                st.markdown(f"📊 Sin evento hace: {row['Dias_Sin_Evento']} días | Predicción: cada {row['Prediccion_Dias']} días")
                
                if row['Transferencia'] == 'T->N':
                    ultimo_T = df_fijos[df_fijos['Tipo_Sorteo'] == 'T'].iloc[-1] if len(df_fijos[df_fijos['Tipo_Sorteo'] == 'T']) > 0 else None
                    if ultimo_T is not None:
                        decena_actual = int(ultimo_T['Numero']) // 10
                        nums_sugeridos = [f"{d*10 + decena_actual:02d}" for d in range(10)]
                        st.markdown(f"🎯 **Fijo Tarde**: {int(ultimo_T['Numero']):02d} → Decena: **{decena_actual}**")
                        st.markdown(f"💰 **Jugar en NOCHE números terminados en {decena_actual}:** {', '.join(nums_sugeridos)}")
                
                elif row['Transferencia'] == 'N->T':
                    ultimo_N = df_fijos[df_fijos['Tipo_Sorteo'] == 'N'].iloc[-1] if len(df_fijos[df_fijos['Tipo_Sorteo'] == 'N']) > 0 else None
                    if ultimo_N is not None:
                        decena_actual = int(ultimo_N['Numero']) // 10
                        nums_sugeridos = [f"{d*10 + decena_actual:02d}" for d in range(10)]
                        st.markdown(f"🎯 **Fijo Noche**: {int(ultimo_N['Numero']):02d} → Decena: **{decena_actual}**")
                        st.markdown(f"💰 **Jugar en TARDE (día siguiente) números terminados en {decena_actual}:** {', '.join(nums_sugeridos)}")
            else:
                st.info(f"⏳ **{row['Transferencia']}** - {row['Estado_Ciclo']}")
                st.markdown(f"📅 Último evento: {row['Ultima_Fecha']} | Días sin evento: {row['Dias_Sin_Evento']}")
                if row['Dias_Estimados'] > 0:
                    st.markdown(f"⏰ **Faltan aproximadamente {row['Dias_Estimados']} días**")
            
            st.markdown("---")
        
        with st.expander("Ver tabla completa"):
            st.dataframe(df_stats, hide_index=True)

# PESTAÑA 2: DÍGITO FALTANTE
with tabs[2]:
    st.subheader("🔢 Análisis de Dígito Faltante")
    
    tab1, tab2, tab3 = st.tabs(["📅 Por Fecha", "📊 Estadísticas", "🧪 Backtest"])
    
    with tab1:
        st.markdown("### Selecciona una fecha para analizar")
        fecha_sel = st.date_input("Fecha:", datetime.now().date(), key="dig_falt_fecha")
        fecha_dt = datetime.combine(fecha_sel, datetime.min.time())
        
        if st.button("Analizar Fecha", key="btn_dig_falt"):
            resultado, error = analizar_dia_completo(df_completo, fecha_dt)
            
            if resultado:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("✅ Dígitos presentes")
                    st.write(resultado['digitos_presentes'])
                with col2:
                    st.subheader("❌ Dígitos faltantes")
                    if resultado['digitos_faltantes']:
                        st.warning(f"Dígitos que NO aparecieron: {resultado['digitos_faltantes']}")
                        st.markdown("**Estrategia**: Estos dígitos pueden aparecer en los próximos sorteos")
                    else:
                        st.success("¡Todos los dígitos aparecieron!")
                
                st.subheader("📋 Detalle por sesión")
                for det in resultado['detalle']:
                    st.markdown(f"**{det['Sesion']}** - Centena: {det['Centena']}, Fijo: {det['Fijo']}, Corridos: {det['1er_Corrido']}, {det['2do_Corrido']}")
                    st.markdown(f"Dígitos extraídos: {det['Digitos']}")
            else:
                st.error(error)
    
    with tab2:
        st.subheader("📊 Estadísticas por dígito")
        dias_stats_dig = st.slider("Días de análisis:", 30, 365, 180, key="dig_stats_dias")
        tipo_stats = st.selectbox("Ver estadísticas de:", ['general', 'centena', 'fijo', 'corrido1', 'corrido2'], key="sel_tipo_stats")
        
        df_stats_input = df_completo.copy()
        if modo == "Tarde":
            df_stats_input = df_completo[df_completo['Tipo_Sorteo'] == 'T'].copy()
        elif modo == "Noche":
            df_stats_input = df_completo[df_completo['Tipo_Sorteo'] == 'N'].copy()
            
        stats = estadisticas_digitos_separadas(df_stats_input, dias_stats_dig)
        
        st.markdown(f"### Estadísticas: {tipo_stats.upper()} (Modo: {modo})")
        st.dataframe(stats[tipo_stats], use_container_width=True, hide_index=True)
        
        st.markdown("### 📊 Resumen")
        df_stats_view = stats['general'].copy()
        df_stats_view['Ausencia_Sort'] = pd.to_numeric(df_stats_view['Ausencia_Maxima'], errors='coerce').fillna(999)
        df_stats_view = df_stats_view.sort_values('Ausencia_Sort', ascending=False)
        df_stats_view = df_stats_view.drop(columns=['Ausencia_Sort'])
        st.markdown("**Dígitos con mayor ausencia máxima:**")
        st.dataframe(df_stats_view.head(5), use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🧪 Backtest del Dígito Faltante")
        dias_backtest = st.slider("Días para backtest:", 30, 180, 90, key="dig_backtest_dias")
        
        if st.button("Ejecutar Backtest", key="btn_dig_backtest"):
            with st.spinner("Analizando..."):
                resultado_bt = backtest_digito_faltante(df_completo, dias_backtest)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total evaluados", resultado_bt['total_evaluados'])
            with col2:
                st.metric("✅ Aciertos", resultado_bt['aciertos'])
            with col3:
                st.metric("🎯 Efectividad", f"{resultado_bt['efectividad']}%")
            
            if resultado_bt['resultados']:
                st.markdown("### 📋 Detalle")
                df_resultados = pd.DataFrame(resultado_bt['resultados'])
                st.dataframe(df_resultados, use_container_width=True, hide_index=True)

# PESTAÑA 3: PATRONES
with tabs[3]:
    st.subheader(f"🔍 Patrones")
    c1, c2 = st.columns(2)
    with c1: n = st.number_input("Disparador:", 0, 99, 40, format="%02d", key="patron_num")
    with c2: v = st.slider("Ventana:", 1, 30, 15, key="patron_ventana")
    
    if st.button("🔍 Analizar", key="b1"):
        st.session_state['sb1'] = True
    
    if st.session_state.get('sb1'):
        r, tot = analizar_siguientes(dfa, n, v)
        if r is None:
            st.warning(f"⚠️ El número {n:02d} no ha salido.")
        else:
            st.success(f"📊 Encontrado {tot} veces.")
            max_val = int(r['Frecuencia'].max()) if not r.empty else 1
            st.dataframe(r.head(20), column_config={
                "Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=max_val)
            }, hide_index=True)

# PESTAÑA 4: ALMANAQUE
with tabs[4]:
    st.subheader("📅 Almanaque")
    
    with st.form("almanaque_form"):
        c_r, c_m = st.columns(2)
        with c_r:
            ca, cb = st.columns(2)
            with ca: dia_inicio = st.number_input("Día Ini:", 1, 31, 16)
            with cb: dia_fin = st.number_input("Día Fin:", 1, 31, 20)
        with c_m: meses_atras = st.slider("Meses Atrás:", 1, 12, 4)
        
        submitted = st.form_submit_button("📊 Analizar", type="primary")
        
        if submitted:
            if dia_inicio > dia_fin:
                st.error("❌ El día de inicio no puede ser mayor al final.")
            else:
                with st.spinner("Analizando..."):
                    res = analizar_almanaque(dfa, int(dia_inicio), int(dia_fin), int(meses_atras), strict_mode=False)
                
                if not res['success']:
                    st.error(f"❌ {res.get('mensaje', 'Error')}")
                else:
                    if res['nombres_bloques']:
                        st.success(f"📅 Periodos: {', '.join(res['nombres_bloques'])}")
                    
                    st.markdown("---")
                    st.subheader("⏱️ Evaluación en Tiempo Real")
                    st.info(f"**Estado:** {res['estado_periodo']}")
                    
                    col_h, col_f = st.columns([2, 1])
                    with col_h:
                        if not res['df_historial_actual'].empty:
                            hist_view = res['df_historial_actual'].copy()
                            hist_view['Fecha'] = hist_view['Fecha'].dt.strftime('%d/%m/%Y')
                            st.markdown("### 📜 Resultados del Mes")
                            st.dataframe(hist_view, use_container_width=True, hide_index=True)
                        else:
                            st.info("No hay resultados aún.")
                    
                    with col_f:
                        st.markdown("### ⏳ Faltantes")
                        if not res['df_faltantes'].empty:
                            st.dataframe(res['df_faltantes'], use_container_width=True, hide_index=True)
                        else:
                            st.success("🎉 ¡Todos salieron!")
                    
                    st.markdown("---")
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown("### 🔢 Decenas")
                        st.dataframe(res['df_dec'], hide_index=True)
                    with col_d2:
                        st.markdown("### 🔢 Unidades")
                        st.dataframe(res['df_uni'], hide_index=True)
                    
                    st.markdown("---")
                    col_t1, col_t2 = st.columns([1, 2])
                    with col_t1:
                        st.markdown("### 🔥 Tendencia")
                        if not res['tend'].empty:
                            mv = int(res['tend']['Frecuencia'].max())
                            st.dataframe(res['tend'], column_config={
                                "Frecuencia": st.column_config.ProgressColumn("Frecuencia", format="%d", min_value=0, max_value=mv)
                            }, hide_index=True)
                            st.info(f"Dominante: **{res['top_p']}**")
                    
                    with col_t2:
                        st.markdown("### 💡 Sugerencias")
                        if not res['df_tend_nums'].empty:
                            st.dataframe(res['df_tend_nums'], hide_index=True)
                    
                    with st.expander("🛡️ Persistencia"):
                        p1, p2 = st.columns(2)
                        with p1:
                            st.markdown("#### 📌 Números")
                            if not res['df_pers_num'].empty:
                                st.dataframe(res['df_pers_num'], hide_index=True)
                            else:
                                st.info("Ninguno.")
                        with p2:
                            st.markdown("#### 🏷️ Perfiles")
                            if res['persistentes_perfiles']:
                                st.dataframe(pd.DataFrame(list(res['persistentes_perfiles']), columns=["Perfil"]), hide_index=True)
                            else:
                                st.info("Ninguno.")
                    
                    with st.expander("📋 Ranking"):
                        if not res['df_rank'].empty:
                            st.dataframe(res['df_rank'].head(20), hide_index=True)

# PESTAÑA 5: PROPUESTA
with tabs[5]:
    st.subheader(f"🧠 Sincronización")
    c1, c2 = st.columns(2)
    with c1: dt = st.number_input("Días Tendencia:", 5, 60, 15, key="prop_dias")
    with c2: dg = st.number_input("Gap Mínimo:", 1, 90, 10, key="prop_gap")
    
    if st.button("🧠 Generar", key="b_pr"):
        st.session_state['spr'] = True
    
    if st.session_state.get('spr'):
        p = generar_sugerencia(dfa, dt, dg)
        if p.empty:
            st.warning("No hay sugerencias.")
        else:
            st.dataframe(p, hide_index=True)

# PESTAÑA 6: SECUENCIA
with tabs[6]:
    st.subheader(f"🔗 Secuencia")
    c1, c2, c3 = st.columns(3)
    with c1: parte = st.selectbox("Parte del número", ["Decena", "Unidad"])
    with c2: tipo = st.selectbox("Tipo de patrón", ["digito", "paridad", "altura"])
    with c3: secuencia = st.text_input("Secuencia:", "0 1 2")
    
    if st.button("Buscar"):
        r, e = buscar_seq(dfa, parte, tipo, secuencia)
        if r is not None:
            st.success(f"Se encontraron {len(r)} coincidencias")
            st.dataframe(r, use_container_width=True, hide_index=True)
        else:
            st.error(e)
    
    st.markdown("""
    **Ayuda:**
    - **Dígito**: Usa valores 0-9
    - **Paridad**: P (par) o I (impar)
    - **Altura**: A (alto: 5-9) o B (bajo: 0-4)
    """)

# PESTAÑA 7: LABORATORIO
with tabs[7]:
    st.subheader("🧪 Simulador")
    
    meses_lab = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 
                 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
    fecha_hoy_lab = datetime.now()
    mes_default_lab = fecha_hoy_lab.month - 1 if fecha_hoy_lab.month > 1 else 12
    anio_default_lab = fecha_hoy_lab.year if fecha_hoy_lab.month > 1 else fecha_hoy_lab.year - 1
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        nombre_mes_sel = st.selectbox("Mes:", list(meses_lab.values()), index=list(meses_lab.keys()).index(mes_default_lab), key="lab_mes")
        mes_sel_num = [k for k, v in meses_lab.items() if v == nombre_mes_sel][0]
    with col_l2:
        anio_sel = st.number_input("Año:", min_value=2020, max_value=2030, value=anio_default_lab, key="lab_anio")
    with col_l3:
        c_dia1, c_dia2 = st.columns(2)
        with c_dia1: dia_ini_lab = st.number_input("Día Ini:", 1, 31, 1, key="lab_dia_ini")
        with c_dia2: dia_fin_lab = st.number_input("Día Fin:", 1, 31, 15, key="lab_dia_fin")
    
    meses_atras_sim = st.slider("Meses atrás:", 2, 6, 3, key="lab_meses_atras")

    if st.button("🚀 Ejecutar", type="primary"):
        with st.spinner("Analizando..."):
            res = backtesting_estrategia(dfa, mes_sel_num, anio_sel, dia_ini_lab, dia_fin_lab, meses_atras_sim)
            
            if res is not None:
                st.success(f"✅ Efectividad: {res['Efectividad']}%")
                
                col_izq, col_der = st.columns(2)
                
                with col_izq:
                    st.markdown("### 📋 Estrategia")
                    st.caption(f"Basada en: {res['Entrenamiento']}")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔥 Calientes D", str(res['hot_d']))
                    c2.metric("🟡 Tibios D", str(res['warm_d']))
                    c3.metric("🧊 Fríos D", str(res['cold_d']))
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("🔥 Calientes U", str(res['hot_u']))
                    c5.metric("🟡 Tibios U", str(res['warm_u']))
                    c6.metric("🧊 Fríos U", str(res['cold_u']))
                    
                    st.markdown("**Perfiles Persistentes:**")
                    for p in res['Perfiles']:
                        st.markdown(f"- 🏷️ {p}")

                with col_der:
                    st.markdown("### 🎲 Resultados")
                    st.caption(f"Periodo: {res['Prueba']}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total", res['Total'])
                    m2.metric("Aciertos", res['Aciertos'])
                    m3.metric("Sufrientes", res['Sufrientes'])
                    
                    st.markdown("#### Detalle:")
                    df_view = res['Detalle'].copy()
                    df_view['Fecha'] = df_view['Fecha'].dt.strftime('%d/%m/%Y')
                    st.dataframe(df_view, use_container_width=True, hide_index=True)
            else:
                st.error(f"🛑 Error en el análisis")

# PESTAÑA 8: ESTABILIDAD
with tabs[8]:
    st.subheader("📉 Estabilidad")
    
    dias_analisis = st.slider("Días de Historial:", 90, 3650, 365, step=30, key="est_dias")
    
    if st.button("📊 Calcular", key="b_est"):
        with st.spinner("Analizando..."):
            df_est = analizar_estabilidad_numeros(dfa, dias_analisis)
            
            if df_est is None:
                st.error("Sin datos suficientes.")
            else:
                st.markdown("### 🏆 Ranking")
                
                st.dataframe(
                    df_est.head(30),
                    column_config={
                        "Estado": st.column_config.TextColumn("Estado"),
                        "Gap Actual": st.column_config.NumberColumn("Días sin salir", format="%d"),
                        "Gap Máximo (Días)": st.column_config.NumberColumn("Max", format="%d"),
                        "Gap Promedio": st.column_config.NumberColumn("Prom", format="%.1f"),
                        "Desviación (Irregularidad)": st.column_config.NumberColumn("Irreg", format="%.1f"),
                        "Última Salida": st.column_config.TextColumn("Último")
                    },
                    hide_index=True
                )
                
                st.info("💡 **Estados:** 🔥 EN RACHA | ✅ NORMAL | ⏳ VENCIDO | 🔴 MUY VENCIDO")

st.markdown("---")
st.caption("Flotodo Suite Ultimate v2.1 | 🦩 Análisis de lotería")