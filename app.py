import streamlit as st
import pandas as pd
import os
import re
import random
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página y tema
st.set_page_config(
    page_title="El Dilema del Veto - ONU",
    page_icon="🌐",
    layout="wide"
)

# --- FORZAR MODO OSCURO CON CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        [data-testid="stSidebar"] {
            background-color: #161B22;
        }
    </style>
""", unsafe_allow_html=True)

# --- DICCIONARIO Y FUNCIÓN DE TRADUCCIÓN DIPLOMÁTICA ---
DICCIONARIO_TRADUCCION = {
    r"\bSecurity Council resolution\b": "Resolución del Consejo de Seguridad",
    r"\bSecurity Council\b": "Consejo de Seguridad",
    r"\bGeneral Assembly\b": "Asamblea General",
    r"\bDraft resolution\b": "Proyecto de resolución",
    r"\bon extension of the mandate of\b": "sobre la prórroga del mandato de",
    r"\bon extension of the mandate\b": "sobre la prórroga del mandato",
    r"\bthe UN Support Mission in\b": "la Misión de Apoyo de la ONU en",
    r"\bLibya\b": "Libia",
    r"\buntil\b": "hasta el",
    r"\bSituation in\b": "Situación en",
    r"\bSituation regarding\b": "Situación con respecto a",
    r"\bQuestion of\b": "Cuestión relativa a",
    r"\bMiddle East\b": "Oriente Medio",
    r"\bPeace and security\b": "Paz y seguridad",
    r"\bNon-proliferation\b": "No proliferación",
    r"\bHuman rights\b": "Derechos humanos",
    r"\bSanctions\b": "Sanciones",
    r"\bTerrorism\b": "Terrorismo",
    r"\bPeacekeeping\b": "Mantenimiento de la paz",
    r"\bCeasefire\b": "Alto el fuego",
    r"\bHumanitarian assistance\b": "Asistencia humanitaria",
    r"\bNuclear weapons\b": "Armas nucleares",
    r"\bArmed conflict\b": "Conflicto armado",
    r"\bProtection of civilians\b": "Protección de civiles",
    r"\bResolution\b": "Resolución",
    r"\bThe situation\b": "La situación",
}

def traducir_titulo(texto):
    if not isinstance(texto, str) or not texto.strip():
        return "Resolución del Consejo de Seguridad"
    texto_traducido = texto
    for eng_pattern, esp_sub in DICCIONARIO_TRADUCCION.items():
        texto_traducido = re.sub(eng_pattern, esp_sub, texto_traducido, flags=re.IGNORECASE)
    return texto_traducido

def clasificar_tipo_crisis(titulo_es):
    t = titulo_es.lower()
    if "prórroga" in t or "mandate" in t or "extension" in t:
        return "prorroga"
    elif "sanciones" in t or "sanctions" in t:
        return "sanciones"
    elif "humanitaria" in t or "assistance" in t or "civilians" in t or "salud" in t:
        return "humanitaria"
    elif "paz" in t or "peace" in t or "ceasefire" in t:
        return "paz"
    else:
        return "general"

def generar_descripcion_resolucion(titulo_es):
    tipo = clasificar_tipo_crisis(titulo_es)
    if tipo == "prorroga":
        return "Esta resolución renueva el apoyo operativo de una misión en terreno para mantener el orden institucional."
    elif tipo == "sanciones":
        return "Establece restricciones económicas directas sobre actores para forzar el cumplimiento de acuerdos internacionales."
    elif tipo == "humanitaria":
        return "Abre rutas seguras e inyecta recursos de emergencia para mitigar la hambruna y epidemias en la región."
    elif tipo == "paz":
        return "Promueve la mediación entre partes en combate para negociar el alto al fuego inmediato."
    else:
        return "Plantea mecanismos de supervisión y gobernabilidad ante riesgos de desestabilización regional."

def generar_noticia_impacto(voto, tipo_crisis, turno):
    if "A favor" in voto:
        if tipo_crisis == "humanitaria":
            return f"📰 NOTICIA (Turno {turno}): 'Corredor médico operativo: 2,000 personas atendidas tras envío de suministros.'"
        elif tipo_crisis == "paz":
            return f"📰 NOTICIA (Turno {turno}): 'Tregua pactada: Se suspenden los bombardeos y se evitan miles de bajas civiles.'"
        else:
            return f"📰 NOTICIA (Turno {turno}): 'Acuerdo internacional firmado con amplio respaldo multilateral.'"
    elif "Veto" in voto:
        if tipo_crisis == "humanitaria":
            return f"📰 NOTICIA (Turno {turno}): 'Bloqueo diplomático: 5,000 civiles quedan desprotegidos al rechazarse la moción.'"
        elif tipo_crisis == "sanciones":
            return f"📰 NOTICIA (Turno {turno}): 'Ruptura en el Consejo: Escalada de tensión militar tras ejercer el Veto.'"
        else:
            return f"📰 NOTICIA (Turno {turno}): 'Parálisis en el órgano de seguridad: Aumenta la inestabilidad en las zonas en conflicto.'"
    else:
        return f"📰 NOTICIA (Turno {turno}): 'Falta de consenso: La abstención detiene las negociaciones clave en terreno.'"

# --- CARGA Y ENTRENAMIENTO ---
@st.cache_resource
def cargar_y_entrenar():
    try:
        archivo = "UN_DATA.csv"
        if not os.path.exists(archivo):
            return None, None, ["No se encontró el archivo"]
            
        df = None
        for enc in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(archivo, header=0, encoding=enc, low_memory=False)
                break
            except:
                continue
                
        if df is None:
            return None, None, ["Error de codificación"]
            
        columnas_limpias = []
        conteo_cols = {}
        for col in df.columns:
            nombre = str(col).strip()
            if nombre in conteo_cols:
                conteo_cols[nombre] += 1
                nombre_nuevo = f"{nombre}_{conteo_cols[nombre]}"
            else:
                conteo_cols[nombre] = 0
                nombre_nuevo = nombre
            columnas_limpias.append(nombre_nuevo)
            
        df.columns = columnas_limpias
        columnas = df.columns.tolist()
        
        if 'Title' in df.columns:
            df['Title_ES'] = df['Title'].astype(str).apply(traducir_titulo)
        
        if {'Council', 'Resolution', 'TOTAL VOTES'}.issubset(df.columns):
            X = pd.DataFrame({
                'council_encoded': pd.factorize(df['Council'])[0],
                'resolution_encoded': pd.factorize(df['Resolution'])[0]
            })
            mediana_votos = df['TOTAL VOTES'].median()
            y = (df['TOTAL VOTES'] > mediana_votos).astype(int)
        else:
            X = pd.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6]})
            y = [1, 0, 1]
            
        modelo = RandomForestClassifier(n_estimators=50, random_state=42)
        modelo.fit(X, y)
        
        return modelo, df, columnas
        
    except Exception as e:
        return None, None, [f"Error: {e}"]

modelo_ia, df_onu, lista_columnas = cargar_y_entrenar()

# --- INICIALIZACIÓN DE ESTADO ---
if "turno" not in st.session_state:
    st.session_state.turno = 1
    st.session_state.vidas_riesgo = 1200000
    st.session_state.acceso_ayuda = 50
    st.session_state.estabilidad = 65
    st.session_state.puntos_diplomacia = 100  # Puntos de Diplomacia (DP)
    st.session_state.tutorial_visto = False
    st.session_state.votos_favor = 0
    st.session_state.votos_veto = 0
    st.session_state.votos_abstencion = 0
    st.session_state.indices_crisis = []
    st.session_state.historial_feed = ["Simulación iniciada. Puntos de diplomacia disponibles."]
    st.session_state.historial_noticias = ["📰 NOTICIA INICIAL: El Consejo de Seguridad abre un nuevo ciclo de sesiones."]
    st.session_state.historico_grafica = pd.DataFrame({
        "Estabilidad Mundial (%)": [65],
        "Población en Riesgo (en miles)": [1200]
    }, index=[0])

if not st.session_state.indices_crisis and df_onu is not None and len(df_onu) > 0:
    st.session_state.indices_crisis = random.sample(range(len(df_onu)), min(15, len(df_onu)))

def reiniciar_juego():
    st.session_state.turno = 1
    st.session_state.vidas_riesgo = 1200000
    st.session_state.acceso_ayuda = 50
    st.session_state.estabilidad = 65
    st.session_state.puntos_diplomacia = 100
    st.session_state.votos_favor = 0
    st.session_state.votos_veto = 0
    st.session_state.votos_abstencion = 0
    if df_onu is not None and len(df_onu) > 0:
        st.session_state.indices_crisis = random.sample(range(len(df_onu)), min(15, len(df_onu)))
    st.session_state.historial_feed = ["Partida reiniciada. Métricas restablecidas."]
    st.session_state.historial_noticias = ["📰 NOTICIA INICIAL: Se reestructuran las comisiones de trabajo en la ONU."]
    st.session_state.historico_grafica = pd.DataFrame({
        "Estabilidad Mundial (%)": [65],
        "Población en Riesgo (en miles)": [1200]
    }, index=[0])

# Tutorial inicial
if not st.session_state.tutorial_visto:
    st.info("📖 **Tutorial Rápido: ¿Cómo jugar?**\n\n"
            "1. **Objetivo:** Sobrevivir 10 turnos acumulando **Puntos de Diplomacia (DP)** y protegiendo vidas.\n"
            "2. **Puntos de Diplomacia:** Cada voto afecta tus DP disponibles. Puedes gastar DP para ejecutar mediaciones directas de urgencia.\n"
            "3. **Evaluación:** Mantén la Estabilidad Mundial por encima de 0% y la Población en Riesgo bajo los 2.5 millones.")
    if st.button("¡Entendido, comenzar a jugar!"):
        st.session_state.tutorial_visto = True
        st.rerun()

st.title("🌐 El Dilema del Veto: Simulador del Consejo de Seguridad")
st.markdown("---")

# --- CONDICIONES DE VICTORIA O DERROTA ---
juego_terminado = False
tipo_final = ""
estado_final = ""

if st.session_state.estabilidad <= 0:
    juego_terminado = True
    tipo_final = "derrota_estabilidad"
    estado_final = "DERROTA CRÍTICA: La estabilidad mundial colapsó por completo y el Consejo perdió toda autoridad."
elif st.session_state.vidas_riesgo >= 2500000:
    juego_terminado = True
    tipo_final = "derrota_humanitaria"
    estado_final = "DERROTA HUMANITARIA: La población en riesgo superó los 2.5 millones de personas."
elif st.session_state.turno > 10:
    juego_terminado = True
    tipo_final = "victoria"
    estado_final = "¡VICTORIA DIPLOMÁTICA! Lograste navegar con éxito 10 crisis internacionales manteniendo el equilibrio global."

if juego_terminado:
    puntaje_base = (st.session_state.estabilidad * 100) + (st.session_state.puntos_diplomacia * 15)
    puntuacion_final = max(0, int(puntaje_base + ((3000000 - st.session_state.vidas_riesgo) / 100)))

    if tipo_final == "victoria":
        st.balloons()
        st.success(f"### 🏆 ¡Misión Cumplida!\n\n{estado_final}")
    else:
        st.snow()
        st.error(f"### ⚠️ Fin del Simulador\n\n{estado_final}")
        
    st.markdown("---")
    st.subheader("📊 Estadísticas de Desempeño Final")
    
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    col_p1.metric("Puntuación Final", f"{puntuacion_final} pts")
    col_p2.metric("DP Restantes", f"{st.session_state.puntos_diplomacia} DP")
    col_p3.metric("Votos a Favor", st.session_state.votos_favor)
    col_p4.metric("Vetos Ejercidos", st.session_state.votos_veto)
    col_p5.metric("Abstenciones", st.session_state.votos_abstencion)
    
    st.markdown("---")
    if st.button("🔄 Iniciar Nueva Partida Diplomática", type="primary"):
        reiniciar_juego()
        st.rerun()
else:
    # --- CONSTRUCCIÓN DINÁMICA DE CRISIS ---
    if df_onu is not None and 'Title' in df_onu.columns and st.session_state.indices_crisis:
        idx_lista = (st.session_state.turno - 1) % len(st.session_state.indices_crisis)
        idx_crisis = st.session_state.indices_crisis[idx_lista]
        fila_actual = df_onu.iloc[idx_crisis]
        
        titulo_orig = str(fila_actual.get('Title', f'Resolución Histórica #{st.session_state.turno}'))
        titulo_es = traducir_titulo(titulo_orig)
        descripcion_contexto = generar_descripcion_resolucion(titulo_es)
        tipo_crisis = clasificar_tipo_crisis(titulo_es)
        
        fecha_res = str(fila_actual.get('Date', 'Año Desconocido'))
        organo_res = str(fila_actual.get('Council', 'Consejo de Seguridad'))
        votos_totales = fila_actual.get('TOTAL VOTES', 10)
        
        gravedad_calculada = min(10, max(1, int(votos_totales % 10) + 1)) if pd.notnull(votos_totales) else 5
        
        crisis_actual = {
            "titulo": f"Resolución: {titulo_es[:85]}..." if len(titulo_es) > 85 else f"Resolución: {titulo_es}",
            "descripcion": f"**Órgano:** {organo_res} | **Fecha:** {fecha_res}\n\n**Proyecto:** *\"{titulo_es}\"*\n\n💡 **Qué representa:** {descripcion_contexto}",
            "council_val": idx_crisis % 5,
            "res_val": gravedad_calculada,
            "tipo": tipo_crisis
        }
    else:
        crisis_actual = {
            "titulo": "Crisis Humanitaria en la Región Fronteriza Z",
            "descripcion": "Se propone una resolución para abrir un corredor humanitario obligatorio.",
            "council_val": 0,
            "res_val": 1,
            "tipo": "humanitaria"
        }

    # --- PREDICCIÓN DE MACHINE LEARNING ---
    if modelo_ia is not None:
        try:
            input_datos = [[crisis_actual["council_val"], crisis_actual["res_val"]]]
            prediccion = modelo_ia.predict(input_datos)[0]
            voto_eeuu_str = "A FAVOR" if prediccion == 1 else "VETO PROBABLE"
            voto_rusia_str = "VETO PROBABLE" if prediccion == 1 else "A FAVOR"
        except:
            voto_eeuu_str = "A FAVOR"
            voto_rusia_str = "VETO PROBABLE"
    else:
        voto_eeuu_str = "A FAVOR"
        voto_rusia_str = "VETO PROBABLE"

    # Layout principal en 2 columnas
    col_izq, col_der = st.columns([1.2, 0.8])

    with col_izq:
        st.subheader(f"📌 Turno {st.session_state.turno} de 10: {crisis_actual['titulo']}")
        st.write(crisis_actual['descripcion'])
        
        st.markdown("#### 🤖 Predicción por Machine Learning (`RandomForest`)")
        col_ia1, col_ia2 = st.columns(2)
        col_ia1.metric("EE. UU. (Modelo ML)", voto_eeuu_str)
        col_ia2.metric("Rusia (Modelo ML)", voto_rusia_str)
        
        st.markdown("---")
        st.subheader("🛠️ Acciones Diplomáticas Especiales")
        col_act1, col_act2 = st.columns(2)
        
        with col_act1:
            if st.button("🕊️ Solicitar Mediación (-25 DP)", disabled=st.session_state.puntos_diplomacia < 25):
                st.session_state.puntos_diplomacia -= 25
                st.session_state.estabilidad = min(100, st.session_state.estabilidad + 8)
                st.session_state.historial_feed.insert(0, f"Turno {st.session_state.turno}: Mediación enviada. Estabilidad +8%.")
                st.success("¡Mediación realizada exitosamente!")
                st.rerun()

        with col_act2:
            if st.button("📦 Enviar Fondos de Emergencia (-30 DP)", disabled=st.session_state.puntos_diplomacia < 30):
                st.session_state.puntos_diplomacia -= 30
                st.session_state.vidas_riesgo = max(0, st.session_state.vidas_riesgo - 120000)
                st.session_state.acceso_ayuda = min(100, st.session_state.acceso_ayuda + 10)
                st.session_state.historial_feed.insert(0, f"Turno {st.session_state.turno}: Fondos asignados. 120,000 personas protegidas.")
                st.success("¡Fondos asignados exitosamente!")
                st.rerun()

        st.markdown("---")
        st.subheader("🗳️ Tu Decisión como Embajador")
        
        voto_jugador = st.radio(
            "Elige tu voto sobre la resolución:",
            ["A favor (Sí)", "Veto (No)", "Abstención"]
        )
        
        if st.button("Emitir Voto y Avanzar Turno", type="primary"):
            t_crisis = crisis_actual["tipo"]
            if "A favor" in voto_jugador:
                st.session_state.votos_favor += 1
                st.session_state.vidas_riesgo = max(0, st.session_state.vidas_riesgo - 80000)
                st.session_state.acceso_ayuda = min(100, st.session_state.acceso_ayuda + 15)
                st.session_state.estabilidad = min(100, st.session_state.estabilidad + 3)
                st.session_state.puntos_diplomacia += 15
                mensaje = f"Turno {st.session_state.turno}: Aprobada la moción (+15 DP)."
            elif "Veto" in voto_jugador:
                st.session_state.votos_veto += 1
                st.session_state.vidas_riesgo += 150000
                st.session_state.acceso_ayuda = max(0, st.session_state.acceso_ayuda - 20)
                st.session_state.estabilidad = max(0, st.session_state.estabilidad - 10)
                st.session_state.puntos_diplomacia = max(0, st.session_state.puntos_diplomacia - 20)
                mensaje = f"Turno {st.session_state.turno}: Has ejercido el Veto (-20 DP)."
            else:
                st.session_state.votos_abstencion += 1
                st.session_state.vidas_riesgo += 40000
                st.session_state.acceso_ayuda = max(0, st.session_state.acceso_ayuda - 5)
                st.session_state.estabilidad = max(0, st.session_state.estabilidad - 2)
                st.session_state.puntos_diplomacia += 5
                mensaje = f"Turno {st.session_state.turno}: Abstención registrada (+5 DP)."
                
            noticia_generada = generar_noticia_impacto(voto_jugador, t_crisis, st.session_state.turno)
            st.session_state.historial_noticias.insert(0, noticia_generada)
            st.session_state.historial_feed.insert(0, mensaje)
            
            # Actualización de la gráfica
            nuevo_registro = pd.DataFrame({
                "Estabilidad Mundial (%)": [st.session_state.estabilidad],
                "Población en Riesgo (en miles)": [st.session_state.vidas_riesgo // 1000]
            }, index=[st.session_state.turno])
            st.session_state.historico_grafica = pd.concat([st.session_state.historico_grafica, nuevo_registro])
            
            st.session_state.turno += 1
            st.rerun()

    with col_der:
        st.subheader("👥 Monitor de Impacto Humano")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Puntos de Diplomacia", f"{st.session_state.puntos_diplomacia} DP")
        col_m2.metric("Estabilidad Mundial", f"{st.session_state.estabilidad}%")
        
        st.metric(
            label="Población Vulnerable en Riesgo", 
            value=f"{st.session_state.vidas_riesgo:,} hab."
        )
        st.progress(st.session_state.acceso_ayuda / 100, text=f"Acceso a Ayuda: {st.session_state.acceso_ayuda}%")
        
        st.markdown("---")
        st.subheader("📊 Tendencia Global de la Crisis")
        st.line_chart(st.session_state.historico_grafica)

    # --- SECCIÓN DE NOTICIAS Y BUSCADOR ---
    st.markdown("---")
    st.subheader("📰 Titulares de Última Hora e Impacto Directo")
    for noticia in st.session_state.historial_noticias[:3]:
        st.info(noticia)

    st.markdown("---")
    with st.expander("🔍 Buscador Histórico de Resoluciones (Archivos de la ONU)"):
        if df_onu is not None:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                busqueda_keyword = st.text_input("Buscar por palabra clave:")
            with col_f2:
                consejos = ['Todos'] + list(df_onu['Council'].dropna().unique()) if 'Council' in df_onu.columns else ['Todos']
                consejo_elegido = st.selectbox("Filtrar por Consejo:", consejos)
                    
            df_filtrado = df_onu.copy()
            if busqueda_keyword:
                cond_es = df_filtrado['Title_ES'].astype(str).str.contains(busqueda_keyword, case=False, na=False) if 'Title_ES' in df_filtrado.columns else False
                cond_en = df_filtrado['Title'].astype(str).str.contains(busqueda_keyword, case=False, na=False) if 'Title' in df_filtrado.columns else False
                df_filtrado = df_filtrado[cond_es | cond_en]
                
            if consejo_elegido != 'Todos' and 'Council' in df_filtrado.columns:
                df_filtrado = df_filtrado[df_filtrado['Council'] == consejo_elegido]
                
            st.write(f"**Resultados encontrados:** {len(df_filtrado):,}")
            
            df_mostrar = pd.DataFrame()
            if 'Title_ES' in df_filtrado.columns:
                df_mostrar['Título (Español)'] = df_filtrado['Title_ES']
            if 'Council' in df_filtrado.columns:
                df_mostrar['Consejo / Órgano'] = df_filtrado['Council']
            if 'Date' in df_filtrado.columns:
                df_mostrar['Fecha'] = df_filtrado['Date']
            if 'TOTAL VOTES' in df_filtrado.columns:
                df_mostrar['Votos Totales'] = df_filtrado['TOTAL VOTES']
                
            st.dataframe(df_mostrar.head(100), use_container_width=True)