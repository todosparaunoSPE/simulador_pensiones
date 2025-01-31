# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:12:11 2025

@author: jperezr
"""


import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Configuración inicial de la app
st.set_page_config(page_title="Simulador de Pensiones", layout="wide", initial_sidebar_state="expanded")
st.title("📊 **Simulador de Pensiones con Predicción de Rentabilidad**")


# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)





# Título atractivo con una breve descripción
st.markdown("""
    ### 💡 Descubre cómo tu ahorro para el retiro se puede incrementar con nuestras simulaciones.
    Estima tu pensión mensual futura basada en tu edad, salario, semanas cotizadas y aportación voluntaria.
""", unsafe_allow_html=True)

# Entrada de datos del usuario
col1, col2 = st.columns(2)
with col1:
    edad = st.number_input("Edad actual", min_value=18, max_value=70, value=30)
    salario = st.number_input("Salario mensual ($)", min_value=5000, max_value=200000, value=20000, step=1000)
with col2:
    semanas_cotizadas = st.number_input("Semanas cotizadas", min_value=0, max_value=2500, value=750, step=50)
    aportacion_voluntaria = st.slider("Aportación voluntaria (%)", 0, 20, 5)

# Cálculo inicial de la pensión
edad_retiro = 65
anios_restantes = max(0, edad_retiro - edad)

def calcular_pension(salario, semanas, aportacion, rendimiento):
    saldo_acumulado = (salario * 0.065 + salario * (aportacion / 100)) * semanas / 52
    saldo_futuro = saldo_acumulado * ((1 + rendimiento) ** anios_restantes)
    pension_mensual = saldo_futuro / (20 * 12)  # Considerando 20 años de retiro
    return pension_mensual

# Generación de datos ficticios para entrenamiento del modelo ML
df = pd.DataFrame({
    "edad": np.random.randint(25, 65, 500),
    "salario": np.random.randint(10000, 100000, 500),
    "semanas": np.random.randint(200, 2500, 500),
    "aportacion": np.random.randint(0, 20, 500),
    "rendimiento": np.random.uniform(0.02, 0.07, 500)
})
df["pension"] = df.apply(lambda x: calcular_pension(x.salario, x.semanas, x.aportacion, x.rendimiento), axis=1)

# Entrenamiento del modelo
X = df[["edad", "salario", "semanas", "aportacion"]]
y = df["pension"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
error = mean_absolute_error(y_test, y_pred)

# Predicción de la pensión para el usuario
entrada_usuario = np.array([[edad, salario, semanas_cotizadas, aportacion_voluntaria]])
pension_predicha = model.predict(entrada_usuario)[0]

# Visualización de resultados
st.markdown("## 📈 **Predicción de tu Pensión**")
st.metric("Pensión mensual estimada ($)", f"{pension_predicha:,.2f}")
st.caption(f"Error promedio del modelo: ±{error:,.2f} MXN")

# Gráfico de evolución del ahorro
rendimiento_escenarios = {"Conservador": 0.03, "Moderado": 0.05, "Agresivo": 0.07}
datos = []
for escenario, rendimiento in rendimiento_escenarios.items():
    saldo = [(salario * 0.065 + salario * (aportacion_voluntaria / 100)) * i for i in range(anios_restantes + 1)]
    saldo_acumulado = [sum(saldo[:i]) * (1 + rendimiento) ** i for i in range(anios_restantes + 1)]
    datos.extend([[i + edad, saldo_acumulado[i], escenario] for i in range(anios_restantes + 1)])

df_plot = pd.DataFrame(datos, columns=["Edad", "Saldo Acumulado", "Escenario"])
fig = px.line(df_plot, x="Edad", y="Saldo Acumulado", color="Escenario", title="Evolución del Ahorro", 
              labels={"Saldo Acumulado": "Saldo acumulado ($)", "Edad": "Edad (años)"}, 
              template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# DataFrame de simulación de saldo acumulado por escenario
st.markdown("### 📊 **Simulación de Saldo Acumulado por Escenario**")
df_simulacion = df_plot.pivot(index="Edad", columns="Escenario", values="Saldo Acumulado").reset_index()
st.dataframe(df_simulacion)

# DataFrame con las predicciones de la pensión
st.markdown("### 📊 **Predicción de la Pensión**")
df_prediccion = pd.DataFrame({
    "Edad": [edad],
    "Salario": [salario],
    "Semanas Cotizadas": [semanas_cotizadas],
    "Aportación Voluntaria (%)": [aportacion_voluntaria],
    "Pensión Predicha ($)": [pension_predicha]
})
st.dataframe(df_prediccion)

# Sección de ayuda
st.sidebar.markdown("## ℹ️ **Ayuda**")
st.sidebar.info("Este simulador permite estimar tu pensión mensual futura basado en tu edad, salario, semanas cotizadas y aportación voluntaria. Utiliza Machine Learning para predecir los valores y proyectar escenarios de inversión.")

# Sección de contacto
st.sidebar.markdown("## 📬 **Contáctanos**")
st.sidebar.info("Si tienes preguntas o deseas más información sobre cómo mejorar tu pensión, ¡no dudes en contactarnos!")

# Copyright
st.sidebar.markdown("---")
st.sidebar.markdown("© **Javier Horacio Pérez Ricárdez**")

# Fondo visual atractivo
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.wallpaperflare.com/static/930/280/676/pensions-wallpaper.jpg");
        background-size: cover;
    }
    </style>
""", unsafe_allow_html=True)
