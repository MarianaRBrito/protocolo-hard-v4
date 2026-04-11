import streamlit as st
import pandas as pd

# 1. Configuração inicial da página
st.set_page_config(page_title="Protocolo Analítico", layout="wide")

# 2. Menu Lateral (Onde você configura os parâmetros)
st.sidebar.title("⚙️ Configurações")
loteria = st.sidebar.selectbox("Selecione a Loteria", ["Lotofácil", "Lotomania"])
concurso_alvo = st.sidebar.number_input("Concurso Alvo para Backtest (0 = Último)", min_value=0, step=1)
threshold_similaridade = st.sidebar.slider("Threshold de Similaridade (%)", 50, 100, 80)

# 3. Cabeçalho do Painel
st.title(f"📊 Dashboard Estrutural - {loteria}")
st.write("Visão consolidada do pipeline analítico e status térmico das dezenas.")
st.markdown("---")

# 4. Dividindo a tela em blocos visuais (Colunas)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌡️ Camada Térmica")
    st.info("Aqui entrará a contagem e os ciclos das dezenas (Quentes, Mornas, Frias e Frias Profundas).")

with col2:
    st.subheader("📐 Camada Estrutural")
    st.info("Aqui entrará a análise de Pares/Ímpares, Soma, Amplitude e Quadrantes/Faixas.")

with col3:
    st.subheader("🎯 Camada de Similaridade")
    st.success(f"Buscando concursos semelhantes com {threshold_similaridade}% de aderência...")

st.markdown("---")

# 5. Tabela de Proposta Estrutural (Isca)
st.subheader("📋 Proposta Estrutural para Isca")
st.write("Estrutura recomendada baseada no consenso dos concursos seguintes (N+1 e N+2):")

# Criando uma tabela visual em branco para receber os dados futuros
dados_isca = pd.DataFrame({
    "Categoria Analítica": ["Candidatas Fortes", "Apoio", "Ruptura", "Exclusão/Espelho"],
    "Quantidade Sugerida": ["Módulo em construção", "Módulo em construção", "Módulo em construção", "Módulo em construção"],
    "Ação Estratégica": ["Buscar aderência térmica", "Completar equilíbrio", "Elasticidade de montagem", "Inversão lógica"]
})

st.dataframe(dados_isca, use_container_width=True, hide_index=True)
