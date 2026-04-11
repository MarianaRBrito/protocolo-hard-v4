import streamlit as st
import pandas as pd
import numpy as np

# --- 1. FUNÇÕES MATEMÁTICAS E REGRAS DE NEGÓCIO ---

# Motor temporário: Simula os últimos 20 concursos para testarmos as regras
def obter_historico(loteria):
    qtd_numeros = 25 if loteria == "Lotofácil" else 100
    qtd_sorteio = 15 if loteria == "Lotofácil" else 20
    # Gera 20 sorteios aleatórios passados
    historico = [np.random.choice(range(1, qtd_numeros + 1), qtd_sorteio, replace=False) for _ in range(20)]
    return historico, qtd_numeros

# Lógica da Camada Térmica: Conta a frequência de cada dezena
def calcular_termica(historico, qtd_numeros):
    contagem = {i: 0 for i in range(1, qtd_numeros + 1)}
    for sorteio in historico:
        for num in sorteio:
            contagem[num] += 1

    # Transforma em tabela e ordena das mais saídas para as menos saídas
    df = pd.DataFrame(list(contagem.items()), columns=["Dezena", "Vezes Sorteadas"])
    df = df.sort_values(by="Vezes Sorteadas", ascending=False).reset_index(drop=True)

    # Separa os extremos
    quentes = df.head(5)["Dezena"].tolist()
    frias = df.tail(5)["Dezena"].tolist()
    return quentes, frias, df


# --- 2. CONFIGURAÇÃO VISUAL DO PAINEL ---

st.set_page_config(page_title="Protocolo Analítico", layout="wide")

# Menu Lateral
st.sidebar.title("⚙️ Configurações")
loteria = st.sidebar.selectbox("Selecione a Loteria", ["Lotofácil", "Lotomania"])
concurso_alvo = st.sidebar.number_input("Concurso Alvo para Backtest (0 = Último)", min_value=0, step=1)
threshold_similaridade = st.sidebar.slider("Threshold de Similaridade (%)", 50, 100, 80)

# Processando os dados com base na escolha do menu
historico_simulado, max_dezenas = obter_historico(loteria)
dezenas_quentes, dezenas_frias, tabela_termica = calcular_termica(historico_simulado, max_dezenas)

# Cabeçalho
st.title(f"📊 Dashboard Estrutural - {loteria}")
st.write("Visão consolidada do pipeline analítico e status térmico das dezenas.")
st.markdown("---")

# Colunas Analíticas
col1, col2, col3 = st.columns(3)

# BLOCO 1: CAMADA TÉRMICA ATUALIZADA
with col1:
    st.subheader("🌡️ Camada Térmica")
    st.write("Análise dos últimos 20 concursos.")
    
    st.write("**🔥 Top 5 Quentes:**")
    st.success(" - ".join([str(n).zfill(2) for n in dezenas_quentes])) # zfill(2) coloca o zero na frente (ex: 05)

    st.write("**🧊 Top 5 Frias:**")
    st.info(" - ".join([str(n).zfill(2) for n in dezenas_frias]))
    
    # Um botão sanfona para ver todas as dezenas
    with st.expander("Ver Frequência de Todas as Dezenas"):
        st.dataframe(tabela_termica, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📐 Camada Estrutural")
    st.info("Aqui entrará a análise de Pares/Ímpares, Soma, Amplitude e Quadrantes/Faixas.")

with col3:
    st.subheader("🎯 Camada de Similaridade")
    st.warning(f"Aguardando banco de dados para buscar aderência de {threshold_similaridade}%...")

st.markdown("---")

# Tabela Isca
st.subheader("📋 Proposta Estrutural para Isca")
st.write("Estrutura recomendada baseada no consenso dos concursos seguintes (N+1 e N+2):")

dados_isca = pd.DataFrame({
    "Categoria Analítica": ["Candidatas Fortes", "Apoio", "Ruptura", "Exclusão/Espelho"],
    "Quantidade Sugerida": ["Módulo em construção", "Módulo em construção", "Módulo em construção", "Módulo em construção"],
    "Ação Estratégica": ["Buscar aderência térmica", "Completar equilíbrio", "Elasticidade de montagem", "Inversão lógica"]
})

st.dataframe(dados_isca, use_container_width=True, hide_index=True)
