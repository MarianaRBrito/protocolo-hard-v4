import streamlit as st
import pandas as pd
import numpy as np

# --- 1. FUNÇÕES MATEMÁTICAS E REGRAS DE NEGÓCIO ---

@st.cache_data # Isso faz o site ler a base só 1 vez e ficar super rápido
def obter_historico_real(loteria):
    if loteria == "Lotofácil":
        try:
            # Tenta ler o arquivo que você subiu no GitHub
            # Usa sep=";" porque as planilhas brasileiras costumam separar colunas por ponto e vírgula
            df = pd.read_csv("base_lotofacil.csv", sep=None, engine='python')
            
            # Pega apenas as colunas que têm as dezenas (ignorando Concurso e Data)
            colunas_numericas = df.select_dtypes(include=[np.number]).columns
            colunas_dezenas = [c for c in colunas_numericas if 'concurso' not in str(c).lower()]
            
            # Pega as 15 últimas colunas (As Bolas sorteadas)
            df_dezenas = df[colunas_dezenas[-15:]]
            
            # Transforma as linhas da planilha no nosso histórico
            historico = df_dezenas.values.tolist()
            return historico, 25
        except Exception as e:
            st.error("⚠️ Arquivo 'base_lotofacil.csv' não encontrado. Suba o arquivo no GitHub com este exato nome.")
            return [], 25
    else:
        # Quando você selecionar Lotomania, ele ainda vai simular até termos a base dela
        historico = [np.random.choice(range(1, 101), 20, replace=False) for _ in range(20)]
        return historico, 100

def calcular_termica(historico, qtd_numeros):
    if not historico:
        return [], [], pd.DataFrame()
        
    contagem = {i: 0 for i in range(1, qtd_numeros + 1)}
    for sorteio in historico:
        for num in sorteio:
            if not pd.isna(num): # Evita erros se houver células vazias na planilha
                contagem[int(num)] += 1

    df = pd.DataFrame(list(contagem.items()), columns=["Dezena", "Vezes Sorteadas"])
    df = df.sort_values(by="Vezes Sorteadas", ascending=False).reset_index(drop=True)

    quentes = df.head(5)["Dezena"].tolist()
    frias = df.tail(5)["Dezena"].tolist()
    return quentes, frias, df

def calcular_estrutural(sorteio):
    if not sorteio:
        return 0, 0, 0, 0, 0, 0
    sorteio_limpo = [int(n) for n in sorteio if not pd.isna(n)]
    pares = sum(1 for n in sorteio_limpo if n % 2 == 0)
    impares = len(sorteio_limpo) - pares
    soma = int(sum(sorteio_limpo)) 
    amplitude = int(max(sorteio_limpo) - min(sorteio_limpo))
    minimo = int(min(sorteio_limpo))
    maximo = int(max(sorteio_limpo))
    return pares, impares, soma, amplitude, minimo, maximo


# --- 2. CONFIGURAÇÃO VISUAL DO PAINEL ---

st.set_page_config(page_title="Protocolo Analítico", layout="wide")

st.sidebar.title("⚙️ Configurações")
loteria = st.sidebar.selectbox("Selecione a Loteria", ["Lotofácil", "Lotomania"])
concurso_alvo = st.sidebar.number_input("Concurso Alvo para Backtest (0 = Último)", min_value=0, step=1)
threshold_similaridade = st.sidebar.slider("Threshold de Similaridade (%)", 50, 100, 80)

# Processando os DADOS REAIS da sua planilha
historico_real, max_dezenas = obter_historico_real(loteria)
dezenas_quentes, dezenas_frias, tabela_termica = calcular_termica(historico_real, max_dezenas)

st.title(f"📊 Dashboard Estrutural - {loteria}")
if loteria == "Lotofácil":
    st.write(f"Analisando base oficial com **{len(historico_real)} concursos**.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌡️ Camada Térmica")
    st.write(f"Análise de todo o histórico.")
    
    if dezenas_quentes:
        st.write("**🔥 Top 5 Quentes:**")
        st.success(" - ".join([str(n).zfill(2) for n in dezenas_quentes]))
        st.write("**🧊 Top 5 Frias:**")
        st.info(" - ".join([str(n).zfill(2) for n in dezenas_frias]))
        
        with st.expander("Ver Frequência Completa"):
            st.dataframe(tabela_termica, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📐 Camada Estrutural")
    if historico_real:
        ultimo_sorteio = historico_real[0] # Pega a linha 1 da sua planilha
        pares, impares, soma, amplitude, minimo, maximo = calcular_estrutural(ultimo_sorteio)
        st.write("Perfil do último concurso da base.")
        st.write(f"**⚖️ Pares/Ímpares:** {pares} pares / {impares} ímpares")
        st.write(f"**➕ Soma Total:** {soma}")
        st.write(f"**📏 Amplitude:** {amplitude} (De {minimo} a {maximo})")
        proporcao_pares = pares / 15 if loteria == "Lotofácil" else pares / 20
        st.progress(proporcao_pares, text=f"Equilíbrio Par/Ímpar")

with col3:
    st.subheader("🎯 Camada de Similaridade")
    st.warning(f"Buscando no histórico de {len(historico_real)} concursos...")

st.markdown("---")

st.subheader("📋 Proposta Estrutural para Isca")
dados_isca = pd.DataFrame({
    "Categoria Analítica": ["Candidatas Fortes", "Apoio", "Ruptura", "Exclusão/Espelho"],
    "Quantidade Sugerida": ["Módulo em construção", "Módulo em construção", "Módulo em construção", "Módulo em construção"],
    "Ação Estratégica": ["Buscar aderência térmica", "Completar equilíbrio", "Elasticidade de montagem", "Inversão lógica"]
})
st.dataframe(dados_isca, use_container_width=True, hide_index=True)
