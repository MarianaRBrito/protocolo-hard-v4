import streamlit as st
import pandas as pd
import numpy as np

# --- 1. FUNÇÕES MATEMÁTICAS E REGRAS DE NEGÓCIO ---

@st.cache_data
def obter_historico_real(loteria):
    if loteria == "Lotofácil":
        try:
            df = pd.read_csv("base_lotofacil.csv", sep=None, engine='python')
            colunas_numericas = df.select_dtypes(include=[np.number]).columns
            colunas_dezenas = [c for c in colunas_numericas if 'concurso' not in str(c).lower()]
            df_dezenas = df[colunas_dezenas[-15:]]
            historico = df_dezenas.values.tolist()
            return historico, 25
        except Exception as e:
            st.error("⚠️ Base de dados não encontrada. Verifique se o nome é base_lotofacil.csv")
            return [], 25
    else:
        historico = [np.random.choice(range(1, 101), 20, replace=False) for _ in range(20)]
        return historico, 100

def calcular_termica(historico, qtd_numeros):
    if not historico: return [], [], pd.DataFrame()
    contagem = {i: 0 for i in range(1, qtd_numeros + 1)}
    for sorteio in historico:
        for num in sorteio:
            if not pd.isna(num): contagem[int(num)] += 1

    df = pd.DataFrame(list(contagem.items()), columns=["Dezena", "Vezes Sorteadas"])
    df = df.sort_values(by="Vezes Sorteadas", ascending=False).reset_index(drop=True)
    return df.head(5)["Dezena"].tolist(), df.tail(5)["Dezena"].tolist(), df

def calcular_estrutural(sorteio):
    if not sorteio: return 0, 0, 0, 0, 0, 0
    sorteio_limpo = [int(n) for n in sorteio if not pd.isna(n)]
    pares = sum(1 for n in sorteio_limpo if n % 2 == 0)
    impares = len(sorteio_limpo) - pares
    return pares, impares, int(sum(sorteio_limpo)), int(max(sorteio_limpo) - min(sorteio_limpo)), int(min(sorteio_limpo)), int(max(sorteio_limpo))

# MOTOR N+1 / N+2: A Inteligência Preditiva
def projetar_isca(alvo_idx, historico, threshold_percent):
    if not historico or alvo_idx >= len(historico): return [], [], []
    
    alvo = historico[alvo_idx]
    alvo_set = set([int(n) for n in alvo if not pd.isna(n)])
    threshold_hits = int(len(alvo_set) * (threshold_percent / 100.0))
    
    similares_encontrados = []
    futuro_n1_n2 = []
    
    for i, sorteio in enumerate(historico):
        if i == alvo_idx: continue # Não compara com ele mesmo
        
        sorteio_set = set([int(n) for n in sorteio if not pd.isna(n)])
        acertos = len(alvo_set.intersection(sorteio_set))
        
        if acertos >= threshold_hits:
            similares_encontrados.append(i)
            # Pega o que aconteceu nos concursos seguintes (N+1 e N+2)
            if i - 1 >= 0: futuro_n1_n2.extend([int(n) for n in historico[i-1] if not pd.isna(n)])
            if i - 2 >= 0: futuro_n1_n2.extend([int(n) for n in historico[i-2] if not pd.isna(n)])
            
    freq_futuro = {}
    for n in futuro_n1_n2: freq_futuro[n] = freq_futuro.get(n, 0) + 1
            
    ordenados = sorted(freq_futuro.items(), key=lambda x: x[1], reverse=True)
    fortes = [k for k, v in ordenados[:6]] # Top 6 Candidatas Fortes
    apoio = [k for k, v in ordenados[6:12]] # Top 6 de Apoio
    
    return similares_encontrados, fortes, apoio


# --- 2. CONFIGURAÇÃO VISUAL DO PAINEL ---

st.set_page_config(page_title="Protocolo Analítico", layout="wide")

st.sidebar.title("⚙️ Configurações")
loteria = st.sidebar.selectbox("Selecione a Loteria", ["Lotofácil", "Lotomania"])
concurso_alvo = st.sidebar.number_input("Concurso Alvo (0 = Último da Base)", min_value=0, step=1)
threshold_similaridade = st.sidebar.slider("Threshold de Similaridade (%)", 50, 100, 80)

# Processamentos Core
historico_real, max_dezenas = obter_historico_real(loteria)
dezenas_quentes, dezenas_frias, tabela_termica = calcular_termica(historico_real, max_dezenas)

# Proteção de Limite
if concurso_alvo >= len(historico_real): concurso_alvo = 0

# Processamento Preditivo (A Mágica)
similares, fortes, apoio = projetar_isca(concurso_alvo, historico_real, threshold_similaridade)

st.title(f"📊 Dashboard Estrutural - {loteria}")
if loteria == "Lotofácil":
    st.write(f"Analisando base oficial com **{len(historico_real)} concursos**. Alvo da análise: Índice {concurso_alvo}.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🌡️ Camada Térmica")
    if dezenas_quentes:
        st.write("**🔥 Top 5 Quentes (Geral):**")
        st.success(" - ".join([str(n).zfill(2) for n in dezenas_quentes]))
        st.write("**🧊 Top 5 Frias (Geral):**")
        st.info(" - ".join([str(n).zfill(2) for n in dezenas_frias]))
        with st.expander("Ver Frequência Completa"):
            st.dataframe(tabela_termica, use_container_width=True, hide_index=True)

with col2:
    st.subheader("📐 Camada Estrutural")
    if historico_real:
        pares, impares, soma, amplitude, minimo, maximo = calcular_estrutural(historico_real[concurso_alvo])
        st.write(f"Perfil do concurso alvo:")
        st.write(f"**⚖️ Pares/Ímpares:** {pares} pares / {impares} ímpares")
        st.write(f"**➕ Soma:** {soma} | **📏 Ampl.:** {amplitude} ({minimo} a {maximo})")
        proporcao_pares = pares / 15 if loteria == "Lotofácil" else pares / 20
        st.progress(proporcao_pares, text=f"Equilíbrio Par/Ímpar")

with col3:
    st.subheader("🎯 Camada de Similaridade")
    st.write(f"Régua de corte: **{threshold_similaridade}%** (Acertos Mínimos)")
    if len(similares) > 0:
        st.success(f"**{len(similares)} concursos** encontrados no passado com padrão estrutural similar!")
    else:
        st.warning("Nenhum concurso similar encontrado. Tente diminuir a % no menu lateral.")

st.markdown("---")

# A TABELA FINAL PREENCHIDA PELA IA
st.subheader("📋 Proposta Estrutural para Isca")
st.write("Dezenas sugeridas baseadas na projeção estatística dos eventos futuros (N+1 e N+2):")

if similares:
    fortes_str = " - ".join([str(n).zfill(2) for n in fortes])
    apoio_str = " - ".join([str(n).zfill(2) for n in apoio])
    
    dados_isca = pd.DataFrame({
        "Categoria Analítica": ["Candidatas Fortes", "Apoio", "Ruptura/Frias"],
        "Dezenas Mapeadas": [fortes_str, apoio_str, "Analisar Frias Térmicas"],
        "Ação Estratégica": ["Base principal dos volantes", "Completar os espaços", "Inversão de tendência"]
    })
    st.dataframe(dados_isca, use_container_width=True, hide_index=True)
else:
    st.info("Aguardando similaridade. Abaixe o threshold no menu para encontrar padrões e gerar a Isca.")
