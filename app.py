import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
from scipy import stats
import math

# ══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Protocolo Hard V4 — 137+ Análises", layout="wide")

COLS_DEZENAS = [f"bola {i}" for i in range(1, 16)]
PRIMOS = {2,3,5,7,11,13,17,19,23}
FIBONACCI = {1,2,3,5,8,13,21}
MULTIPLOS_3 = {3,6,9,12,15,18,21,24}

# ══════════════════════════════════════════════════════════════
# CARGA DE DADOS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def carregar_base():
    df = pd.read_csv("base_lotofacil.csv")
    df = df.sort_values("Concurso").reset_index(drop=True)
    return df

@st.cache_data
def extrair_sorteios(df):
    return df[COLS_DEZENAS].values.tolist()

# ══════════════════════════════════════════════════════════════
# BLOCO 01 — FREQUÊNCIA SIMPLES E POR JANELAS
# ══════════════════════════════════════════════════════════════
def freq_simples(sorteios, janela=None):
    dados = sorteios[-janela:] if janela else sorteios
    cont = Counter()
    for s in dados:
        cont.update(s)
    return cont

def tabela_frequencia(sorteios, janelas=[50, 100, 200, 500]):
    rows = []
    for n in range(1, 26):
        row = {"Dezena": n}
        for j in janelas:
            dados = sorteios[-j:]
            c = sum(1 for s in dados if n in s)
            row[f"Freq {j}"] = c
            row[f"% {j}"] = round(c / j * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# BLOCO 02 — ATRASO E CICLO
# ══════════════════════════════════════════════════════════════
def calcular_atrasos(sorteios):
    ultimo_idx = {}
    for i, s in enumerate(sorteios):
        for n in s:
            ultimo_idx[n] = i
    total = len(sorteios)
    resultado = []
    for n in range(1, 26):
        idx = ultimo_idx.get(n, -1)
        atraso = total - 1 - idx if idx >= 0 else total
        resultado.append({"Dezena": n, "Último Concurso": idx + 1, "Atraso Atual": atraso})
    return pd.DataFrame(resultado).sort_values("Atraso Atual", ascending=False)

def ciclo_medio(sorteios, dezena):
    aparicoes = [i for i, s in enumerate(sorteios) if dezena in s]
    if len(aparicoes) < 2:
        return None
    diffs = [aparicoes[i+1] - aparicoes[i] for i in range(len(aparicoes)-1)]
    return round(np.mean(diffs), 2)

# ══════════════════════════════════════════════════════════════
# BLOCO 03 — ESTRUTURAL: SOMA, AMPLITUDE, PARES/ÍMPARES
# ══════════════════════════════════════════════════════════════
def perfil_estrutural(jogo):
    jogo = sorted(jogo)
    pares = sum(1 for n in jogo if n % 2 == 0)
    impares = 15 - pares
    soma = sum(jogo)
    amplitude = max(jogo) - min(jogo)
    return {"pares": pares, "impares": impares, "soma": soma, "amplitude": amplitude,
            "minimo": min(jogo), "maximo": max(jogo)}

def faixa_soma_historica(sorteios):
    somas = [sum(s) for s in sorteios]
    return int(np.percentile(somas, 5)), int(np.percentile(somas, 95)), round(np.mean(somas), 1)

def distribuicao_pares_historica(sorteios):
    dist = Counter(sum(1 for n in s if n % 2 == 0) for s in sorteios)
    total = len(sorteios)
    return {k: round(v/total*100, 1) for k, v in sorted(dist.items())}

# ══════════════════════════════════════════════════════════════
# BLOCO 04 — SEQUÊNCIAS E CONSECUTIVOS
# ══════════════════════════════════════════════════════════════
def contar_consecutivos(jogo):
    jogo = sorted(jogo)
    max_seq = seq_atual = 1
    pares_consec = 0
    for i in range(1, len(jogo)):
        if jogo[i] == jogo[i-1] + 1:
            seq_atual += 1
            max_seq = max(max_seq, seq_atual)
            pares_consec += 1
        else:
            seq_atual = 1
    return max_seq, pares_consec

def dist_consecutivos_historica(sorteios):
    dist = Counter(contar_consecutivos(s)[1] for s in sorteios)
    total = len(sorteios)
    return {k: round(v/total*100,1) for k, v in sorted(dist.items())}

# ══════════════════════════════════════════════════════════════
# BLOCO 05 — FAIXAS / QUINTETOS / MOLDURA-MIOLO
# ══════════════════════════════════════════════════════════════
def distribuicao_faixas(jogo):
    f = [0]*5
    for n in jogo:
        f[(n-1)//5] += 1
    return f  # [1-5, 6-10, 11-15, 16-20, 21-25]

def perfil_moldura_miolo(jogo):
    moldura = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
    miolo = set(range(1,26)) - moldura
    m = sum(1 for n in jogo if n in moldura)
    return m, 15 - m

def dist_faixas_historica(sorteios):
    dist = [[] for _ in range(5)]
    for s in sorteios:
        f = distribuicao_faixas(s)
        for i in range(5): dist[i].append(f[i])
    return [{"faixa": f"{i*5+1}-{i*5+5}", "media": round(np.mean(dist[i]),2),
             "min": min(dist[i]), "max": max(dist[i])} for i in range(5)]

# ══════════════════════════════════════════════════════════════
# BLOCO 06 — PRIMOS, FIBONACCI, MÚLTIPLOS
# ══════════════════════════════════════════════════════════════
def composicao_numerica(jogo):
    j = set(jogo)
    return {
        "primos": len(j & PRIMOS),
        "fibonacci": len(j & FIBONACCI),
        "multiplos_3": len(j & MULTIPLOS_3),
    }

def dist_composicao_historica(sorteios):
    dist_p = Counter(len(set(s) & PRIMOS) for s in sorteios)
    dist_f = Counter(len(set(s) & FIBONACCI) for s in sorteios)
    total = len(sorteios)
    return (
        {k: round(v/total*100,1) for k,v in sorted(dist_p.items())},
        {k: round(v/total*100,1) for k,v in sorted(dist_f.items())}
    )

# ══════════════════════════════════════════════════════════════
# BLOCO 07 — GAPS E DISTÂNCIAS
# ══════════════════════════════════════════════════════════════
def gaps_jogo(jogo):
    jogo = sorted(jogo)
    gaps = [jogo[i+1]-jogo[i] for i in range(len(jogo)-1)]
    return {"gaps": gaps, "media": round(np.mean(gaps),2),
            "max_gap": max(gaps), "min_gap": min(gaps)}

def dist_gaps_historica(sorteios):
    todos_gaps = []
    for s in sorteios:
        todos_gaps.extend(gaps_jogo(s)["gaps"])
    c = Counter(todos_gaps)
    total = sum(c.values())
    return {k: round(v/total*100,1) for k,v in sorted(c.items()) if k<=10}

# ══════════════════════════════════════════════════════════════
# BLOCO 08 — PARIDADE POR FAIXA
# ══════════════════════════════════════════════════════════════
def paridade_por_faixa(jogo):
    resultado = {}
    for i in range(5):
        faixa = [n for n in jogo if i*5+1 <= n <= i*5+5]
        pares = sum(1 for n in faixa if n%2==0)
        resultado[f"F{i+1}({i*5+1}-{i*5+5})"] = {"total": len(faixa), "pares": pares, "impares": len(faixa)-pares}
    return resultado

# ══════════════════════════════════════════════════════════════
# BLOCO 09 — COOCORRÊNCIA E PARES FREQUENTES
# ══════════════════════════════════════════════════════════════
@st.cache_data
def calcular_coocorrencia(sorteios_tuple, top_n=20):
    pares = Counter()
    for s in sorteios_tuple:
        for a, b in combinations(sorted(s), 2):
            pares[(a,b)] += 1
    return pares.most_common(top_n)

@st.cache_data
def calcular_trios_recorrentes(sorteios_tuple, top_n=15):
    trios = Counter()
    for s in sorteios_tuple:
        for trio in combinations(sorted(s), 3):
            trios[trio] += 1
    return trios.most_common(top_n)

# ══════════════════════════════════════════════════════════════
# BLOCO 10 — MOTOR N+1/N+2 (MANTIDO DO ORIGINAL)
# ══════════════════════════════════════════════════════════════
def projetar_isca(alvo_idx, sorteios, threshold_percent):
    if not sorteios or alvo_idx >= len(sorteios): return [], [], []
    alvo_set = set(sorteios[alvo_idx])
    threshold_hits = int(len(alvo_set) * (threshold_percent / 100.0))
    similares = []
    futuro = []
    for i, s in enumerate(sorteios):
        if i == alvo_idx: continue
        if len(alvo_set & set(s)) >= threshold_hits:
            similares.append(i)
            if i+1 < len(sorteios): futuro.extend(sorteios[i+1])
            if i+2 < len(sorteios): futuro.extend(sorteios[i+2])
    freq = Counter(futuro)
    ordenados = freq.most_common()
    fortes = [k for k,_ in ordenados[:6]]
    apoio = [k for k,_ in ordenados[6:12]]
    return similares, fortes, apoio

# ══════════════════════════════════════════════════════════════
# BLOCO 11 — ENTROPIA DE SHANNON
# ══════════════════════════════════════════════════════════════
def entropia_shannon(jogo, sorteios):
    freq = freq_simples(sorteios)
    total = sum(freq.values())
    H = 0
    for n in jogo:
        p = freq[n] / total
        if p > 0: H -= p * math.log2(p)
    return round(H, 4)

def faixa_entropia_historica(sorteios):
    entropias = []
    freq = freq_simples(sorteios)
    total = sum(freq.values())
    for s in sorteios[-500:]:
        H = -sum((freq[n]/total)*math.log2(freq[n]/total) for n in s if freq[n]>0)
        entropias.append(H)
    return round(np.mean(entropias),4), round(np.std(entropias),4)

# ══════════════════════════════════════════════════════════════
# BLOCO 12 — QUI-QUADRADO E ADERÊNCIA
# ══════════════════════════════════════════════════════════════
def teste_qui_quadrado(sorteios, janela=500):
    dados = sorteios[-janela:]
    observado = np.array([sum(1 for s in dados if n in s) for n in range(1,26)])
    esperado = np.full(25, len(dados) * 15 / 25)
    chi2, p = stats.chisquare(observado, esperado)
    return round(chi2, 2), round(p, 4)

# ══════════════════════════════════════════════════════════════
# BLOCO 13 — MONTE CARLO
# ══════════════════════════════════════════════════════════════
def monte_carlo_validar(jogo, sorteios, n_sim=10000):
    acertos = []
    for _ in range(n_sim):
        aleatorio = set(random.sample(range(1,26), 15))
        acertos.append(len(set(jogo) & aleatorio))
    media = np.mean(acertos)
    meu_acerto = np.mean([len(set(jogo) & set(s)) for s in sorteios[-200:]])
    return round(media, 2), round(meu_acerto, 2)

# ══════════════════════════════════════════════════════════════
# BLOCO 14 — BACKTESTING
# ══════════════════════════════════════════════════════════════
def backtesting_jogo(jogo, sorteios, ultimos=100):
    dados = sorteios[-ultimos:]
    acertos = [len(set(jogo) & set(s)) for s in dados]
    dist = Counter(acertos)
    return {
        "media": round(np.mean(acertos), 2),
        "max": max(acertos),
        "min": min(acertos),
        "dist": dict(sorted(dist.items())),
        "pct_11+": round(sum(1 for a in acertos if a >= 11) / len(acertos) * 100, 1),
        "pct_13+": round(sum(1 for a in acertos if a >= 13) / len(acertos) * 100, 1),
    }

# ══════════════════════════════════════════════════════════════
# BLOCO 15 — ROLLING WINDOW / TENDÊNCIA
# ══════════════════════════════════════════════════════════════
def tendencia_dezena(dezena, sorteios, janelas=[20, 50, 100, 200]):
    rows = []
    for j in janelas:
        dados = sorteios[-j:]
        freq = sum(1 for s in dados if dezena in s)
        rows.append({"Janela": j, "Freq": freq, "%": round(freq/j*100, 1)})
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════
# BLOCO 16 — MARKOV (TRANSIÇÕES)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def matriz_markov_simples(sorteios_tuple, janela=200):
    sorteios = list(sorteios_tuple)[-janela:]
    trans = {n: Counter() for n in range(1,26)}
    for i in range(len(sorteios)-1):
        atual = set(sorteios[i])
        prox = set(sorteios[i+1])
        for n in atual:
            trans[n].update(prox)
    return trans

# ══════════════════════════════════════════════════════════════
# BLOCO 17 — FILTRO COMPLETO DE JOGO
# ══════════════════════════════════════════════════════════════
def filtrar_jogo(jogo, sorteios, config):
    jogo = list(jogo)
    alertas = []
    soma_min, soma_max, soma_media = faixa_soma_historica(sorteios)
    soma = sum(jogo)
    if soma < soma_min or soma > soma_max:
        alertas.append(f"⚠️ Soma {soma} fora da faixa histórica [{soma_min}–{soma_max}]")

    max_seq, pares_cons = contar_consecutivos(jogo)
    if max_seq > config.get("max_seq", 4):
        alertas.append(f"⚠️ Sequência de {max_seq} consecutivos (limite: {config['max_seq']})")

    pares = sum(1 for n in jogo if n%2==0)
    if pares < config.get("min_pares",5) or pares > config.get("max_pares",10):
        alertas.append(f"⚠️ {pares} pares fora do intervalo [{config['min_pares']}–{config['max_pares']}]")

    faixas = distribuicao_faixas(jogo)
    for i, f in enumerate(faixas):
        if f == 0:
            alertas.append(f"⚠️ Faixa {i*5+1}-{i*5+5} vazia")
        if f > config.get("max_por_faixa", 5):
            alertas.append(f"⚠️ Faixa {i*5+1}-{i*5+5} com {f} dezenas (excesso)")

    comp = composicao_numerica(jogo)
    if comp["primos"] > config.get("max_primos", 5):
        alertas.append(f"⚠️ {comp['primos']} primos (excesso)")
    if comp["fibonacci"] > config.get("max_fibonacci", 5):
        alertas.append(f"⚠️ {comp['fibonacci']} Fibonacci (excesso)")

    return alertas

# ══════════════════════════════════════════════════════════════
# BLOCO 18 — GERADOR DE JOGOS
# ══════════════════════════════════════════════════════════════
def gerar_jogo_filtrado(sorteios, fortes, apoio, config, tentativas=10000):
    soma_min, soma_max, _ = faixa_soma_historica(sorteios)
    for _ in range(tentativas):
        pool = list(set(fortes + apoio))
        if len(pool) < 15:
            pool += random.sample([n for n in range(1,26) if n not in pool], 15-len(pool))
        jogo = sorted(random.sample(pool[:20] if len(pool)>=20 else pool+list(range(1,26))[:20-len(pool)], 15))
        alertas = filtrar_jogo(jogo, sorteios, config)
        if not alertas:
            return jogo
    return None

# ══════════════════════════════════════════════════════════════
# BLOCO 19 — VALOR ESPERADO
# ══════════════════════════════════════════════════════════════
PREMIOS = {15: 1_500_000, 14: 1500, 13: 25, 12: 10, 11: 5}
PROB = {15: 1/3_268_760, 14: 15/3_268_760, 13: 105/3_268_760,
        12: 455/3_268_760, 11: 1365/3_268_760}

def valor_esperado(custo=3.5):
    ev = sum(PREMIOS[k] * PROB[k] for k in PREMIOS)
    return round(ev, 4), round(ev / custo, 4)

# ══════════════════════════════════════════════════════════════
# BLOCO 20 — ANÁLISE DE DEZENAS FRIAS EM PICO
# ══════════════════════════════════════════════════════════════
def dezenas_em_pico_atraso(sorteios, percentil=85):
    df_atraso = calcular_atrasos(sorteios)
    limite = np.percentile(df_atraso["Atraso Atual"], percentil)
    return df_atraso[df_atraso["Atraso Atual"] >= limite]

# ══════════════════════════════════════════════════════════════
# BLOCO 21 — COBERTURA HAMMING
# ══════════════════════════════════════════════════════════════
def distancia_hamming(j1, j2):
    return 15 - len(set(j1) & set(j2))

def cobertura_entre_jogos(jogos):
    if len(jogos) < 2: return []
    result = []
    for i in range(len(jogos)):
        for j in range(i+1, len(jogos)):
            d = distancia_hamming(jogos[i], jogos[j])
            result.append({"Par": f"J{i+1} x J{j+1}", "Distância Hamming": d, "Dezenas em comum": 15-d})
    return pd.DataFrame(result)

# ══════════════════════════════════════════════════════════════
# BLOCO 22 — SCORE FINAL INTEGRADO
# ══════════════════════════════════════════════════════════════
def score_jogo(jogo, sorteios, fortes, apoio):
    score = 0
    j = set(jogo)

    # Térmica: quantas fortes/apoio contém
    score += len(j & set(fortes)) * 3
    score += len(j & set(apoio)) * 1.5

    # Estrutural
    soma = sum(jogo)
    soma_min, soma_max, soma_med = faixa_soma_historica(sorteios)
    if soma_min <= soma <= soma_max: score += 5
    score += 3 if abs(soma - soma_med) < 10 else 0

    pares = sum(1 for n in jogo if n%2==0)
    score += 5 if 6 <= pares <= 9 else 0

    faixas = distribuicao_faixas(jogo)
    score += 3 if all(f > 0 for f in faixas) else -5

    max_seq, _ = contar_consecutivos(jogo)
    score += 3 if max_seq <= 3 else 0

    # Backtesting
    bt = backtesting_jogo(jogo, sorteios, ultimos=200)
    score += bt["pct_11+"] / 10
    score += bt["pct_13+"] / 5

    return round(score, 2)

# ══════════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════
df = carregar_base()
sorteios = extrair_sorteios(df)
sorteios_tuple = tuple(tuple(s) for s in sorteios)

st.sidebar.title("⚙️ Protocolo Hard V4")
st.sidebar.markdown(f"**Base:** {len(sorteios)} concursos | C1–C{df['Concurso'].max()}")

janela_analise = st.sidebar.slider("Janela de Análise (concursos)", 50, len(sorteios), 500, step=50)
threshold_sim = st.sidebar.slider("Threshold Similaridade N+1/N+2 (%)", 50, 100, 80)
concurso_ref = st.sidebar.number_input("Índice de Referência (0=último)", 0, len(sorteios)-1, len(sorteios)-1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Meus Jogos (Backtesting)")
jogo1_input = st.sidebar.text_input("Jogo 1 (ex: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)")
jogo2_input = st.sidebar.text_input("Jogo 2")
jogo3_input = st.sidebar.text_input("Jogo 3")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Config. Filtros")
config_filtro = {
    "max_seq": st.sidebar.slider("Máx. consecutivos", 2, 6, 4),
    "min_pares": st.sidebar.slider("Mín. pares", 4, 8, 5),
    "max_pares": st.sidebar.slider("Máx. pares", 7, 11, 10),
    "max_por_faixa": st.sidebar.slider("Máx. por faixa", 4, 7, 5),
    "max_primos": st.sidebar.slider("Máx. primos", 3, 7, 5),
    "max_fibonacci": st.sidebar.slider("Máx. Fibonacci", 3, 7, 5),
}

def parse_jogo(s):
    try:
        nums = [int(x) for x in s.strip().split()]
        if len(nums) == 15 and all(1 <= n <= 25 for n in nums):
            return sorted(nums)
    except:
        pass
    return None

jogos_usuario = [j for j in [parse_jogo(jogo1_input), parse_jogo(jogo2_input), parse_jogo(jogo3_input)] if j]

sorteios_janela = sorteios[-janela_analise:]

# ══════════════════════════════════════════════════════════════
# TABS PRINCIPAIS
# ══════════════════════════════════════════════════════════════
st.title("📊 Protocolo Hard V4 — 137+ Análises | Lotofácil")
st.markdown(f"Base ativa: **{len(sorteios)} concursos** (C1 – C{df['Concurso'].max()}) | Janela: **{janela_analise}**")

tabs = st.tabs([
    "🌡️ Térmica", "⏱️ Atraso & Ciclo", "📐 Estrutural",
    "🔢 Composição", "📏 Sequências & Gaps", "🎯 Similaridade N+1/N+2",
    "🔗 Coocorrência", "📊 Backtesting", "🧪 Filtros & Score",
    "🎲 Monte Carlo", "📈 Tendência Rolling", "⚖️ Valor Esperado",
    "🎰 Gerar Jogos", "🏆 Relatório Final"
])

# ──────────────────────────────────────────────────────────────
with tabs[0]:  # TÉRMICA
    st.header("🌡️ Camada Térmica — Frequência por Janelas")
    df_freq = tabela_frequencia(sorteios)
    st.dataframe(df_freq, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        f = freq_simples(sorteios, janela_analise)
        top5 = [n for n,_ in f.most_common(5)]
        bot5 = [n for n,_ in f.most_common()[:-6:-1]]
        st.success(f"🔥 Top 5 Quentes (janela {janela_analise}): {' - '.join(str(n).zfill(2) for n in top5)}")
        st.info(f"🧊 Top 5 Frias: {' - '.join(str(n).zfill(2) for n in bot5)}")
    with col2:
        pico = dezenas_em_pico_atraso(sorteios)
        st.warning("⚠️ Dezenas em Pico de Atraso (top 15%)")
        st.dataframe(pico, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
with tabs[1]:  # ATRASO & CICLO
    st.header("⏱️ Atraso Atual & Ciclo Médio")
    df_atraso = calcular_atrasos(sorteios)
    ciclos = []
    for n in range(1,26):
        cm = ciclo_medio(sorteios, n)
        ciclos.append({"Dezena": n, "Ciclo Médio": cm})
    df_ciclo = pd.DataFrame(ciclos)
    df_completo = df_atraso.merge(df_ciclo, on="Dezena")
    df_completo["Status"] = df_completo.apply(
        lambda r: "🔴 VENCIDA" if r["Atraso Atual"] > (r["Ciclo Médio"] or 99)*1.5
        else ("🟡 No ciclo" if r["Atraso Atual"] > (r["Ciclo Médio"] or 99)*0.8 else "🟢 Ok"), axis=1)
    st.dataframe(df_completo.sort_values("Atraso Atual", ascending=False), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
with tabs[2]:  # ESTRUTURAL
    st.header("📐 Camada Estrutural")
    soma_min, soma_max, soma_med = faixa_soma_historica(sorteios_janela)
    st.info(f"📊 Faixa de Soma Histórica (p5–p95): **{soma_min} – {soma_max}** | Média: **{soma_med}**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição de Pares (histórico)")
        dist_p = distribuicao_pares_historica(sorteios_janela)
        df_p = pd.DataFrame(list(dist_p.items()), columns=["Pares", "% Histórico"])
        st.dataframe(df_p, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Distribuição por Faixas (médias)")
        df_faixas = pd.DataFrame(dist_faixas_historica(sorteios_janela))
        st.dataframe(df_faixas, use_container_width=True, hide_index=True)

    st.subheader("Moldura vs Miolo — Últimos concursos")
    molduras = [perfil_moldura_miolo(s) for s in sorteios_janela[-50:]]
    m_vals = [m[0] for m in molduras]
    st.write(f"Moldura — Média: **{round(np.mean(m_vals),1)}** | Min: {min(m_vals)} | Max: {max(m_vals)}")

# ──────────────────────────────────────────────────────────────
with tabs[3]:  # COMPOSIÇÃO
    st.header("🔢 Composição Numérica: Primos, Fibonacci, Múltiplos")
    dist_pr, dist_fi = dist_composicao_historica(sorteios_janela)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Primos por jogo (%)")
        st.dataframe(pd.DataFrame(list(dist_pr.items()), columns=["Qtd", "%"]), hide_index=True)
        st.caption(f"Primos até 25: {sorted(PRIMOS)}")
    with col2:
        st.subheader("Fibonacci por jogo (%)")
        st.dataframe(pd.DataFrame(list(dist_fi.items()), columns=["Qtd", "%"]), hide_index=True)
        st.caption(f"Fibonacci até 25: {sorted(FIBONACCI)}")
    with col3:
        st.subheader("Múltiplos de 3 (%)")
        dist_m3 = Counter(len(set(s) & MULTIPLOS_3) for s in sorteios_janela)
        total = len(sorteios_janela)
        df_m3 = pd.DataFrame([(k, round(v/total*100,1)) for k,v in sorted(dist_m3.items())], columns=["Qtd", "%"])
        st.dataframe(df_m3, hide_index=True)
        st.caption(f"Múltiplos de 3 até 25: {sorted(MULTIPLOS_3)}")

# ──────────────────────────────────────────────────────────────
with tabs[4]:  # SEQUÊNCIAS & GAPS
    st.header("📏 Sequências Consecutivas & Gaps")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição de Consecutivos (%)")
        dist_cons = dist_consecutivos_historica(sorteios_janela)
        df_cons = pd.DataFrame(list(dist_cons.items()), columns=["Pares Consec.", "% Histórico"])
        st.dataframe(df_cons, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Distribuição de Gaps (%)")
        dist_g = dist_gaps_historica(sorteios_janela)
        df_g = pd.DataFrame(list(dist_g.items()), columns=["Gap", "% Histórico"])
        st.dataframe(df_g, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
with tabs[5]:  # SIMILARIDADE N+1/N+2
    st.header("🎯 Motor de Similaridade N+1/N+2")
    similares, fortes, apoio = projetar_isca(concurso_ref, sorteios, threshold_sim)
    concurso_nome = df.iloc[concurso_ref]["Concurso"] if concurso_ref < len(df) else "?"
    st.write(f"Concurso de referência: **{concurso_nome}** | Threshold: **{threshold_sim}%**")
    if similares:
        st.success(f"✅ {len(similares)} concursos similares encontrados no histórico")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔥 Candidatas Fortes (base do jogo):**")
            st.success(" · ".join(str(n).zfill(2) for n in sorted(fortes)))
        with col2:
            st.write("**🤝 Apoio (complemento):**")
            st.info(" · ".join(str(n).zfill(2) for n in sorted(apoio)))

        dados_isca = pd.DataFrame({
            "Categoria": ["Candidatas Fortes", "Apoio", "Frias em Pico"],
            "Dezenas": [
                " ".join(str(n).zfill(2) for n in sorted(fortes)),
                " ".join(str(n).zfill(2) for n in sorted(apoio)),
                " ".join(str(n).zfill(2) for n in dezenas_em_pico_atraso(sorteios)["Dezena"].tolist())
            ],
            "Ação": ["Base principal", "Completar espaços", "Ruptura de tendência"]
        })
        st.dataframe(dados_isca, use_container_width=True, hide_index=True)
    else:
        st.warning("Nenhum similar encontrado. Reduza o threshold no menu lateral.")

# ──────────────────────────────────────────────────────────────
with tabs[6]:  # COOCORRÊNCIA
    st.header("🔗 Coocorrência & Trios Recorrentes")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 20 Pares Mais Frequentes")
        pares = calcular_coocorrencia(sorteios_tuple)
        df_pares = pd.DataFrame([(f"{a:02d}+{b:02d}", c) for (a,b),c in pares], columns=["Par","Freq"])
        st.dataframe(df_pares, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Top 15 Trios Mais Frequentes")
        trios = calcular_trios_recorrentes(sorteios_tuple)
        df_trios = pd.DataFrame([(f"{a:02d}+{b:02d}+{c:02d}", cnt) for (a,b,c),cnt in trios], columns=["Trio","Freq"])
        st.dataframe(df_trios, use_container_width=True, hide_index=True)

    st.subheader("Markov — Top Sucessores por Dezena")
    dezena_markov = st.selectbox("Selecione uma dezena:", range(1, 26))
    trans = matriz_markov_simples(sorteios_tuple)
    top_suc = trans[dezena_markov].most_common(10)
    df_markov = pd.DataFrame(top_suc, columns=["Sucessor","Freq"])
    st.dataframe(df_markov, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
with tabs[7]:  # BACKTESTING
    st.header("📊 Backtesting dos Jogos")
    if jogos_usuario:
        for i, jogo in enumerate(jogos_usuario):
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            bt = backtesting_jogo(jogo, sorteios, ultimos=200)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Média Acertos", bt["media"])
            col2.metric("% com 11+", f"{bt['pct_11+']}%")
            col3.metric("% com 13+", f"{bt['pct_13+']}%")
            col4.metric("Máx Acertos", bt["max"])
            df_dist = pd.DataFrame(list(bt["dist"].items()), columns=["Acertos","Freq"])
            st.bar_chart(df_dist.set_index("Acertos"))
            st.markdown("---")
    else:
        st.info("Insira seus jogos no menu lateral para ver o backtesting.")
        st.write("**Exemplo de backtesting com jogo aleatório:**")
        jogo_demo = random.sample(range(1,26), 15)
        bt = backtesting_jogo(jogo_demo, sorteios, ultimos=200)
        st.write(f"Jogo: {sorted(jogo_demo)}")
        st.write(f"Média: {bt['media']} | 11+: {bt['pct_11+']}% | 13+: {bt['pct_13+']}%")

# ──────────────────────────────────────────────────────────────
with tabs[8]:  # FILTROS & SCORE
    st.header("🧪 Filtros Hard V4 + Score Integrado")
    if jogos_usuario:
        for i, jogo in enumerate(jogos_usuario):
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            alertas = filtrar_jogo(jogo, sorteios, config_filtro)
            perf = perfil_estrutural(jogo)
            paridade = paridade_por_faixa(jogo)
            comp = composicao_numerica(jogo)
            gaps = gaps_jogo(jogo)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Soma:** {perf['soma']} | **Amplitude:** {perf['amplitude']}")
                st.write(f"**Pares:** {perf['pares']} | **Ímpares:** {perf['impares']}")
                max_seq, pares_cons = contar_consecutivos(jogo)
                st.write(f"**Maior sequência:** {max_seq} | **Pares consec.:** {pares_cons}")
            with col2:
                st.write(f"**Primos:** {comp['primos']} | **Fibonacci:** {comp['fibonacci']} | **Múlt.3:** {comp['multiplos_3']}")
                faixas = distribuicao_faixas(jogo)
                st.write(f"**Faixas [1-5|6-10|11-15|16-20|21-25]:** {faixas}")
                mold, miolo = perfil_moldura_miolo(jogo)
                st.write(f"**Moldura:** {mold} | **Miolo:** {miolo}")
            with col3:
                st.write(f"**Gap médio:** {gaps['media']} | **Max gap:** {gaps['max_gap']}")
                H = entropia_shannon(jogo, sorteios)
                H_med, H_std = faixa_entropia_historica(sorteios)
                st.write(f"**Entropia Shannon:** {H}")
                st.write(f"**Entropia Histórica:** {H_med} ± {H_std}")
                sc = score_jogo(jogo, sorteios, fortes if similares else [], [])
                st.metric("🏅 Score Final", sc)

            if alertas:
                for a in alertas:
                    st.warning(a)
            else:
                st.success("✅ Jogo passou em todos os filtros Hard V4!")
            st.markdown("---")

        # Cobertura entre jogos
        if len(jogos_usuario) > 1:
            st.subheader("📐 Cobertura Hamming entre Jogos")
            st.dataframe(cobertura_entre_jogos(jogos_usuario), use_container_width=True, hide_index=True)
    else:
        st.info("Insira seus jogos no menu lateral.")

# ──────────────────────────────────────────────────────────────
with tabs[9]:  # MONTE CARLO
    st.header("🎲 Validação Monte Carlo")
    if jogos_usuario:
        for i, jogo in enumerate(jogos_usuario):
            media_aleat, media_hist = monte_carlo_validar(jogo, sorteios, n_sim=5000)
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            col1, col2 = st.columns(2)
            col1.metric("Média acertos vs aleatório (Monte Carlo)", media_aleat)
            col2.metric("Média acertos vs histórico real (200 conc.)", media_hist)
            delta = round(media_hist - media_aleat, 2)
            if delta > 0:
                st.success(f"✅ Jogo performa **+{delta}** acertos acima do aleatório puro")
            else:
                st.warning(f"⚠️ Jogo performa {delta} acertos vs aleatório")
            st.markdown("---")
    else:
        st.info("Insira seus jogos no menu lateral.")

    st.subheader("📊 Teste Qui-Quadrado (aderência da base)")
    chi2, p = teste_qui_quadrado(sorteios, janela=janela_analise)
    st.write(f"χ² = **{chi2}** | p-valor = **{p}**")
    if p > 0.05:
        st.success("✅ Base não apresenta desvio estatístico significativo (p > 0.05)")
    else:
        st.warning("⚠️ Desvio detectado na base — interprete frequências com cautela")

# ──────────────────────────────────────────────────────────────
with tabs[10]:  # TENDÊNCIA ROLLING
    st.header("📈 Tendência Rolling por Dezena")
    dezena_tend = st.selectbox("Dezena para análise de tendência:", range(1, 26), key="tend")
    df_tend = tendencia_dezena(dezena_tend, sorteios)
    st.dataframe(df_tend, use_container_width=True, hide_index=True)

    freq_recente = df_tend[df_tend["Janela"]==50]["Freq"].values[0]
    freq_longa = df_tend[df_tend["Janela"]==200]["Freq"].values[0]
    taxa_50 = freq_recente / 50
    taxa_200 = freq_longa / 200

    if taxa_50 > taxa_200 * 1.2:
        st.success(f"🔼 Dezena {dezena_tend:02d} em ALTA — frequência recente acima da média histórica")
    elif taxa_50 < taxa_200 * 0.8:
        st.warning(f"🔽 Dezena {dezena_tend:02d} em BAIXA — frequência recente abaixo da média histórica")
    else:
        st.info(f"➡️ Dezena {dezena_tend:02d} ESTÁVEL")

# ──────────────────────────────────────────────────────────────
with tabs[11]:  # VALOR ESPERADO
    st.header("⚖️ Valor Esperado & EV por Jogo")
    ev, ev_ratio = valor_esperado()
    col1, col2, col3 = st.columns(3)
    col1.metric("EV por jogo (R$)", f"R$ {ev:.4f}")
    col2.metric("Custo por jogo", "R$ 3,50")
    col3.metric("Ratio EV/Custo", f"{ev_ratio:.4f}")
    st.info("📌 EV calculado sobre prêmios fixos estimados. O rateio real reduz o EV em concursos acumulados.")

    st.subheader("Probabilidades por faixa")
    df_ev = pd.DataFrame([
        {"Faixa": k, "Prob": f"1 em {int(1/v):,}", "Prêmio Est.": f"R$ {PREMIOS[k]:,.0f}",
         "EV parcial": f"R$ {PREMIOS[k]*v:.4f}"}
        for k, v in sorted(PROB.items(), reverse=True)
    ])
    st.dataframe(df_ev, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────
with tabs[12]:  # GERAR JOGOS — MOTOR AGRESSIVO
    st.header("🎰 Gerador Agressivo — Protocolo Hard V4")
    st.info("Motor de score ponderado: todas as análises convergem para rankear as dezenas antes de gerar a combinação.")

    # ── Pesos do cenário agressivo ──────────────────────────────
    # G1 Comportamento recente (N+1/N+2 + freq + rolling)  → 40%
    # G2 Ciclo e atraso                                     → 30%
    # G3 Padrão histórico (coocorrência + Markov + trios)   → 20%
    # G4 Estrutural (faixas, primos, Fibonacci)             →  5%
    # G5 Entropia                                           →  5%
    PESOS = {"g1": 0.40, "g2": 0.30, "g3": 0.20, "g4": 0.05, "g5": 0.05}

    def normalizar(dct):
        """Normaliza dict {dezena: valor} para [0,1]"""
        vals = list(dct.values())
        mn, mx = min(vals), max(vals)
        if mx == mn: return {k: 0.5 for k in dct}
        return {k: (v - mn) / (mx - mn) for k, v in dct.items()}

    def score_agressivo(sorteios, janela, concurso_ref, threshold_sim, pesos):
        """Calcula score ponderado para cada dezena 1-25."""
        scores = {n: 0.0 for n in range(1, 26)}

        # ── G1: Comportamento recente (40%) ──────────────────
        # N+1/N+2
        sim, fortes_n, apoio_n = projetar_isca(concurso_ref, sorteios, threshold_sim)
        g1_n12 = {n: 0.0 for n in range(1, 26)}
        for n in fortes_n: g1_n12[n] = 1.0
        for n in apoio_n:  g1_n12[n] = max(g1_n12[n], 0.6)

        # Frequência janela curta (50) vs longa (janela)
        f50  = freq_simples(sorteios, 50)
        f_longa = freq_simples(sorteios, janela)
        g1_freq = normalizar({n: f50[n] for n in range(1,26)})

        # Tendência rolling: alta = freq recente > histórica
        g1_roll = {}
        for n in range(1, 26):
            t50  = f50[n]  / 50
            tlong = f_longa[n] / janela
            g1_roll[n] = max(0.0, (t50 - tlong) / (tlong + 1e-9))
        g1_roll = normalizar(g1_roll)

        g1 = {n: (g1_n12[n]*0.50 + g1_freq[n]*0.30 + g1_roll[n]*0.20) for n in range(1,26)}

        # ── G2: Ciclo e atraso (30%) ──────────────────────────
        total = len(sorteios)
        ultimo_idx = {}
        for i, s in enumerate(sorteios):
            for x in s: ultimo_idx[x] = i

        g2_atraso = {}
        g2_ciclo  = {}
        for n in range(1, 26):
            idx = ultimo_idx.get(n, -1)
            atraso = total - 1 - idx if idx >= 0 else total
            cm = ciclo_medio(sorteios, n) or atraso
            # Quanto mais vencida (atraso > ciclo), maior o score
            ratio = atraso / cm if cm > 0 else 1.0
            g2_atraso[n] = min(ratio, 3.0)   # cap em 3x o ciclo
            g2_ciclo[n]  = ratio

        g2_atraso = normalizar(g2_atraso)
        g2_ciclo  = normalizar(g2_ciclo)
        g2 = {n: (g2_atraso[n]*0.60 + g2_ciclo[n]*0.40) for n in range(1,26)}

        # ── G3: Padrão histórico (20%) ────────────────────────
        # Coocorrência: soma das frequências de todos os pares que contêm n
        pares_freq = Counter()
        for s in sorteios[-500:]:
            for a, b in combinations(sorted(s), 2):
                pares_freq[(a,b)] += 1

        g3_cooc = {n: 0.0 for n in range(1,26)}
        for (a,b), cnt in pares_freq.items():
            g3_cooc[a] += cnt
            g3_cooc[b] += cnt
        g3_cooc = normalizar(g3_cooc)

        # Markov: probabilidade de n aparecer no próximo concurso
        trans = matriz_markov_simples(tuple(tuple(s) for s in sorteios), janela=min(janela,500))
        ultimo_sorteio = set(sorteios[concurso_ref])
        g3_markov = {n: 0.0 for n in range(1,26)}
        total_trans = sum(sum(trans[x].values()) for x in ultimo_sorteio if x in trans)
        for x in ultimo_sorteio:
            if x in trans:
                for suc, cnt in trans[x].items():
                    g3_markov[suc] = g3_markov.get(suc, 0) + cnt
        g3_markov = normalizar(g3_markov)

        # Trios: dezenas que participam de trios frequentes
        trios_freq = Counter()
        for s in sorteios[-500:]:
            for trio in combinations(sorted(s), 3):
                trios_freq[trio] += 1
        g3_trios = {n: 0.0 for n in range(1,26)}
        for trio, cnt in trios_freq.most_common(100):
            for x in trio: g3_trios[x] += cnt
        g3_trios = normalizar(g3_trios)

        g3 = {n: (g3_cooc[n]*0.40 + g3_markov[n]*0.40 + g3_trios[n]*0.20) for n in range(1,26)}

        # ── G4: Estrutural / composição (5%) ─────────────────
        # Dezenas bem distribuídas por faixa recebem bônus
        # Proxy: dezenas que apareceram em jogos com boas faixas
        g4 = {n: 0.5 for n in range(1,26)}  # neutro por padrão

        # ── G5: Entropia (5%) ─────────────────────────────────
        freq_all = freq_simples(sorteios)
        total_all = sum(freq_all.values())
        g5_raw = {n: freq_all[n]/total_all for n in range(1,26)}
        # Dezenas com freq próxima da esperada (1/25 = 0.04) = mais entrópicas
        esperado = 1/25
        g5 = normalizar({n: 1 - abs(v - esperado)/esperado for n,v in g5_raw.items()})

        # ── Score Final ───────────────────────────────────────
        for n in range(1, 26):
            scores[n] = (
                pesos["g1"] * g1[n] +
                pesos["g2"] * g2[n] +
                pesos["g3"] * g3[n] +
                pesos["g4"] * g4[n] +
                pesos["g5"] * g5[n]
            )

        return scores, sim, fortes_n, apoio_n

    # ── Interface ─────────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        qtd_jogos = st.radio("Quantos jogos gerar?", [1, 2, 3], index=1, horizontal=True)
    with col_cfg2:
        seed_val = st.number_input("Seed (0 = aleatório)", min_value=0, value=0, step=1)
    with col_cfg3:
        top_pool = st.slider("Tamanho do pool (top N dezenas)", 15, 22, 18)

    st.markdown("**Dezenas fixas obrigatórias** (opcional — separe por espaço):")
    fixas_input = st.text_input("Ex: 01 05 13 18", key="fixas_gen")
    fixas = []
    if fixas_input.strip():
        try:
            fixas = [int(x) for x in fixas_input.strip().split() if 1 <= int(x) <= 25]
        except:
            st.warning("Formato inválido nas fixas.")

    # Mostra o ranking antes de gerar
    with st.expander("📊 Ver ranking de score por dezena (antes de gerar)", expanded=False):
        with st.spinner("Calculando scores..."):
            scores_preview, _, _, _ = score_agressivo(sorteios, janela_analise, concurso_ref, threshold_sim, PESOS)
        ranking_df = pd.DataFrame([
            {"Dezena": n, "Score": round(scores_preview[n], 4),
             "Rank": i+1}
            for i, (n, _) in enumerate(sorted(scores_preview.items(), key=lambda x: -x[1]))
        ])
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.dataframe(ranking_df.head(13), use_container_width=True, hide_index=True)
        with col_r2:
            st.dataframe(ranking_df.tail(12), use_container_width=True, hide_index=True)
        st.caption(f"Pesos: G1 Recente={PESOS['g1']*100:.0f}% | G2 Ciclo={PESOS['g2']*100:.0f}% | G3 Histórico={PESOS['g3']*100:.0f}% | G4 Estrutural={PESOS['g4']*100:.0f}% | G5 Entropia={PESOS['g5']*100:.0f}%")

    if st.button("🎯 Gerar Combinações Agressivas", type="primary"):
        if seed_val > 0:
            random.seed(seed_val)

        with st.spinner("Calculando score de todas as análises..."):
            scores, _sim, _fortes, _apoio = score_agressivo(
                sorteios, janela_analise, concurso_ref, threshold_sim, PESOS)

        # Pool = top N dezenas por score + fixas obrigatórias
        ranking_ord = sorted(scores.items(), key=lambda x: -x[1])
        pool = [n for n, _ in ranking_ord[:top_pool]]
        for f in fixas:
            if f not in pool: pool.append(f)

        st.write(f"**Pool de {len(pool)} dezenas:** {' · '.join(f'{n:02d}' for n in sorted(pool))}")

        jogos_gerados = []
        tentativas_total = 0

        while len(jogos_gerados) < qtd_jogos and tentativas_total < 100000:
            tentativas_total += 1

            # Amostragem ponderada pelo score dentro do pool
            pool_scores = np.array([scores[n] for n in pool])
            pool_scores = pool_scores / pool_scores.sum()

            base = fixas[:15]
            restante_pool = [n for n in pool if n not in base]
            restante_scores = np.array([scores[n] for n in restante_pool])
            restante_scores = restante_scores / restante_scores.sum()

            faltam = 15 - len(base)
            if len(restante_pool) >= faltam:
                escolhidos = list(np.random.choice(restante_pool, size=faltam, replace=False, p=restante_scores))
                candidato = sorted(base + escolhidos)
            else:
                extras = [n for n in range(1,26) if n not in base and n not in restante_pool]
                candidato = sorted(base + restante_pool + random.sample(extras, faltam - len(restante_pool)))

            alertas = filtrar_jogo(candidato, sorteios, config_filtro)
            if not alertas and candidato not in jogos_gerados:
                jogos_gerados.append(candidato)

        if jogos_gerados:
            st.success(f"✅ {len(jogos_gerados)} combinação(ões) gerada(s) em {tentativas_total} tentativas")
            st.markdown("---")

            for i, jogo in enumerate(jogos_gerados):
                perf = perfil_estrutural(jogo)
                bt = backtesting_jogo(jogo, sorteios, ultimos=200)
                comp = composicao_numerica(jogo)
                faixas = distribuicao_faixas(jogo)
                max_seq, _ = contar_consecutivos(jogo)
                score_jogo_val = round(sum(scores[n] for n in jogo), 4)
                alertas_final = filtrar_jogo(jogo, sorteios, config_filtro)

                st.subheader(f"🃏 Combinação {i+1}")
                nums_str = "  ·  ".join(f"**{n:02d}**" for n in jogo)
                st.markdown(f"## {nums_str}")

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Score Total", round(score_jogo_val, 3))
                col2.metric("Soma", perf["soma"])
                col3.metric("Pares / Ímpares", f"{perf['pares']} / {perf['impares']}")
                col4.metric("Backtesting 11+", f"{bt['pct_11+']}%")
                col5.metric("Backtesting 13+", f"{bt['pct_13+']}%")

                col6, col7, col8 = st.columns(3)
                col6.write(f"**Faixas [F1→F5]:** {faixas}")
                col7.write(f"**Primos:** {comp['primos']} | **Fib:** {comp['fibonacci']} | **Mult3:** {comp['multiplos_3']}")
                col8.write(f"**Maior seq.:** {max_seq} | **Amplitude:** {perf['amplitude']}")

                # Breakdown do score por dezena
                with st.expander(f"🔬 Breakdown do score — Combinação {i+1}"):
                    breakdown = pd.DataFrame([
                        {"Dezena": n,
                         "Score": round(scores[n], 4),
                         "Rank Geral": sorted(scores, key=lambda x: -scores[x]).index(n)+1,
                         "G1 Recente": "✅" if scores[n] >= sorted(scores.values(), reverse=True)[int(25*0.4)] else "—",
                         "G2 Ciclo":   "✅" if n in dezenas_em_pico_atraso(sorteios)["Dezena"].tolist() else "—",
                         "G3 Histórico": "✅" if n in (_fortes + _apoio) else "—",
                        }
                        for n in sorted(jogo)
                    ])
                    st.dataframe(breakdown, use_container_width=True, hide_index=True)

                if not alertas_final:
                    st.success("✅ Passou em todos os filtros Hard V4")
                st.markdown("---")

            custo_total = len(jogos_gerados) * 3.5
            st.info(f"💰 Custo total: **{len(jogos_gerados)} jogo(s) × R$ 3,50 = R$ {custo_total:.2f}**")

            if len(jogos_gerados) > 1:
                st.subheader("📐 Cobertura Hamming entre os jogos gerados")
                st.dataframe(cobertura_entre_jogos(jogos_gerados), use_container_width=True, hide_index=True)
        else:
            st.error("❌ Não foi possível gerar combinações aprovadas. Relaxe os filtros no menu lateral.")


# ──────────────────────────────────────────────────────────────
with tabs[13]:  # RELATÓRIO FINAL
    st.header("🏆 Relatório Final Integrado")
    st.markdown(f"""
    **Protocolo Hard V4 — Relatório de Sessão**
    - Base: **{len(sorteios)} concursos** | Último: C{df['Concurso'].max()} ({df.iloc[-1]['Data']})
    - Janela ativa: **{janela_analise} concursos**
    - Threshold similaridade: **{threshold_sim}%**
    - Similares encontrados: **{len(similares) if 'similares' in dir() else 0}**
    """)

    soma_min, soma_max, soma_med = faixa_soma_historica(sorteios_janela)
    chi2, p = teste_qui_quadrado(sorteios, janela=janela_analise)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📐 Parâmetros Estruturais Validados")
        st.write(f"- Soma: **{soma_min} – {soma_max}** (média {soma_med})")
        dist_pc = distribuicao_pares_historica(sorteios_janela)
        top_pares = max(dist_pc, key=dist_pc.get)
        st.write(f"- Pares mais frequentes: **{top_pares}** ({dist_pc[top_pares]}%)")
        dist_cons = dist_consecutivos_historica(sorteios_janela)
        top_cons = max(dist_cons, key=dist_cons.get)
        st.write(f"- Consecutivos mais comuns: **{top_cons}** ({dist_cons[top_cons]}%)")
        dist_pr, _ = dist_composicao_historica(sorteios_janela)
        top_prim = max(dist_pr, key=dist_pr.get)
        st.write(f"- Primos mais comuns por jogo: **{top_prim}** ({dist_pr[top_prim]}%)")
    with col2:
        st.subheader("🎯 Dezenas Prioritárias")
        f = freq_simples(sorteios, janela_analise)
        top10 = [n for n,_ in f.most_common(10)]
        pico = dezenas_em_pico_atraso(sorteios)["Dezena"].tolist()
        st.write(f"- Quentes (top 10): **{' '.join(str(n).zfill(2) for n in top10)}**")
        st.write(f"- Em pico de atraso: **{' '.join(str(n).zfill(2) for n in pico)}**")
        if 'fortes' in dir() and fortes:
            st.write(f"- N+1/N+2 Fortes: **{' '.join(str(n).zfill(2) for n in sorted(fortes))}**")
            st.write(f"- N+1/N+2 Apoio: **{' '.join(str(n).zfill(2) for n in sorted(apoio))}**")

    if jogos_usuario:
        st.subheader("🃏 Seus Jogos — Resumo")
        for i, jogo in enumerate(jogos_usuario):
            bt = backtesting_jogo(jogo, sorteios, ultimos=200)
            alertas = filtrar_jogo(jogo, sorteios, config_filtro)
            sc = score_jogo(jogo, sorteios, fortes if 'fortes' in dir() and fortes else [], [])
            status = "✅ APROVADO" if not alertas else f"⚠️ {len(alertas)} alertas"
            st.write(f"**J{i+1}:** {' '.join(str(n).zfill(2) for n in jogo)} | Score: **{sc}** | Média: **{bt['media']}** | 11+: **{bt['pct_11+']}%** | {status}")

    st.markdown("---")
    st.caption("Protocolo Hard V4 — MarianaRBrito | github.com/MarianaRBrito/protocolo-hard-v4")
