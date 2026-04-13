import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
from scipy import stats
import math

st.set_page_config(page_title="Protocolo Hard V4 — 137+ Análises", layout="wide")

COLS_DEZENAS = [f"bola {i}" for i in range(1, 16)]
PRIMOS      = {2,3,5,7,11,13,17,19,23}
FIBONACCI   = {1,2,3,5,8,13,21}
MULTIPLOS_3 = {3,6,9,12,15,18,21,24}

# ══════════════════════════════════════════════════════════════
# SESSION STATE — objeto central que conecta análises ao gerador
# ══════════════════════════════════════════════════════════════
ETAPAS = [
    "termica", "atraso", "estrutural", "composicao",
    "sequencias", "n12", "coocorrencia", "monte_carlo"
]

def init_session():
    if "analise" not in st.session_state:
        st.session_state["analise"] = {e: None for e in ETAPAS}
    if "scores_cache" not in st.session_state:
        st.session_state["scores_cache"] = None

def etapa_ok(nome):
    return st.session_state["analise"].get(nome) is not None

def todas_etapas_ok():
    return all(etapa_ok(e) for e in ETAPAS)

def salvar_etapa(nome, dados):
    st.session_state["analise"][nome] = dados
    st.session_state["scores_cache"] = None  # invalida cache de scores

def status_etapas():
    total = len(ETAPAS)
    ok = sum(1 for e in ETAPAS if etapa_ok(e))
    return ok, total

init_session()

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
# FUNÇÕES DE ANÁLISE
# ══════════════════════════════════════════════════════════════
def freq_simples(sorteios, janela=None):
    dados = sorteios[-janela:] if janela else sorteios
    cont = Counter()
    for s in dados: cont.update(s)
    return cont

def tabela_frequencia(sorteios, janelas=[50, 100, 200, 500]):
    rows = []
    for n in range(1, 26):
        row = {"Dezena": n}
        for j in janelas:
            dados = sorteios[-j:]
            c = sum(1 for s in dados if n in s)
            row[f"Freq {j}"] = c
            row[f"% {j}"] = round(c/j*100, 1)
        rows.append(row)
    return pd.DataFrame(rows)

def calcular_atrasos(sorteios):
    ultimo_idx = {}
    for i, s in enumerate(sorteios):
        for n in s: ultimo_idx[n] = i
    total = len(sorteios)
    resultado = []
    for n in range(1, 26):
        idx = ultimo_idx.get(n, -1)
        atraso = total - 1 - idx if idx >= 0 else total
        resultado.append({"Dezena": n, "Último Concurso": idx+1, "Atraso Atual": atraso})
    return pd.DataFrame(resultado).sort_values("Atraso Atual", ascending=False)

def ciclo_medio(sorteios, dezena):
    aparicoes = [i for i, s in enumerate(sorteios) if dezena in s]
    if len(aparicoes) < 2: return None
    diffs = [aparicoes[i+1]-aparicoes[i] for i in range(len(aparicoes)-1)]
    return round(np.mean(diffs), 2)

def perfil_estrutural(jogo):
    jogo = sorted(jogo)
    pares = sum(1 for n in jogo if n%2==0)
    soma  = sum(jogo)
    return {"pares": pares, "impares": 15-pares, "soma": soma,
            "amplitude": max(jogo)-min(jogo), "minimo": min(jogo), "maximo": max(jogo)}

def faixa_soma_historica(sorteios):
    somas = [sum(s) for s in sorteios]
    return int(np.percentile(somas,5)), int(np.percentile(somas,95)), round(np.mean(somas),1)

def distribuicao_pares_historica(sorteios):
    dist = Counter(sum(1 for n in s if n%2==0) for s in sorteios)
    total = len(sorteios)
    return {k: round(v/total*100,1) for k,v in sorted(dist.items())}

def contar_consecutivos(jogo):
    jogo = sorted(jogo)
    max_seq = seq_atual = 1
    pares_consec = 0
    for i in range(1, len(jogo)):
        if jogo[i] == jogo[i-1]+1:
            seq_atual += 1
            max_seq = max(max_seq, seq_atual)
            pares_consec += 1
        else:
            seq_atual = 1
    return max_seq, pares_consec

def dist_consecutivos_historica(sorteios):
    dist = Counter(contar_consecutivos(s)[1] for s in sorteios)
    total = len(sorteios)
    return {k: round(v/total*100,1) for k,v in sorted(dist.items())}

def distribuicao_faixas(jogo):
    f = [0]*5
    for n in jogo: f[(n-1)//5] += 1
    return f

def perfil_moldura_miolo(jogo):
    moldura = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
    m = sum(1 for n in jogo if n in moldura)
    return m, 15-m

def dist_faixas_historica(sorteios):
    dist = [[] for _ in range(5)]
    for s in sorteios:
        f = distribuicao_faixas(s)
        for i in range(5): dist[i].append(f[i])
    return [{"faixa": f"{i*5+1}-{i*5+5}", "media": round(np.mean(dist[i]),2),
             "min": min(dist[i]), "max": max(dist[i])} for i in range(5)]

def composicao_numerica(jogo):
    j = set(jogo)
    return {"primos": len(j&PRIMOS), "fibonacci": len(j&FIBONACCI), "multiplos_3": len(j&MULTIPLOS_3)}

def dist_composicao_historica(sorteios):
    dist_p = Counter(len(set(s)&PRIMOS) for s in sorteios)
    dist_f = Counter(len(set(s)&FIBONACCI) for s in sorteios)
    total = len(sorteios)
    return ({k: round(v/total*100,1) for k,v in sorted(dist_p.items())},
            {k: round(v/total*100,1) for k,v in sorted(dist_f.items())})

def gaps_jogo(jogo):
    jogo = sorted(jogo)
    gaps = [jogo[i+1]-jogo[i] for i in range(len(jogo)-1)]
    return {"gaps": gaps, "media": round(np.mean(gaps),2), "max_gap": max(gaps), "min_gap": min(gaps)}

def dist_gaps_historica(sorteios):
    todos = []
    for s in sorteios: todos.extend(gaps_jogo(s)["gaps"])
    c = Counter(todos)
    total = sum(c.values())
    return {k: round(v/total*100,1) for k,v in sorted(c.items()) if k<=10}

@st.cache_data
def calcular_coocorrencia(sorteios_tuple, top_n=20):
    pares = Counter()
    for s in sorteios_tuple:
        for a,b in combinations(sorted(s),2): pares[(a,b)] += 1
    return pares.most_common(top_n)

@st.cache_data
def calcular_trios_recorrentes(sorteios_tuple, top_n=15):
    trios = Counter()
    for s in sorteios_tuple:
        for trio in combinations(sorted(s),3): trios[trio] += 1
    return trios.most_common(top_n)

def projetar_isca(alvo_idx, sorteios, threshold_percent):
    if not sorteios or alvo_idx >= len(sorteios): return [], [], []
    alvo_set = set(sorteios[alvo_idx])
    threshold_hits = int(len(alvo_set)*(threshold_percent/100.0))
    similares, futuro = [], []
    for i, s in enumerate(sorteios):
        if i == alvo_idx: continue
        if len(alvo_set & set(s)) >= threshold_hits:
            similares.append(i)
            if i+1 < len(sorteios): futuro.extend(sorteios[i+1])
            if i+2 < len(sorteios): futuro.extend(sorteios[i+2])
    freq = Counter(futuro)
    ordenados = freq.most_common()
    return similares, [k for k,_ in ordenados[:6]], [k for k,_ in ordenados[6:12]]

def entropia_shannon(jogo, sorteios):
    freq = freq_simples(sorteios)
    total = sum(freq.values())
    H = 0
    for n in jogo:
        p = freq[n]/total
        if p > 0: H -= p*math.log2(p)
    return round(H, 4)

def teste_qui_quadrado(sorteios, janela=500):
    dados = sorteios[-janela:]
    observado = np.array([sum(1 for s in dados if n in s) for n in range(1,26)])
    esperado  = np.full(25, len(dados)*15/25)
    chi2, p = stats.chisquare(observado, esperado)
    return round(chi2,2), round(p,4)

def monte_carlo_validar(jogo, sorteios, n_sim=5000):
    acertos = [len(set(jogo) & set(random.sample(range(1,26),15))) for _ in range(n_sim)]
    media = np.mean(acertos)
    meu   = np.mean([len(set(jogo)&set(s)) for s in sorteios[-200:]])
    return round(media,2), round(meu,2)

def backtesting_jogo(jogo, sorteios, ultimos=200):
    dados   = sorteios[-ultimos:]
    acertos = [len(set(jogo)&set(s)) for s in dados]
    dist    = Counter(acertos)
    return {"media": round(np.mean(acertos),2), "max": max(acertos), "min": min(acertos),
            "dist": dict(sorted(dist.items())),
            "pct_11+": round(sum(1 for a in acertos if a>=11)/len(acertos)*100,1),
            "pct_13+": round(sum(1 for a in acertos if a>=13)/len(acertos)*100,1)}

def tendencia_dezena(dezena, sorteios, janelas=[20,50,100,200]):
    rows = []
    for j in janelas:
        freq = sum(1 for s in sorteios[-j:] if dezena in s)
        rows.append({"Janela": j, "Freq": freq, "%": round(freq/j*100,1)})
    return pd.DataFrame(rows)

@st.cache_data
def matriz_markov_simples(sorteios_tuple, janela=200):
    sorteios = list(sorteios_tuple)[-janela:]
    trans = {n: Counter() for n in range(1,26)}
    for i in range(len(sorteios)-1):
        for n in set(sorteios[i]): trans[n].update(set(sorteios[i+1]))
    return trans

def dezenas_em_pico_atraso(sorteios, percentil=85):
    df_a = calcular_atrasos(sorteios)
    limite = np.percentile(df_a["Atraso Atual"], percentil)
    return df_a[df_a["Atraso Atual"] >= limite]

def distancia_hamming(j1, j2):
    return 15 - len(set(j1)&set(j2))

def cobertura_entre_jogos(jogos):
    if len(jogos) < 2: return pd.DataFrame()
    result = []
    for i in range(len(jogos)):
        for j in range(i+1, len(jogos)):
            d = distancia_hamming(jogos[i], jogos[j])
            result.append({"Par": f"J{i+1} x J{j+1}", "Hamming": d, "Em comum": 15-d})
    return pd.DataFrame(result)

# ══════════════════════════════════════════════════════════════
# FILTROS AUTO-CALIBRADOS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def calibrar_filtros(sorteios_tuple):
    sorteios = list(sorteios_tuple)
    somas      = [sum(s) for s in sorteios]
    pares_list = [sum(1 for n in s if n%2==0) for s in sorteios]
    cons_list  = [contar_consecutivos(s)[0] for s in sorteios]
    primos_list= [len(set(s)&PRIMOS) for s in sorteios]
    fib_list   = [len(set(s)&FIBONACCI) for s in sorteios]
    faixas_max = [max(distribuicao_faixas(s)) for s in sorteios]
    return {
        "soma_min":      int(np.percentile(somas, 5)),
        "soma_max":      int(np.percentile(somas, 95)),
        "min_pares":     int(np.percentile(pares_list, 5)),
        "max_pares":     int(np.percentile(pares_list, 95)),
        "max_seq":       int(np.percentile(cons_list, 95)),
        "max_primos":    int(np.percentile(primos_list, 95)),
        "max_fibonacci": int(np.percentile(fib_list, 95)),
        "max_por_faixa": int(np.percentile(faixas_max, 95)),
    }

def filtrar_jogo(jogo, sorteios, config):
    alertas = []
    soma = sum(jogo)
    if soma < config["soma_min"] or soma > config["soma_max"]:
        alertas.append(f"Soma {soma} fora [{config['soma_min']}–{config['soma_max']}]")
    max_seq, _ = contar_consecutivos(jogo)
    if max_seq > config["max_seq"]:
        alertas.append(f"Sequência {max_seq} > {config['max_seq']}")
    pares = sum(1 for n in jogo if n%2==0)
    if pares < config["min_pares"] or pares > config["max_pares"]:
        alertas.append(f"{pares} pares fora [{config['min_pares']}–{config['max_pares']}]")
    faixas = distribuicao_faixas(jogo)
    for i, f in enumerate(faixas):
        if f == 0: alertas.append(f"Faixa {i*5+1}-{i*5+5} vazia")
        if f > config["max_por_faixa"]: alertas.append(f"Faixa {i*5+1}-{i*5+5} excesso ({f})")
    comp = composicao_numerica(jogo)
    if comp["primos"]    > config["max_primos"]:    alertas.append(f"{comp['primos']} primos > {config['max_primos']}")
    if comp["fibonacci"] > config["max_fibonacci"]: alertas.append(f"{comp['fibonacci']} Fibonacci > {config['max_fibonacci']}")
    return alertas

# ══════════════════════════════════════════════════════════════
# MOTOR DE SCORE PONDERADO
# ══════════════════════════════════════════════════════════════
def normalizar(dct):
    vals = list(dct.values())
    mn, mx = min(vals), max(vals)
    if mx == mn: return {k: 0.5 for k in dct}
    return {k: (v-mn)/(mx-mn) for k,v in dct.items()}

def calcular_scores_completo(sorteios, janela, concurso_ref, threshold_sim, pesos, analise):
    """
    Usa os resultados JÁ CALCULADOS nas abas de análise (session_state)
    para compor o score de cada dezena — as análises conversam com o gerador.
    """
    scores = {n: 0.0 for n in range(1, 26)}

    # ── G1: Comportamento recente — usa resultados das abas Térmica + N+1/N+2 + Rolling
    g1_n12  = {n: 0.0 for n in range(1,26)}
    fortes_n, apoio_n, sim = [], [], []
    if analise.get("n12"):
        fortes_n = analise["n12"]["fortes"]
        apoio_n  = analise["n12"]["apoio"]
        sim      = analise["n12"]["similares"]
        for n in fortes_n: g1_n12[n] = 1.0
        for n in apoio_n:  g1_n12[n] = max(g1_n12[n], 0.6)

    f50    = freq_simples(sorteios, 50)
    f_long = freq_simples(sorteios, janela)

    # Usa quentes da aba Térmica se disponível
    if analise.get("termica"):
        quentes = analise["termica"]["quentes"]
        for i, n in enumerate(quentes):
            g1_n12[n] = max(g1_n12[n], 1.0 - i*0.05)

    g1_freq = normalizar({n: f50[n] for n in range(1,26)})
    g1_roll = {}
    for n in range(1,26):
        t50   = f50[n]/50
        tlong = f_long[n]/janela
        g1_roll[n] = max(0.0, (t50-tlong)/(tlong+1e-9))
    g1_roll = normalizar(g1_roll)
    g1 = {n: (g1_n12[n]*0.50 + g1_freq[n]*0.30 + g1_roll[n]*0.20) for n in range(1,26)}

    # ── G2: Ciclo e atraso — usa resultados da aba Atraso & Ciclo
    if analise.get("atraso"):
        df_at = analise["atraso"]["df_completo"]
        g2_raw = {}
        for _, row in df_at.iterrows():
            n = int(row["Dezena"])
            atraso = row["Atraso Atual"]
            cm     = row["Ciclo Médio"] or atraso
            g2_raw[n] = min(atraso/cm if cm > 0 else 1.0, 3.0)
    else:
        total = len(sorteios)
        ultimo_idx = {}
        for i, s in enumerate(sorteios):
            for x in s: ultimo_idx[x] = i
        g2_raw = {}
        for n in range(1,26):
            idx    = ultimo_idx.get(n,-1)
            atraso = total-1-idx if idx>=0 else total
            cm     = ciclo_medio(sorteios, n) or atraso
            g2_raw[n] = min(atraso/cm if cm>0 else 1.0, 3.0)
    g2 = normalizar(g2_raw)

    # ── G3: Padrão histórico — usa resultados da aba Coocorrência
    if analise.get("coocorrencia"):
        g3_cooc  = analise["coocorrencia"]["g3_cooc"]
        g3_markov= analise["coocorrencia"]["g3_markov"]
        g3_trios = analise["coocorrencia"]["g3_trios"]
    else:
        pares_freq = Counter()
        for s in sorteios[-500:]:
            for a,b in combinations(sorted(s),2): pares_freq[(a,b)] += 1
        g3_cooc = {n: 0.0 for n in range(1,26)}
        for (a,b), cnt in pares_freq.items():
            g3_cooc[a] += cnt; g3_cooc[b] += cnt
        g3_cooc = normalizar(g3_cooc)
        trans = matriz_markov_simples(tuple(tuple(s) for s in sorteios), janela=min(janela,500))
        ultimo_s = set(sorteios[concurso_ref])
        g3_markov = {n: 0.0 for n in range(1,26)}
        for x in ultimo_s:
            if x in trans:
                for suc,cnt in trans[x].items(): g3_markov[suc] = g3_markov.get(suc,0)+cnt
        g3_markov = normalizar(g3_markov)
        trios_freq = Counter()
        for s in sorteios[-500:]:
            for trio in combinations(sorted(s),3): trios_freq[trio] += 1
        g3_trios = {n: 0.0 for n in range(1,26)}
        for trio,cnt in trios_freq.most_common(100):
            for x in trio: g3_trios[x] += cnt
        g3_trios = normalizar(g3_trios)
    g3 = {n: (g3_cooc[n]*0.40 + g3_markov[n]*0.40 + g3_trios[n]*0.20) for n in range(1,26)}

    # ── G4: Estrutural — usa faixas da aba Estrutural
    g4 = {n: 0.5 for n in range(1,26)}
    if analise.get("estrutural"):
        soma_med = analise["estrutural"]["soma_med"]
        # Dezenas que historicamente participam de jogos com soma próxima da média recebem bônus
        for n in range(1,26):
            g4[n] = 1.0 - abs(n - soma_med/15) / 25
        g4 = normalizar(g4)

    # ── G5: Composição — usa dados da aba Composição
    freq_all  = freq_simples(sorteios)
    total_all = sum(freq_all.values())
    esperado  = 1/25
    g5 = normalizar({n: 1-abs(freq_all[n]/total_all - esperado)/esperado for n in range(1,26)})

    # ── Score final ponderado
    for n in range(1,26):
        scores[n] = (pesos["g1"]*g1[n] + pesos["g2"]*g2[n] +
                     pesos["g3"]*g3[n] + pesos["g4"]*g4[n] + pesos["g5"]*g5[n])

    return scores, sim, fortes_n, apoio_n

# ══════════════════════════════════════════════════════════════
# PERFIS
# ══════════════════════════════════════════════════════════════
PERFIS = {
    "🎯 Agressiva": {
        "pesos":       {"g1":0.40,"g2":0.30,"g3":0.20,"g4":0.05,"g5":0.05},
        "top_pool":    15,
        "max_comum":   8,
        "descricao":   "Concentra nas dezenas top rankeadas. Ideal para poucos jogos com força máxima.",
        "aviso_acima": 10,
    },
    "⚖️ Híbrida": {
        "pesos":       {"g1":0.30,"g2":0.25,"g3":0.25,"g4":0.10,"g5":0.10},
        "top_pool":    20,
        "max_comum":   11,
        "descricao":   "Equilibra força e cobertura. Ideal para bolões e concursos especiais.",
        "aviso_acima": None,
    },
}

PREMIOS = {15:1_500_000, 14:1500, 13:25, 12:10, 11:5}
PROB    = {15:1/3_268_760, 14:15/3_268_760, 13:105/3_268_760, 12:455/3_268_760, 11:1365/3_268_760}

def valor_esperado(custo=3.5):
    ev = sum(PREMIOS[k]*PROB[k] for k in PREMIOS)
    return round(ev,4), round(ev/custo,4)

# ══════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════
df       = carregar_base()
sorteios = extrair_sorteios(df)
sorteios_tuple = tuple(tuple(s) for s in sorteios)
config_filtro  = calibrar_filtros(sorteios_tuple)

st.sidebar.title("⚙️ Protocolo Hard V4")
st.sidebar.markdown(f"**Base:** {len(sorteios)} concursos | C1–C{df['Concurso'].max()}")

# Status das etapas na sidebar
ok_count, total_count = status_etapas()
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Status das Etapas")
for e in ETAPAS:
    icon = "✅" if etapa_ok(e) else "⭕"
    label = {"termica":"Térmica","atraso":"Atraso & Ciclo","estrutural":"Estrutural",
             "composicao":"Composição","sequencias":"Sequências & Gaps",
             "n12":"Similaridade N+1/N+2","coocorrencia":"Coocorrência",
             "monte_carlo":"Monte Carlo"}[e]
    st.sidebar.write(f"{icon} {label}")

if todas_etapas_ok():
    st.sidebar.success("🎯 Todas etapas OK — Gerador liberado!")
else:
    st.sidebar.warning(f"⏳ {ok_count}/{total_count} etapas concluídas")

st.sidebar.markdown("---")
with st.sidebar.expander("📐 Filtros Calibrados"):
    st.write(f"Soma: {config_filtro['soma_min']} – {config_filtro['soma_max']}")
    st.write(f"Pares: {config_filtro['min_pares']} – {config_filtro['max_pares']}")
    st.write(f"Máx. consecutivos: {config_filtro['max_seq']}")
    st.write(f"Máx. por faixa: {config_filtro['max_por_faixa']}")
    st.write(f"Máx. primos: {config_filtro['max_primos']}")
    st.write(f"Máx. Fibonacci: {config_filtro['max_fibonacci']}")

janela_analise = st.sidebar.slider("Janela de Análise", 50, len(sorteios), 500, step=50)
threshold_sim  = st.sidebar.slider("Threshold N+1/N+2 (%)", 50, 100, 80)
concurso_ref   = st.sidebar.number_input("Índice Referência (0=último)", 0, len(sorteios)-1, len(sorteios)-1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Meus Jogos (Backtesting)")
jogo1_input = st.sidebar.text_input("Jogo 1 (15 números separados por espaço)")
jogo2_input = st.sidebar.text_input("Jogo 2")
jogo3_input = st.sidebar.text_input("Jogo 3")

def parse_jogo(s):
    try:
        nums = [int(x) for x in s.strip().split()]
        if len(nums)==15 and all(1<=n<=25 for n in nums): return sorted(nums)
    except: pass
    return None

jogos_usuario  = [j for j in [parse_jogo(jogo1_input), parse_jogo(jogo2_input), parse_jogo(jogo3_input)] if j]
sorteios_janela = sorteios[-janela_analise:]

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
st.title("📊 Protocolo Hard V4 — 137+ Análises | Lotofácil")
st.markdown(f"Base: **{len(sorteios)} concursos** | C{df['Concurso'].min()}–C{df['Concurso'].max()} | Janela: **{janela_analise}**")

# Barra de progresso geral
prog_geral = ok_count / total_count
st.progress(prog_geral, text=f"Protocolo: {ok_count}/{total_count} etapas concluídas {'🎯 Pronto para gerar!' if todas_etapas_ok() else '— Execute as abas abaixo'}")

tabs = st.tabs([
    "🌡️ Térmica", "⏱️ Atraso & Ciclo", "📐 Estrutural",
    "🔢 Composição", "📏 Sequências & Gaps", "🎯 N+1/N+2",
    "🔗 Coocorrência", "🎲 Monte Carlo", "📊 Backtesting",
    "🧪 Filtros & Score", "📈 Tendência", "⚖️ Valor Esperado",
    "🎰 Gerar Jogos", "🏆 Relatório Final"
])

# ──────────────────────────────────────────────────────────────
# ABA 0 — TÉRMICA
# ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("🌡️ Camada Térmica")
    df_freq = tabela_frequencia(sorteios)
    st.dataframe(df_freq, use_container_width=True, hide_index=True)

    f = freq_simples(sorteios, janela_analise)
    top10  = [n for n,_ in f.most_common(10)]
    top5   = top10[:5]
    bot5   = [n for n,_ in f.most_common()[:-6:-1]]
    pico   = dezenas_em_pico_atraso(sorteios)["Dezena"].tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🔥 Top 5 Quentes: {' - '.join(str(n).zfill(2) for n in top5)}")
        st.info(f"🧊 Top 5 Frias: {' - '.join(str(n).zfill(2) for n in bot5)}")
    with col2:
        st.warning("⚠️ Em Pico de Atraso")
        st.dataframe(dezenas_em_pico_atraso(sorteios), use_container_width=True, hide_index=True)

    if st.button("✅ Confirmar análise Térmica", key="btn_termica"):
        salvar_etapa("termica", {
            "quentes": top10, "frias": bot5, "pico_atraso": pico,
            "freq": {n: f[n] for n in range(1,26)}
        })
        st.success("Etapa Térmica registrada! ✅")
        st.rerun()

    if etapa_ok("termica"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 1 — ATRASO & CICLO
# ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("⏱️ Atraso & Ciclo")
    df_atraso = calcular_atrasos(sorteios)
    ciclos = [{"Dezena": n, "Ciclo Médio": ciclo_medio(sorteios, n)} for n in range(1,26)]
    df_ciclo = pd.DataFrame(ciclos)
    df_completo = df_atraso.merge(df_ciclo, on="Dezena")
    df_completo["Status"] = df_completo.apply(
        lambda r: "🔴 VENCIDA" if r["Atraso Atual"] > (r["Ciclo Médio"] or 99)*1.5
        else ("🟡 No ciclo" if r["Atraso Atual"] > (r["Ciclo Médio"] or 99)*0.8 else "🟢 Ok"), axis=1)
    st.dataframe(df_completo.sort_values("Atraso Atual", ascending=False), use_container_width=True, hide_index=True)

    vencidas = df_completo[df_completo["Status"]=="🔴 VENCIDA"]["Dezena"].tolist()
    no_ciclo = df_completo[df_completo["Status"]=="🟡 No ciclo"]["Dezena"].tolist()
    st.write(f"🔴 Vencidas: **{' '.join(str(n).zfill(2) for n in vencidas)}**")
    st.write(f"🟡 No ciclo: **{' '.join(str(n).zfill(2) for n in no_ciclo)}**")

    if st.button("✅ Confirmar análise Atraso & Ciclo", key="btn_atraso"):
        salvar_etapa("atraso", {
            "df_completo": df_completo,
            "vencidas": vencidas,
            "no_ciclo": no_ciclo
        })
        st.success("Etapa Atraso & Ciclo registrada! ✅")
        st.rerun()

    if etapa_ok("atraso"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 2 — ESTRUTURAL
# ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("📐 Estrutural")
    soma_min, soma_max, soma_med = faixa_soma_historica(sorteios_janela)
    st.info(f"Soma histórica (p5–p95): **{soma_min} – {soma_max}** | Média: **{soma_med}**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição de Pares")
        dist_p = distribuicao_pares_historica(sorteios_janela)
        st.dataframe(pd.DataFrame(list(dist_p.items()), columns=["Pares","%"]), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Distribuição por Faixas")
        st.dataframe(pd.DataFrame(dist_faixas_historica(sorteios_janela)), use_container_width=True, hide_index=True)

    molduras = [perfil_moldura_miolo(s) for s in sorteios_janela[-50:]]
    m_vals = [m[0] for m in molduras]
    st.write(f"**Moldura** — Média: {round(np.mean(m_vals),1)} | Min: {min(m_vals)} | Max: {max(m_vals)}")

    if st.button("✅ Confirmar análise Estrutural", key="btn_estrutural"):
        salvar_etapa("estrutural", {
            "soma_min": soma_min, "soma_max": soma_max, "soma_med": soma_med,
            "dist_pares": dist_p,
            "faixas": dist_faixas_historica(sorteios_janela)
        })
        st.success("Etapa Estrutural registrada! ✅")
        st.rerun()

    if etapa_ok("estrutural"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 3 — COMPOSIÇÃO
# ──────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("🔢 Composição: Primos, Fibonacci, Múltiplos")
    dist_pr, dist_fi = dist_composicao_historica(sorteios_janela)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Primos (%)")
        st.dataframe(pd.DataFrame(list(dist_pr.items()), columns=["Qtd","%"]), hide_index=True)
        st.caption(f"Primos: {sorted(PRIMOS)}")
    with col2:
        st.subheader("Fibonacci (%)")
        st.dataframe(pd.DataFrame(list(dist_fi.items()), columns=["Qtd","%"]), hide_index=True)
        st.caption(f"Fibonacci: {sorted(FIBONACCI)}")
    with col3:
        st.subheader("Múltiplos de 3 (%)")
        dist_m3 = Counter(len(set(s)&MULTIPLOS_3) for s in sorteios_janela)
        total_j = len(sorteios_janela)
        st.dataframe(pd.DataFrame([(k,round(v/total_j*100,1)) for k,v in sorted(dist_m3.items())], columns=["Qtd","%"]), hide_index=True)

    top_prim = max(dist_pr, key=dist_pr.get)
    top_fib  = max(dist_fi, key=dist_fi.get)

    if st.button("✅ Confirmar análise Composição", key="btn_composicao"):
        salvar_etapa("composicao", {
            "dist_primos": dist_pr, "dist_fibonacci": dist_fi,
            "top_primos": top_prim, "top_fibonacci": top_fib
        })
        st.success("Etapa Composição registrada! ✅")
        st.rerun()

    if etapa_ok("composicao"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 4 — SEQUÊNCIAS & GAPS
# ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("📏 Sequências & Gaps")
    col1, col2 = st.columns(2)
    dist_cons = dist_consecutivos_historica(sorteios_janela)
    dist_g    = dist_gaps_historica(sorteios_janela)
    with col1:
        st.subheader("Consecutivos (%)")
        st.dataframe(pd.DataFrame(list(dist_cons.items()), columns=["Pares Consec.","%"]), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Gaps (%)")
        st.dataframe(pd.DataFrame(list(dist_g.items()), columns=["Gap","%"]), use_container_width=True, hide_index=True)

    top_cons = max(dist_cons, key=dist_cons.get)
    top_gap  = max(dist_g, key=dist_g.get)

    if st.button("✅ Confirmar análise Sequências & Gaps", key="btn_sequencias"):
        salvar_etapa("sequencias", {
            "dist_consecutivos": dist_cons, "dist_gaps": dist_g,
            "top_consecutivos": top_cons, "top_gap": top_gap
        })
        st.success("Etapa Sequências registrada! ✅")
        st.rerun()

    if etapa_ok("sequencias"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 5 — N+1/N+2
# ──────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("🎯 Similaridade N+1/N+2")
    similares, fortes, apoio = projetar_isca(concurso_ref, sorteios, threshold_sim)
    concurso_nome = df.iloc[concurso_ref]["Concurso"] if concurso_ref < len(df) else "?"
    st.write(f"Referência: **C{concurso_nome}** | Threshold: **{threshold_sim}%**")

    if similares:
        st.success(f"✅ {len(similares)} concursos similares encontrados")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**🔥 Candidatas Fortes:**")
            st.success(" · ".join(str(n).zfill(2) for n in sorted(fortes)))
        with col2:
            st.write("**🤝 Apoio:**")
            st.info(" · ".join(str(n).zfill(2) for n in sorted(apoio)))
        dados_isca = pd.DataFrame({
            "Categoria": ["Candidatas Fortes","Apoio","Frias em Pico"],
            "Dezenas": [
                " ".join(str(n).zfill(2) for n in sorted(fortes)),
                " ".join(str(n).zfill(2) for n in sorted(apoio)),
                " ".join(str(n).zfill(2) for n in dezenas_em_pico_atraso(sorteios)["Dezena"].tolist())
            ],
            "Ação": ["Base principal","Completar espaços","Ruptura de tendência"]
        })
        st.dataframe(dados_isca, use_container_width=True, hide_index=True)

        if st.button("✅ Confirmar análise N+1/N+2", key="btn_n12"):
            salvar_etapa("n12", {
                "similares": similares, "fortes": fortes, "apoio": apoio,
                "concurso_ref": int(concurso_nome), "threshold": threshold_sim
            })
            st.success("Etapa N+1/N+2 registrada! ✅")
            st.rerun()
    else:
        st.warning("Nenhum similar encontrado. Reduza o threshold.")

    if etapa_ok("n12"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 6 — COOCORRÊNCIA
# ──────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("🔗 Coocorrência & Trios")
    col1, col2 = st.columns(2)
    pares_cooc = calcular_coocorrencia(sorteios_tuple)
    trios_cooc = calcular_trios_recorrentes(sorteios_tuple)
    with col1:
        st.subheader("Top 20 Pares")
        st.dataframe(pd.DataFrame([(f"{a:02d}+{b:02d}",c) for (a,b),c in pares_cooc], columns=["Par","Freq"]), use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Top 15 Trios")
        st.dataframe(pd.DataFrame([(f"{a:02d}+{b:02d}+{c:02d}",cnt) for (a,b,c),cnt in trios_cooc], columns=["Trio","Freq"]), use_container_width=True, hide_index=True)

    st.subheader("Markov — Sucessores")
    dezena_markov = st.selectbox("Dezena:", range(1,26))
    trans = matriz_markov_simples(sorteios_tuple)
    st.dataframe(pd.DataFrame(trans[dezena_markov].most_common(10), columns=["Sucessor","Freq"]), use_container_width=True, hide_index=True)

    # Calcula os vetores para o score
    pares_freq = Counter()
    for s in sorteios[-500:]:
        for a,b in combinations(sorted(s),2): pares_freq[(a,b)] += 1
    g3_cooc = {n: 0.0 for n in range(1,26)}
    for (a,b),cnt in pares_freq.items(): g3_cooc[a]+=cnt; g3_cooc[b]+=cnt
    g3_cooc = normalizar(g3_cooc)

    ultimo_s = set(sorteios[concurso_ref])
    g3_markov = {n: 0.0 for n in range(1,26)}
    for x in ultimo_s:
        if x in trans:
            for suc,cnt in trans[x].items(): g3_markov[suc]=g3_markov.get(suc,0)+cnt
    g3_markov = normalizar(g3_markov)

    trios_freq = Counter()
    for s in sorteios[-500:]:
        for trio in combinations(sorted(s),3): trios_freq[trio]+=1
    g3_trios = {n: 0.0 for n in range(1,26)}
    for trio,cnt in trios_freq.most_common(100):
        for x in trio: g3_trios[x]+=cnt
    g3_trios = normalizar(g3_trios)

    if st.button("✅ Confirmar análise Coocorrência", key="btn_cooc"):
        salvar_etapa("coocorrencia", {
            "g3_cooc": g3_cooc, "g3_markov": g3_markov, "g3_trios": g3_trios,
            "pares_top": pares_cooc[:10], "trios_top": trios_cooc[:5]
        })
        st.success("Etapa Coocorrência registrada! ✅")
        st.rerun()

    if etapa_ok("coocorrencia"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 7 — MONTE CARLO
# ──────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("🎲 Monte Carlo & Qui-Quadrado")
    chi2, p = teste_qui_quadrado(sorteios, janela=janela_analise)
    st.write(f"**Qui-Quadrado:** χ²={chi2} | p={p}")
    if p > 0.05:
        st.success("✅ Base sem desvio estatístico significativo")
    else:
        st.warning("⚠️ Desvio detectado na base")

    if jogos_usuario:
        for i, jogo in enumerate(jogos_usuario):
            media_aleat, media_hist = monte_carlo_validar(jogo, sorteios)
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            col1, col2 = st.columns(2)
            col1.metric("Média vs aleatório", media_aleat)
            col2.metric("Média vs histórico", media_hist)
            delta = round(media_hist - media_aleat, 2)
            if delta > 0:
                st.success(f"✅ +{delta} acertos acima do aleatório")
            else:
                st.warning(f"⚠️ {delta} acertos vs aleatório")
    else:
        st.info("Insira jogos no menu lateral para validação Monte Carlo.")

    if st.button("✅ Confirmar análise Monte Carlo", key="btn_mc"):
        salvar_etapa("monte_carlo", {
            "chi2": chi2, "p_valor": p,
            "base_ok": p > 0.05
        })
        st.success("Etapa Monte Carlo registrada! ✅")
        st.rerun()

    if etapa_ok("monte_carlo"):
        st.info("✅ Etapa já concluída — dados salvos no protocolo.")

# ──────────────────────────────────────────────────────────────
# ABA 8 — BACKTESTING
# ──────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("📊 Backtesting")
    if jogos_usuario:
        for i, jogo in enumerate(jogos_usuario):
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            bt = backtesting_jogo(jogo, sorteios, ultimos=200)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Média", bt["media"])
            col2.metric("11+", f"{bt['pct_11+']}%")
            col3.metric("13+", f"{bt['pct_13+']}%")
            col4.metric("Máx", bt["max"])
            st.bar_chart(pd.DataFrame(list(bt["dist"].items()), columns=["Acertos","Freq"]).set_index("Acertos"))
            st.markdown("---")
    else:
        st.info("Insira seus jogos no menu lateral.")

# ──────────────────────────────────────────────────────────────
# ABA 9 — FILTROS & SCORE
# ──────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("🧪 Filtros & Score")
    st.info(f"Filtros calibrados: Soma {config_filtro['soma_min']}–{config_filtro['soma_max']} | Pares {config_filtro['min_pares']}–{config_filtro['max_pares']} | Máx seq {config_filtro['max_seq']}")
    if jogos_usuario:
        for i, jogo in enumerate(jogos_usuario):
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            alertas = filtrar_jogo(jogo, sorteios, config_filtro)
            perf = perfil_estrutural(jogo)
            comp = composicao_numerica(jogo)
            gaps = gaps_jogo(jogo)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Soma:** {perf['soma']} | **Amplitude:** {perf['amplitude']}")
                st.write(f"**Pares:** {perf['pares']} | **Ímpares:** {perf['impares']}")
                max_seq_j, _ = contar_consecutivos(jogo)
                st.write(f"**Maior seq.:** {max_seq_j}")
            with col2:
                st.write(f"**Primos:** {comp['primos']} | **Fib:** {comp['fibonacci']} | **M3:** {comp['multiplos_3']}")
                st.write(f"**Faixas:** {distribuicao_faixas(jogo)}")
                mold, miolo = perfil_moldura_miolo(jogo)
                st.write(f"**Moldura:** {mold} | **Miolo:** {miolo}")
            with col3:
                st.write(f"**Gap médio:** {gaps['media']} | **Max gap:** {gaps['max_gap']}")
                H = entropia_shannon(jogo, sorteios)
                st.write(f"**Entropia:** {H}")
            if alertas:
                for a in alertas: st.warning(a)
            else:
                st.success("✅ Passou em todos os filtros!")
            st.markdown("---")
        if len(jogos_usuario) > 1:
            st.subheader("📐 Hamming entre Jogos")
            st.dataframe(cobertura_entre_jogos(jogos_usuario), use_container_width=True, hide_index=True)
    else:
        st.info("Insira seus jogos no menu lateral.")

# ──────────────────────────────────────────────────────────────
# ABA 10 — TENDÊNCIA ROLLING
# ──────────────────────────────────────────────────────────────
with tabs[10]:
    st.header("📈 Tendência Rolling")
    dezena_tend = st.selectbox("Dezena:", range(1,26), key="tend")
    df_tend = tendencia_dezena(dezena_tend, sorteios)
    st.dataframe(df_tend, use_container_width=True, hide_index=True)
    taxa_50  = df_tend[df_tend["Janela"]==50]["Freq"].values[0] / 50
    taxa_200 = df_tend[df_tend["Janela"]==200]["Freq"].values[0] / 200
    if taxa_50 > taxa_200*1.2:
        st.success(f"🔼 Dezena {dezena_tend:02d} em ALTA")
    elif taxa_50 < taxa_200*0.8:
        st.warning(f"🔽 Dezena {dezena_tend:02d} em BAIXA")
    else:
        st.info(f"➡️ Dezena {dezena_tend:02d} ESTÁVEL")

# ──────────────────────────────────────────────────────────────
# ABA 11 — VALOR ESPERADO
# ──────────────────────────────────────────────────────────────
with tabs[11]:
    st.header("⚖️ Valor Esperado")
    ev, ev_ratio = valor_esperado()
    col1, col2, col3 = st.columns(3)
    col1.metric("EV por jogo", f"R$ {ev:.4f}")
    col2.metric("Custo", "R$ 3,50")
    col3.metric("Ratio EV/Custo", f"{ev_ratio:.4f}")
    df_ev = pd.DataFrame([
        {"Faixa":k,"Prob":f"1 em {int(1/v):,}","Prêmio":f"R$ {PREMIOS[k]:,.0f}","EV":f"R$ {PREMIOS[k]*v:.4f}"}
        for k,v in sorted(PROB.items(), reverse=True)
    ])
    st.dataframe(df_ev, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# ABA 12 — GERADOR (só libera quando todas etapas OK)
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    st.header("🎰 Gerador — Protocolo Hard V4")

    ok_count, total_count = status_etapas()

    if not todas_etapas_ok():
        st.error(f"⛔ Gerador bloqueado — {total_count - ok_count} etapa(s) pendente(s).")
        st.markdown("**Complete as etapas abaixo antes de gerar:**")
        for e in ETAPAS:
            icon = "✅" if etapa_ok(e) else "❌"
            label = {"termica":"Térmica","atraso":"Atraso & Ciclo","estrutural":"Estrutural",
                     "composicao":"Composição","sequencias":"Sequências & Gaps",
                     "n12":"Similaridade N+1/N+2","coocorrencia":"Coocorrência",
                     "monte_carlo":"Monte Carlo"}[e]
            st.write(f"{icon} {label}")

        if st.button("🔄 Executar todas as etapas automaticamente"):
            with st.spinner("Executando todas as análises..."):
                # Térmica
                f_auto = freq_simples(sorteios, janela_analise)
                top10_auto = [n for n,_ in f_auto.most_common(10)]
                salvar_etapa("termica", {"quentes": top10_auto, "frias": [n for n,_ in f_auto.most_common()[:-6:-1]],
                                          "pico_atraso": dezenas_em_pico_atraso(sorteios)["Dezena"].tolist(),
                                          "freq": {n: f_auto[n] for n in range(1,26)}})
                # Atraso
                df_at_auto = calcular_atrasos(sorteios)
                ciclos_auto = [{"Dezena": n, "Ciclo Médio": ciclo_medio(sorteios,n)} for n in range(1,26)]
                df_c_auto = df_at_auto.merge(pd.DataFrame(ciclos_auto), on="Dezena")
                df_c_auto["Status"] = df_c_auto.apply(
                    lambda r: "🔴 VENCIDA" if r["Atraso Atual"]>(r["Ciclo Médio"] or 99)*1.5
                    else ("🟡 No ciclo" if r["Atraso Atual"]>(r["Ciclo Médio"] or 99)*0.8 else "🟢 Ok"), axis=1)
                salvar_etapa("atraso", {"df_completo": df_c_auto,
                                         "vencidas": df_c_auto[df_c_auto["Status"]=="🔴 VENCIDA"]["Dezena"].tolist(),
                                         "no_ciclo": df_c_auto[df_c_auto["Status"]=="🟡 No ciclo"]["Dezena"].tolist()})
                # Estrutural
                sm, sx, smed = faixa_soma_historica(sorteios_janela)
                salvar_etapa("estrutural", {"soma_min":sm,"soma_max":sx,"soma_med":smed,
                                             "dist_pares": distribuicao_pares_historica(sorteios_janela),
                                             "faixas": dist_faixas_historica(sorteios_janela)})
                # Composição
                dp, df_fi = dist_composicao_historica(sorteios_janela)
                salvar_etapa("composicao", {"dist_primos":dp,"dist_fibonacci":df_fi,
                                             "top_primos":max(dp,key=dp.get),"top_fibonacci":max(df_fi,key=df_fi.get)})
                # Sequências
                dc = dist_consecutivos_historica(sorteios_janela)
                dg = dist_gaps_historica(sorteios_janela)
                salvar_etapa("sequencias", {"dist_consecutivos":dc,"dist_gaps":dg,
                                             "top_consecutivos":max(dc,key=dc.get),"top_gap":max(dg,key=dg.get)})
                # N+1/N+2
                sim_auto, fortes_auto, apoio_auto = projetar_isca(concurso_ref, sorteios, threshold_sim)
                salvar_etapa("n12", {"similares":sim_auto,"fortes":fortes_auto,"apoio":apoio_auto,
                                      "concurso_ref":int(df.iloc[concurso_ref]["Concurso"]),"threshold":threshold_sim})
                # Coocorrência
                pf_auto = Counter()
                for s in sorteios[-500:]:
                    for a,b in combinations(sorted(s),2): pf_auto[(a,b)]+=1
                g3c = {n:0.0 for n in range(1,26)}
                for (a,b),cnt in pf_auto.items(): g3c[a]+=cnt; g3c[b]+=cnt
                g3c = normalizar(g3c)
                tr_auto = matriz_markov_simples(sorteios_tuple)
                ult_s = set(sorteios[concurso_ref])
                g3m = {n:0.0 for n in range(1,26)}
                for x in ult_s:
                    if x in tr_auto:
                        for suc,cnt in tr_auto[x].items(): g3m[suc]=g3m.get(suc,0)+cnt
                g3m = normalizar(g3m)
                tf_auto = Counter()
                for s in sorteios[-500:]:
                    for trio in combinations(sorted(s),3): tf_auto[trio]+=1
                g3t = {n:0.0 for n in range(1,26)}
                for trio,cnt in tf_auto.most_common(100):
                    for x in trio: g3t[x]+=cnt
                g3t = normalizar(g3t)
                salvar_etapa("coocorrencia", {"g3_cooc":g3c,"g3_markov":g3m,"g3_trios":g3t,
                                               "pares_top":pf_auto.most_common(10),"trios_top":tf_auto.most_common(5)})
                # Monte Carlo
                chi2_auto, p_auto = teste_qui_quadrado(sorteios, janela=janela_analise)
                salvar_etapa("monte_carlo", {"chi2":chi2_auto,"p_valor":p_auto,"base_ok":p_auto>0.05})

            st.success("✅ Todas as etapas executadas!")
            st.rerun()

    else:
        # ── GERADOR LIBERADO ──────────────────────────────────
        analise = st.session_state["analise"]
        st.success("🎯 Todas as etapas concluídas — Gerador ativo!")

        # Resumo do que foi capturado
        with st.expander("📋 Resumo das análises que alimentam este gerador"):
            col1, col2 = st.columns(2)
            with col1:
                if analise["termica"]:
                    st.write(f"🌡️ Quentes top 10: **{' '.join(str(n).zfill(2) for n in analise['termica']['quentes'])}**")
                if analise["atraso"]:
                    st.write(f"⏱️ Vencidas: **{' '.join(str(n).zfill(2) for n in analise['atraso']['vencidas'])}**")
                if analise["n12"]:
                    st.write(f"🎯 N+1/N+2 Fortes: **{' '.join(str(n).zfill(2) for n in sorted(analise['n12']['fortes']))}**")
                    st.write(f"🤝 N+1/N+2 Apoio: **{' '.join(str(n).zfill(2) for n in sorted(analise['n12']['apoio']))}**")
            with col2:
                if analise["estrutural"]:
                    st.write(f"📐 Soma: {analise['estrutural']['soma_min']}–{analise['estrutural']['soma_max']} (med {analise['estrutural']['soma_med']})")
                if analise["composicao"]:
                    st.write(f"🔢 Primos mais freq: **{analise['composicao']['top_primos']}** | Fib: **{analise['composicao']['top_fibonacci']}**")
                if analise["monte_carlo"]:
                    status_base = "✅ OK" if analise["monte_carlo"]["base_ok"] else "⚠️ Desvio"
                    st.write(f"🎲 Base estatística: **{status_base}** (χ²={analise['monte_carlo']['chi2']} p={analise['monte_carlo']['p_valor']})")

        st.markdown("---")

        # Seletor de perfil
        perfil_nome = st.radio("Perfil de combinação:", list(PERFIS.keys()), horizontal=True)
        perfil = PERFIS[perfil_nome]
        st.info(f"**{perfil_nome}** — {perfil['descricao']}")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- Pool base: top **{perfil['top_pool']}** dezenas | Similaridade máx: **{perfil['max_comum']}** em comum")
            st.write(f"- Pesos: G1={perfil['pesos']['g1']*100:.0f}% | G2={perfil['pesos']['g2']*100:.0f}% | G3={perfil['pesos']['g3']*100:.0f}%")
        with col2:
            st.write(f"- Soma: {config_filtro['soma_min']}–{config_filtro['soma_max']} | Pares: {config_filtro['min_pares']}–{config_filtro['max_pares']}")
            st.write(f"- Máx consecutivos: {config_filtro['max_seq']} | Máx por faixa: {config_filtro['max_por_faixa']}")

        st.markdown("---")

        col3, col4 = st.columns(2)
        with col3:
            qtd_jogos = st.number_input("Quantos jogos gerar?", min_value=1, value=3, step=1)
            custo_prev = qtd_jogos * 3.5
            st.warning(f"💰 Custo estimado: **{qtd_jogos} × R$ 3,50 = R$ {custo_prev:.2f}**")
        with col4:
            seed_val    = st.number_input("Seed (0 = aleatório)", min_value=0, value=0, step=1)
            fixas_input = st.text_input("Dezenas fixas (opcional):", placeholder="Ex: 05 13 22")

        if perfil["aviso_acima"] and qtd_jogos > perfil["aviso_acima"]:
            st.warning(f"⚠️ Agressiva com {qtd_jogos} jogos — pool vai expandir automaticamente.")

        fixas = []
        if fixas_input.strip():
            try:
                fixas = [int(x) for x in fixas_input.strip().split() if 1<=int(x)<=25]
            except:
                st.warning("Formato inválido.")

        with st.expander("📊 Ranking de score das dezenas"):
            if st.session_state["scores_cache"] is None:
                with st.spinner("Calculando scores com base nas análises..."):
                    scores_prev, _, _, _ = calcular_scores_completo(
                        sorteios, janela_analise, concurso_ref, threshold_sim, perfil["pesos"], analise)
                    st.session_state["scores_cache"] = scores_prev
            scores_prev = st.session_state["scores_cache"]
            rk = pd.DataFrame([{"Rank":i+1,"Dezena":n,"Score":round(scores_prev[n],4)}
                                for i,(n,_) in enumerate(sorted(scores_prev.items(), key=lambda x:-x[1]))])
            col_r1, col_r2 = st.columns(2)
            with col_r1: st.dataframe(rk.head(13), use_container_width=True, hide_index=True)
            with col_r2: st.dataframe(rk.tail(12), use_container_width=True, hide_index=True)

        if st.button("🎯 Gerar Combinações", type="primary"):
            if seed_val > 0: random.seed(seed_val)

            prog = st.progress(0, text="Calculando scores finais...")
            scores, _sim, _fortes, _apoio = calcular_scores_completo(
                sorteios, janela_analise, concurso_ref, threshold_sim, perfil["pesos"], analise)

            ranking_ord   = sorted(scores.items(), key=lambda x: -x[1])
            pool_dinamico = max(perfil["top_pool"], min(qtd_jogos+5, 25))
            pool = [n for n,_ in ranking_ord[:pool_dinamico]]
            for f in fixas:
                if f not in pool: pool.append(f)

            st.write(f"**Pool ({len(pool)} dez):** {' · '.join(f'{n:02d}' for n in sorted(pool))}")

            jogos_gerados  = []
            tentativas_total = 0
            max_tentativas = max(200000, qtd_jogos*10000)
            max_comum_atual = perfil["max_comum"]
            ultimo_gerado = 0
            travamentos   = 0

            prog.progress(10, text="Gerando combinações...")

            while len(jogos_gerados) < qtd_jogos and tentativas_total < max_tentativas:
                tentativas_total += 1

                if tentativas_total - ultimo_gerado > 20000 and len(jogos_gerados) > 0:
                    travamentos += 1
                    ultimo_gerado = tentativas_total
                    max_comum_atual = min(perfil["max_comum"]+travamentos, 13)
                    if pool_dinamico < 25:
                        pool_dinamico = min(pool_dinamico+2, 25)
                        pool = [n for n,_ in ranking_ord[:pool_dinamico]]
                        for f in fixas:
                            if f not in pool: pool.append(f)
                    prog.progress(min(10+int(len(jogos_gerados)/qtd_jogos*85),95),
                                  text=f"Gerados {len(jogos_gerados)}/{qtd_jogos} — ajustando (pool {pool_dinamico}, sim {max_comum_atual})...")

                base = fixas[:15]
                restante = [n for n in pool if n not in base]
                rest_scores = np.array([scores[n] for n in restante])
                rest_scores = rest_scores / rest_scores.sum()

                faltam = 15 - len(base)
                if len(restante) >= faltam:
                    escolhidos = list(np.random.choice(restante, size=faltam, replace=False, p=rest_scores))
                    candidato  = sorted(base + escolhidos)
                else:
                    extras = [n for n in range(1,26) if n not in base and n not in restante]
                    candidato = sorted(base + restante + random.sample(extras, faltam-len(restante)))

                if filtrar_jogo(candidato, sorteios, config_filtro): continue

                muito_similar = any(15-distancia_hamming(candidato,j) > max_comum_atual for j in jogos_gerados)
                if muito_similar: continue

                if candidato not in jogos_gerados:
                    jogos_gerados.append(candidato)
                    ultimo_gerado = tentativas_total
                    prog.progress(min(10+int(len(jogos_gerados)/qtd_jogos*85),95),
                                  text=f"Gerados {len(jogos_gerados)}/{qtd_jogos}...")

            prog.progress(100, text="Concluído!")

            if len(jogos_gerados) < qtd_jogos:
                st.warning(f"⚠️ Gerados {len(jogos_gerados)} de {qtd_jogos}. Tente Híbrida ou reduza a quantidade.")

            if jogos_gerados:
                st.success(f"✅ {len(jogos_gerados)} combinação(ões) em {tentativas_total:,} tentativas")
                st.markdown("---")

                for i, jogo in enumerate(jogos_gerados):
                    perf_j  = perfil_estrutural(jogo)
                    bt      = backtesting_jogo(jogo, sorteios, ultimos=200)
                    comp    = composicao_numerica(jogo)
                    faixas_j= distribuicao_faixas(jogo)
                    max_seq_j, _ = contar_consecutivos(jogo)
                    score_val = round(sum(scores[n] for n in jogo), 4)

                    st.subheader(f"🃏 Combinação {i+1}")
                    st.markdown("## " + "  ·  ".join(f"**{n:02d}**" for n in jogo))

                    col1,col2,col3,col4,col5 = st.columns(5)
                    col1.metric("Score", score_val)
                    col2.metric("Soma", perf_j["soma"])
                    col3.metric("Pares/Ímp", f"{perf_j['pares']}/{perf_j['impares']}")
                    col4.metric("11+", f"{bt['pct_11+']}%")
                    col5.metric("13+", f"{bt['pct_13+']}%")

                    col6,col7,col8 = st.columns(3)
                    col6.write(f"**Faixas:** {faixas_j}")
                    col7.write(f"**Primos:** {comp['primos']} | **Fib:** {comp['fibonacci']} | **M3:** {comp['multiplos_3']}")
                    col8.write(f"**Seq.:** {max_seq_j} | **Ampl.:** {perf_j['amplitude']}")

                    cobertura_f = len(set(jogo)&set(_fortes))
                    cobertura_a = len(set(jogo)&set(_apoio))
                    st.write(f"🎯 N+1/N+2 — Fortes: **{cobertura_f}/6** | Apoio: **{cobertura_a}/6**")

                    with st.expander(f"🔬 Breakdown — Combinação {i+1}"):
                        bd = pd.DataFrame([
                            {"Dezena":n, "Score":round(scores[n],4),
                             "Rank": sorted(scores,key=lambda x:-scores[x]).index(n)+1,
                             "G1 Recente":"✅" if scores[n]>=sorted(scores.values(),reverse=True)[9] else "—",
                             "G2 Ciclo":"✅" if n in analise["atraso"]["vencidas"]+analise["atraso"]["no_ciclo"] else "—",
                             "G3 Histórico":"✅" if n in (_fortes+_apoio) else "—"}
                            for n in sorted(jogo)
                        ])
                        st.dataframe(bd, use_container_width=True, hide_index=True)

                    st.success("✅ Aprovado em todos os filtros Hard V4")
                    st.markdown("---")

                custo_total = len(jogos_gerados)*3.5
                st.info(f"💰 **{len(jogos_gerados)} jogos × R$ 3,50 = R$ {custo_total:.2f}**")

                if len(jogos_gerados) > 1:
                    st.subheader("📐 Cobertura Hamming")
                    st.dataframe(cobertura_entre_jogos(jogos_gerados), use_container_width=True, hide_index=True)
            else:
                st.error("❌ Nenhuma combinação gerada. Tente o perfil Híbrida.")

# ──────────────────────────────────────────────────────────────
# ABA 13 — RELATÓRIO FINAL
# ──────────────────────────────────────────────────────────────
with tabs[13]:
    st.header("🏆 Relatório Final")
    analise = st.session_state["analise"]
    ok_c, tot_c = status_etapas()
    st.markdown(f"**Base:** {len(sorteios)} concursos | C{df['Concurso'].max()} | Janela: {janela_analise} | Etapas: {ok_c}/{tot_c}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📐 Parâmetros Calibrados")
        st.write(f"- Soma: **{config_filtro['soma_min']} – {config_filtro['soma_max']}**")
        if analise["estrutural"]:
            st.write(f"- Média histórica: **{analise['estrutural']['soma_med']}**")
        if analise["composicao"]:
            st.write(f"- Primos mais freq: **{analise['composicao']['top_primos']}** | Fib: **{analise['composicao']['top_fibonacci']}**")
        if analise["monte_carlo"]:
            st.write(f"- Base: **{'✅ OK' if analise['monte_carlo']['base_ok'] else '⚠️ Desvio'}**")
    with col2:
        st.subheader("🎯 Dezenas Prioritárias")
        if analise["termica"]:
            st.write(f"- Quentes top 10: **{' '.join(str(n).zfill(2) for n in analise['termica']['quentes'])}**")
            st.write(f"- Em pico atraso: **{' '.join(str(n).zfill(2) for n in analise['termica']['pico_atraso'])}**")
        if analise["n12"]:
            st.write(f"- N+1/N+2 Fortes: **{' '.join(str(n).zfill(2) for n in sorted(analise['n12']['fortes']))}**")
            st.write(f"- N+1/N+2 Apoio: **{' '.join(str(n).zfill(2) for n in sorted(analise['n12']['apoio']))}**")
        if analise["atraso"]:
            st.write(f"- Vencidas: **{' '.join(str(n).zfill(2) for n in analise['atraso']['vencidas'])}**")

    if jogos_usuario:
        st.subheader("🃏 Seus Jogos")
        for i, jogo in enumerate(jogos_usuario):
            bt = backtesting_jogo(jogo, sorteios, ultimos=200)
            alertas = filtrar_jogo(jogo, sorteios, config_filtro)
            status = "✅ APROVADO" if not alertas else f"⚠️ {len(alertas)} alertas"
            st.write(f"**J{i+1}:** {' '.join(str(n).zfill(2) for n in jogo)} | Média: **{bt['media']}** | 11+: **{bt['pct_11+']}%** | {status}")

    st.markdown("---")
    st.caption("Protocolo Hard V4 — MarianaRBrito | github.com/MarianaRBrito/protocolo-hard-v4")
