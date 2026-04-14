import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
from scipy import stats
import math
import hashlib
from io import StringIO

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Protocolo Hard V4.2", layout="wide")

COLS_DEZENAS = [f"bola {i}" for i in range(1, 16)]
PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23}
FIBONACCI = {1, 2, 3, 5, 8, 13, 21}
MULTIPLOS_3 = {3, 6, 9, 12, 15, 18, 21, 24}

QUINTETOS = {
    "Q1": set(range(1, 6)),
    "Q2": set(range(6, 11)),
    "Q3": set(range(11, 16)),
    "Q4": set(range(16, 21)),
    "Q5": set(range(21, 26)),
}

MOLDURA = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
MIOLO = set(range(1, 26)) - MOLDURA

LINHAS = {
    "L1": set(range(1, 6)),
    "L2": set(range(6, 11)),
    "L3": set(range(11, 16)),
    "L4": set(range(16, 21)),
    "L5": set(range(21, 26)),
}

COLUNAS = {
    "C1": {1, 6, 11, 16, 21},
    "C2": {2, 7, 12, 17, 22},
    "C3": {3, 8, 13, 18, 23},
    "C4": {4, 9, 14, 19, 24},
    "C5": {5, 10, 15, 20, 25},
}

ETAPAS = [
    "termica",
    "atraso",
    "estrutural",
    "composicao",
    "sequencias",
    "n12",
    "coocorrencia",
    "monte_carlo",
]

ETAPA_LABELS = {
    "termica": "1️⃣ Térmica",
    "atraso": "2️⃣ Atraso & Ciclo",
    "estrutural": "3️⃣ Estrutural",
    "composicao": "4️⃣ Composição",
    "sequencias": "5️⃣ Sequências & Gaps",
    "n12": "6️⃣ N+1/N+2",
    "coocorrencia": "7️⃣ Coocorrência",
    "monte_carlo": "8️⃣ Monte Carlo",
}

PERFIS = {
    "🎯 Agressiva": {
        "pesos": {"g1": 0.34, "g2": 0.26, "g3": 0.22, "g4": 0.10, "g5": 0.08},
        "top_pool": 17,
        "max_comum": 6,
        "descricao": "Mais forte sem virar clone. Núcleo menor e repetição controlada.",
        "aviso": 8,
    },
    "⚖️ Híbrida": {
        "pesos": {"g1": 0.30, "g2": 0.24, "g3": 0.24, "g4": 0.12, "g5": 0.10},
        "top_pool": 20,
        "max_comum": 8,
        "descricao": "Cobertura mais ampla, com trava de semelhança.",
        "aviso": None,
    },
}

PREMIOS = {15: 1500000, 14: 1500, 13: 25, 12: 10, 11: 5}
PROB = {15: 1 / 3268760, 14: 15 / 3268760, 13: 105 / 3268760, 12: 455 / 3268760, 11: 1365 / 3268760}


# ============================================================
# SESSION STATE
# ============================================================
def param_hash(janela: int, threshold: int, ref_ui: int) -> str:
    return hashlib.md5(f"{janela}_{threshold}_{ref_ui}".encode()).hexdigest()[:8]


def init_session() -> None:
    defaults = {
        "analise": {e: None for e in ETAPAS},
        "pipeline": [],
        "param_hash": None,
        "jogos_gerados": [],
        "base_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def verificar_params(janela: int, threshold: int, ref_ui: int) -> None:
    h = param_hash(janela, threshold, ref_ui)
    if st.session_state["param_hash"] != h:
        st.session_state["analise"] = {e: None for e in ETAPAS}
        st.session_state["pipeline"] = []
        st.session_state["param_hash"] = h
        st.session_state["jogos_gerados"] = []


def salvar_etapa(nome: str, dados: dict, pipeline_update=None) -> None:
    st.session_state["analise"][nome] = dados
    if pipeline_update is not None:
        st.session_state["pipeline"] = pipeline_update


def etapa_ok(nome: str) -> bool:
    return st.session_state["analise"].get(nome) is not None


def todas_ok() -> bool:
    return all(etapa_ok(e) for e in ETAPAS)


def proxima_etapa():
    for e in ETAPAS:
        if not etapa_ok(e):
            return e
    return None


def status_count():
    ok = sum(1 for e in ETAPAS if etapa_ok(e))
    return ok, len(ETAPAS)


init_session()


# ============================================================
# BASE
# ============================================================
def padronizar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    for c in df.columns:
        c_low = c.strip().lower()
        if c_low == "concurso":
            rename_map[c] = "Concurso"
        elif c_low == "data":
            rename_map[c] = "Data"
        elif c_low.startswith("bola "):
            rename_map[c] = c_low
        elif c_low.startswith("bola"):
            sufixo = c_low.replace("bola", "").strip()
            if sufixo.isdigit():
                rename_map[c] = f"bola {int(sufixo)}"

    df = df.rename(columns=rename_map)

    obrig = {"Concurso", "Data", *COLS_DEZENAS}
    faltando = obrig - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltando)}")

    for c in ["Concurso", *COLS_DEZENAS]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df = df.dropna(subset=["Concurso", *COLS_DEZENAS]).copy()
    df["Concurso"] = df["Concurso"].astype(int)

    for c in COLS_DEZENAS:
        df[c] = df[c].astype(int)

    df = df.sort_values("Concurso").drop_duplicates(subset=["Concurso"], keep="last").reset_index(drop=True)
    return df


@st.cache_data
def carregar_base_local() -> pd.DataFrame:
    return padronizar_base(pd.read_csv("base_lotofacil.csv"))


def carregar_base_upload(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return carregar_base_local()
    content = uploaded.getvalue()
    return padronizar_base(pd.read_csv(StringIO(content.decode("utf-8", errors="ignore"))))


def extrair_sorteios(df: pd.DataFrame):
    return df[COLS_DEZENAS].values.tolist()


def incorporar_resultado_manual(df: pd.DataFrame, concurso: int, data: str, dezenas: list[int]) -> pd.DataFrame:
    if len(dezenas) != 15 or len(set(dezenas)) != 15:
        raise ValueError("Informe exatamente 15 dezenas distintas.")
    if any(n < 1 or n > 25 for n in dezenas):
        raise ValueError("As dezenas devem ficar entre 1 e 25.")
    dezenas = sorted(dezenas)

    row = {"Concurso": concurso, "Data": data}
    for i, n in enumerate(dezenas, start=1):
        row[f"bola {i}"] = n

    novo = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    novo = novo.sort_values("Concurso").drop_duplicates(subset=["Concurso"], keep="last").reset_index(drop=True)
    return padronizar_base(novo)


# ============================================================
# HELPERS ANALÍTICOS
# ============================================================
def concurso_ref_ui_to_idx(ref_ui: int, total: int) -> int:
    return total - 1 - ref_ui


def normalizar(d: dict) -> dict:
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 0.5 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def freq_simples(sorteios, janela=None):
    dados = sorteios[-janela:] if janela else sorteios
    c = Counter()
    for s in dados:
        c.update(s)
    return c


def tabela_frequencia(sorteios, janelas=(50, 100, 200, 500)):
    rows = []
    for n in range(1, 26):
        row = {"Dezena": n}
        for j in janelas:
            j2 = min(j, len(sorteios))
            c = sum(1 for s in sorteios[-j2:] if n in s)
            row[f"Freq {j2}"] = c
            row[f"% {j2}"] = round(c / j2 * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def calcular_atrasos(sorteios):
    ultimo = {}
    for i, s in enumerate(sorteios):
        for n in s:
            ultimo[n] = i
    total = len(sorteios)
    rows = []
    for n in range(1, 26):
        idx = ultimo.get(n, -1)
        atraso = total - 1 - idx if idx >= 0 else total
        rows.append({"Dezena": n, "Último": idx + 1 if idx >= 0 else None, "Atraso": atraso})
    return pd.DataFrame(rows).sort_values("Atraso", ascending=False).reset_index(drop=True)


@st.cache_data
def calcular_percentis_atraso(sorteios_tuple):
    sorteios = list(sorteios_tuple)
    gaps = []
    for n in range(1, 26):
        ap = [i for i, s in enumerate(sorteios) if n in s]
        for k in range(1, len(ap)):
            gaps.append(ap[k] - ap[k - 1])
    if not gaps:
        return {"p75": 4, "p85": 5, "p90": 6, "p95": 7}
    return {
        "p75": int(np.percentile(gaps, 75)),
        "p85": int(np.percentile(gaps, 85)),
        "p90": int(np.percentile(gaps, 90)),
        "p95": int(np.percentile(gaps, 95)),
    }


def classificar_atraso(atraso: int, percentis: dict) -> str:
    if atraso >= percentis["p95"]:
        return "🔴 VENCIDA"
    if atraso >= percentis["p85"]:
        return "🟡 No ciclo"
    if atraso >= percentis["p75"]:
        return "🟠 Atrasada"
    return "🟢 Ok"


def ciclo_medio(sorteios, dezena: int):
    ap = [i for i, s in enumerate(sorteios) if dezena in s]
    if len(ap) < 2:
        return None
    return round(np.mean([ap[i + 1] - ap[i] for i in range(len(ap) - 1)]), 2)


def faixa_soma(sorteios):
    somas = [sum(s) for s in sorteios]
    return int(np.percentile(somas, 5)), int(np.percentile(somas, 95)), round(float(np.mean(somas)), 1)


def dist_pares(sorteios):
    d = Counter(sum(1 for n in s if n % 2 == 0) for s in sorteios)
    t = len(sorteios)
    return {k: round(v / t * 100, 1) for k, v in sorted(d.items())}


def contar_consec(jogo):
    jogo = sorted(jogo)
    mx = cur = 1
    pares = 0
    for i in range(1, len(jogo)):
        if jogo[i] == jogo[i - 1] + 1:
            cur += 1
            mx = max(mx, cur)
            pares += 1
        else:
            cur = 1
    return mx, pares


def dist_consec(sorteios):
    d = Counter(contar_consec(s)[1] for s in sorteios)
    t = len(sorteios)
    return {k: round(v / t * 100, 1) for k, v in sorted(d.items())}


def gaps(jogo):
    j = sorted(jogo)
    g = [j[i + 1] - j[i] for i in range(len(j) - 1)]
    return {"lista": g, "media": round(float(np.mean(g)), 2), "max": max(g), "min": min(g)}


def dist_gaps(sorteios):
    todos = []
    for s in sorteios:
        todos.extend(gaps(s)["lista"])
    c = Counter(todos)
    t = sum(c.values())
    return {k: round(v / t * 100, 1) for k, v in sorted(c.items()) if k <= 10}


def dist_faixas(jogo):
    return [sum(1 for n in jogo if i * 5 + 1 <= n <= i * 5 + 5) for i in range(5)]


def dist_faixas_hist(sorteios):
    dist = [[] for _ in range(5)]
    for s in sorteios:
        f = dist_faixas(s)
        for i in range(5):
            dist[i].append(f[i])
    out = []
    for i in range(5):
        out.append({
            "Faixa": f"{i*5+1}-{i*5+5}",
            "Média": round(float(np.mean(dist[i])), 2),
            "Min": int(min(dist[i])),
            "Max": int(max(dist[i])),
        })
    return out


def composicao(jogo):
    s = set(jogo)
    return {
        "primos": len(s & PRIMOS),
        "fibonacci": len(s & FIBONACCI),
        "mult3": len(s & MULTIPLOS_3),
    }


def dist_comp(sorteios):
    dp = Counter(len(set(s) & PRIMOS) for s in sorteios)
    df = Counter(len(set(s) & FIBONACCI) for s in sorteios)
    t = len(sorteios)
    return (
        {k: round(v / t * 100, 1) for k, v in sorted(dp.items())},
        {k: round(v / t * 100, 1) for k, v in sorted(df.items())},
    )


def dist_quintetos(jogo):
    s = set(jogo)
    return [len(s & QUINTETOS[f"Q{i}"]) for i in range(1, 6)]


def moldura_miolo(jogo):
    s = set(jogo)
    return len(s & MOLDURA), len(s & MIOLO)


def dist_linhas(jogo):
    s = set(jogo)
    return [len(s & LINHAS[f"L{i}"]) for i in range(1, 6)]


def dist_colunas(jogo):
    s = set(jogo)
    return [len(s & COLUNAS[f"C{i}"]) for i in range(1, 6)]


def repeticao_ultimo(jogo, sorteio_anterior):
    return len(set(jogo) & set(sorteio_anterior))


def projetar_isca(alvo_idx, sorteios, threshold):
    if not sorteios or alvo_idx >= len(sorteios):
        return [], [], []
    alvo = set(sorteios[alvo_idx])
    th = int(len(alvo) * (threshold / 100))
    sim = []
    fut = []
    for i, s in enumerate(sorteios):
        if i == alvo_idx:
            continue
        if len(alvo & set(s)) >= th:
            sim.append(i)
            if i + 1 < len(sorteios):
                fut.extend(sorteios[i + 1])
            if i + 2 < len(sorteios):
                fut.extend(sorteios[i + 2])
    freq = Counter(fut).most_common()
    return sim, [k for k, _ in freq[:6]], [k for k, _ in freq[6:12]]


@st.cache_data
def calc_cooc(sorteios_tuple, top=20):
    p = Counter()
    for s in sorteios_tuple:
        for a, b in combinations(sorted(s), 2):
            p[(a, b)] += 1
    return p.most_common(top)


@st.cache_data
def calc_trios(sorteios_tuple, top=15):
    t = Counter()
    for s in sorteios_tuple:
        for trio in combinations(sorted(s), 3):
            t[trio] += 1
    return t.most_common(top)


@st.cache_data
def calc_markov(sorteios_tuple, janela=200):
    sl = list(sorteios_tuple)[-janela:]
    tr = {n: Counter() for n in range(1, 26)}
    for i in range(len(sl) - 1):
        for n in set(sl[i]):
            tr[n].update(set(sl[i + 1]))
    return tr


def qui_quadrado(sorteios, janela=500):
    dados = sorteios[-janela:]
    obs = np.array([sum(1 for s in dados if n in s) for n in range(1, 26)])
    esp = np.full(25, len(dados) * 15 / 25)
    chi2, p = stats.chisquare(obs, esp)
    return round(float(chi2), 2), round(float(p), 4)


def backtesting(jogo, sorteios, ultimos=200):
    dados = sorteios[-ultimos:]
    ac = [len(set(jogo) & set(s)) for s in dados]
    d = Counter(ac)
    return {
        "media": round(float(np.mean(ac)), 2),
        "max": int(max(ac)),
        "min": int(min(ac)),
        "dist": dict(sorted(d.items())),
        "pct11": round(sum(1 for a in ac if a >= 11) / len(ac) * 100, 1),
        "pct13": round(sum(1 for a in ac if a >= 13) / len(ac) * 100, 1),
    }


def dezenas_pico(sorteios, pct=85):
    df = calcular_atrasos(sorteios)
    lim = np.percentile(df["Atraso"], pct)
    return df[df["Atraso"] >= lim]["Dezena"].tolist()


def hamming(j1, j2):
    return 15 - len(set(j1) & set(j2))


def cobertura_jogos(jogos):
    if len(jogos) < 2:
        return pd.DataFrame()
    rows = []
    for i in range(len(jogos)):
        for j in range(i + 1, len(jogos)):
            d = hamming(jogos[i], jogos[j])
            rows.append({"Par": f"J{i+1}×J{j+1}", "Hamming": d, "Em comum": 15 - d})
    return pd.DataFrame(rows)


def auditoria_portfolio(jogos, fortes, apoio, vencidas, sorteio_anterior):
    if not jogos:
        return {"cegas_criticas": [], "nucleo_comum": [], "repeticoes": []}

    uniao = set().union(*map(set, jogos))
    inter = set(jogos[0]).intersection(*map(set, jogos[1:])) if len(jogos) > 1 else set(jogos[0])
    criticas = set(fortes) | set(apoio[:4]) | set(vencidas[:4]) | set(sorteio_anterior)
    cegas = sorted(criticas - uniao)

    return {
        "cegas_criticas": cegas,
        "nucleo_comum": sorted(inter),
        "repeticoes": [repeticao_ultimo(j, sorteio_anterior) for j in jogos],
    }


# ============================================================
# CALIBRAÇÃO E FILTROS
# ============================================================
@st.cache_data
def calibrar(sorteios_tuple):
    sl = list(sorteios_tuple)
    repeticoes = [len(set(sl[i]) & set(sl[i - 1])) for i in range(1, len(sl))]
    molduras = [moldura_miolo(s)[0] for s in sl]
    quint_max = [max(dist_quintetos(s)) for s in sl]

    return {
        "soma_min": int(np.percentile([sum(s) for s in sl], 5)),
        "soma_max": int(np.percentile([sum(s) for s in sl], 95)),
        "min_pares": int(np.percentile([sum(1 for n in s if n % 2 == 0) for s in sl], 5)),
        "max_pares": int(np.percentile([sum(1 for n in s if n % 2 == 0) for s in sl], 95)),
        "max_seq": int(np.percentile([contar_consec(s)[0] for s in sl], 95)),
        "max_primo": int(np.percentile([len(set(s) & PRIMOS) for s in sl], 95)),
        "max_fib": int(np.percentile([len(set(s) & FIBONACCI) for s in sl], 95)),
        "max_faixa": int(np.percentile([max(dist_faixas(s)) for s in sl], 95)),
        "rep_min": int(np.percentile(repeticoes, 10)) if repeticoes else 6,
        "rep_max": int(np.percentile(repeticoes, 90)) if repeticoes else 10,
        "mold_min": int(np.percentile(molduras, 10)),
        "mold_max": int(np.percentile(molduras, 90)),
        "quint_max": int(np.percentile(quint_max, 95)),
    }


def filtrar(
    jogo,
    sorteios,
    cfg,
    sorteio_anterior=None,
    meta_rep_min=None,
    meta_rep_max=None,
    meta_qmax=None,
    exigir_linhas=True,
    exigir_colunas=True,
):
    erros = []

    soma = sum(jogo)
    if soma < cfg["soma_min"] or soma > cfg["soma_max"]:
        erros.append(f"Soma {soma} fora [{cfg['soma_min']}–{cfg['soma_max']}]")

    mx, _ = contar_consec(jogo)
    if mx > cfg["max_seq"]:
        erros.append(f"Seq {mx}>{cfg['max_seq']}")

    p = sum(1 for n in jogo if n % 2 == 0)
    if p < cfg["min_pares"] or p > cfg["max_pares"]:
        erros.append(f"{p} pares fora [{cfg['min_pares']}–{cfg['max_pares']}]")

    fq = dist_faixas(jogo)
    for i, f in enumerate(fq):
        if f == 0:
            erros.append(f"Faixa {i*5+1}-{i*5+5} vazia")
        if f > cfg["max_faixa"]:
            erros.append(f"Faixa {i*5+1}-{i*5+5} excesso")

    c = composicao(jogo)
    if c["primos"] > cfg["max_primo"]:
        erros.append(f"{c['primos']} primos>{cfg['max_primo']}")
    if c["fibonacci"] > cfg["max_fib"]:
        erros.append(f"{c['fibonacci']} fib>{cfg['max_fib']}")

    mold, miolo = moldura_miolo(jogo)
    if mold < cfg["mold_min"] or mold > cfg["mold_max"]:
        erros.append(f"Moldura {mold} fora [{cfg['mold_min']}–{cfg['mold_max']}]")

    q = dist_quintetos(jogo)
    qmax_lim = meta_qmax if meta_qmax is not None else cfg["quint_max"]
    if max(q) > qmax_lim:
        erros.append(f"Quinteto inflado {max(q)}>{qmax_lim}")
    if min(q) == 0:
        erros.append("Quinteto vazio")

    if exigir_linhas:
        linhas = dist_linhas(jogo)
        if any(x == 0 for x in linhas):
            erros.append(f"Linha vazia {linhas}")

    if exigir_colunas:
        cols = dist_colunas(jogo)
        if any(x == 0 for x in cols):
            erros.append(f"Coluna vazia {cols}")

    if sorteio_anterior is not None:
        rep = repeticao_ultimo(jogo, sorteio_anterior)
        rmin = cfg["rep_min"] if meta_rep_min is None else meta_rep_min
        rmax = cfg["rep_max"] if meta_rep_max is None else meta_rep_max
        if rep < rmin or rep > rmax:
            erros.append(f"Repetição {rep} fora [{rmin}–{rmax}]")

    return erros


# ============================================================
# SCORE
# ============================================================
def calcular_scores(sorteios, janela, ref, threshold, pesos, analise):
    g1 = {n: 0.0 for n in range(1, 26)}
    if analise.get("termica"):
        quentes = analise["termica"]["quentes"]
        for i, n in enumerate(quentes):
            g1[n] = max(g1[n], 1.0 - i * 0.04)

    fortes_n, apoio_n = [], []
    if analise.get("n12"):
        fortes_n = analise["n12"]["fortes"]
        apoio_n = analise["n12"]["apoio"]
        for n in fortes_n:
            g1[n] = max(g1[n], 1.0)
        for n in apoio_n:
            g1[n] = max(g1[n], 0.7)

    f50 = freq_simples(sorteios, min(50, len(sorteios)))
    fl = freq_simples(sorteios, janela)
    roll = {}
    for n in range(1, 26):
        t50 = f50[n] / min(50, len(sorteios))
        tl = fl[n] / janela
        roll[n] = max(0.0, (t50 - tl) / (tl + 1e-9))
    roll = normalizar(roll)
    g1 = normalizar(g1)
    G1 = {n: g1[n] * 0.60 + roll[n] * 0.40 for n in range(1, 26)}

    g2 = {n: 0.0 for n in range(1, 26)}
    vencidas, no_ciclo = [], []
    if analise.get("atraso"):
        df_at = analise["atraso"]["df_completo"]
        vencidas = analise["atraso"]["vencidas"]
        no_ciclo = analise["atraso"]["no_ciclo"]
        for _, row in df_at.iterrows():
            n = int(row["Dezena"])
            atraso = row["Atraso"]
            cm = row["Ciclo Médio"] if pd.notna(row["Ciclo Médio"]) else atraso
            ratio = min(atraso / cm if cm and cm > 0 else 1.0, 3.0)
            if n in vencidas:
                g2[n] = 3.0
            elif n in no_ciclo:
                g2[n] = max(ratio, 1.5)
            else:
                g2[n] = ratio
    G2 = normalizar(g2)

    if analise.get("coocorrencia"):
        G3 = {
            n: (
                analise["coocorrencia"]["g3_cooc"][n] * 0.40
                + analise["coocorrencia"]["g3_markov"][n] * 0.40
                + analise["coocorrencia"]["g3_trios"][n] * 0.20
            )
            for n in range(1, 26)
        }
    else:
        G3 = {n: 0.5 for n in range(1, 26)}

    freq_all = freq_simples(sorteios)
    tot = sum(freq_all.values())
    G4 = normalizar({n: 1 - abs(freq_all[n] / tot - 1 / 25) / (1 / 25) for n in range(1, 26)})
    G5 = normalizar({n: freq_all[n] / tot for n in range(1, 26)})

    scores = {
        n: (
            pesos["g1"] * G1[n]
            + pesos["g2"] * G2[n]
            + pesos["g3"] * G3[n]
            + pesos["g4"] * G4[n]
            + pesos["g5"] * G5[n]
        )
        for n in range(1, 26)
    }
    return scores, fortes_n, apoio_n, vencidas, no_ciclo


# ============================================================
# UI - BASE
# ============================================================
st.sidebar.title("⚙️ Protocolo Hard V4.2")

uploaded = st.sidebar.file_uploader("Base CSV da Lotofácil", type=["csv"])

try:
    df = carregar_base_upload(uploaded)
except Exception as e:
    st.error(f"Falha ao carregar a base: {e}")
    st.stop()

with st.sidebar.expander("➕ Atualização manual da base"):
    concurso_manual = st.number_input("Concurso novo", min_value=1, value=int(df["Concurso"].max()) + 1, step=1)
    data_manual = st.text_input("Data", value="")
    dezenas_manual = st.text_input("15 dezenas", value="", placeholder="01 02 03 ... 15")
    if st.button("Incorporar resultado manual"):
        try:
            dezenas = sorted([int(x) for x in dezenas_manual.strip().split()])
            df = incorporar_resultado_manual(df, int(concurso_manual), data_manual, dezenas)
            st.session_state["base_df"] = df
            st.success(f"Resultado do concurso {int(concurso_manual)} incorporado.")
        except Exception as e:
            st.error(f"Não foi possível incorporar: {e}")

if st.session_state["base_df"] is not None:
    df = st.session_state["base_df"]

sorteios = extrair_sorteios(df)
st_tuple = tuple(tuple(s) for s in sorteios)

janela = st.sidebar.slider("Janela de análise", 50, len(sorteios), min(500, len(sorteios)), step=50)
threshold = st.sidebar.slider("Threshold N+1/N+2 (%)", 50, 100, 80)
ref_ui = st.sidebar.number_input("Índice Referência (0=último)", 0, len(sorteios) - 1, 0)
ref = concurso_ref_ui_to_idx(ref_ui, len(sorteios))

verificar_params(janela, threshold, ref_ui)
cfg = calibrar(st_tuple)

with st.sidebar.expander("📐 Filtros calibrados"):
    st.write(f"Soma: {cfg['soma_min']}–{cfg['soma_max']}")
    st.write(f"Pares: {cfg['min_pares']}–{cfg['max_pares']}")
    st.write(f"Repetição: {cfg['rep_min']}–{cfg['rep_max']}")
    st.write(f"Moldura: {cfg['mold_min']}–{cfg['mold_max']}")
    st.write(f"Máx seq: {cfg['max_seq']} | Máx faixa: {cfg['max_faixa']} | Máx quinteto: {cfg['quint_max']}")
    st.write(f"Máx primos: {cfg['max_primo']} | Máx fib: {cfg['max_fib']}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Status do protocolo")
ok, tot = status_count()
for e in ETAPAS:
    icon = "✅" if etapa_ok(e) else "⭕"
    st.sidebar.write(f"{icon} {ETAPA_LABELS[e]}")
if todas_ok():
    st.sidebar.success("🎯 Pronto para gerar")
else:
    st.sidebar.warning(f"⏳ {ok}/{tot} — próxima: {ETAPA_LABELS[proxima_etapa()]}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Backtesting manual")
j1 = st.sidebar.text_input("Jogo 1 (15 números)")
j2 = st.sidebar.text_input("Jogo 2")
j3 = st.sidebar.text_input("Jogo 3")


def parse_jogo(txt: str):
    try:
        nums = sorted([int(x) for x in txt.strip().split()])
        if len(nums) == 15 and len(set(nums)) == 15 and all(1 <= x <= 25 for x in nums):
            return nums
    except Exception:
        pass
    return None


jogos_bt = [j for j in [parse_jogo(j1), parse_jogo(j2), parse_jogo(j3)] if j]
sorteios_j = sorteios[-janela:]

# ============================================================
# TOPO
# ============================================================
st.title("📊 Protocolo Hard V4.2 — Lotofácil")
st.markdown(
    f"Base: **{len(sorteios)} concursos** | "
    f"C{df['Concurso'].min()}–C{df['Concurso'].max()} | "
    f"Ref UI: **{ref_ui}** | Concurso ref: **C{df.iloc[ref]['Concurso']}** | "
    f"Janela: **{janela}**"
)

if len(df) >= 2:
    st.caption(
        f"Último concurso na base: C{df.iloc[-1]['Concurso']} | "
        f"Penúltimo: C{df.iloc[-2]['Concurso']}"
    )

ok, tot = status_count()
st.progress(ok / tot, text=f"Protocolo: {ok}/{tot} etapas {'✅ pronto para gerar' if todas_ok() else ''}")

tabs = st.tabs([
    "1️⃣ Térmica",
    "2️⃣ Atraso & Ciclo",
    "3️⃣ Estrutural",
    "4️⃣ Composição",
    "5️⃣ Sequências",
    "6️⃣ N+1/N+2",
    "7️⃣ Coocorrência",
    "8️⃣ Monte Carlo",
    "📊 Backtesting",
    "🧪 Filtros",
    "📈 Tendência",
    "⚖️ EV",
    "🏆 Relatório",
    "🎰 Gerar",
])

# ============================================================
# ETAPA 1
# ============================================================
with tabs[0]:
    st.header("1️⃣ Térmica — Frequência & temperatura")
    if etapa_ok("termica"):
        d = st.session_state["analise"]["termica"]
        st.success("✅ Etapa concluída")
        st.write(f"🔥 Quentes: {' '.join(str(n).zfill(2) for n in d['quentes'])}")
        st.write(f"🧊 Frias: {' '.join(str(n).zfill(2) for n in d['frias'])}")
        st.write(f"⚠️ Pico atraso: {' '.join(str(n).zfill(2) for n in d['pico'])}")
    else:
        st.dataframe(tabela_frequencia(sorteios), use_container_width=True, hide_index=True)
        f = freq_simples(sorteios, janela)
        top10 = [n for n, _ in f.most_common(10)]
        bot5 = [n for n, _ in f.most_common()[:-6:-1]]
        pico = dezenas_pico(sorteios)
        candidatas = list(dict.fromkeys(top10 + pico))
        st.success(f"🔥 Quentes: {' '.join(str(n).zfill(2) for n in top10)}")
        st.info(f"🧊 Frias: {' '.join(str(n).zfill(2) for n in bot5)}")
        st.warning(f"⚠️ Pico atraso: {' '.join(str(n).zfill(2) for n in pico)}")
        st.info(f"Candidatas: {' '.join(str(n).zfill(2) for n in candidatas)}")
        if st.button("✅ Confirmar Etapa 1", type="primary"):
            salvar_etapa("termica", {
                "quentes": top10,
                "frias": bot5,
                "pico": pico,
                "candidatas": candidatas,
                "freq": {n: f[n] for n in range(1, 26)},
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 2
# ============================================================
with tabs[1]:
    st.header("2️⃣ Atraso & ciclo")
    st.caption(f"Base ativa até C{df['Concurso'].max()}")
    if not etapa_ok("termica"):
        st.warning("Execute a Etapa 1 primeiro.")
    elif etapa_ok("atraso"):
        d = st.session_state["analise"]["atraso"]
        st.success("✅ Etapa concluída")
        if d["vencidas"]:
            st.error(f"🔴 Vencidas: {' '.join(str(n).zfill(2) for n in d['vencidas'])}")
        if d["no_ciclo"]:
            st.warning(f"🟡 No ciclo: {' '.join(str(n).zfill(2) for n in d['no_ciclo'])}")
        if d["atrasadas"]:
            st.warning(f"🟠 Atrasadas: {' '.join(str(n).zfill(2) for n in d['atrasadas'])}")
    else:
        percentis = calcular_percentis_atraso(st_tuple)
        df_at = calcular_atrasos(sorteios)
        ciclos = [{"Dezena": n, "Ciclo Médio": ciclo_medio(sorteios, n)} for n in range(1, 26)]
        df_comp = df_at.merge(pd.DataFrame(ciclos), on="Dezena")
        df_comp["Status"] = df_comp["Atraso"].apply(lambda x: classificar_atraso(x, percentis))
        st.caption(
            f"Limites históricos: atrasada≥{percentis['p75']} | "
            f"no ciclo≥{percentis['p85']} | vencida≥{percentis['p95']}"
        )
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        vencidas = df_comp[df_comp["Status"] == "🔴 VENCIDA"]["Dezena"].tolist()
        no_ciclo = df_comp[df_comp["Status"] == "🟡 No ciclo"]["Dezena"].tolist()
        atrasadas = df_comp[df_comp["Status"] == "🟠 Atrasada"]["Dezena"].tolist()
        top5 = df_comp.sort_values("Atraso", ascending=False).head(5)["Dezena"].tolist()

        prioridade = vencidas if vencidas else (no_ciclo + atrasadas if (no_ciclo or atrasadas) else top5)
        candidatas = list(dict.fromkeys(st.session_state["pipeline"] + prioridade))
        st.info(f"Prioridade: {' '.join(str(n).zfill(2) for n in prioridade)}")
        st.info(f"Candidatas: {' '.join(str(n).zfill(2) for n in candidatas)}")

        if st.button("✅ Confirmar Etapa 2", type="primary"):
            salvar_etapa("atraso", {
                "df_completo": df_comp,
                "vencidas": vencidas,
                "no_ciclo": no_ciclo,
                "atrasadas": atrasadas,
                "top5_atraso": top5,
                "prioridade": prioridade,
                "candidatas": candidatas,
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 3
# ============================================================
with tabs[2]:
    st.header("3️⃣ Estrutural")
    if not etapa_ok("atraso"):
        st.warning("Execute as etapas anteriores primeiro.")
    elif etapa_ok("estrutural"):
        d = st.session_state["analise"]["estrutural"]
        st.success("✅ Etapa concluída")
        st.write(f"Soma: {d['soma_min']}–{d['soma_max']} | média {d['soma_med']}")
        st.write(f"Pares mais frequentes: {d['top_pares']} ({d['pct_top_pares']}%)")
    else:
        sm, sx, smed = faixa_soma(sorteios_j)
        dp = dist_pares(sorteios_j)
        top_pares = max(dp, key=dp.get)
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"Soma: {sm}–{sx} | média {smed}")
            st.dataframe(pd.DataFrame(list(dp.items()), columns=["Pares", "%"]), use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(pd.DataFrame(dist_faixas_hist(sorteios_j)), use_container_width=True, hide_index=True)

        candidatas = st.session_state["pipeline"]
        if st.button("✅ Confirmar Etapa 3", type="primary"):
            salvar_etapa("estrutural", {
                "soma_min": sm,
                "soma_max": sx,
                "soma_med": smed,
                "top_pares": top_pares,
                "pct_top_pares": dp[top_pares],
                "dist_pares": dp,
                "candidatas": candidatas,
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 4
# ============================================================
with tabs[3]:
    st.header("4️⃣ Composição")
    if not etapa_ok("estrutural"):
        st.warning("Execute as etapas anteriores primeiro.")
    elif etapa_ok("composicao"):
        d = st.session_state["analise"]["composicao"]
        st.success("✅ Etapa concluída")
        st.write(f"Primos mais freq: {d['top_primo']} | Fibonacci: {d['top_fib']}")
    else:
        dp_c, df_c = dist_comp(sorteios_j)
        top_primo = max(dp_c, key=dp_c.get)
        top_fib = max(df_c, key=df_c.get)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Primos (%)")
            st.dataframe(pd.DataFrame(list(dp_c.items()), columns=["Qtd", "%"]), hide_index=True)
        with c2:
            st.subheader("Fibonacci (%)")
            st.dataframe(pd.DataFrame(list(df_c.items()), columns=["Qtd", "%"]), hide_index=True)
        with c3:
            dm3 = Counter(len(set(s) & MULTIPLOS_3) for s in sorteios_j)
            t = len(sorteios_j)
            st.subheader("Múltiplos de 3 (%)")
            st.dataframe(pd.DataFrame([(k, round(v / t * 100, 1)) for k, v in sorted(dm3.items())], columns=["Qtd", "%"]), hide_index=True)

        candidatas = st.session_state["pipeline"]
        if st.button("✅ Confirmar Etapa 4", type="primary"):
            salvar_etapa("composicao", {
                "top_primo": top_primo,
                "top_fib": top_fib,
                "dist_primo": dp_c,
                "dist_fib": df_c,
                "candidatas": candidatas,
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 5
# ============================================================
with tabs[4]:
    st.header("5️⃣ Sequências & gaps")
    if not etapa_ok("composicao"):
        st.warning("Execute as etapas anteriores primeiro.")
    elif etapa_ok("sequencias"):
        d = st.session_state["analise"]["sequencias"]
        st.success("✅ Etapa concluída")
        st.write(f"Consecutivos mais comuns: {d['top_consec']} ({d['pct_consec']}%)")
        st.write(f"Gap mais comum: {d['top_gap']} ({d['pct_gap']}%)")
    else:
        dc = dist_consec(sorteios_j)
        dg = dist_gaps(sorteios_j)
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(pd.DataFrame(list(dc.items()), columns=["Pares consec.", "%"]), use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(pd.DataFrame(list(dg.items()), columns=["Gap", "%"]), use_container_width=True, hide_index=True)

        candidatas = st.session_state["pipeline"]
        if st.button("✅ Confirmar Etapa 5", type="primary"):
            salvar_etapa("sequencias", {
                "top_consec": max(dc, key=dc.get),
                "pct_consec": dc[max(dc, key=dc.get)],
                "top_gap": max(dg, key=dg.get),
                "pct_gap": dg[max(dg, key=dg.get)],
                "dist_consec": dc,
                "dist_gap": dg,
                "candidatas": candidatas,
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 6
# ============================================================
with tabs[5]:
    st.header("6️⃣ N+1/N+2")
    if not etapa_ok("sequencias"):
        st.warning("Execute as etapas anteriores primeiro.")
    elif etapa_ok("n12"):
        d = st.session_state["analise"]["n12"]
        st.success("✅ Etapa concluída")
        st.write(f"Similares: {len(d['similares'])}")
        st.write(f"🔥 Fortes: {' '.join(str(n).zfill(2) for n in sorted(d['fortes']))}")
        st.write(f"🤝 Apoio: {' '.join(str(n).zfill(2) for n in sorted(d['apoio']))}")
    else:
        sim, fortes, apoio = projetar_isca(ref, sorteios, threshold)
        st.write(f"Referência: C{df.iloc[ref]['Concurso']} | threshold {threshold}% | similares {len(sim)}")

        if sim:
            sim_data = []
            for idx in sim[:20]:
                sim_data.append({
                    "Concurso": int(df.iloc[idx]["Concurso"]),
                    "Data": df.iloc[idx]["Data"],
                    "Em comum": len(set(sorteios[idx]) & set(sorteios[ref])),
                    "Dezenas": " ".join(str(n).zfill(2) for n in sorteios[idx]),
                })
            st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)

        st.success(f"🔥 Fortes: {' '.join(str(n).zfill(2) for n in sorted(fortes))}")
        st.info(f"🤝 Apoio: {' '.join(str(n).zfill(2) for n in sorted(apoio))}")

        candidatas = list(dict.fromkeys(st.session_state["pipeline"] + fortes + apoio))
        candidatas = sorted(candidatas, key=lambda n: (0 if n in fortes else 1 if n in apoio else 2, n))

        if st.button("✅ Confirmar Etapa 6", type="primary"):
            salvar_etapa("n12", {
                "similares": sim,
                "fortes": fortes,
                "apoio": apoio,
                "conc_ref": int(df.iloc[ref]["Concurso"]),
                "threshold": threshold,
                "candidatas": candidatas,
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 7
# ============================================================
with tabs[6]:
    st.header("7️⃣ Coocorrência")
    if not etapa_ok("n12"):
        st.warning("Execute as etapas anteriores primeiro.")
    elif etapa_ok("coocorrencia"):
        st.success("✅ Etapa concluída")
        st.write(f"Candidatas ordenadas: {len(st.session_state['analise']['coocorrencia']['candidatas'])}")
    else:
        pares_top = calc_cooc(st_tuple)
        trios_top = calc_trios(st_tuple)
        tr = calc_markov(st_tuple)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top pares")
            st.dataframe(pd.DataFrame([(f"{a:02d}+{b:02d}", c) for (a, b), c in pares_top], columns=["Par", "Freq"]), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Top trios")
            st.dataframe(pd.DataFrame([(f"{a:02d}+{b:02d}+{c:02d}", cnt) for (a, b, c), cnt in trios_top], columns=["Trio", "Freq"]), use_container_width=True, hide_index=True)

        pf = Counter()
        for s in sorteios[-min(500, len(sorteios)):]:
            for a, b in combinations(sorted(s), 2):
                pf[(a, b)] += 1
        gc = {n: 0.0 for n in range(1, 26)}
        for (a, b), cnt in pf.items():
            gc[a] += cnt
            gc[b] += cnt
        gc = normalizar(gc)

        ult_s = set(sorteios[ref])
        gm = {n: 0.0 for n in range(1, 26)}
        for x in ult_s:
            if x in tr:
                for suc, cnt in tr[x].items():
                    gm[suc] = gm.get(suc, 0) + cnt
        gm = normalizar(gm)

        tf = Counter()
        for s in sorteios[-min(500, len(sorteios)):]:
            for trio in combinations(sorted(s), 3):
                tf[trio] += 1
        gt = {n: 0.0 for n in range(1, 26)}
        for trio, cnt in tf.most_common(100):
            for x in trio:
                gt[x] += cnt
        gt = normalizar(gt)

        score_cooc = {n: gc[n] * 0.40 + gm[n] * 0.40 + gt[n] * 0.20 for n in range(1, 26)}
        candidatas = sorted(st.session_state["pipeline"], key=lambda n: -score_cooc[n])

        st.dataframe(pd.DataFrame([
            {"Dezena": n, "Score Cooc.": round(score_cooc[n], 4)}
            for n in candidatas
        ]), use_container_width=True, hide_index=True)

        if st.button("✅ Confirmar Etapa 7", type="primary"):
            salvar_etapa("coocorrencia", {
                "g3_cooc": gc,
                "g3_markov": gm,
                "g3_trios": gt,
                "score_cooc": score_cooc,
                "candidatas": candidatas,
            }, pipeline_update=candidatas)
            st.rerun()

# ============================================================
# ETAPA 8
# ============================================================
with tabs[7]:
    st.header("8️⃣ Monte Carlo / auditoria final")
    if not etapa_ok("coocorrencia"):
        st.warning("Execute as etapas anteriores primeiro.")
    elif etapa_ok("monte_carlo"):
        d = st.session_state["analise"]["monte_carlo"]
        st.success("✅ Etapa concluída")
        st.write(f"χ²={d['chi2']} | p={d['p_valor']} | Base {'✅ OK' if d['base_ok'] else '⚠️ desvio'}")
        st.write(f"Pool final: {' '.join(str(n).zfill(2) for n in d['pool_final'])}")
    else:
        chi2, p = qui_quadrado(sorteios, janela)
        st.write(f"Qui-quadrado: χ²={chi2} | p={p}")
        st.success("✅ Base sem desvio significativo" if p > 0.05 else "⚠️ Base com desvio")

        analise = st.session_state["analise"]
        f_sc = freq_simples(sorteios, janela)
        cand7 = st.session_state["pipeline"]
        score_final = {}
        for n in cand7:
            sc = 0.0
            if analise.get("termica") and n in analise["termica"]["quentes"]:
                sc += (10 - analise["termica"]["quentes"].index(n)) * 0.3 if n in analise["termica"]["quentes"][:10] else 0
            if analise.get("atraso"):
                if n in analise["atraso"]["vencidas"]:
                    sc += 5.0
                elif n in analise["atraso"]["no_ciclo"]:
                    sc += 2.5
            if analise.get("n12"):
                if n in analise["n12"]["fortes"]:
                    sc += 4.0
                elif n in analise["n12"]["apoio"]:
                    sc += 2.0
            if analise.get("coocorrencia"):
                sc += analise["coocorrencia"]["score_cooc"].get(n, 0) * 3.0
            sc += f_sc[n] / janela * 2.0
            score_final[n] = round(sc, 3)

        pool_ordenado = sorted(cand7, key=lambda n: -score_final[n])
        st.dataframe(pd.DataFrame([
            {"Rank": i + 1, "Dezena": n, "Score Final": score_final[n]}
            for i, n in enumerate(pool_ordenado)
        ]), use_container_width=True, hide_index=True)

        if st.button("✅ Confirmar Etapa 8", type="primary"):
            salvar_etapa("monte_carlo", {
                "chi2": chi2,
                "p_valor": p,
                "base_ok": p > 0.05,
                "pool_final": pool_ordenado,
                "score_final": score_final,
            }, pipeline_update=pool_ordenado)
            st.rerun()

# ============================================================
# BACKTESTING
# ============================================================
with tabs[8]:
    st.header("📊 Backtesting")
    if jogos_bt:
        for i, jogo in enumerate(jogos_bt):
            bt = backtesting(jogo, sorteios, min(200, len(sorteios)))
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Média", bt["media"])
            c2.metric("11+", f"{bt['pct11']}%")
            c3.metric("13+", f"{bt['pct13']}%")
            c4.metric("Máx", bt["max"])
            st.bar_chart(pd.DataFrame(list(bt["dist"].items()), columns=["Acertos", "Freq"]).set_index("Acertos"))
            st.markdown("---")
    else:
        st.info("Digite jogos no menu lateral.")

# ============================================================
# FILTROS
# ============================================================
with tabs[9]:
    st.header("🧪 Filtros")
    st.info(
        f"Soma {cfg['soma_min']}–{cfg['soma_max']} | "
        f"Pares {cfg['min_pares']}–{cfg['max_pares']} | "
        f"Repetição {cfg['rep_min']}–{cfg['rep_max']}"
    )
    if jogos_bt:
        for i, jogo in enumerate(jogos_bt):
            erros = filtrar(jogo, sorteios, cfg, sorteio_anterior=sorteios[ref])
            c = composicao(jogo)
            g = gaps(jogo)
            mold, miolo = moldura_miolo(jogo)
            st.subheader(f"Jogo {i+1}")
            c1, c2, c3 = st.columns(3)
            c1.write(f"Soma:{sum(jogo)} | Pares:{sum(1 for n in jogo if n%2==0)} | Seq:{contar_consec(jogo)[0]}")
            c2.write(f"Faixas:{dist_faixas(jogo)} | Quint:{dist_quintetos(jogo)}")
            c3.write(f"Primos:{c['primos']} | Fib:{c['fibonacci']} | Mold/Miolo:{mold}/{miolo}")
            if erros:
                for e in erros:
                    st.warning(e)
            else:
                st.success("✅ Passou em todos os filtros")
            st.markdown("---")
    else:
        st.info("Digite jogos no menu lateral.")

# ============================================================
# TENDÊNCIA
# ============================================================
with tabs[10]:
    st.header("📈 Tendência rolling")
    THRESH = 0.05
    esp = 15 / 25
    rows = []
    for n in range(1, 26):
        t20 = sum(1 for s in sorteios[-min(20, len(sorteios)):] if n in s) / min(20, len(sorteios))
        t50 = sum(1 for s in sorteios[-min(50, len(sorteios)):] if n in s) / min(50, len(sorteios))
        t200 = sum(1 for s in sorteios[-min(200, len(sorteios)):] if n in s) / min(200, len(sorteios))
        dcl = t20 - t200
        dml = t50 - t200
        if dcl > THRESH and dml > 0:
            status = "🔼 ALTA"
        elif dcl < -THRESH and dml < 0:
            status = "🔽 BAIXA"
        elif t50 > esp * 1.05:
            status = "🟢 ACIMA"
        elif t50 < esp * 0.95:
            status = "🟡 ABAIXO"
        else:
            status = "➡️ ESTÁVEL"
        rows.append({
            "Dezena": n,
            "J20%": round(t20 * 100, 1),
            "J50%": round(t50 * 100, 1),
            "J200%": round(t200 * 100, 1),
            "Δ(20-200)": f"{dcl * 100:+.1f}pp",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# EV
# ============================================================
with tabs[11]:
    st.header("⚖️ Valor esperado")
    ev = sum(PREMIOS[k] * PROB[k] for k in PREMIOS)
    c1, c2, c3 = st.columns(3)
    c1.metric("EV/jogo", f"R$ {round(ev, 4)}")
    c2.metric("Custo", "R$ 3,50")
    c3.metric("Ratio", round(ev / 3.5, 4))
    st.dataframe(pd.DataFrame([
        {
            "Faixa": k,
            "Prob": f"1 em {int(1 / v):,}",
            "Prêmio": f"R$ {PREMIOS[k]:,.0f}",
            "EV": f"R$ {PREMIOS[k] * v:.4f}",
        }
        for k, v in sorted(PROB.items(), reverse=True)
    ]), use_container_width=True, hide_index=True)

# ============================================================
# RELATÓRIO
# ============================================================
with tabs[12]:
    st.header("🏆 Relatório final")
    analise = st.session_state["analise"]
    ok, tot = status_count()
    st.markdown(f"**Base:** {len(sorteios)} concursos | C{df['Concurso'].max()} | Janela:{janela} | Etapas:{ok}/{tot}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Cascata")
        for e in ETAPAS:
            icon = "✅" if etapa_ok(e) else "⭕"
            d = analise.get(e)
            if d and "candidatas" in d:
                st.write(f"{icon} {ETAPA_LABELS[e]} → {len(d['candidatas'])} candidatas")
            else:
                st.write(f"{icon} {ETAPA_LABELS[e]}")

    with c2:
        st.subheader("🎯 Pool final")
        if etapa_ok("monte_carlo"):
            d = analise["monte_carlo"]
            for i, n in enumerate(d["pool_final"][:20]):
                st.write(f"#{i+1} {n:02d} — score {d['score_final'][n]}")

    if st.session_state["jogos_gerados"]:
        st.subheader("🩺 Auditoria do portfólio")
        sorteio_anterior = sorteios[ref]
        audit = auditoria_portfolio(
            jogos=st.session_state["jogos_gerados"],
            fortes=sorted(list(analise["n12"]["fortes"])) if etapa_ok("n12") else [],
            apoio=sorted(list(analise["n12"]["apoio"])) if etapa_ok("n12") else [],
            vencidas=sorted(list(analise["atraso"]["vencidas"])) if etapa_ok("atraso") else [],
            sorteio_anterior=sorteio_anterior,
        )
        st.write(f"Núcleo comum: {' '.join(str(n).zfill(2) for n in audit['nucleo_comum'])}")
        st.write(f"Repetições por jogo: {audit['repeticoes']}")
        if audit["cegas_criticas"]:
            st.warning("Dezenas críticas fora do portfólio: " + " ".join(str(n).zfill(2) for n in audit["cegas_criticas"]))
        else:
            st.success("Nenhuma dezena crítica ficou cega no portfólio.")

# ============================================================
# GERADOR
# ============================================================
with tabs[13]:
    st.header("🎰 Gerador — Protocolo Hard V4.2")

    if not todas_ok():
        st.error(f"⛔ {tot-ok} etapa(s) pendente(s)")
        for e in ETAPAS:
            st.write(f"{'✅' if etapa_ok(e) else '❌'} {ETAPA_LABELS[e]}")

        if st.button("⚡ Executar todas as etapas automaticamente", type="primary"):
            f = freq_simples(sorteios, janela)
            top10 = [n for n, _ in f.most_common(10)]
            bot5 = [n for n, _ in f.most_common()[:-6:-1]]
            pico = dezenas_pico(sorteios)
            cand1 = list(dict.fromkeys(top10 + pico))
            salvar_etapa("termica", {"quentes": top10, "frias": bot5, "pico": pico, "candidatas": cand1, "freq": {n: f[n] for n in range(1, 26)}}, pipeline_update=cand1)

            percentis = calcular_percentis_atraso(st_tuple)
            df_at = calcular_atrasos(sorteios)
            ciclos = [{"Dezena": n, "Ciclo Médio": ciclo_medio(sorteios, n)} for n in range(1, 26)]
            df_comp = df_at.merge(pd.DataFrame(ciclos), on="Dezena")
            df_comp["Status"] = df_comp["Atraso"].apply(lambda x: classificar_atraso(x, percentis))
            vencidas = df_comp[df_comp["Status"] == "🔴 VENCIDA"]["Dezena"].tolist()
            no_ciclo = df_comp[df_comp["Status"] == "🟡 No ciclo"]["Dezena"].tolist()
            atrasadas = df_comp[df_comp["Status"] == "🟠 Atrasada"]["Dezena"].tolist()
            top5 = df_comp.sort_values("Atraso", ascending=False).head(5)["Dezena"].tolist()
            prioridade = vencidas if vencidas else (no_ciclo + atrasadas if (no_ciclo or atrasadas) else top5)
            cand2 = list(dict.fromkeys(cand1 + prioridade))
            salvar_etapa("atraso", {"df_completo": df_comp, "vencidas": vencidas, "no_ciclo": no_ciclo, "atrasadas": atrasadas, "top5_atraso": top5, "prioridade": prioridade, "candidatas": cand2}, pipeline_update=cand2)

            sm, sx, smed = faixa_soma(sorteios_j)
            dp = dist_pares(sorteios_j)
            top_pares = max(dp, key=dp.get)
            salvar_etapa("estrutural", {"soma_min": sm, "soma_max": sx, "soma_med": smed, "top_pares": top_pares, "pct_top_pares": dp[top_pares], "dist_pares": dp, "candidatas": cand2}, pipeline_update=cand2)

            dp_c, df_c = dist_comp(sorteios_j)
            salvar_etapa("composicao", {"top_primo": max(dp_c, key=dp_c.get), "top_fib": max(df_c, key=df_c.get), "dist_primo": dp_c, "dist_fib": df_c, "candidatas": cand2}, pipeline_update=cand2)

            dc = dist_consec(sorteios_j)
            dg = dist_gaps(sorteios_j)
            salvar_etapa("sequencias", {"top_consec": max(dc, key=dc.get), "pct_consec": dc[max(dc, key=dc.get)], "top_gap": max(dg, key=dg.get), "pct_gap": dg[max(dg, key=dg.get)], "dist_consec": dc, "dist_gap": dg, "candidatas": cand2}, pipeline_update=cand2)

            sim, fortes, apoio = projetar_isca(ref, sorteios, threshold)
            cand6 = list(dict.fromkeys(cand2 + fortes + apoio))
            cand6 = sorted(cand6, key=lambda n: (0 if n in fortes else 1 if n in apoio else 2, n))
            salvar_etapa("n12", {"similares": sim, "fortes": fortes, "apoio": apoio, "conc_ref": int(df.iloc[ref]["Concurso"]), "threshold": threshold, "candidatas": cand6}, pipeline_update=cand6)

            pf = Counter()
            for s in sorteios[-min(500, len(sorteios)):]:
                for a, b in combinations(sorted(s), 2):
                    pf[(a, b)] += 1
            gc = {n: 0.0 for n in range(1, 26)}
            for (a, b), cnt in pf.items():
                gc[a] += cnt
                gc[b] += cnt
            gc = normalizar(gc)

            tr = calc_markov(st_tuple)
            ult_s = set(sorteios[ref])
            gm = {n: 0.0 for n in range(1, 26)}
            for x in ult_s:
                if x in tr:
                    for suc, cnt in tr[x].items():
                        gm[suc] = gm.get(suc, 0) + cnt
            gm = normalizar(gm)

            tf = Counter()
            for s in sorteios[-min(500, len(sorteios)):]:
                for trio in combinations(sorted(s), 3):
                    tf[trio] += 1
            gt = {n: 0.0 for n in range(1, 26)}
            for trio, cnt in tf.most_common(100):
                for x in trio:
                    gt[x] += cnt
            gt = normalizar(gt)

            score_cooc = {n: gc[n] * 0.40 + gm[n] * 0.40 + gt[n] * 0.20 for n in range(1, 26)}
            cand7 = sorted(cand6, key=lambda n: -score_cooc[n])
            salvar_etapa("coocorrencia", {"g3_cooc": gc, "g3_markov": gm, "g3_trios": gt, "score_cooc": score_cooc, "candidatas": cand7}, pipeline_update=cand7)

            chi2, p = qui_quadrado(sorteios, janela)
            f_sc = freq_simples(sorteios, janela)
            score_final = {}
            for n in cand7:
                sc = 0.0
                if n in top10:
                    sc += (10 - top10.index(n)) * 0.3
                if n in vencidas:
                    sc += 5.0
                elif n in no_ciclo:
                    sc += 2.5
                if n in fortes:
                    sc += 4.0
                elif n in apoio:
                    sc += 2.0
                sc += score_cooc.get(n, 0) * 3.0
                sc += f_sc[n] / janela * 2.0
                score_final[n] = round(sc, 3)
            pool_final = sorted(cand7, key=lambda n: -score_final[n])
            salvar_etapa("monte_carlo", {"chi2": chi2, "p_valor": p, "base_ok": p > 0.05, "pool_final": pool_final, "score_final": score_final}, pipeline_update=pool_final)
            st.rerun()

    else:
        analise = st.session_state["analise"]
        pool_final = analise["monte_carlo"]["pool_final"]
        score_final = analise["monte_carlo"]["score_final"]
        vencidas_g = analise["atraso"]["vencidas"]
        fortes_g = analise["n12"]["fortes"]
        apoio_g = analise["n12"]["apoio"]

        st.success("🎯 Protocolo completo")
        perfil_nome = st.radio("Perfil", list(PERFIS.keys()), horizontal=True)
        perfil = PERFIS[perfil_nome]
        st.info(perfil["descricao"])

        c1, c2 = st.columns(2)
        with c1:
            qtd = st.number_input("Quantos jogos?", min_value=1, value=3, step=1)
            st.warning(f"💰 {qtd} × R$ 3,50 = R$ {qtd*3.5:.2f}")
        with c2:
            seed = st.number_input("Seed (0=aleatório)", min_value=0, value=0, step=1)
            fixas_inp = st.text_input("Dezenas fixas", placeholder="Ex: 05 13")

        fixas = []
        if fixas_inp.strip():
            try:
                fixas = [int(x) for x in fixas_inp.strip().split() if 1 <= int(x) <= 25]
            except Exception:
                fixas = []

        if st.button("🎯 Gerar combinações", type="primary"):
            if seed > 0:
                random.seed(seed)
                np.random.seed(seed)

            prog = st.progress(0, text="Calculando scores...")
            scores, fortes_calc, apoio_calc, vencidas_calc, _ = calcular_scores(
                sorteios, janela, ref, threshold, perfil["pesos"], analise
            )

            ranking = sorted(scores.items(), key=lambda x: -x[1])
            pool_obrig = list(dict.fromkeys(vencidas_g))
            pool_score = [n for n, _ in ranking if n not in pool_obrig]

            if "Agressiva" in perfil_nome:
                pool_din = max(perfil["top_pool"], min(qtd + 3, 22))
            else:
                pool_din = max(perfil["top_pool"], min(qtd + 5, 25))

            sorteio_anterior = sorteios[ref]
            familias = [
                {"nome": "protecao", "rep_min": 8, "rep_max": 10, "qmax": 4},
                {"nome": "equilibrado", "rep_min": 7, "rep_max": 9, "qmax": 4},
                {"nome": "agressivo", "rep_min": 6, "rep_max": 9, "qmax": 5},
            ]

            pool = pool_obrig + [n for n in pool_score if n not in pool_obrig][:pool_din]
            pool = list(dict.fromkeys(pool))
            for f in fixas:
                if f not in pool:
                    pool.append(f)

            st.write(f"**Pool ({len(pool)}):** {' · '.join(f'{n:02d}' for n in sorted(pool))}")

            jogos = []
            jogos_meta = []
            tent = 0
            max_t = max(200000, qtd * 12000)
            mc_atual = perfil["max_comum"]

            prog.progress(10, text="Gerando com repetição e estrutura...")
            while len(jogos) < qtd and tent < max_t:
                tent += 1
                meta = familias[min(len(jogos), len(familias) - 1)]

                base = fixas[:15]
                obrigatorias_jogo = []
                if vencidas_g:
                    obrigatorias_jogo.extend(vencidas_g[:1])
                if fortes_g:
                    obrigatorias_jogo.extend(sorted(list(fortes_g))[:2])
                if apoio_g and len(obrigatorias_jogo) < 4:
                    obrigatorias_jogo.extend(sorted(list(apoio_g))[:1])

                for n in obrigatorias_jogo:
                    if n not in base and len(base) < 15:
                        base.append(n)

                rest = [n for n in pool if n not in base]
                if len(rest) < 15 - len(base):
                    ext = [n for n in range(1, 26) if n not in base and n not in rest]
                    rest += ext

                faltam = 15 - len(base)
                universo = rest[:max(faltam + 12, 18)]
                rest_sc = np.array([scores[n] for n in universo], dtype=float)
                rest_sc = rest_sc / rest_sc.sum()

                escolh = list(np.random.choice(universo, size=faltam, replace=False, p=rest_sc))
                cand = sorted(base + escolh)

                erros = filtrar(
                    cand,
                    sorteios,
                    cfg,
                    sorteio_anterior=sorteio_anterior,
                    meta_rep_min=meta["rep_min"],
                    meta_rep_max=meta["rep_max"],
                    meta_qmax=meta["qmax"],
                    exigir_linhas=True,
                    exigir_colunas=True,
                )
                if erros:
                    continue

                if any((15 - hamming(cand, j)) > mc_atual for j in jogos):
                    continue

                if jogos:
                    comum_global = set(cand)
                    for j in jogos:
                        comum_global &= set(j)
                    if len(comum_global) > mc_atual:
                        continue

                if cand not in jogos:
                    jogos.append(cand)
                    jogos_meta.append(meta)
                    prog.progress(min(10 + int(len(jogos) / qtd * 85), 95), text=f"Gerados {len(jogos)}/{qtd}")

            prog.progress(100, text="Concluído")

            if len(jogos) < qtd:
                st.warning(f"Gerados {len(jogos)}/{qtd}. Ajuste o perfil ou a seed.")

            if jogos:
                st.success(f"✅ {len(jogos)} combinação(ões) — {tent:,} tentativas")

                audit = auditoria_portfolio(
                    jogos=jogos,
                    fortes=sorted(list(fortes_g)),
                    apoio=sorted(list(apoio_g)),
                    vencidas=sorted(list(vencidas_g)),
                    sorteio_anterior=sorteio_anterior,
                )
                if audit["cegas_criticas"]:
                    st.warning("⚠️ Dezenas críticas fora do portfólio: " + " ".join(f"{n:02d}" for n in audit["cegas_criticas"]))
                st.info(f"Núcleo comum: {' '.join(f'{n:02d}' for n in audit['nucleo_comum'])} ({len(audit['nucleo_comum'])})")
                st.info("Repetição por jogo com o último concurso: " + " | ".join(str(x) for x in audit["repeticoes"]))

                st.markdown("---")
                for i, jogo in enumerate(jogos):
                    bt = backtesting(jogo, sorteios, min(200, len(sorteios)))
                    c = composicao(jogo)
                    f_j = dist_faixas(jogo)
                    q_j = dist_quintetos(jogo)
                    mold, miolo = moldura_miolo(jogo)
                    mx_j, _ = contar_consec(jogo)
                    sc_j = round(sum(scores[n] for n in jogo), 4)
                    rep_j = repeticao_ultimo(jogo, sorteio_anterior)

                    st.subheader(f"🃏 Combinação {i+1} — {jogos_meta[i]['nome'].upper()}")
                    st.markdown("## " + " · ".join(f"**{n:02d}**" for n in jogo))

                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Score", sc_j)
                    k2.metric("Soma", sum(jogo))
                    k3.metric("Par/Ímp", f"{sum(1 for n in jogo if n%2==0)}/{sum(1 for n in jogo if n%2!=0)}")
                    k4.metric("11+", f"{bt['pct11']}%")
                    k5.metric("Repetição", rep_j)

                    c6, c7, c8 = st.columns(3)
                    c6.write(f"Faixas:{f_j} | Quint:{q_j}")
                    c7.write(f"Mold/Miolo:{mold}/{miolo}")
                    c8.write(f"Primos:{c['primos']} | Fib:{c['fibonacci']} | Seq:{mx_j}")

                    st.success("✅ Aprovado — filtros V4.2")
                    st.markdown("---")

                if len(jogos) > 1:
                    st.subheader("📐 Distância entre jogos")
                    st.dataframe(cobertura_jogos(jogos), use_container_width=True, hide_index=True)

                st.session_state["jogos_gerados"] = jogos
            else:
                st.error("Nenhuma combinação gerada.")
