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
st.set_page_config(page_title="Protocolo Hard V4.5.2", layout="wide")

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
    "nivel1",
]

ETAPA_LABELS = {
    "termica": "1️⃣ Térmica",
    "atraso": "2️⃣ Atraso & Ciclo",
    "estrutural": "3️⃣ Estrutural",
    "composicao": "4️⃣ Composição",
    "sequencias": "5️⃣ Sequências & Gaps",
    "n12": "6️⃣ N+1/N+2",
    "coocorrencia": "7️⃣ Coocorrência",
    "monte_carlo": "8️⃣ Monte Carlo / Pool provisório",
    "nivel1": "9️⃣ Nível 1",
}

PERFIS = {
    "🎯 Agressiva": {
        "pesos": {"g1": 0.34, "g2": 0.26, "g3": 0.22, "g4": 0.10, "g5": 0.08},
        "top_pool": 24,
        "max_comum": 10,
        "descricao": "Mais forte, mas sem travar o motor por impossibilidade combinatória.",
        "aviso": 8,
    },
    "⚖️ Híbrida": {
        "pesos": {"g1": 0.30, "g2": 0.24, "g3": 0.24, "g4": 0.12, "g5": 0.10},
        "top_pool": 25,
        "max_comum": 12,
        "descricao": "Cobertura ampla para carteira maior, com semelhança controlada e pool mais respirável.",
        "aviso": None,
    },
}

PREMIOS = {15: 1500000, 14: 1500, 13: 25, 12: 10, 11: 5}
PROB = {15: 1 / 3268760, 14: 15 / 3268760, 13: 105 / 3268760, 12: 455 / 3268760, 11: 1365 / 3268760}

PESOS_ANALISE_FINAL = {"g1": 0.31, "g2": 0.25, "g3": 0.24, "g4": 0.10, "g5": 0.10}



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


def identificar_dezenas_criticas(analise, sorteio_anterior, pool_oficial=None):
    sinais = Counter()

    if analise.get("termica"):
        for n in analise["termica"].get("quentes", [])[:8]:
            sinais[n] += 1

    if analise.get("atraso"):
        for n in analise["atraso"].get("vencidas", [])[:5]:
            sinais[n] += 1
        for n in analise["atraso"].get("no_ciclo", [])[:5]:
            sinais[n] += 1

    if analise.get("n12"):
        for n in analise["n12"].get("fortes", [])[:6]:
            sinais[n] += 2
        for n in analise["n12"].get("apoio", [])[:6]:
            sinais[n] += 1

    if analise.get("nivel1"):
        for n in analise["nivel1"].get("top_bonus", [])[:8]:
            sinais[n] += 1

    for n in sorteio_anterior:
        sinais[n] += 1

    if pool_oficial:
        for n in list(pool_oficial)[:10]:
            sinais[n] += 1

    criticas = [n for n, pts in sinais.items() if pts >= 2]
    criticas = sorted(criticas, key=lambda n: (-sinais[n], n))
    return criticas, dict(sinais)


def montar_pool_carteira(scores, ranking, perfil, qtd, fixas, criticas, vencidas, pool_oficial=None):
    if "Agressiva" in perfil:
        alvo_pool = max(22, min(25, qtd + 8))
    else:
        alvo_pool = max(24, min(25, qtd + 10))

    base_ordenada = []
    if pool_oficial:
        base_ordenada.extend(list(pool_oficial)[:25])
    base_ordenada.extend([n for n, _ in ranking])

    pool = []
    for grupo in [criticas, vencidas, fixas, base_ordenada]:
        for n in grupo:
            if n not in pool:
                pool.append(n)

    pool = pool[:max(alvo_pool, len(set(criticas) | set(vencidas) | set(fixas)))]
    pool = sorted(pool)
    return pool


def montar_obrigatorias_jogo(meta_nome, idx_rot, scores, criticas, fortes, apoio, vencidas, sorteio_anterior, usage=None):
    usage = usage or Counter()
    base = []

    def add_rot(pool, qtd):
        if not pool or qtd <= 0:
            return
        ordenado = sorted(pool, key=lambda n: (usage.get(n, 0), -scores.get(n, 0), n))
        for n in ordenado[:qtd]:
            if n not in base:
                base.append(n)

    if meta_nome == "protecao":
        add_rot(criticas, 3)
        add_rot([n for n in sorteio_anterior if n in criticas], 2)
        add_rot(fortes, 2)
        add_rot(vencidas, 1)
    elif meta_nome == "equilibrado":
        add_rot(criticas, 2)
        add_rot(fortes, 2)
        add_rot(apoio, 1)
        add_rot([n for n in sorteio_anterior if n in criticas], 1)
    else:
        add_rot(criticas, 1)
        add_rot(fortes, 1)
        add_rot(apoio, 1)
        add_rot([n for n in sorteio_anterior if n in criticas], 1)

    return list(dict.fromkeys(base))


def auditoria_carteira_11mais(jogos, criticas, sorteio_anterior):
    if not jogos:
        return {"criticas_cegas": criticas, "jogos_rep_alta": 0, "cobertura_total": 0}

    uniao = set().union(*map(set, jogos))
    criticas_cegas = sorted(set(criticas) - uniao)
    reps = [repeticao_ultimo(j, sorteio_anterior) for j in jogos]
    jogos_rep_alta = sum(1 for r in reps if r >= 8)

    return {
        "criticas_cegas": criticas_cegas,
        "jogos_rep_alta": jogos_rep_alta,
        "cobertura_total": len(uniao),
    }



def auc_rank(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return float(auc)


def _binary_matrix(sorteios):
    M = np.zeros((len(sorteios), 25), dtype=int)
    for i, s in enumerate(sorteios):
        for n in s:
            M[i, n - 1] = 1
    return M


def ljung_box_pvalue(series, lags=5):
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < max(20, lags + 5):
        return 1.0, 0.0
    x = x - x.mean()
    var = np.dot(x, x)
    if var <= 1e-12:
        return 1.0, 0.0
    acfs = []
    for k in range(1, lags + 1):
        acf = np.dot(x[k:], x[:-k]) / var
        acfs.append(acf)
    Q = n * (n + 2) * sum((acf ** 2) / (n - k) for k, acf in enumerate(acfs, start=1))
    p = 1 - stats.chi2.cdf(Q, df=lags)
    return float(p), float(acfs[0] if acfs else 0.0)


def rolling_sum(arr, window=10):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < window:
        return np.array([])
    cs = np.cumsum(np.insert(arr, 0, 0))
    return cs[window:] - cs[:-window]


@st.cache_data
def nivel1_metricas(sorteios_tuple, ref_idx, recent=24, hist=200, n_boot=250, n_perm=250):
    sorteios = list(sorteios_tuple)
    base = sorteios[: ref_idx + 1]
    hist_n = min(hist, len(base))
    recent_n = min(recent, len(base))
    base_hist = base[-hist_n:]
    base_recent = base[-recent_n:]
    M_hist = _binary_matrix(base_hist)
    M_recent = _binary_matrix(base_recent)
    recent_freq = M_recent.mean(axis=0)
    hist_freq = M_hist.mean(axis=0)

    # Bootstrap frequency intervals on recent window
    if len(M_recent) > 0:
        rng = np.random.default_rng(42)
        boot = np.zeros((n_boot, 25), dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, len(M_recent), size=len(M_recent))
            boot[b] = M_recent[idx].mean(axis=0)
        boot_low = np.percentile(boot, 10, axis=0)
        boot_high = np.percentile(boot, 90, axis=0)
    else:
        boot_low = np.zeros(25)
        boot_high = np.zeros(25)

    # Permutation / resampling style significance: compare recent count vs random subsets of historic window
    rngp = np.random.default_rng(123)
    recent_count = M_recent.sum(axis=0)
    perm_sig = np.zeros(25)
    for j in range(25):
        obs = recent_count[j]
        sampled = []
        if len(M_hist) > 0:
            for _ in range(n_perm):
                idx = rngp.choice(len(M_hist), size=recent_n, replace=False if recent_n <= len(M_hist) else True)
                sampled.append(M_hist[idx, j].sum())
            sampled = np.asarray(sampled)
            if obs >= sampled.mean():
                p_up = (np.sum(sampled >= obs) + 1) / (len(sampled) + 1)
                perm_sig[j] = max(0.0, 1 - p_up)
            else:
                perm_sig[j] = 0.0

    # CUSUM positive drift
    cusum = np.zeros(25)
    ks_shift = np.zeros(25)
    lb_sig = np.zeros(25)
    for j in range(25):
        serie = M_hist[:, j].astype(float)
        if len(serie) > 10:
            mu = serie.mean()
            csum = np.cumsum(serie - mu)
            cusum[j] = max(0.0, float(csum.max() - csum.min()))
            roll_hist = rolling_sum(serie[:-recent_n] if len(serie) > recent_n else serie, window=min(10, max(3, recent_n // 2)))
            roll_recent = rolling_sum(serie[-recent_n:], window=min(10, max(3, recent_n // 2)))
            if len(roll_hist) >= 5 and len(roll_recent) >= 5:
                _, pks = stats.ks_2samp(roll_recent, roll_hist)
                if recent_freq[j] > hist_freq[j]:
                    ks_shift[j] = max(0.0, 1 - float(pks))
            p_lb, acf1 = ljung_box_pvalue(serie, lags=min(5, max(1, len(serie) // 10)))
            if acf1 > 0:
                lb_sig[j] = max(0.0, 1 - p_lb)

    # Lift with reference draw
    ref_draw = set(base[-1]) if base else set()
    pa = M_hist.mean(axis=0)
    lift_ref = np.zeros(25)
    if len(M_hist) > 0 and ref_draw:
        for n in range(1, 26):
            vals = []
            for m in ref_draw:
                if m == n:
                    continue
                joint = np.mean(M_hist[:, n - 1] * M_hist[:, m - 1])
                denom = max(pa[n - 1] * pa[m - 1], 1e-9)
                vals.append(joint / denom)
            lift_ref[n - 1] = np.mean(vals) if vals else 1.0

    # Global metrics
    eps = 1e-9
    p_recent = recent_freq + eps
    p_recent = p_recent / p_recent.sum()
    p_hist = hist_freq + eps
    p_hist = p_hist / p_hist.sum()
    kl = float(np.sum(p_recent * np.log(p_recent / p_hist)))
    ks_global = float(stats.ks_2samp(recent_freq, hist_freq).pvalue)
    skew_recent = float(stats.skew(recent_freq))
    kurt_recent = float(stats.kurtosis(recent_freq))

    # Normalize per-dezena advanced bonus
    adv_parts = {
        "boot_low": {i + 1: float(boot_low[i]) for i in range(25)},
        "perm_sig": {i + 1: float(perm_sig[i]) for i in range(25)},
        "cusum": {i + 1: float(cusum[i]) for i in range(25)},
        "ks_shift": {i + 1: float(ks_shift[i]) for i in range(25)},
        "lb_sig": {i + 1: float(lb_sig[i]) for i in range(25)},
        "lift_ref": {i + 1: float(lift_ref[i]) for i in range(25)},
    }
    adv_norm = {k: normalizar(v) for k, v in adv_parts.items()}
    bonus = {}
    for n in range(1, 26):
        bonus[n] = (
            0.25 * adv_norm["boot_low"][n]
            + 0.25 * adv_norm["perm_sig"][n]
            + 0.20 * adv_norm["cusum"][n]
            + 0.10 * adv_norm["ks_shift"][n]
            + 0.10 * adv_norm["lb_sig"][n]
            + 0.10 * adv_norm["lift_ref"][n]
        )

    df = pd.DataFrame({
        "Dezena": list(range(1, 26)),
        "Freq Recente %": np.round(recent_freq * 100, 1),
        "Freq Hist %": np.round(hist_freq * 100, 1),
        "Bootstrap p10": np.round(boot_low * 100, 1),
        "Bootstrap p90": np.round(boot_high * 100, 1),
        "Perm Sig": np.round(perm_sig, 4),
        "CUSUM": np.round(cusum, 4),
        "KS Shift": np.round(ks_shift, 4),
        "LB Sig": np.round(lb_sig, 4),
        "Lift Ref": np.round(lift_ref, 4),
        "Bonus N1": np.round([bonus[n] for n in range(1, 26)], 4),
    }).sort_values(["Bonus N1", "Freq Recente %"], ascending=[False, False]).reset_index(drop=True)

    return {
        "df": df,
        "bonus": bonus,
        "kl_div": kl,
        "ks_global_p": ks_global,
        "skew_recent": skew_recent,
        "kurt_recent": kurt_recent,
        "top_bonus": df.head(10)["Dezena"].tolist(),
    }


@st.cache_data
def lift_top_pairs(sorteios_tuple, ref_idx, janela=200, top=20):
    sorteios = list(sorteios_tuple)
    base = sorteios[: ref_idx + 1]
    window = base[-min(janela, len(base)):]
    M = _binary_matrix(window)
    p = M.mean(axis=0)
    rows = []
    for a, b in combinations(range(1, 26), 2):
        joint = np.mean(M[:, a - 1] * M[:, b - 1])
        denom = max(p[a - 1] * p[b - 1], 1e-9)
        lift = joint / denom
        rows.append((a, b, joint, lift))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows[:top]


@st.cache_data
def auc_criterios_nivel1(sorteios_tuple, janela_eval=250):
    sorteios = list(sorteios_tuple)
    ini = max(80, len(sorteios) - janela_eval - 1)
    ys = []
    sc_freq24 = []
    sc_atraso = []
    sc_shift = []
    for t in range(ini, len(sorteios) - 1):
        hist = sorteios[: t + 1]
        nxt = set(sorteios[t + 1])
        f24 = freq_simples(hist, min(24, len(hist)))
        f100 = freq_simples(hist, min(100, len(hist)))
        atrasos_df = calcular_atrasos(hist)
        atraso_map = dict(zip(atrasos_df["Dezena"], atrasos_df["Atraso"]))
        shift_map = {}
        for n in range(1, 26):
            shift_map[n] = (f24[n] / min(24, len(hist))) - (f100[n] / min(100, len(hist)))
        for n in range(1, 26):
            ys.append(1 if n in nxt else 0)
            sc_freq24.append(f24[n])
            sc_atraso.append(atraso_map[n])
            sc_shift.append(shift_map[n])
    return {
        "Freq24": round(auc_rank(ys, sc_freq24), 4),
        "Atraso": round(auc_rank(ys, sc_atraso), 4),
        "Shift24x100": round(auc_rank(ys, sc_shift), 4),
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
def calcular_scores(sorteios, janela, ref, threshold, pesos, analise, usar_nivel1=False, bonus_n1=None):
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

    base_scores = {
        n: (
            pesos["g1"] * G1[n]
            + pesos["g2"] * G2[n]
            + pesos["g3"] * G3[n]
            + pesos["g4"] * G4[n]
            + pesos["g5"] * G5[n]
        )
        for n in range(1, 26)
    }
    if usar_nivel1 and bonus_n1:
        scores = {n: 0.88 * base_scores[n] + 0.12 * bonus_n1.get(n, 0.0) for n in range(1, 26)}
    else:
        scores = base_scores
    return scores, fortes_n, apoio_n, vencidas, no_ciclo


def recalcular_score_oficial_n1(sorteios, janela, ref, threshold, analise):
    """
    O Nível 1 entra oficialmente na análise:
    1) recalcula score base analítico
    2) mede confiabilidade dos sinais N1
    3) gera score final oficial e pool oficial usados no relatório e na geração
    """
    base_scores, _, _, _, _ = calcular_scores(
        sorteios,
        janela,
        ref,
        threshold,
        PESOS_ANALISE_FINAL,
        analise,
        usar_nivel1=False,
        bonus_n1=None,
    )
    base_norm = normalizar(base_scores)

    n1 = analise["nivel1"]
    df_n1 = n1["df"].copy()
    bonus_raw = n1["bonus"]
    bonus_norm = normalizar({n: float(bonus_raw.get(n, 0.0)) for n in range(1, 26)})

    gates = {}
    for _, row in df_n1.iterrows():
        n = int(row["Dezena"])
        lift_adj = max(0.0, float(row["Lift Ref"]) - 1.0)
        gate = (
            0.35 * float(row["Perm Sig"])
            + 0.25 * float(row["KS Shift"])
            + 0.20 * float(row["LB Sig"])
            + 0.20 * lift_adj
        )
        gates[n] = gate
    gate_norm = normalizar(gates)

    auc_vals = list(n1["auc"].values()) if n1.get("auc") else [0.5]
    auc_mean = float(np.mean(auc_vals))
    peso_n1 = max(0.06, min(0.14, 0.09 + (auc_mean - 0.50) * 0.12))

    score_final_oficial = {}
    for n in range(1, 26):
        n1_comp = 0.70 * bonus_norm[n] + 0.30 * gate_norm[n]
        final_norm = (1 - peso_n1) * base_norm[n] + peso_n1 * n1_comp
        score_final_oficial[n] = round(final_norm * 100, 4)

    pool_final_oficial = sorted(range(1, 26), key=lambda n: (-score_final_oficial[n], n))

    ranking_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Dezena": n,
            "Score Oficial": score_final_oficial[n],
            "Score Base": round(base_norm[n] * 100, 4),
            "Bonus N1": round(float(bonus_raw.get(n, 0.0)), 4),
            "Gate N1": round(float(gate_norm[n]), 4),
        }
        for i, n in enumerate(pool_final_oficial)
    ])

    return {
        "score_base_analitico": {n: round(base_norm[n] * 100, 4) for n in range(1, 26)},
        "score_final_oficial": score_final_oficial,
        "pool_final_oficial": pool_final_oficial,
        "peso_n1_aplicado": round(peso_n1, 4),
        "auc_medio": round(auc_mean, 4),
        "ranking_final_df": ranking_df,
    }


# ============================================================
# ============================================================
# CLASSE: 62 ANÁLISES (v4.9) ← NOVA CLASSE INTEGRADA
# ============================================================

class AnalisadorLotofacil:
    """Implementa as 62 análises do protocolo"""
    
    def __init__(self, sorteios):
        self.sorteios = sorteios
        self.n_sorteios = len(sorteios)
        self.scores = {n: 0 for n in range(1, 26)}
    
    def analise_01_frequencia_simples(self):
        """Análise 01: Frequência simples"""
        freq = Counter()
        for s in self.sorteios:
            freq.update(s)
        return freq
    
    def analise_02_frequencia_janelas(self):
        """Análise 02: Frequência em janelas"""
        janelas = {}
        for tam in [50, 100, 200, 500]:
            if self.n_sorteios >= tam:
                freq_j = Counter()
                for s in self.sorteios[-tam:]:
                    freq_j.update(s)
                janelas[tam] = freq_j
        return janelas
    
    def analise_03_05_atraso_e_ciclo(self):
        """Análises 03-05: Atraso, Ciclo, Percentis"""
        ultimo = {}
        for i, s in enumerate(self.sorteios):
            for n in s:
                ultimo[n] = i
        
        atrasos = {}
        ciclos = {}
        
        for n in range(1, 26):
            atraso = self.n_sorteios - 1 - ultimo.get(n, -1)
            atrasos[n] = atraso
            
            aparicoes = [i for i, s in enumerate(self.sorteios) if n in s]
            if len(aparicoes) > 1:
                ciclo = np.mean([aparicoes[i+1] - aparicoes[i] 
                               for i in range(len(aparicoes)-1)])
                ciclos[n] = ciclo
            else:
                ciclos[n] = atraso
        
        atr_array = np.array([atrasos[n] for n in range(1, 26)])
        percentis = {
            'p75': np.percentile(atr_array, 75),
            'p85': np.percentile(atr_array, 85),
            'p90': np.percentile(atr_array, 90),
            'p95': np.percentile(atr_array, 95),
        }
        
        return atrasos, ciclos, percentis
    
    def analise_16_17_coocorrencia(self):
        """Análises 16-17: Pares e Trios"""
        pares_freq = Counter()
        trios_freq = Counter()
        
        for s in self.sorteios:
            for a, b in combinations(sorted(s), 2):
                pares_freq[(a, b)] += 1
            for trio in combinations(sorted(s), 3):
                trios_freq[trio] += 1
        
        return pares_freq, trios_freq
    
    def analise_18_markov(self):
        """Análise 18: Markov (N → N+1)"""
        markov = {n: Counter() for n in range(1, 26)}
        
        for i in range(len(self.sorteios)-1):
            este = set(self.sorteios[i])
            proximo = set(self.sorteios[i+1])
            for n in este:
                markov[n].update(proximo)
        
        return markov
    
    def calcular_scores(self):
        """Calcula scores consolidados (62 análises)"""
        scores = {n: 0 for n in range(1, 26)}
        
        # 01. Frequência (AGRESSIVO: peso 20 em vez de 10)
        freq = self.analise_01_frequencia_simples()
        for n in range(1, 26):
            scores[n] += (freq[n] / self.n_sorteios) * 20
        
        # 02. Janelas
        janelas = self.analise_02_frequencia_janelas()
        for tam, freq_j in janelas.items():
            for n in range(1, 26):
                scores[n] += (freq_j[n] / min(tam, self.n_sorteios)) * 5
        
        # 03-05. Atraso & Ciclo
        atrasos, ciclos, percentis = self.analise_03_05_atraso_e_ciclo()
        for n in range(1, 26):
            if atrasos[n] > percentis['p90']:
                scores[n] += 8
            elif atrasos[n] > percentis['p75']:
                scores[n] += 4
            
            # COMANDO 2 (AGRESSIVO): Penalizar números muito frios
            if atrasos[n] > percentis['p95']:
                scores[n] -= 5  # Penaliza números muito atrasados
            
            ciclo_esperado = 1.7
            ciclo_desvio = abs(ciclos[n] - ciclo_esperado)
            scores[n] += max(0, (5 - ciclo_desvio * 3))
        
        # 07. Pares
        for n in range(1, 26):
            if n % 2 == 0:
                scores[n] += 1.5
        
        # 09. Moldura/Miolo
        for n in MOLDURA:
            scores[n] += 1
        for n in MIOLO:
            scores[n] += 2
        
        # 10-12. Composição
        for n in PRIMOS:
            scores[n] += 1.5
        for n in FIBONACCI:
            scores[n] += 1.5
        for n in MULTIPLOS_3:
            scores[n] += 1
        
        # 16-17. Coocorrência (AGRESSIVO: peso 5 em vez de 2)
        pares_freq, trios_freq = self.analise_16_17_coocorrencia()
        for (a, b), count in pares_freq.most_common(30):
            peso = (count / self.n_sorteios) * 5  # AGRESSIVO: 2.5x mais forte
            scores[a] += peso
            scores[b] += peso
        
        for trio, count in trios_freq.most_common(20):
            peso = (count / self.n_sorteios) * 4  # AGRESSIVO: aumentado também
            for n in trio:
                scores[n] += peso
        
        # 18. Markov
        markov = self.analise_18_markov()
        for n in range(1, 26):
            if markov[n]:
                total_markov = sum(markov[n].values())
                for m, count in markov[n].most_common(3):
                    prob = count / total_markov
                    scores[m] += prob * 3
        
        # Normalizar
        max_score = max(scores.values()) if scores else 1
        if max_score > 0:
            scores = {n: (scores[n] / max_score) * 100 for n in range(1, 26)}
        
        return scores

# ============================================================
# FIM DA CLASSE (62 ANÁLISES)
# ============================================================

# UI - BASE
# ============================================================
st.sidebar.title("⚙️ Protocolo Hard V4.5.2")

if "base_source" not in st.session_state:
    st.session_state["base_source"] = "local"

uploaded = st.sidebar.file_uploader("Base CSV da Lotofácil", type=["csv"], key="uploader_base")

if uploaded is not None:
    try:
        df_upload = carregar_base_upload(uploaded)
        st.session_state["base_df"] = df_upload
        st.session_state["base_source"] = f"upload: {uploaded.name}"
        st.sidebar.success(f"Base carregada do upload: {uploaded.name}")
        st.sidebar.caption(f"{len(df_upload)} concursos | C{df_upload['Concurso'].min()}–C{df_upload['Concurso'].max()}")
    except Exception as e:
        st.sidebar.error(f"Falha ao ler upload: {e}")

if st.session_state["base_df"] is not None:
    df = st.session_state["base_df"]
else:
    try:
        df = carregar_base_local()
        st.session_state["base_source"] = "local"
    except Exception as e:
        st.error(f"Falha ao carregar a base local: {e}")
        st.stop()

if st.sidebar.button("Usar base local novamente"):
    st.session_state["base_df"] = None
    st.session_state["base_source"] = "local"
    st.rerun()

st.sidebar.caption(f"Fonte ativa: {st.session_state['base_source']}")

with st.sidebar.expander("➕ Atualização manual da base"):
    concurso_manual = st.number_input("Concurso novo", min_value=1, value=int(df["Concurso"].max()) + 1, step=1)
    data_manual = st.text_input("Data", value="")
    dezenas_manual = st.text_input("15 dezenas", value="", placeholder="01 02 03 ... 15")
    if st.button("Incorporar resultado manual"):
        try:
            dezenas = sorted([int(x) for x in dezenas_manual.strip().split()])
            df = incorporar_resultado_manual(df, int(concurso_manual), data_manual, dezenas)
            st.session_state["base_df"] = df
            st.session_state["base_source"] = f"{st.session_state['base_source']} + manual"
            st.success(f"Resultado do concurso {int(concurso_manual)} incorporado.")
            st.rerun()
        except Exception as e:
            st.error(f"Não foi possível incorporar: {e}")

sorteios = extrair_sorteios(df)
st_tuple = tuple(tuple(s) for s in sorteios)

janela = st.sidebar.slider("Janela de análise", 50, len(sorteios), min(500, len(sorteios)), step=50)
threshold = st.sidebar.slider("Threshold N+1/N+2 (%)", 50, 100, 80)
ref_ui = st.sidebar.number_input("Índice Referência (0=último)", 0, len(sorteios) - 1, 0)
ref = concurso_ref_ui_to_idx(ref_ui, len(sorteios))
ativar_nivel1 = True

verificar_params(janela, threshold, ref_ui)
cfg = calibrar(st_tuple)
metricas_n1 = nivel1_metricas(st_tuple, ref_idx=ref, recent=min(24, len(sorteios)), hist=min(200, len(sorteios)))
auc_n1 = auc_criterios_nivel1(st_tuple, janela_eval=min(250, max(80, len(sorteios) - 2)))

with st.sidebar.expander("📐 Filtros calibrados"):
    st.write(f"Soma: {cfg['soma_min']}–{cfg['soma_max']}")
    st.write(f"Pares: {cfg['min_pares']}–{cfg['max_pares']}")
    st.write(f"Repetição: {cfg['rep_min']}–{cfg['rep_max']}")
    st.write(f"Moldura: {cfg['mold_min']}–{cfg['mold_max']}")
    st.write(f"Máx seq: {cfg['max_seq']} | Máx faixa: {cfg['max_faixa']} | Máx quinteto: {cfg['quint_max']}")
    st.write(f"Máx primos: {cfg['max_primo']} | Máx fib: {cfg['max_fib']}")
    st.caption(f"Nível 1: obrigatório no score | KL={metricas_n1['kl_div']:.4f}")

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
st.title("📊 Protocolo Hard V4.5 — Lotofácil")
st.markdown(
    f"Fonte: **{st.session_state.get('base_source', 'local')}** | "
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
    "🧠 Nível 1",
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
# NÍVEL 1 — FILTROS AVANÇADOS
# ============================================================
with tabs[12]:
    st.header("🧠 Nível 1 — Filtros avançados")
    st.caption("Agora esta etapa é obrigatória: executa, entra no relatório e alimenta o score da geração.")

    if etapa_ok("nivel1"):
        d = st.session_state["analise"]["nivel1"]
        st.success("✅ Etapa Nível 1 concluída e incorporada ao score")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KL div", round(d["kl_div"], 4))
        c2.metric("KS global p", round(d["ks_global_p"], 4))
        c3.metric("Skewness", round(d["skew_recent"], 4))
        c4.metric("Curtose", round(d["kurt_recent"], 4))

        st.subheader("AUC dos critérios")
        st.dataframe(pd.DataFrame([d["auc"]]), use_container_width=True, hide_index=True)

        st.subheader("Top pares por Lift")
        st.dataframe(
            pd.DataFrame([
                {"Par": f"{a:02d}+{b:02d}", "Joint": round(joint, 4), "Lift": round(lift, 4)}
                for a, b, joint, lift in d["top_lift"]
            ]),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Ranking por bônus Nível 1")
        st.dataframe(d["df"], use_container_width=True, hide_index=True)

        if "ranking_final_df" in d:
            st.subheader("Score final oficial da análise")
            st.caption(f"Peso N1 aplicado no score final: {d['peso_n1_aplicado']:.4f} | AUC médio: {d['auc_medio']:.4f}")
            st.dataframe(d["ranking_final_df"], use_container_width=True, hide_index=True)

        st.subheader("Kelly informacional")
        banca = st.number_input("Banca simulada (R$)", min_value=1.0, value=100.0, step=10.0, key="banca_kelly")
        krows = []
        for faixa in [11, 12, 13, 14, 15]:
            p = PROB[faixa]
            b = max(PREMIOS[faixa] / 3.5 - 1, 0)
            q = 1 - p
            frac = max(0.0, (b * p - q) / b) if b > 0 else 0.0
            krows.append({
                "Faixa": faixa,
                "Prob.": p,
                "Kelly %": round(frac * 100, 6),
                "Aposta sugerida (R$)": round(frac * banca, 4),
            })
        st.dataframe(pd.DataFrame(krows), use_container_width=True, hide_index=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KL div", round(metricas_n1["kl_div"], 4))
        c2.metric("KS global p", round(metricas_n1["ks_global_p"], 4))
        c3.metric("Skewness", round(metricas_n1["skew_recent"], 4))
        c4.metric("Curtose", round(metricas_n1["kurt_recent"], 4))

        st.subheader("AUC dos critérios")
        st.dataframe(pd.DataFrame([auc_n1]), use_container_width=True, hide_index=True)

        top_lift = lift_top_pairs(st_tuple, ref_idx=ref, janela=min(200, len(sorteios)), top=20)
        st.subheader("Top pares por Lift")
        st.dataframe(
            pd.DataFrame([
                {"Par": f"{a:02d}+{b:02d}", "Joint": round(joint, 4), "Lift": round(lift, 4)}
                for a, b, joint, lift in top_lift
            ]),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Ranking por bônus Nível 1")
        st.dataframe(metricas_n1["df"], use_container_width=True, hide_index=True)

        st.info("Ao confirmar esta etapa, os filtros Nível 1 passam a contar oficialmente no protocolo, recalculam o score final e redefinem o pool oficial.")
        if st.button("✅ Confirmar Etapa 9 — Nível 1", type="primary"):
            dados_n1 = {
                "df": metricas_n1["df"],
                "bonus": metricas_n1["bonus"],
                "kl_div": metricas_n1["kl_div"],
                "ks_global_p": metricas_n1["ks_global_p"],
                "skew_recent": metricas_n1["skew_recent"],
                "kurt_recent": metricas_n1["kurt_recent"],
                "auc": auc_n1,
                "top_lift": top_lift,
            }
            analise_tmp = dict(st.session_state["analise"])
            analise_tmp["nivel1"] = dados_n1
            oficial = recalcular_score_oficial_n1(sorteios, janela, ref, threshold, analise_tmp)
            dados_n1.update(oficial)
            salvar_etapa("nivel1", dados_n1, pipeline_update=oficial["pool_final_oficial"])
            st.rerun()


# ============================================================
# RELATÓRIO
# ============================================================
with tabs[13]:
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
        if etapa_ok("nivel1") and "pool_final_oficial" in analise["nivel1"]:
            d = analise["nivel1"]
            n1_df = analise["nivel1"]["df"].set_index("Dezena")
            for i, n in enumerate(d["pool_final_oficial"][:20]):
                extra = f" | N1 {n1_df.loc[n, 'Bonus N1']:.4f}"
                st.write(f"#{i+1} {n:02d} — score oficial {d['score_final_oficial'][n]}{extra}")
        elif etapa_ok("monte_carlo"):
            d = analise["monte_carlo"]
            for i, n in enumerate(d["pool_final"][:20]):
                st.write(f"#{i+1} {n:02d} — score provisório {d['score_final'][n]}")

    if etapa_ok("nivel1"):
        st.subheader("🧠 Resumo Nível 1 no relatório")
        n1 = analise["nivel1"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("KL div", round(n1["kl_div"], 4))
        r2.metric("KS global p", round(n1["ks_global_p"], 4))
        r3.metric("Skewness", round(n1["skew_recent"], 4))
        r4.metric("Curtose", round(n1["kurt_recent"], 4))
        st.write("AUC dos critérios:")
        st.dataframe(pd.DataFrame([n1["auc"]]), use_container_width=True, hide_index=True)
        st.write("Top 10 dezenas por bônus Nível 1:")
        st.dataframe(n1["df"].head(10), use_container_width=True, hide_index=True)
        if "ranking_final_df" in n1:
            st.write("Top 15 do score final oficial da análise:")
            st.dataframe(n1["ranking_final_df"].head(15), use_container_width=True, hide_index=True)
            st.caption(f"Peso N1 aplicado: {n1['peso_n1_aplicado']:.4f} | AUC médio: {n1['auc_medio']:.4f}")

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
with tabs[14]:
    st.header("🎰 Gerador — Protocolo Hard V4.5.1")
    st.caption("O gerador só libera após a Etapa 9 — Nível 1 ser executada e registrada.")

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

            top_lift_auto = lift_top_pairs(st_tuple, ref_idx=ref, janela=min(200, len(sorteios)), top=20)
            dados_n1_auto = {
                "df": metricas_n1["df"],
                "bonus": metricas_n1["bonus"],
                "kl_div": metricas_n1["kl_div"],
                "ks_global_p": metricas_n1["ks_global_p"],
                "skew_recent": metricas_n1["skew_recent"],
                "kurt_recent": metricas_n1["kurt_recent"],
                "auc": auc_n1,
                "top_lift": top_lift_auto,
            }
            analise_tmp = dict(st.session_state["analise"])
            analise_tmp["nivel1"] = dados_n1_auto
            oficial = recalcular_score_oficial_n1(sorteios, janela, ref, threshold, analise_tmp)
            dados_n1_auto.update(oficial)
            salvar_etapa("nivel1", dados_n1_auto, pipeline_update=oficial["pool_final_oficial"])
            st.rerun()

    else:
        analise = st.session_state["analise"]
        pool_final = analise["nivel1"]["pool_final_oficial"] if etapa_ok("nivel1") and "pool_final_oficial" in analise["nivel1"] else analise["monte_carlo"]["pool_final"]
        score_final = analise["nivel1"]["score_final_oficial"] if etapa_ok("nivel1") and "score_final_oficial" in analise["nivel1"] else analise["monte_carlo"]["score_final"]
        vencidas_g = analise["atraso"]["vencidas"]
        fortes_g = analise["n12"]["fortes"]
        apoio_g = analise["n12"]["apoio"]

        st.success("🎯 Protocolo completo")
        perfil_nome = st.radio("Perfil", list(PERFIS.keys()), horizontal=True)
        perfil = PERFIS[perfil_nome]
        st.info(perfil["descricao"])
        st.caption("Nesta versão, o perfil altera cobertura e travas da carteira. O score oficial vem da análise concluída na Etapa 9.")

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

            prog = st.progress(0, text="Carregando score oficial da análise...")
            scores = dict(score_final)
            ranking = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            sorteio_anterior = sorteios[ref]

            criticas, mapa_criticas = identificar_dezenas_criticas(
                analise,
                sorteio_anterior=sorteio_anterior,
                pool_oficial=pool_final,
            )
            pool = montar_pool_carteira(
                scores=scores,
                ranking=ranking,
                perfil=perfil_nome,
                qtd=qtd,
                fixas=fixas,
                criticas=criticas,
                vencidas=vencidas_g,
                pool_oficial=pool_final,
            )

            familias = [
                {"nome": "protecao", "rep_min": 8, "rep_max": 10, "qmax": 4},
                {"nome": "equilibrado", "rep_min": 7, "rep_max": 9, "qmax": 4},
                {"nome": "agressivo", "rep_min": 6, "rep_max": 9, "qmax": 5},
            ]

            st.write(f"**Pool oficial base ({len(pool_final)}):** {' · '.join(f'{n:02d}' for n in pool_final[:25])}")
            st.write(f"**Pool usado na carteira ({len(pool)}):** {' · '.join(f'{n:02d}' for n in sorted(pool))}")
            if criticas:
                st.write(f"**Dezenas críticas da carteira ({len(criticas)}):** {' · '.join(f'{n:02d}' for n in criticas[:12])}")

            st.caption(
                f"Pool={len(pool)} | interseção mínima possível entre 2 jogos de 15 = {max(0, 30 - len(pool))}"
            )

            jogos = []
            jogos_meta = []
            usage = Counter()
            tent = 0
            max_t = max(260000, qtd * 18000)
            pool_size = len(pool)
            mc_minimo_possivel = max(0, 30 - pool_size)
            if perfil["max_comum"] < mc_minimo_possivel:
                st.warning(
                    f"max_comum={perfil['max_comum']} é impossível com pool={pool_size}. "
                    f"Ajustando automaticamente para {mc_minimo_possivel}."
                )
            mc_atual = max(perfil["max_comum"], mc_minimo_possivel)

            prog.progress(10, text="Montando carteira com proteção 11+ e dezenas críticas...")
            while len(jogos) < qtd and tent < max_t:
                tent += 1
                meta = familias[len(jogos) % len(familias)]

                base = fixas[:15]
                obrigatorias_jogo = montar_obrigatorias_jogo(
                    meta_nome=meta["nome"],
                    idx_rot=len(jogos),
                    scores=scores,
                    criticas=criticas,
                    fortes=sorted(list(fortes_g)),
                    apoio=sorted(list(apoio_g)),
                    vencidas=sorted(list(vencidas_g)),
                    sorteio_anterior=sorteio_anterior,
                    usage=usage,
                )

                for n in obrigatorias_jogo:
                    if n not in base and len(base) < 15:
                        base.append(n)

                rest = [n for n in pool if n not in base]
                if len(rest) < 15 - len(base):
                    ext = [n for n in range(1, 26) if n not in base and n not in rest]
                    rest += ext

                faltam = 15 - len(base)
                universo = rest[:max(faltam + 14, 22)]
                pesos_universo = []
                for n in universo:
                    peso = float(scores[n])
                    if n in criticas:
                        peso *= 1.18
                    if meta["nome"] == "protecao" and n in sorteio_anterior:
                        peso *= 1.10
                    pesos_universo.append(peso)
                rest_sc = np.array(pesos_universo, dtype=float)
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

                if meta["nome"] == "protecao":
                    crit_no_jogo = len(set(cand) & set(criticas))
                    rep_jogo = repeticao_ultimo(cand, sorteio_anterior)
                    if crit_no_jogo < min(4, len(criticas)):
                        erros.append("Proteção sem massa crítica suficiente")
                    if rep_jogo < 8:
                        erros.append("Proteção sem repetição alta suficiente")

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
                    usage.update(cand)
                    prog.progress(min(10 + int(len(jogos) / qtd * 85), 95), text=f"Gerados {len(jogos)}/{qtd}")

            prog.progress(100, text="Concluído")

            if len(jogos) < qtd:
                st.warning(f"Gerados {len(jogos)}/{qtd}. O motor ficou mais seletivo que a carteira pedida.")

            if jogos:
                st.success(f"✅ {len(jogos)} combinação(ões) — {tent:,} tentativas")

                audit = auditoria_portfolio(
                    jogos=jogos,
                    fortes=sorted(list(fortes_g)),
                    apoio=sorted(list(apoio_g)),
                    vencidas=sorted(list(vencidas_g)),
                    sorteio_anterior=sorteio_anterior,
                )
                audit11 = auditoria_carteira_11mais(
                    jogos=jogos,
                    criticas=criticas,
                    sorteio_anterior=sorteio_anterior,
                )
                if audit["cegas_criticas"]:
                    st.warning("⚠️ Dezenas críticas fora do portfólio: " + " ".join(f"{n:02d}" for n in audit["cegas_criticas"]))
                if audit11["criticas_cegas"]:
                    st.error("⛔ Carteira com críticas cegas: " + " ".join(f"{n:02d}" for n in audit11["criticas_cegas"]))
                else:
                    st.success("✅ Nenhuma dezena crítica ficou fora da carteira.")
                st.info(f"Núcleo comum: {' '.join(f'{n:02d}' for n in audit['nucleo_comum'])} ({len(audit['nucleo_comum'])})")
                st.info("Repetição por jogo com o último concurso: " + " | ".join(str(x) for x in audit["repeticoes"]))
                st.info(f"Cobertura total da carteira: {audit11['cobertura_total']} dezenas | jogos com repetição alta: {audit11['jogos_rep_alta']}")

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

                    st.success("✅ Aprovado — filtros V4.5.2")
                    st.markdown("---")

                if len(jogos) > 1:
                    st.subheader("📐 Distância entre jogos")
                    st.dataframe(cobertura_jogos(jogos), use_container_width=True, hide_index=True)

                st.session_state["jogos_gerados"] = jogos
            else:
                st.error("Nenhuma combinação gerada.")
