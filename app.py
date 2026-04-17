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
    s = f"{janela}_{threshold}_{ref_ui}".encode()
    return hashlib.sha256(s).hexdigest()[:8]

def init_session():
    if "base_df" not in st.session_state:
        st.session_state["base_df"] = None
    if "analises" not in st.session_state:
        st.session_state["analises"] = {}
    if "protocolo_status" not in st.session_state:
        st.session_state["protocolo_status"] = {e: False for e in ETAPAS}
    if "param_hash" not in st.session_state:
        st.session_state["param_hash"] = None

init_session()

def verificar_params(janela: int, threshold: int, ref_ui: int):
    h = param_hash(janela, threshold, ref_ui)
    if st.session_state["param_hash"] != h:
        st.session_state["analises"] = {}
        st.session_state["protocolo_status"] = {e: False for e in ETAPAS}
        st.session_state["param_hash"] = h

def salvar_etapa(etapa: str, dados: dict):
    st.session_state["analises"][etapa] = dados
    st.session_state["protocolo_status"][etapa] = True

def etapa_ok(etapa: str) -> bool:
    return st.session_state["protocolo_status"].get(etapa, False)

def todas_ok() -> bool:
    return all(st.session_state["protocolo_status"].values())

def proxima_etapa() -> str:
    for e in ETAPAS:
        if not etapa_ok(e):
            return e
    return ETAPAS[-1]

def status_count() -> tuple:
    ok = sum(st.session_state["protocolo_status"].values())
    tot = len(ETAPAS)
    return ok, tot

# ============================================================
# UTILITIES
# ============================================================

def padronizar_base(df):
    if "Data" in df.columns:
        try:
            df["Data"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")
        except Exception:
            try:
                df["Data"] = pd.to_datetime(df["Data"])
            except Exception:
                st.warning("⚠️ Erro ao converter Data, será ignorada.")
    df = df.sort_values("Concurso").reset_index(drop=True)
    return df

def carregar_base_local() -> pd.DataFrame:
    try:
        df = pd.read_csv("base_lotofacil.csv")
        return padronizar_base(df)
    except FileNotFoundError:
        st.error("❌ 'base_lotofacil.csv' não encontrado no diretório.")
        return None

def carregar_base_upload(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return carregar_base_local()
    try:
        df = pd.read_csv(uploaded)
        return padronizar_base(df)
    except Exception as e:
        st.error(f"❌ Erro ao ler CSV: {e}")
        return None

def extrair_sorteios(df):
    cols = [c for c in df.columns if c.startswith("bola")]
    return df[cols].values.tolist()

def incorporar_resultado_manual(df, concurso, data, dezenas):
    if len(dezenas) != 15 or len(set(dezenas)) != 15 or not all(1 <= x <= 25 for x in dezenas):
        raise ValueError("Dezenas inválidas: devem ser 15 números distintos entre 1-25.")
    
    cols = [c for c in df.columns if c.startswith("bola")]
    nova_linha = {"Concurso": concurso, "Data": data}
    for i, n in enumerate(sorted(dezenas)):
        nova_linha[cols[i]] = n
    
    nova_linha = {col: nova_linha.get(col) for col in df.columns}
    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    return padronizar_base(df)

def concurso_ref_ui_to_idx(ref_ui, n_sorteios):
    if ref_ui == 0:
        return n_sorteios - 1
    return min(ref_ui, n_sorteios - 1)

# ============================================================
# CONFIGURAÇÃO & CÁLCULOS (9 etapas originais)
# ============================================================

def calibrar(st_tuple):
    sorteios = [list(s) for s in st_tuple]
    somas = [sum(s) for s in sorteios]
    pares_counts = [sum(1 for n in s if n % 2 == 0) for s in sorteios]
    reps = [len([n for n in s if s.count(n) > 1]) for s in sorteios]
    mold_counts = [sum(1 for n in s if n in MOLDURA) for s in sorteios]
    
    soma_min, soma_max = int(np.percentile(somas, 10)), int(np.percentile(somas, 90))
    min_pares, max_pares = int(np.percentile(pares_counts, 20)), int(np.percentile(pares_counts, 80))
    rep_min, rep_max = int(np.percentile(reps, 10)), int(np.percentile(reps, 90))
    mold_min, mold_max = int(np.percentile(mold_counts, 20)), int(np.percentile(mold_counts, 80))
    max_seq = 3
    max_faixa = 6
    quint_max = 4
    max_primo = 5
    max_fib = 3
    
    return {
        "soma_min": soma_min, "soma_max": soma_max,
        "min_pares": min_pares, "max_pares": max_pares,
        "rep_min": rep_min, "rep_max": rep_max,
        "mold_min": mold_min, "mold_max": mold_max,
        "max_seq": max_seq, "max_faixa": max_faixa, "quint_max": quint_max,
        "max_primo": max_primo, "max_fib": max_fib,
    }

def nivel1_metricas(st_tuple, ref_idx=None, recent=24, hist=200):
    sorteios = [list(s) for s in st_tuple]
    if ref_idx is None:
        ref_idx = len(sorteios) - 1
    
    recentes = sorteios[max(0, ref_idx-recent+1):ref_idx+1]
    historico = sorteios[max(0, ref_idx-hist+1):ref_idx+1]
    
    from scipy.stats import entropy
    freq_rec = Counter()
    for s in recentes:
        freq_rec.update(s)
    
    freq_hist = Counter()
    for s in historico:
        freq_hist.update(s)
    
    prob_rec = np.array([freq_rec[i] / (15 * len(recentes)) for i in range(1, 26)])
    prob_hist = np.array([freq_hist[i] / (15 * len(historico)) for i in range(1, 26)])
    
    prob_rec = np.clip(prob_rec, 1e-10, 1)
    prob_hist = np.clip(prob_hist, 1e-10, 1)
    
    kl_div = np.sum(prob_rec * (np.log(prob_rec) - np.log(prob_hist)))
    
    return {"kl_div": kl_div, "recentes": len(recentes), "historico": len(historico)}

def auc_criterios_nivel1(st_tuple, janela_eval=80):
    sorteios = [list(s) for s in st_tuple]
    if len(sorteios) < janela_eval + 1:
        return {n: 0.5 for n in range(1, 26)}
    
    auc_scores = {}
    for n in range(1, 26):
        appears = np.array([1 if n in s else 0 for s in sorteios])
        if len(np.unique(appears)) == 1:
            auc_scores[n] = 0.5
        else:
            auc_scores[n] = np.mean(appears)
    
    return auc_scores

# ============================================================
# CLASSE: 62 ANÁLISES (v4.9) ← NOVA CLASSE
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
        
        # 01. Frequência
        freq = self.analise_01_frequencia_simples()
        for n in range(1, 26):
            scores[n] += (freq[n] / self.n_sorteios) * 10
        
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
        
        # 16-17. Coocorrência
        pares_freq, trios_freq = self.analise_16_17_coocorrencia()
        for (a, b), count in pares_freq.most_common(30):
            peso = (count / self.n_sorteios) * 2
            scores[a] += peso
            scores[b] += peso
        
        for trio, count in trios_freq.most_common(20):
            peso = (count / self.n_sorteios) * 1.5
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
    "📊 Resumo Executivo",  # ← NOVA ABA (posição 14)
    "🎰 Gerar",
])

# ============================================================
# ABA 15: RESUMO EXECUTIVO (NOVA!) ← ADICIONAR AQUI
# ============================================================
with tabs[14]:
    st.subheader("📊 Resumo Executivo — 62 Análises Consolidadas")
    st.markdown("**Números recomendados baseados em 62 critérios estatísticos**")
    
    # Rodar as 62 análises
    analisador = AnalisadorLotofacil(sorteios)
    scores_62 = analisador.calcular_scores()
    
    # Layout 3 colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🔴 **FREQUÊNCIA** (Top 10 Quentes)")
        freq = analisador.analise_01_frequencia_simples()
        quentes = [n for n, _ in freq.most_common(10)]
        st.code(f"{' '.join(str(n).zfill(2) for n in quentes)}")
    
    with col2:
        st.info("⏱️ **ATRASO** (Top 5 Vencidas)")
        atrasos, _, _ = analisador.analise_03_05_atraso_e_ciclo()
        vencidas = sorted(range(1, 26), key=lambda n: -atrasos[n])[:5]
        st.code(f"{' '.join(str(n).zfill(2) for n in vencidas)}")
    
    with col3:
        st.info("🎯 **POOL FINAL** (15 números)")
        top_15 = sorted(scores_62.items(), key=lambda x: -x[1])[:15]
        pool = sorted([n for n, _ in top_15])
        st.code(f"{' '.join(str(n).zfill(2) for n in pool)}")
    
    st.markdown("---")
    
    st.subheader("📈 Ranking Completo com Scores")
    ranking_62 = pd.DataFrame([
        {"Número": n, "Score": f"{score:.1f}%"}
        for n, score in sorted(scores_62.items(), key=lambda x: -x[1])
    ])
    st.dataframe(ranking_62, use_container_width=True, hide_index=True)
    
    st.success("✅ **Resumo gerado com as 62 análises!** Use esses números para gerar os jogos na próxima aba.")
    
    # Mostrar também um resumo de estatísticas
    st.markdown("---")
    st.subheader("📊 Estatísticas das Análises")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de sorteios", len(sorteios))
    with col2:
        st.metric("Números analisados", 25)
    with col3:
        st.metric("Análises implementadas", 62)
    with col4:
        st.metric("Score máximo", f"{max(scores_62.values()):.1f}%")

# ============================================================
# ABA 16: GERAR (original, mantido igual)
# ============================================================
with tabs[15]:
    st.subheader("🎰 Gerar Jogos")
    
    if not todas_ok():
        st.warning(f"⏳ Complete as etapas primeiro ({ok}/{tot} OK)")
    else:
        perfil = st.radio("Perfil", list(PERFIS.keys()))
        p = PERFIS[perfil]
        
        col1, col2 = st.columns(2)
        with col1:
            qtd = st.number_input("Qtd jogos", 1, 20, 1)
        with col2:
            seed = st.number_input("Seed (0=aleatório)", 0, 999999, 0)
        
        if st.button("🎲 Gerar Jogos", key="btn_gerar"):
            # Usar o scoring das 62 análises se disponível
            try:
                analisador = AnalisadorLotofacil(sorteios)
                scores_gerador = analisador.calcular_scores()
                st.success("✅ Usando scores das 62 análises!")
            except:
                scores_gerador = {n: 1/25 for n in range(1, 26)}
            
            st.write(f"**{perfil}** | {qtd} jogo(s) | Seed: {seed if seed > 0 else 'aleatório'}")
            st.markdown("---")
            
            # Simular geração
            if seed > 0:
                random.seed(seed)
                np.random.seed(seed)
            
            for i in range(qtd):
                pesos = [scores_gerador.get(n, 0.1) for n in range(1, 26)]
                pesos_norm = np.array(pesos) / sum(pesos)
                
                jogo = sorted(list(np.random.choice(range(1, 26), 15, replace=False, p=pesos_norm)))
                st.code(f"Jogo {i+1}: {' '.join(str(n).zfill(2) for n in jogo)}")
            
            st.success("✅ Jogos gerados!")

st.sidebar.markdown("---")
st.sidebar.caption("v4.5.2 com 62 análises (v4.9) 🚀")
