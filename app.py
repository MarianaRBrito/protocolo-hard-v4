import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import random
from scipy import stats
import math
import hashlib

st.set_page_config(page_title="Protocolo Hard V4", layout="wide")

COLS_DEZENAS = [f"bola {i}" for i in range(1, 16)]
PRIMOS       = {2,3,5,7,11,13,17,19,23}
FIBONACCI    = {1,2,3,5,8,13,21}
MULTIPLOS_3  = {3,6,9,12,15,18,21,24}

# ══════════════════════════════════════════════════════════════
# SESSION STATE CENTRAL
# ══════════════════════════════════════════════════════════════
ETAPAS = ["termica","atraso","estrutural","composicao",
          "sequencias","n12","coocorrencia","monte_carlo"]

ETAPA_LABELS = {
    "termica":     "1️⃣ Térmica",
    "atraso":      "2️⃣ Atraso & Ciclo",
    "estrutural":  "3️⃣ Estrutural",
    "composicao":  "4️⃣ Composição",
    "sequencias":  "5️⃣ Sequências & Gaps",
    "n12":         "6️⃣ N+1/N+2",
    "coocorrencia":"7️⃣ Coocorrência",
    "monte_carlo": "8️⃣ Monte Carlo",
}

def param_hash(janela, threshold, ref):
    """Hash dos parâmetros — se mudar, invalida todas as etapas."""
    return hashlib.md5(f"{janela}_{threshold}_{ref}".encode()).hexdigest()[:8]

def init_session():
    defaults = {
        "analise":      {e: None for e in ETAPAS},
        "pipeline":     [],          # dezenas candidatas em cascata
        "param_hash":   None,
        "scores_cache": None,
        "jogos_gerados": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def verificar_params(janela, threshold, ref):
    """Invalida etapas se parâmetros mudaram."""
    h = param_hash(janela, threshold, ref)
    if st.session_state["param_hash"] != h:
        st.session_state["analise"]      = {e: None for e in ETAPAS}
        st.session_state["pipeline"]     = []
        st.session_state["param_hash"]   = h
        st.session_state["scores_cache"] = None
        st.session_state["jogos_gerados"] = []

def salvar_etapa(nome, dados, pipeline_update=None):
    st.session_state["analise"][nome] = dados
    st.session_state["scores_cache"]  = None
    if pipeline_update is not None:
        st.session_state["pipeline"] = pipeline_update

def etapa_ok(nome):
    return st.session_state["analise"].get(nome) is not None

def todas_ok():
    return all(etapa_ok(e) for e in ETAPAS)

def proxima_etapa():
    for e in ETAPAS:
        if not etapa_ok(e): return e
    return None

def status_count():
    return sum(1 for e in ETAPAS if etapa_ok(e)), len(ETAPAS)

init_session()

# ══════════════════════════════════════════════════════════════
# CARGA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def carregar_base():
    df = pd.read_csv("base_lotofacil.csv")
    return df.sort_values("Concurso").reset_index(drop=True)

@st.cache_data
def extrair_sorteios(df):
    return df[COLS_DEZENAS].values.tolist()

# ══════════════════════════════════════════════════════════════
# FUNÇÕES ANALÍTICAS
# ══════════════════════════════════════════════════════════════
def freq_simples(sorteios, janela=None):
    dados = sorteios[-janela:] if janela else sorteios
    c = Counter()
    for s in dados: c.update(s)
    return c

def tabela_frequencia(sorteios, janelas=[50,100,200,500]):
    rows = []
    for n in range(1,26):
        row = {"Dezena": n}
        for j in janelas:
            c = sum(1 for s in sorteios[-j:] if n in s)
            row[f"Freq {j}"] = c
            row[f"% {j}"]   = round(c/j*100,1)
        rows.append(row)
    return pd.DataFrame(rows)

def calcular_atrasos(sorteios):
    ultimo = {}
    for i,s in enumerate(sorteios):
        for n in s: ultimo[n] = i
    total = len(sorteios)
    res = []
    for n in range(1,26):
        idx = ultimo.get(n,-1)
        res.append({"Dezena":n,"Último":idx+1,"Atraso":total-1-idx if idx>=0 else total})
    return pd.DataFrame(res).sort_values("Atraso",ascending=False)

def ciclo_medio(sorteios, dezena):
    ap = [i for i,s in enumerate(sorteios) if dezena in s]
    if len(ap)<2: return None
    return round(np.mean([ap[i+1]-ap[i] for i in range(len(ap)-1)]),2)

def faixa_soma(sorteios):
    somas = [sum(s) for s in sorteios]
    return int(np.percentile(somas,5)), int(np.percentile(somas,95)), round(np.mean(somas),1)

def dist_pares(sorteios):
    d = Counter(sum(1 for n in s if n%2==0) for s in sorteios)
    t = len(sorteios)
    return {k:round(v/t*100,1) for k,v in sorted(d.items())}

def contar_consec(jogo):
    jogo = sorted(jogo)
    mx = cur = 1; pc = 0
    for i in range(1,len(jogo)):
        if jogo[i]==jogo[i-1]+1: cur+=1; mx=max(mx,cur); pc+=1
        else: cur=1
    return mx, pc

def dist_consec(sorteios):
    d = Counter(contar_consec(s)[1] for s in sorteios)
    t = len(sorteios)
    return {k:round(v/t*100,1) for k,v in sorted(d.items())}

def dist_faixas_hist(sorteios):
    dist = [[] for _ in range(5)]
    for s in sorteios:
        f = [sum(1 for n in s if i*5+1<=n<=i*5+5) for i in range(5)]
        for i in range(5): dist[i].append(f[i])
    return [{"Faixa":f"{i*5+1}-{i*5+5}","Média":round(np.mean(dist[i]),2),
             "Min":min(dist[i]),"Max":max(dist[i])} for i in range(5)]

def dist_faixas(jogo):
    return [sum(1 for n in jogo if i*5+1<=n<=i*5+5) for i in range(5)]

def composicao(jogo):
    j=set(jogo)
    return {"primos":len(j&PRIMOS),"fibonacci":len(j&FIBONACCI),"mult3":len(j&MULTIPLOS_3)}

def dist_comp(sorteios):
    dp = Counter(len(set(s)&PRIMOS)    for s in sorteios)
    df = Counter(len(set(s)&FIBONACCI) for s in sorteios)
    t  = len(sorteios)
    return ({k:round(v/t*100,1) for k,v in sorted(dp.items())},
            {k:round(v/t*100,1) for k,v in sorted(df.items())})

def gaps(jogo):
    j=sorted(jogo); g=[j[i+1]-j[i] for i in range(len(j)-1)]
    return {"lista":g,"media":round(np.mean(g),2),"max":max(g),"min":min(g)}

def dist_gaps(sorteios):
    todos=[]
    for s in sorteios: todos.extend(gaps(s)["lista"])
    c=Counter(todos); t=sum(c.values())
    return {k:round(v/t*100,1) for k,v in sorted(c.items()) if k<=10}

def projetar_isca(alvo_idx, sorteios, threshold):
    if not sorteios or alvo_idx>=len(sorteios): return [],[],[]
    alvo = set(sorteios[alvo_idx])
    th   = int(len(alvo)*(threshold/100))
    sim=[]; fut=[]
    for i,s in enumerate(sorteios):
        if i==alvo_idx: continue
        if len(alvo&set(s))>=th:
            sim.append(i)
            if i+1<len(sorteios): fut.extend(sorteios[i+1])
            if i+2<len(sorteios): fut.extend(sorteios[i+2])
    freq=Counter(fut).most_common()
    return sim,[k for k,_ in freq[:6]],[k for k,_ in freq[6:12]]

@st.cache_data
def calc_cooc(sorteios_tuple, top=20):
    p=Counter()
    for s in sorteios_tuple:
        for a,b in combinations(sorted(s),2): p[(a,b)]+=1
    return p.most_common(top)

@st.cache_data
def calc_trios(sorteios_tuple, top=15):
    t=Counter()
    for s in sorteios_tuple:
        for trio in combinations(sorted(s),3): t[trio]+=1
    return t.most_common(top)

@st.cache_data
def calc_markov(sorteios_tuple, janela=200):
    sl=list(sorteios_tuple)[-janela:]
    tr={n:Counter() for n in range(1,26)}
    for i in range(len(sl)-1):
        for n in set(sl[i]): tr[n].update(set(sl[i+1]))
    return tr

def entropia(jogo, sorteios):
    f=freq_simples(sorteios); t=sum(f.values())
    return round(-sum((f[n]/t)*math.log2(f[n]/t) for n in jogo if f[n]>0),4)

def qui_quadrado(sorteios, janela=500):
    dados=sorteios[-janela:]
    obs=np.array([sum(1 for s in dados if n in s) for n in range(1,26)])
    esp=np.full(25,len(dados)*15/25)
    chi2,p=stats.chisquare(obs,esp)
    return round(chi2,2),round(p,4)

def monte_carlo(jogo, sorteios, n=5000):
    ac=[len(set(jogo)&set(random.sample(range(1,26),15))) for _ in range(n)]
    meu=np.mean([len(set(jogo)&set(s)) for s in sorteios[-200:]])
    return round(np.mean(ac),2),round(meu,2)

def backtesting(jogo, sorteios, ultimos=200):
    dados=sorteios[-ultimos:]
    ac=[len(set(jogo)&set(s)) for s in dados]
    d=Counter(ac)
    return {"media":round(np.mean(ac),2),"max":max(ac),"min":min(ac),
            "dist":dict(sorted(d.items())),
            "pct11":round(sum(1 for a in ac if a>=11)/len(ac)*100,1),
            "pct13":round(sum(1 for a in ac if a>=13)/len(ac)*100,1)}

def dezenas_pico(sorteios, pct=85):
    df=calcular_atrasos(sorteios)
    lim=np.percentile(df["Atraso"],pct)
    return df[df["Atraso"]>=lim]["Dezena"].tolist()

def hamming(j1,j2): return 15-len(set(j1)&set(j2))

def cobertura_jogos(jogos):
    if len(jogos)<2: return pd.DataFrame()
    res=[]
    for i in range(len(jogos)):
        for j in range(i+1,len(jogos)):
            d=hamming(jogos[i],jogos[j])
            res.append({"Par":f"J{i+1}×J{j+1}","Hamming":d,"Em comum":15-d})
    return pd.DataFrame(res)

# ══════════════════════════════════════════════════════════════
# FILTROS AUTO-CALIBRADOS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def calibrar(sorteios_tuple):
    sl=list(sorteios_tuple)
    return {
        "soma_min":  int(np.percentile([sum(s) for s in sl],5)),
        "soma_max":  int(np.percentile([sum(s) for s in sl],95)),
        "min_pares": int(np.percentile([sum(1 for n in s if n%2==0) for s in sl],5)),
        "max_pares": int(np.percentile([sum(1 for n in s if n%2==0) for s in sl],95)),
        "max_seq":   int(np.percentile([contar_consec(s)[0] for s in sl],95)),
        "max_primo": int(np.percentile([len(set(s)&PRIMOS) for s in sl],95)),
        "max_fib":   int(np.percentile([len(set(s)&FIBONACCI) for s in sl],95)),
        "max_faixa": int(np.percentile([max(dist_faixas(s)) for s in sl],95)),
    }

def filtrar(jogo, sorteios, cfg):
    erros=[]
    soma=sum(jogo)
    if soma<cfg["soma_min"] or soma>cfg["soma_max"]:
        erros.append(f"Soma {soma} fora [{cfg['soma_min']}–{cfg['soma_max']}]")
    mx,_=contar_consec(jogo)
    if mx>cfg["max_seq"]: erros.append(f"Seq {mx}>{cfg['max_seq']}")
    p=sum(1 for n in jogo if n%2==0)
    if p<cfg["min_pares"] or p>cfg["max_pares"]:
        erros.append(f"{p} pares fora [{cfg['min_pares']}–{cfg['max_pares']}]")
    for i,f in enumerate(dist_faixas(jogo)):
        if f==0: erros.append(f"Faixa {i*5+1}-{i*5+5} vazia")
        if f>cfg["max_faixa"]: erros.append(f"Faixa {i*5+1}-{i*5+5} excesso")
    c=composicao(jogo)
    if c["primos"]>cfg["max_primo"]:  erros.append(f"{c['primos']} primos>{cfg['max_primo']}")
    if c["fibonacci"]>cfg["max_fib"]: erros.append(f"{c['fibonacci']} fib>{cfg['max_fib']}")
    return erros

# ══════════════════════════════════════════════════════════════
# MOTOR DE SCORE — usa resultados das etapas em cascata
# ══════════════════════════════════════════════════════════════
def normalizar(d):
    v=list(d.values()); mn,mx=min(v),max(v)
    if mx==mn: return {k:0.5 for k in d}
    return {k:(v-mn)/(mx-mn) for k,v in d.items()}

def calcular_scores(sorteios, janela, ref, threshold, pesos, analise):
    """
    Score em cascata — cada grupo usa resultados das etapas anteriores.
    Dezenas vencidas (Etapa 2) têm peso garantido no pool final.
    """
    # ── G1: Térmica + N+1/N+2 + Rolling (40%) ────────────────
    g1 = {n:0.0 for n in range(1,26)}

    # Quentes da Etapa 1
    if analise.get("termica"):
        quentes = analise["termica"]["quentes"]
        for i,n in enumerate(quentes):
            g1[n] = max(g1[n], 1.0 - i*0.04)

    # N+1/N+2 da Etapa 6
    fortes_n,apoio_n,sim = [],[],[]
    if analise.get("n12"):
        fortes_n = analise["n12"]["fortes"]
        apoio_n  = analise["n12"]["apoio"]
        sim      = analise["n12"]["similares"]
        for n in fortes_n: g1[n] = max(g1[n], 1.0)
        for n in apoio_n:  g1[n] = max(g1[n], 0.7)

    # Rolling
    f50   = freq_simples(sorteios,50)
    f_lon = freq_simples(sorteios,janela)
    roll  = {}
    for n in range(1,26):
        t50  = f50[n]/50
        tlon = f_lon[n]/janela
        roll[n] = max(0.0,(t50-tlon)/(tlon+1e-9))
    roll = normalizar(roll)
    g1   = normalizar(g1)
    G1   = {n: g1[n]*0.60 + roll[n]*0.40 for n in range(1,26)}

    # ── G2: Atraso & Ciclo (30%) — VENCIDAS = score máximo ───
    g2 = {n:0.0 for n in range(1,26)}
    vencidas,no_ciclo = [],[]

    if analise.get("atraso"):
        df_at = analise["atraso"]["df_completo"]
        vencidas  = analise["atraso"]["vencidas"]
        no_ciclo  = analise["atraso"]["no_ciclo"]
        for _,row in df_at.iterrows():
            n      = int(row["Dezena"])
            atraso = row["Atraso"]
            cm     = row["Ciclo Médio"] or atraso
            ratio  = min(atraso/cm if cm>0 else 1.0, 3.0)
            # Vencidas recebem score máximo garantido
            if n in vencidas:  g2[n] = 3.0
            elif n in no_ciclo: g2[n] = max(ratio, 1.5)
            else: g2[n] = ratio
    else:
        total=len(sorteios); ult={}
        for i,s in enumerate(sorteios):
            for x in s: ult[x]=i
        for n in range(1,26):
            idx=ult.get(n,-1); at=total-1-idx if idx>=0 else total
            cm=ciclo_medio(sorteios,n) or at
            g2[n]=min(at/cm if cm>0 else 1.0,3.0)
    G2 = normalizar(g2)

    # ── G3: Coocorrência + Markov + Trios (20%) ──────────────
    if analise.get("coocorrencia"):
        G3 = {n: (analise["coocorrencia"]["g3_cooc"][n]*0.40 +
                  analise["coocorrencia"]["g3_markov"][n]*0.40 +
                  analise["coocorrencia"]["g3_trios"][n]*0.20)
              for n in range(1,26)}
    else:
        pf=Counter()
        for s in sorteios[-500:]:
            for a,b in combinations(sorted(s),2): pf[(a,b)]+=1
        gc={n:0.0 for n in range(1,26)}
        for (a,b),c in pf.items(): gc[a]+=c; gc[b]+=c
        gc=normalizar(gc)
        tr=calc_markov(tuple(tuple(s) for s in sorteios))
        ult_s=set(sorteios[ref])
        gm={n:0.0 for n in range(1,26)}
        for x in ult_s:
            if x in tr:
                for suc,c in tr[x].items(): gm[suc]=gm.get(suc,0)+c
        gm=normalizar(gm)
        tf=Counter()
        for s in sorteios[-500:]:
            for trio in combinations(sorted(s),3): tf[trio]+=1
        gt={n:0.0 for n in range(1,26)}
        for trio,c in tf.most_common(100):
            for x in trio: gt[x]+=c
        gt=normalizar(gt)
        G3={n:gc[n]*0.40+gm[n]*0.40+gt[n]*0.20 for n in range(1,26)}

    # ── G4: Estrutural (5%) ───────────────────────────────────
    freq_all=freq_simples(sorteios); tot=sum(freq_all.values())
    esp=1/25
    G4=normalizar({n:1-abs(freq_all[n]/tot-esp)/esp for n in range(1,26)})

    # ── G5: Entropia (5%) ─────────────────────────────────────
    G5=normalizar({n:freq_all[n]/tot for n in range(1,26)})

    # ── Score final ───────────────────────────────────────────
    scores={n:(pesos["g1"]*G1[n]+pesos["g2"]*G2[n]+
               pesos["g3"]*G3[n]+pesos["g4"]*G4[n]+
               pesos["g5"]*G5[n])
            for n in range(1,26)}

    return scores, sim, fortes_n, apoio_n, vencidas, no_ciclo

# ══════════════════════════════════════════════════════════════
# PERFIS
# ══════════════════════════════════════════════════════════════
PERFIS = {
    "🎯 Agressiva": {
        "pesos":    {"g1":0.40,"g2":0.30,"g3":0.20,"g4":0.05,"g5":0.05},
        "top_pool": 15,
        "max_comum":8,
        "descricao":"Força máxima — pool restrito às dezenas top rankeadas.",
        "aviso":    10,
    },
    "⚖️ Híbrida": {
        "pesos":    {"g1":0.30,"g2":0.25,"g3":0.25,"g4":0.10,"g5":0.10},
        "top_pool": 20,
        "max_comum":11,
        "descricao":"Cobertura ampla — ideal para bolões e concursos especiais.",
        "aviso":    None,
    },
}

PREMIOS={15:1_500_000,14:1500,13:25,12:10,11:5}
PROB={15:1/3_268_760,14:15/3_268_760,13:105/3_268_760,12:455/3_268_760,11:1365/3_268_760}

# ══════════════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════════════
df       = carregar_base()
sorteios = extrair_sorteios(df)
st_tuple = tuple(tuple(s) for s in sorteios)

# Sidebar
st.sidebar.title("⚙️ Protocolo Hard V4")
st.sidebar.markdown(f"**Base:** {len(sorteios)} concursos | C{df['Concurso'].max()}")

janela    = st.sidebar.slider("Janela de Análise", 50, len(sorteios), 500, step=50)
threshold = st.sidebar.slider("Threshold N+1/N+2 (%)", 50, 100, 80)
ref       = st.sidebar.number_input("Índice Referência (0=último)", 0, len(sorteios)-1, len(sorteios)-1)

# Invalida etapas se params mudaram
verificar_params(janela, threshold, ref)

cfg = calibrar(st_tuple)

with st.sidebar.expander("📐 Filtros Calibrados (auto)"):
    st.write(f"Soma: {cfg['soma_min']}–{cfg['soma_max']}")
    st.write(f"Pares: {cfg['min_pares']}–{cfg['max_pares']}")
    st.write(f"Máx seq: {cfg['max_seq']} | Máx faixa: {cfg['max_faixa']}")
    st.write(f"Máx primos: {cfg['max_primo']} | Máx Fib: {cfg['max_fib']}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Status do Protocolo")
ok,tot = status_count()
for e in ETAPAS:
    icon = "✅" if etapa_ok(e) else "⭕"
    st.sidebar.write(f"{icon} {ETAPA_LABELS[e]}")
if todas_ok():
    st.sidebar.success("🎯 Pronto para gerar!")
else:
    prox = proxima_etapa()
    st.sidebar.warning(f"⏳ {ok}/{tot} — próxima: {ETAPA_LABELS[prox]}")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Backtesting Manual")
j1 = st.sidebar.text_input("Jogo 1 (15 números)")
j2 = st.sidebar.text_input("Jogo 2")
j3 = st.sidebar.text_input("Jogo 3")

def parse_jogo(s):
    try:
        n=[int(x) for x in s.strip().split()]
        if len(n)==15 and all(1<=x<=25 for x in n): return sorted(n)
    except: pass
    return None

jogos_bt = [j for j in [parse_jogo(j1),parse_jogo(j2),parse_jogo(j3)] if j]
sorteios_j = sorteios[-janela:]

# ══════════════════════════════════════════════════════════════
# TÍTULO + BARRA DE PROGRESSO GERAL
# ══════════════════════════════════════════════════════════════
st.title("📊 Protocolo Hard V4 — Lotofácil")
st.markdown(f"Base: **{len(sorteios)} concursos** | C{df['Concurso'].min()}–C{df['Concurso'].max()} | Janela: **{janela}**")

ok,tot = status_count()
st.progress(ok/tot, text=f"Protocolo: {ok}/{tot} etapas {'✅ Pronto para gerar!' if todas_ok() else '— Execute as etapas abaixo'}")

if not todas_ok():
    prox = proxima_etapa()
    st.info(f"▶️ Próxima etapa a executar: **{ETAPA_LABELS[prox]}**")

tabs = st.tabs([
    "1️⃣ Térmica","2️⃣ Atraso & Ciclo","3️⃣ Estrutural",
    "4️⃣ Composição","5️⃣ Sequências","6️⃣ N+1/N+2",
    "7️⃣ Coocorrência","8️⃣ Monte Carlo",
    "📊 Backtesting","🧪 Filtros","📈 Tendência","⚖️ EV",
    "🏆 Relatório","🎰 Gerar"
])

# ══════════════════════════════════════════════════════════════
# ETAPA 1 — TÉRMICA
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("1️⃣ Térmica — Frequência & Temperatura")
    if etapa_ok("termica"):
        st.success("✅ Etapa concluída — dados salvos no protocolo.")
        d = st.session_state["analise"]["termica"]
        st.write(f"🔥 Quentes top 10: **{' '.join(str(n).zfill(2) for n in d['quentes'])}**")
        st.write(f"🧊 Frias top 5: **{' '.join(str(n).zfill(2) for n in d['frias'])}**")
        st.write(f"⚠️ Pico de atraso: **{' '.join(str(n).zfill(2) for n in d['pico'])}**")
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 2")
        if st.button("🔄 Refazer Etapa 1"):
            salvar_etapa("termica", None)
            # Invalida etapas seguintes em cascata
            for e in ETAPAS[1:]:
                st.session_state["analise"][e] = None
            st.session_state["pipeline"] = []
            st.rerun()
    else:
        st.dataframe(tabela_frequencia(sorteios), use_container_width=True, hide_index=True)
        f = freq_simples(sorteios, janela)
        top10 = [n for n,_ in f.most_common(10)]
        bot5  = [n for n,_ in f.most_common()[:-6:-1]]
        pico  = dezenas_pico(sorteios)

        col1,col2 = st.columns(2)
        with col1:
            st.success(f"🔥 Top 10 Quentes: {' '.join(str(n).zfill(2) for n in top10)}")
            st.info(f"🧊 Top 5 Frias: {' '.join(str(n).zfill(2) for n in bot5)}")
        with col2:
            st.warning(f"⚠️ Pico de Atraso: {' '.join(str(n).zfill(2) for n in pico)}")
            st.dataframe(calcular_atrasos(sorteios).head(8), use_container_width=True, hide_index=True)

        # Candidatas = quentes + pico (sem duplicatas)
        candidatas = list(dict.fromkeys(top10 + pico))
        st.info(f"➡️ **{len(candidatas)} dezenas candidatas** para Etapa 2: {' '.join(str(n).zfill(2) for n in candidatas)}")

        if st.button("✅ Confirmar Etapa 1 — Térmica", type="primary"):
            salvar_etapa("termica", {
                "quentes": top10, "frias": bot5, "pico": pico,
                "candidatas": candidatas,
                "freq": {n: f[n] for n in range(1,26)}
            }, pipeline_update=candidatas)
            for e in ETAPAS[1:]:
                st.session_state["analise"][e] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ETAPA 2 — ATRASO & CICLO
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("2️⃣ Atraso & Ciclo — Prioridade Máxima para Vencidas")
    if not etapa_ok("termica"):
        st.warning("⚠️ Execute a Etapa 1 primeiro.")
    elif etapa_ok("atraso"):
        st.success("✅ Etapa concluída.")
        d = st.session_state["analise"]["atraso"]
        st.write(f"🔴 Vencidas (prioridade máxima): **{' '.join(str(n).zfill(2) for n in d['vencidas'])}**")
        st.write(f"🟡 No ciclo: **{' '.join(str(n).zfill(2) for n in d['no_ciclo'])}**")
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 3")
        if st.button("🔄 Refazer Etapa 2"):
            for e in ETAPAS[1:]:
                st.session_state["analise"][e] = None
            st.rerun()
    else:
        candidatas_e1 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 1: **{' '.join(str(n).zfill(2) for n in candidatas_e1)}**")

        df_at = calcular_atrasos(sorteios)
        ciclos_l = [{"Dezena":n,"Ciclo Médio":ciclo_medio(sorteios,n)} for n in range(1,26)]
        df_cic = pd.DataFrame(ciclos_l)
        df_comp = df_at.merge(df_cic, on="Dezena")
        df_comp["Status"] = df_comp.apply(
            lambda r: "🔴 VENCIDA" if r["Atraso"]>(r["Ciclo Médio"] or 99)*1.5
            else ("🟡 No ciclo" if r["Atraso"]>(r["Ciclo Médio"] or 99)*0.8 else "🟢 Ok"), axis=1)
        st.dataframe(df_comp.sort_values("Atraso", ascending=False), use_container_width=True, hide_index=True)

        vencidas = df_comp[df_comp["Status"]=="🔴 VENCIDA"]["Dezena"].tolist()
        no_ciclo = df_comp[df_comp["Status"]=="🟡 No ciclo"]["Dezena"].tolist()

        st.error(f"🔴 Vencidas (entram OBRIGATORIAMENTE no pool): **{' '.join(str(n).zfill(2) for n in vencidas)}**")
        st.warning(f"🟡 No ciclo (alta prioridade): **{' '.join(str(n).zfill(2) for n in no_ciclo)}**")

        # Candidatas = e1 + vencidas (vencidas entram mesmo que não estejam nas quentes)
        candidatas_e2 = list(dict.fromkeys(candidatas_e1 + vencidas + no_ciclo[:3]))
        st.info(f"➡️ **{len(candidatas_e2)} dezenas candidatas** para Etapa 3: {' '.join(str(n).zfill(2) for n in candidatas_e2)}")

        if st.button("✅ Confirmar Etapa 2 — Atraso & Ciclo", type="primary"):
            salvar_etapa("atraso", {
                "df_completo": df_comp, "vencidas": vencidas,
                "no_ciclo": no_ciclo, "candidatas": candidatas_e2
            }, pipeline_update=candidatas_e2)
            for e in ETAPAS[2:]:
                st.session_state["analise"][e] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ETAPA 3 — ESTRUTURAL
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("3️⃣ Estrutural — Soma, Pares, Faixas")
    if not etapa_ok("atraso"):
        st.warning("⚠️ Execute as etapas anteriores primeiro.")
    elif etapa_ok("estrutural"):
        st.success("✅ Etapa concluída.")
        d = st.session_state["analise"]["estrutural"]
        st.write(f"Soma: **{d['soma_min']}–{d['soma_max']}** (média {d['soma_med']})")
        st.write(f"Pares mais freq: **{d['top_pares']}** ({d['pct_top_pares']}%)")
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 4")
        if st.button("🔄 Refazer Etapa 3"):
            for e in ETAPAS[2:]:
                st.session_state["analise"][e] = None
            st.rerun()
    else:
        candidatas_e2 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 2: **{' '.join(str(n).zfill(2) for n in candidatas_e2)}**")

        sm,sx,smed = faixa_soma(sorteios_j)
        dp = dist_pares(sorteios_j)
        top_pares = max(dp, key=dp.get)

        col1,col2 = st.columns(2)
        with col1:
            st.info(f"Faixa de soma: **{sm}–{sx}** | Média: **{smed}**")
            st.dataframe(pd.DataFrame(list(dp.items()),columns=["Pares","%"]), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame(dist_faixas_hist(sorteios_j)), use_container_width=True, hide_index=True)

        # Filtra candidatas: mantém as que contribuem para soma na faixa
        # Heurística: remove dezenas que individualmente forçam soma fora da faixa
        soma_media_por_dezena = smed / 15
        candidatas_e3 = [n for n in candidatas_e2
                         if abs(n - soma_media_por_dezena) <= soma_media_por_dezena * 0.8]
        # Garante mínimo de candidatas
        if len(candidatas_e3) < 18:
            candidatas_e3 = candidatas_e2

        st.info(f"➡️ **{len(candidatas_e3)} dezenas candidatas** para Etapa 4: {' '.join(str(n).zfill(2) for n in candidatas_e3)}")

        if st.button("✅ Confirmar Etapa 3 — Estrutural", type="primary"):
            salvar_etapa("estrutural", {
                "soma_min":sm,"soma_max":sx,"soma_med":smed,
                "top_pares":top_pares,"pct_top_pares":dp[top_pares],
                "dist_pares":dp,"candidatas":candidatas_e3
            }, pipeline_update=candidatas_e3)
            for e in ETAPAS[3:]:
                st.session_state["analise"][e] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ETAPA 4 — COMPOSIÇÃO
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("4️⃣ Composição — Primos, Fibonacci, Múltiplos")
    if not etapa_ok("estrutural"):
        st.warning("⚠️ Execute as etapas anteriores primeiro.")
    elif etapa_ok("composicao"):
        st.success("✅ Etapa concluída.")
        d = st.session_state["analise"]["composicao"]
        st.write(f"Primos mais freq: **{d['top_primo']}** | Fibonacci: **{d['top_fib']}**")
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 5")
        if st.button("🔄 Refazer Etapa 4"):
            for e in ETAPAS[3:]:
                st.session_state["analise"][e] = None
            st.rerun()
    else:
        candidatas_e3 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 3: **{' '.join(str(n).zfill(2) for n in candidatas_e3)}**")

        dp_c, df_c = dist_comp(sorteios_j)
        top_primo = max(dp_c, key=dp_c.get)
        top_fib   = max(df_c, key=df_c.get)

        col1,col2,col3 = st.columns(3)
        with col1:
            st.subheader("Primos (%)")
            st.dataframe(pd.DataFrame(list(dp_c.items()),columns=["Qtd","%"]),hide_index=True)
            st.caption(f"Primos: {sorted(PRIMOS)}")
        with col2:
            st.subheader("Fibonacci (%)")
            st.dataframe(pd.DataFrame(list(df_c.items()),columns=["Qtd","%"]),hide_index=True)
            st.caption(f"Fibonacci: {sorted(FIBONACCI)}")
        with col3:
            st.subheader("Múlt.3 (%)")
            dm3=Counter(len(set(s)&MULTIPLOS_3) for s in sorteios_j)
            t=len(sorteios_j)
            st.dataframe(pd.DataFrame([(k,round(v/t*100,1)) for k,v in sorted(dm3.items())],columns=["Qtd","%"]),hide_index=True)

        # Filtra: equilibra primos e fibonacci dentro do esperado
        primos_cands   = [n for n in candidatas_e3 if n in PRIMOS]
        nprimos_cands  = [n for n in candidatas_e3 if n not in PRIMOS]
        fib_cands      = [n for n in candidatas_e3 if n in FIBONACCI]

        # Candidatas = mantém equilíbrio
        candidatas_e4 = candidatas_e3  # mantém todas, composição vira filtro no gerador

        st.info(f"➡️ **{len(candidatas_e4)} dezenas candidatas** para Etapa 5: {' '.join(str(n).zfill(2) for n in candidatas_e4)}")
        st.write(f"Primos nas candidatas: **{' '.join(str(n).zfill(2) for n in primos_cands)}** ({len(primos_cands)} dez)")
        st.write(f"Fibonacci nas candidatas: **{' '.join(str(n).zfill(2) for n in fib_cands)}** ({len(fib_cands)} dez)")

        if st.button("✅ Confirmar Etapa 4 — Composição", type="primary"):
            salvar_etapa("composicao", {
                "top_primo":top_primo,"top_fib":top_fib,
                "dist_primo":dp_c,"dist_fib":df_c,
                "primos_candidatas":primos_cands,
                "candidatas":candidatas_e4
            }, pipeline_update=candidatas_e4)
            for e in ETAPAS[4:]:
                st.session_state["analise"][e] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ETAPA 5 — SEQUÊNCIAS & GAPS
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("5️⃣ Sequências & Gaps")
    if not etapa_ok("composicao"):
        st.warning("⚠️ Execute as etapas anteriores primeiro.")
    elif etapa_ok("sequencias"):
        st.success("✅ Etapa concluída.")
        d = st.session_state["analise"]["sequencias"]
        st.write(f"Consecutivos mais comuns: **{d['top_consec']}** ({d['pct_consec']}%)")
        st.write(f"Gap mais comum: **{d['top_gap']}** ({d['pct_gap']}%)")
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 6")
        if st.button("🔄 Refazer Etapa 5"):
            for e in ETAPAS[4:]:
                st.session_state["analise"][e] = None
            st.rerun()
    else:
        candidatas_e4 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 4: **{' '.join(str(n).zfill(2) for n in candidatas_e4)}**")

        dc = dist_consec(sorteios_j)
        dg = dist_gaps(sorteios_j)
        top_c = max(dc, key=dc.get)
        top_g = max(dg, key=dg.get)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Consecutivos (%)")
            st.dataframe(pd.DataFrame(list(dc.items()),columns=["Pares Consec.","%"]),use_container_width=True,hide_index=True)
        with col2:
            st.subheader("Gaps (%)")
            st.dataframe(pd.DataFrame(list(dg.items()),columns=["Gap","%"]),use_container_width=True,hide_index=True)

        candidatas_e5 = candidatas_e4

        st.info(f"➡️ **{len(candidatas_e5)} dezenas candidatas** para Etapa 6: {' '.join(str(n).zfill(2) for n in candidatas_e5)}")

        if st.button("✅ Confirmar Etapa 5 — Sequências", type="primary"):
            salvar_etapa("sequencias", {
                "top_consec":top_c,"pct_consec":dc[top_c],
                "top_gap":top_g,"pct_gap":dg[top_g],
                "dist_consec":dc,"dist_gap":dg,
                "candidatas":candidatas_e5
            }, pipeline_update=candidatas_e5)
            for e in ETAPAS[5:]:
                st.session_state["analise"][e] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ETAPA 6 — N+1/N+2
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("6️⃣ N+1/N+2 — Projeção de Dezenas Prováveis")
    if not etapa_ok("sequencias"):
        st.warning("⚠️ Execute as etapas anteriores primeiro.")
    elif etapa_ok("n12"):
        st.success("✅ Etapa concluída.")
        d = st.session_state["analise"]["n12"]
        st.write(f"Concursos similares: **{len(d['similares'])}**")
        st.write(f"🔥 Fortes: **{' '.join(str(n).zfill(2) for n in sorted(d['fortes']))}**")
        st.write(f"🤝 Apoio: **{' '.join(str(n).zfill(2) for n in sorted(d['apoio']))}**")
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 7")
        if st.button("🔄 Refazer Etapa 6"):
            for e in ETAPAS[5:]:
                st.session_state["analise"][e] = None
            st.rerun()
    else:
        candidatas_e5 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 5: **{' '.join(str(n).zfill(2) for n in candidatas_e5)}**")

        sim, fortes, apoio = projetar_isca(ref, sorteios, threshold)
        conc_ref = df.iloc[ref]["Concurso"]
        st.write(f"Referência: **C{conc_ref}** | Threshold: **{threshold}%** | Similares: **{len(sim)}**")

        if sim:
            # Mostra os concursos similares
            with st.expander(f"📋 Ver {len(sim)} concursos similares encontrados"):
                sim_data = []
                for idx in sim[:20]:
                    conc_n = df.iloc[idx]["Concurso"]
                    data_n = df.iloc[idx]["Data"]
                    nums   = sorteios[idx]
                    em_comum = len(set(nums) & set(sorteios[ref]))
                    sim_data.append({
                        "Concurso": conc_n, "Data": data_n,
                        "Em comum c/ ref": em_comum,
                        "Dezenas": " ".join(str(n).zfill(2) for n in nums)
                    })
                st.dataframe(pd.DataFrame(sim_data), use_container_width=True, hide_index=True)
                if len(sim) > 20:
                    st.caption(f"... e mais {len(sim)-20} concursos similares")

            col1,col2 = st.columns(2)
            with col1:
                st.success(f"🔥 Fortes: {' · '.join(str(n).zfill(2) for n in sorted(fortes))}")
            with col2:
                st.info(f"🤝 Apoio: {' · '.join(str(n).zfill(2) for n in sorted(apoio))}")

            # Candidatas = intersecção de e5 + fortes + apoio
            # Fortes e apoio entram mesmo que não estejam em e5
            candidatas_e6 = list(dict.fromkeys(
                candidatas_e5 + fortes + apoio
            ))
            # Prioriza: fortes no topo
            candidatas_e6 = sorted(candidatas_e6,
                key=lambda n: (0 if n in fortes else 1 if n in apoio else 2, n))

            st.info(f"➡️ **{len(candidatas_e6)} dezenas candidatas** para Etapa 7: {' '.join(str(n).zfill(2) for n in candidatas_e6)}")

            if st.button("✅ Confirmar Etapa 6 — N+1/N+2", type="primary"):
                salvar_etapa("n12", {
                    "similares":sim,"fortes":fortes,"apoio":apoio,
                    "conc_ref":int(conc_ref),"threshold":threshold,
                    "candidatas":candidatas_e6
                }, pipeline_update=candidatas_e6)
                for e in ETAPAS[6:]:
                    st.session_state["analise"][e] = None
                st.rerun()
        else:
            st.warning("Nenhum similar encontrado. Reduza o threshold no menu lateral.")

# ══════════════════════════════════════════════════════════════
# ETAPA 7 — COOCORRÊNCIA
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("7️⃣ Coocorrência — Confirma Pares e Trios Frequentes")
    if not etapa_ok("n12"):
        st.warning("⚠️ Execute as etapas anteriores primeiro.")
    elif etapa_ok("coocorrencia"):
        st.success("✅ Etapa concluída.")
        d = st.session_state["analise"]["coocorrencia"]
        st.write(f"➡️ **{len(d['candidatas'])} dezenas candidatas** passam para Etapa 8")
        if st.button("🔄 Refazer Etapa 7"):
            for e in ETAPAS[6:]:
                st.session_state["analise"][e] = None
            st.rerun()
    else:
        candidatas_e6 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 6: **{' '.join(str(n).zfill(2) for n in candidatas_e6)}**")

        pares_top = calc_cooc(st_tuple)
        trios_top = calc_trios(st_tuple)
        tr        = calc_markov(st_tuple)

        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Top 20 Pares")
            st.dataframe(pd.DataFrame([(f"{a:02d}+{b:02d}",c) for (a,b),c in pares_top],columns=["Par","Freq"]),use_container_width=True,hide_index=True)
        with col2:
            st.subheader("Top 15 Trios")
            st.dataframe(pd.DataFrame([(f"{a:02d}+{b:02d}+{c:02d}",cnt) for (a,b,c),cnt in trios_top],columns=["Trio","Freq"]),use_container_width=True,hide_index=True)

        # Score de coocorrência para candidatas
        pf=Counter()
        for s in sorteios[-500:]:
            for a,b in combinations(sorted(s),2): pf[(a,b)]+=1
        gc={n:0.0 for n in range(1,26)}
        for (a,b),cnt in pf.items(): gc[a]+=cnt; gc[b]+=cnt
        gc=normalizar(gc)

        ult_s=set(sorteios[ref])
        gm={n:0.0 for n in range(1,26)}
        for x in ult_s:
            if x in tr:
                for suc,cnt in tr[x].items(): gm[suc]=gm.get(suc,0)+cnt
        gm=normalizar(gm)

        tf=Counter()
        for s in sorteios[-500:]:
            for trio in combinations(sorted(s),3): tf[trio]+=1
        gt={n:0.0 for n in range(1,26)}
        for trio,cnt in tf.most_common(100):
            for x in trio: gt[x]+=cnt
        gt=normalizar(gt)

        # Score combinado para candidatas
        score_cooc = {n: gc[n]*0.40+gm[n]*0.40+gt[n]*0.20 for n in range(1,26)}

        # Reordena candidatas por score de coocorrência
        candidatas_e7 = sorted(candidatas_e6, key=lambda n: -score_cooc[n])

        # Mostra score das candidatas
        df_sc = pd.DataFrame([
            {"Dezena":n,"Score Cooc.":round(score_cooc[n],4),
             "Pares freq":"✅" if any(n in (a,b) for (a,b),_ in pares_top[:10]) else "—",
             "Trios freq":"✅" if any(n in trio for trio,_ in trios_top[:5]) else "—"}
            for n in candidatas_e7
        ])
        st.subheader("Score de Coocorrência das Candidatas")
        st.dataframe(df_sc, use_container_width=True, hide_index=True)

        st.info(f"➡️ **{len(candidatas_e7)} dezenas candidatas** para Etapa 8: {' '.join(str(n).zfill(2) for n in candidatas_e7)}")

        if st.button("✅ Confirmar Etapa 7 — Coocorrência", type="primary"):
            salvar_etapa("coocorrencia", {
                "g3_cooc":gc,"g3_markov":gm,"g3_trios":gt,
                "score_cooc":score_cooc,
                "candidatas":candidatas_e7
            }, pipeline_update=candidatas_e7)
            for e in ETAPAS[7:]:
                st.session_state["analise"][e] = None
            st.rerun()

# ══════════════════════════════════════════════════════════════
# ETAPA 8 — MONTE CARLO
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("8️⃣ Monte Carlo — Validação Estatística Final")
    if not etapa_ok("coocorrencia"):
        st.warning("⚠️ Execute as etapas anteriores primeiro.")
    elif etapa_ok("monte_carlo"):
        st.success("✅ Etapa concluída — Protocolo pronto para gerar!")
        d = st.session_state["analise"]["monte_carlo"]
        st.write(f"χ²={d['chi2']} | p={d['p_valor']} | Base: {'✅ OK' if d['base_ok'] else '⚠️ Desvio'}")
        st.write(f"Pool final: **{' '.join(str(n).zfill(2) for n in d['pool_final'])}** ({len(d['pool_final'])} dezenas)")
        if st.button("🔄 Refazer Etapa 8"):
            st.session_state["analise"]["monte_carlo"] = None
            st.rerun()
    else:
        candidatas_e7 = st.session_state["pipeline"]
        st.info(f"Entrada da Etapa 7: **{' '.join(str(n).zfill(2) for n in candidatas_e7)}**")

        chi2,p = qui_quadrado(sorteios, janela)
        st.write(f"**Qui-Quadrado:** χ²={chi2} | p={p}")
        if p>0.05:
            st.success("✅ Base sem desvio estatístico significativo — análises confiáveis")
        else:
            st.warning("⚠️ Desvio detectado — interprete com cautela")

        # Pool final = candidatas ordenadas pelo score combinado de todas as etapas
        analise_atual = st.session_state["analise"]

        # Score rápido para ordenar pool final
        f_sc = freq_simples(sorteios, janela)
        score_final = {}
        for n in candidatas_e7:
            sc = 0
            # Térmica
            if analise_atual.get("termica"):
                quentes = analise_atual["termica"]["quentes"]
                if n in quentes: sc += (10-quentes.index(n))*0.3 if n in quentes[:10] else 0
            # Atraso
            if analise_atual.get("atraso"):
                if n in analise_atual["atraso"]["vencidas"]: sc += 5.0
                elif n in analise_atual["atraso"]["no_ciclo"]: sc += 2.5
            # N+1/N+2
            if analise_atual.get("n12"):
                if n in analise_atual["n12"]["fortes"]: sc += 4.0
                elif n in analise_atual["n12"]["apoio"]: sc += 2.0
            # Coocorrência
            if analise_atual.get("coocorrencia"):
                sc += analise_atual["coocorrencia"]["score_cooc"].get(n,0)*3.0
            # Frequência
            sc += f_sc[n]/janela*2.0
            score_final[n] = round(sc,3)

        pool_ordenado = sorted(candidatas_e7, key=lambda n: -score_final[n])

        st.subheader("🏆 Pool Final — Dezenas ordenadas por score cascata")
        df_pool = pd.DataFrame([
            {"Rank":i+1,"Dezena":n,"Score Final":score_final[n],
             "Vencida":"🔴" if analise_atual.get("atraso") and n in analise_atual["atraso"]["vencidas"] else "",
             "N+1/N+2":"🔥" if analise_atual.get("n12") and n in analise_atual["n12"]["fortes"] else
                        "🤝" if analise_atual.get("n12") and n in analise_atual["n12"]["apoio"] else "",
             "Quente":"🌡️" if analise_atual.get("termica") and n in analise_atual["termica"]["quentes"][:5] else ""}
            for i,n in enumerate(pool_ordenado)
        ])
        st.dataframe(df_pool, use_container_width=True, hide_index=True)

        st.success(f"✅ Pool final de **{len(pool_ordenado)} dezenas** pronto para o gerador!")

        if st.button("✅ Confirmar Etapa 8 — Monte Carlo", type="primary"):
            salvar_etapa("monte_carlo", {
                "chi2":chi2,"p_valor":p,"base_ok":p>0.05,
                "pool_final":pool_ordenado,
                "score_final":score_final
            }, pipeline_update=pool_ordenado)
            st.rerun()

# ══════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("📊 Backtesting")
    if jogos_bt:
        for i,jogo in enumerate(jogos_bt):
            st.subheader(f"Jogo {i+1}: {' · '.join(str(n).zfill(2) for n in jogo)}")
            bt=backtesting(jogo,sorteios,200)
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Média",bt["media"]); c2.metric("11+",f"{bt['pct11']}%")
            c3.metric("13+",f"{bt['pct13']}%"); c4.metric("Máx",bt["max"])
            st.bar_chart(pd.DataFrame(list(bt["dist"].items()),columns=["Acertos","Freq"]).set_index("Acertos"))
            st.markdown("---")
    else:
        st.info("Insira seus jogos no menu lateral.")

# ══════════════════════════════════════════════════════════════
# FILTROS & SCORE
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    st.header("🧪 Filtros Hard V4")
    st.info(f"Calibrados: Soma {cfg['soma_min']}–{cfg['soma_max']} | Pares {cfg['min_pares']}–{cfg['max_pares']} | Seq {cfg['max_seq']}")
    if jogos_bt:
        for i,jogo in enumerate(jogos_bt):
            st.subheader(f"Jogo {i+1}")
            erros=filtrar(jogo,sorteios,cfg)
            perf={"soma":sum(jogo),"pares":sum(1 for n in jogo if n%2==0)}
            c=composicao(jogo); g=gaps(jogo); mx,_=contar_consec(jogo)
            c1,c2,c3=st.columns(3)
            c1.write(f"Soma:{perf['soma']} | Pares:{perf['pares']} | Seq:{mx}")
            c2.write(f"Faixas:{dist_faixas(jogo)}")
            c3.write(f"Primos:{c['primos']} | Fib:{c['fibonacci']} | Gap md:{g['media']}")
            if erros:
                for e in erros: st.warning(e)
            else: st.success("✅ Passou em todos os filtros!")
            st.markdown("---")
    else:
        st.info("Insira seus jogos no menu lateral.")

# ══════════════════════════════════════════════════════════════
# TENDÊNCIA ROLLING
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    st.header("📈 Tendência Rolling")
    THRESHOLD_TEND = 0.05
    esp = 15/25

    rows_all=[]
    for n in range(1,26):
        df_n=pd.DataFrame([{"Janela":j,"Freq":sum(1 for s in sorteios[-j:] if n in s)} for j in [20,50,100,200]])
        t20=df_n[df_n["Janela"]==20]["Freq"].values[0]/20
        t50=df_n[df_n["Janela"]==50]["Freq"].values[0]/50
        t200=df_n[df_n["Janela"]==200]["Freq"].values[0]/200
        dcl=t20-t200; dml=t50-t200
        if dcl>THRESHOLD_TEND and dml>0:   status="🔼 ALTA"
        elif dcl<-THRESHOLD_TEND and dml<0: status="🔽 BAIXA"
        elif t50>esp*1.05:  status="🟢 ACIMA"
        elif t50<esp*0.95:  status="🟡 ABAIXO"
        else: status="➡️ ESTÁVEL"
        rows_all.append({"Dezena":n,"J20%":round(t20*100,1),"J50%":round(t50*100,1),
                          "J200%":round(t200*100,1),"Δ(20-200)":f"{dcl*100:+.1f}pp","Status":status})

    df_tend=pd.DataFrame(rows_all)
    st.dataframe(df_tend,use_container_width=True,hide_index=True)

    altas =[r["Dezena"] for r in rows_all if "ALTA"  in r["Status"]]
    baixas=[r["Dezena"] for r in rows_all if "BAIXA" in r["Status"]]
    acimas=[r["Dezena"] for r in rows_all if "ACIMA" in r["Status"]]
    if altas:  st.success(f"🔼 ALTA: {' '.join(str(n).zfill(2) for n in altas)}")
    if baixas: st.warning(f"🔽 BAIXA: {' '.join(str(n).zfill(2) for n in baixas)}")
    if acimas: st.info(f"🟢 ACIMA: {' '.join(str(n).zfill(2) for n in acimas)}")

# ══════════════════════════════════════════════════════════════
# VALOR ESPERADO
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    st.header("⚖️ Valor Esperado")
    ev=sum(PREMIOS[k]*PROB[k] for k in PREMIOS)
    c1,c2,c3=st.columns(3)
    c1.metric("EV/jogo",f"R$ {round(ev,4)}")
    c2.metric("Custo","R$ 3,50")
    c3.metric("Ratio",round(ev/3.5,4))
    st.dataframe(pd.DataFrame([
        {"Faixa":k,"Prob":f"1 em {int(1/v):,}","Prêmio":f"R$ {PREMIOS[k]:,.0f}","EV":f"R$ {PREMIOS[k]*v:.4f}"}
        for k,v in sorted(PROB.items(),reverse=True)
    ]),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════
# GERADOR — só libera com 8/8 etapas
# ══════════════════════════════════════════════════════════════
with tabs[13]:
    st.header("🎰 Gerador — Protocolo Hard V4")
    ok,tot=status_count()

    if not todas_ok():
        st.error(f"⛔ {tot-ok} etapa(s) pendente(s)")
        for e in ETAPAS:
            st.write(f"{'✅' if etapa_ok(e) else '❌'} {ETAPA_LABELS[e]}")

        if st.button("⚡ Executar todas as etapas automaticamente", type="primary"):
            with st.spinner("Executando protocolo completo..."):
                # E1 Térmica
                f_a=freq_simples(sorteios,janela)
                top10_a=[n for n,_ in f_a.most_common(10)]
                bot5_a=[n for n,_ in f_a.most_common()[:-6:-1]]
                pico_a=dezenas_pico(sorteios)
                cand1=list(dict.fromkeys(top10_a+pico_a))
                salvar_etapa("termica",{"quentes":top10_a,"frias":bot5_a,"pico":pico_a,
                    "candidatas":cand1,"freq":{n:f_a[n] for n in range(1,26)}},
                    pipeline_update=cand1)

                # E2 Atraso
                df_at_a=calcular_atrasos(sorteios)
                cl_a=[{"Dezena":n,"Ciclo Médio":ciclo_medio(sorteios,n)} for n in range(1,26)]
                df_c_a=df_at_a.merge(pd.DataFrame(cl_a),on="Dezena")
                df_c_a["Status"]=df_c_a.apply(
                    lambda r:"🔴 VENCIDA" if r["Atraso"]>(r["Ciclo Médio"] or 99)*1.5
                    else("🟡 No ciclo" if r["Atraso"]>(r["Ciclo Médio"] or 99)*0.8 else "🟢 Ok"),axis=1)
                ven_a=df_c_a[df_c_a["Status"]=="🔴 VENCIDA"]["Dezena"].tolist()
                noc_a=df_c_a[df_c_a["Status"]=="🟡 No ciclo"]["Dezena"].tolist()
                cand2=list(dict.fromkeys(cand1+ven_a+noc_a[:3]))
                salvar_etapa("atraso",{"df_completo":df_c_a,"vencidas":ven_a,
                    "no_ciclo":noc_a,"candidatas":cand2},pipeline_update=cand2)

                # E3 Estrutural
                sm_a,sx_a,smed_a=faixa_soma(sorteios_j)
                dp_a=dist_pares(sorteios_j)
                tp_a=max(dp_a,key=dp_a.get)
                cand3=cand2
                salvar_etapa("estrutural",{"soma_min":sm_a,"soma_max":sx_a,"soma_med":smed_a,
                    "top_pares":tp_a,"pct_top_pares":dp_a[tp_a],"dist_pares":dp_a,
                    "candidatas":cand3},pipeline_update=cand3)

                # E4 Composição
                dp_ca,df_ca=dist_comp(sorteios_j)
                tpr_a=max(dp_ca,key=dp_ca.get); tfb_a=max(df_ca,key=df_ca.get)
                pc_a=[n for n in cand3 if n in PRIMOS]
                cand4=cand3
                salvar_etapa("composicao",{"top_primo":tpr_a,"top_fib":tfb_a,
                    "dist_primo":dp_ca,"dist_fib":df_ca,
                    "primos_candidatas":pc_a,"candidatas":cand4},pipeline_update=cand4)

                # E5 Sequências
                dc_a=dist_consec(sorteios_j); dg_a=dist_gaps(sorteios_j)
                cand5=cand4
                salvar_etapa("sequencias",{"top_consec":max(dc_a,key=dc_a.get),
                    "pct_consec":dc_a[max(dc_a,key=dc_a.get)],
                    "top_gap":max(dg_a,key=dg_a.get),"pct_gap":dg_a[max(dg_a,key=dg_a.get)],
                    "dist_consec":dc_a,"dist_gap":dg_a,"candidatas":cand5},pipeline_update=cand5)

                # E6 N+1/N+2
                sim_a,fortes_a,apoio_a=projetar_isca(ref,sorteios,threshold)
                cand6=list(dict.fromkeys(cand5+fortes_a+apoio_a))
                cand6=sorted(cand6,key=lambda n:(0 if n in fortes_a else 1 if n in apoio_a else 2,n))
                salvar_etapa("n12",{"similares":sim_a,"fortes":fortes_a,"apoio":apoio_a,
                    "conc_ref":int(df.iloc[ref]["Concurso"]),"threshold":threshold,
                    "candidatas":cand6},pipeline_update=cand6)

                # E7 Coocorrência
                pf_a=Counter()
                for s in sorteios[-500:]:
                    for a,b in combinations(sorted(s),2): pf_a[(a,b)]+=1
                gc_a={n:0.0 for n in range(1,26)}
                for (a,b),c in pf_a.items(): gc_a[a]+=c; gc_a[b]+=c
                gc_a=normalizar(gc_a)
                tr_a=calc_markov(st_tuple)
                ult_s_a=set(sorteios[ref])
                gm_a={n:0.0 for n in range(1,26)}
                for x in ult_s_a:
                    if x in tr_a:
                        for suc,c in tr_a[x].items(): gm_a[suc]=gm_a.get(suc,0)+c
                gm_a=normalizar(gm_a)
                tf_a=Counter()
                for s in sorteios[-500:]:
                    for trio in combinations(sorted(s),3): tf_a[trio]+=1
                gt_a={n:0.0 for n in range(1,26)}
                for trio,c in tf_a.most_common(100):
                    for x in trio: gt_a[x]+=c
                gt_a=normalizar(gt_a)
                sc_cooc={n:gc_a[n]*0.40+gm_a[n]*0.40+gt_a[n]*0.20 for n in range(1,26)}
                cand7=sorted(cand6,key=lambda n:-sc_cooc[n])
                salvar_etapa("coocorrencia",{"g3_cooc":gc_a,"g3_markov":gm_a,"g3_trios":gt_a,
                    "score_cooc":sc_cooc,"candidatas":cand7},pipeline_update=cand7)

                # E8 Monte Carlo
                chi2_a,p_a=qui_quadrado(sorteios,janela)
                analise_a=st.session_state["analise"]
                f_sc_a=freq_simples(sorteios,janela)
                sf_a={}
                for n in cand7:
                    sc=0
                    if n in top10_a: sc+=(10-top10_a.index(n))*0.3
                    if n in ven_a: sc+=5.0
                    elif n in noc_a: sc+=2.5
                    if n in fortes_a: sc+=4.0
                    elif n in apoio_a: sc+=2.0
                    sc+=sc_cooc.get(n,0)*3.0
                    sc+=f_sc_a[n]/janela*2.0
                    sf_a[n]=round(sc,3)
                pool_f=sorted(cand7,key=lambda n:-sf_a[n])
                salvar_etapa("monte_carlo",{"chi2":chi2_a,"p_valor":p_a,"base_ok":p_a>0.05,
                    "pool_final":pool_f,"score_final":sf_a},pipeline_update=pool_f)

            st.success("✅ Protocolo executado! Gerando combinações...")
            st.rerun()

    else:
        analise = st.session_state["analise"]
        pool_final   = analise["monte_carlo"]["pool_final"]
        score_final  = analise["monte_carlo"]["score_final"]
        vencidas_g   = analise["atraso"]["vencidas"]
        fortes_g     = analise["n12"]["fortes"]
        apoio_g      = analise["n12"]["apoio"]

        st.success("🎯 Protocolo completo — todas as 8 etapas em cascata!")

        with st.expander("📋 Pool depurado pelas 8 etapas"):
            df_pf=pd.DataFrame([
                {"Rank":i+1,"Dezena":n,"Score":score_final[n],
                 "🔴 Vencida":"✅" if n in vencidas_g else "",
                 "🔥 N+1/N+2 Forte":"✅" if n in fortes_g else "",
                 "🤝 Apoio":"✅" if n in apoio_g else ""}
                for i,n in enumerate(pool_final)
            ])
            st.dataframe(df_pf,use_container_width=True,hide_index=True)

        st.markdown("---")
        perfil_nome=st.radio("Perfil:",list(PERFIS.keys()),horizontal=True)
        perfil=PERFIS[perfil_nome]
        st.info(f"**{perfil_nome}** — {perfil['descricao']}")

        col1,col2=st.columns(2)
        with col1:
            qtd=st.number_input("Quantos jogos?",min_value=1,value=3,step=1)
            st.warning(f"💰 {qtd} × R$ 3,50 = **R$ {qtd*3.5:.2f}**")
        with col2:
            seed=st.number_input("Seed (0=aleatório)",min_value=0,value=0,step=1)
            fixas_inp=st.text_input("Dezenas fixas:",placeholder="Ex: 05 13")

        if perfil["aviso"] and qtd>perfil["aviso"]:
            st.warning(f"⚠️ Agressiva com {qtd} jogos — pool expande automaticamente se necessário.")

        fixas=[]
        if fixas_inp.strip():
            try: fixas=[int(x) for x in fixas_inp.strip().split() if 1<=int(x)<=25]
            except: pass

        if st.button("🎯 Gerar Combinações", type="primary"):
            if seed>0: random.seed(seed)

            prog=st.progress(0,text="Calculando scores em cascata...")
            scores,_sim,_f,_a,_v,_nc=calcular_scores(
                sorteios,janela,ref,threshold,perfil["pesos"],analise)

            # Pool = pool depurado pelas 8 etapas + ajuste por perfil
            ranking=sorted(scores.items(),key=lambda x:-x[1])

            # Vencidas SEMPRE entram no pool independente do perfil
            pool_obrig = list(vencidas_g)  # obrigatórias
            pool_score = [n for n,_ in ranking if n not in pool_obrig]

            if "Agressiva" in perfil_nome:
                pool_din = max(perfil["top_pool"], min(qtd+3,22))
            else:
                pool_din = max(perfil["top_pool"], min(qtd+5,25))

            # Pool = obrigatórias + top por score
            pool = pool_obrig + [n for n in pool_score if n not in pool_obrig][:pool_din]
            pool = pool[:pool_din+len(pool_obrig)]
            for f in fixas:
                if f not in pool: pool.append(f)

            st.write(f"**Pool ({len(pool)} dez):** {' · '.join(f'{n:02d}' for n in sorted(pool))}")
            if vencidas_g:
                st.write(f"🔴 Vencidas (obrigatórias no pool): **{' '.join(str(n).zfill(2) for n in vencidas_g)}**")

            jogos=[]
            tent=0; max_t=max(200000,qtd*10000)
            mc_atual=perfil["max_comum"]
            ult_g=0; trav=0
            prog.progress(10,text="Gerando...")

            while len(jogos)<qtd and tent<max_t:
                tent+=1
                if tent-ult_g>20000 and len(jogos)>0:
                    trav+=1; ult_g=tent
                    mc_atual=min(perfil["max_comum"]+trav,13)
                    if pool_din<25:
                        pool_din=min(pool_din+2,25)
                        pool=pool_obrig+[n for n in pool_score if n not in pool_obrig][:pool_din]
                        for f in fixas:
                            if f not in pool: pool.append(f)
                    prog.progress(min(10+int(len(jogos)/qtd*85),95),
                        text=f"{len(jogos)}/{qtd} — ajustando (pool {len(pool)}, sim {mc_atual})...")

                base=fixas[:15]
                rest=[n for n in pool if n not in base]
                if len(rest)<15-len(base):
                    ext=[n for n in range(1,26) if n not in base and n not in rest]
                    rest+=ext
                faltam=15-len(base)
                rest_sc=np.array([scores[n] for n in rest[:faltam+10]])
                rest_sc=rest_sc/rest_sc.sum()
                escolh=list(np.random.choice(rest[:faltam+10],size=faltam,replace=False,p=rest_sc))
                cand=sorted(base+escolh)

                if filtrar(cand,sorteios,cfg): continue
                if any(15-hamming(cand,j)>mc_atual for j in jogos): continue
                if cand not in jogos:
                    jogos.append(cand)
                    ult_g=tent
                    prog.progress(min(10+int(len(jogos)/qtd*85),95),
                        text=f"Gerados {len(jogos)}/{qtd}...")

            prog.progress(100,text="Concluído!")

            if len(jogos)<qtd:
                st.warning(f"⚠️ Gerados {len(jogos)}/{qtd}. Tente Híbrida.")

            if jogos:
                st.success(f"✅ {len(jogos)} combinação(ões) — {tent:,} tentativas")
                st.markdown("---")
                for i,jogo in enumerate(jogos):
                    bt=backtesting(jogo,sorteios,200)
                    c=composicao(jogo); f_j=dist_faixas(jogo); mx_j,_=contar_consec(jogo)
                    sc_j=round(sum(scores[n] for n in jogo),4)
                    st.subheader(f"🃏 Combinação {i+1}")
                    st.markdown("## "+"  ·  ".join(f"**{n:02d}**" for n in jogo))
                    c1,c2,c3,c4,c5=st.columns(5)
                    c1.metric("Score",sc_j)
                    c2.metric("Soma",sum(jogo))
                    c3.metric("Par/Ímp",f"{sum(1 for n in jogo if n%2==0)}/{sum(1 for n in jogo if n%2!=0)}")
                    c4.metric("11+",f"{bt['pct11']}%")
                    c5.metric("13+",f"{bt['pct13']}%")
                    col6,col7,col8=st.columns(3)
                    col6.write(f"Faixas:{f_j}")
                    col7.write(f"Primos:{c['primos']}|Fib:{c['fibonacci']}|M3:{c['mult3']}")
                    col8.write(f"Seq:{mx_j}|Ampl:{max(jogo)-min(jogo)}")
                    cob_v=len(set(jogo)&set(vencidas_g))
                    cob_f=len(set(jogo)&set(fortes_g))
                    st.write(f"🔴 Vencidas cobertas: **{cob_v}/{len(vencidas_g)}** | 🔥 Fortes N+1/N+2: **{cob_f}/6**")
                    with st.expander(f"🔬 Breakdown — Combinação {i+1}"):
                        bd=pd.DataFrame([
                            {"Dezena":n,"Score":round(scores[n],4),
                             "Rank":sorted(scores,key=lambda x:-scores[x]).index(n)+1,
                             "Vencida":"🔴" if n in vencidas_g else "",
                             "N+1/N+2":"🔥" if n in fortes_g else "🤝" if n in apoio_g else "",
                             "Score Final":score_final.get(n,0)}
                            for n in sorted(jogo)
                        ])
                        st.dataframe(bd,use_container_width=True,hide_index=True)
                    st.success("✅ Aprovado — filtros Hard V4")
                    st.markdown("---")

                st.info(f"💰 **{len(jogos)} × R$ 3,50 = R$ {len(jogos)*3.5:.2f}**")
                if len(jogos)>1:
                    st.subheader("📐 Hamming")
                    st.dataframe(cobertura_jogos(jogos),use_container_width=True,hide_index=True)

                st.session_state["jogos_gerados"] = jogos
            else:
                st.error("❌ Nenhuma combinação gerada. Tente Híbrida.")

# ══════════════════════════════════════════════════════════════
# RELATÓRIO FINAL
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    st.header("🏆 Relatório Final")
    analise=st.session_state["analise"]
    ok,tot=status_count()
    st.markdown(f"**Base:** {len(sorteios)} concursos | C{df['Concurso'].max()} | Janela:{janela} | Etapas:{ok}/{tot}")
    st.progress(ok/tot)

    col1,col2=st.columns(2)
    with col1:
        st.subheader("📊 Cascata de Análises")
        for e in ETAPAS:
            icon="✅" if etapa_ok(e) else "⭕"
            d=analise.get(e)
            if d and "candidatas" in d:
                st.write(f"{icon} **{ETAPA_LABELS[e]}** → {len(d['candidatas'])} candidatas")
                if e=="termica":
                    st.caption(f"Quentes: {' '.join(str(n).zfill(2) for n in d['quentes'][:5])}")
                elif e=="atraso":
                    st.caption(f"Vencidas: {' '.join(str(n).zfill(2) for n in d['vencidas'])}")
                elif e=="n12":
                    st.caption(f"Fortes: {' '.join(str(n).zfill(2) for n in sorted(d['fortes']))}")
                elif e=="monte_carlo":
                    st.caption(f"Base: {'✅ OK' if d['base_ok'] else '⚠️ Desvio'} | χ²={d['chi2']}")
            else:
                st.write(f"{icon} **{ETAPA_LABELS[e]}** — pendente")

    with col2:
        st.subheader("🎯 Pool Final Depurado")
        if etapa_ok("monte_carlo"):
            d=analise["monte_carlo"]
            pool=d["pool_final"]; sf=d["score_final"]
            st.write(f"**{len(pool)} dezenas rankeadas pelas 8 etapas:**")
            for i,n in enumerate(pool[:20]):
                tags=""
                if analise.get("atraso") and n in analise["atraso"]["vencidas"]: tags+="🔴"
                if analise.get("n12") and n in analise["n12"]["fortes"]: tags+="🔥"
                if analise.get("n12") and n in analise["n12"]["apoio"]: tags+="🤝"
                st.write(f"#{i+1} **{n:02d}** — score {sf[n]} {tags}")
        else:
            st.info("Execute todas as etapas para ver o pool final.")

    if st.session_state["jogos_gerados"]:
        st.subheader("🃏 Combinações Geradas")
        for i,jogo in enumerate(st.session_state["jogos_gerados"]):
            bt=backtesting(jogo,sorteios,200)
            erros=filtrar(jogo,sorteios,cfg)
            status="✅" if not erros else "⚠️"
            st.write(f"**J{i+1}:** {' '.join(str(n).zfill(2) for n in jogo)} | Média:{bt['media']} | 11+:{bt['pct11']}% | {status}")

    st.markdown("---")
    st.caption("Protocolo Hard V4 — MarianaRBrito | github.com/MarianaRBrito/protocolo-hard-v4")
