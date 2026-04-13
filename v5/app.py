# =============================================================================
# app.py — Protocolo Lotofácil V5
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import hashlib

# Garante path correto
v5_dir = os.path.dirname(os.path.abspath(__file__))
if v5_dir not in sys.path:
    sys.path.insert(0, v5_dir)

from engine_core import (
    STEP_REGISTRY, DEZENAS, PipelineExecutor,
    PRIMOS, FIBONACCI, MULTIPLOS_3
)
from etapa_termica import executar_etapa_termica, get_termica
from etapa_atraso   import executar_etapa_atraso,  get_atraso
from etapas_estatisticas import executar_etapas_estatisticas
from etapas_heuristicas  import (
    executar_etapas_heuristicas,
    run_hamming, run_hamming_minima, run_cobertura_minima
)
from gerador_v5 import (
    gerar_jogos_v5, comparar_jogos, breakdown_jogo,
    calibrar_filtros, filtrar_jogo
)

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(
    page_title="Protocolo Lotofácil V5",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
.dez-box {
    display: inline-block; padding: 5px 11px; margin: 3px;
    border-radius: 50%; background: #1a1a2e;
    border: 2px solid #00FFCC; font-family: monospace;
    font-weight: bold; color: #00FFCC; font-size: 16px;
    min-width: 36px; text-align: center;
}
.dez-hot  { border-color: #FF4B4B; color: #FF4B4B; }
.dez-warm { border-color: #FFA500; color: #FFA500; }
.dez-cold { border-color: #00BFFF; color: #00BFFF; }
.dez-venc { border-color: #FF0000; color: #FF0000; background: #2a0000; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARGA DE DADOS — definida UMA única vez
# =============================================================================
@st.cache_data(ttl=300)
def carregar_base():
    caminhos = [
        "base_lotofacil.csv",
        "../base_lotofacil.csv",
        os.path.join(os.path.dirname(__file__), "..", "base_lotofacil.csv"),
        "/mount/src/protocolo-hard-v4/base_lotofacil.csv",
    ]
    for caminho in caminhos:
        if os.path.exists(caminho):
            df = pd.read_csv(caminho)
            return df.sort_values("Concurso").reset_index(drop=True)
    st.error("❌ base_lotofacil.csv não encontrado!")
    st.stop()

@st.cache_data
def extrair_sorteios(df):
    cols = [c for c in df.columns if c.startswith("bola")]
    return df[cols].values.tolist()

def _max_consec(jogo):
    jogo = sorted(jogo)
    mx = cur = 1
    for i in range(1, len(jogo)):
        if jogo[i] == jogo[i-1]+1:
            cur += 1; mx = max(mx, cur)
        else:
            cur = 1
    return mx

# =============================================================================
# SESSION STATE
# =============================================================================
def init_state():
    if "ctx" not in st.session_state:
        st.session_state["ctx"] = {}
    if "executor" not in st.session_state:
        st.session_state["executor"] = PipelineExecutor(st.session_state["ctx"])
    if "jogos_gerados" not in st.session_state:
        st.session_state["jogos_gerados"] = []
    if "param_hash" not in st.session_state:
        st.session_state["param_hash"] = ""
    if "exp_habilitados" not in st.session_state:
        st.session_state["exp_habilitados"] = False

def get_param_hash(janela, ref, threshold):
    return hashlib.md5(f"{janela}_{ref}_{threshold}".encode()).hexdigest()[:8]

def resetar_pipeline():
    st.session_state["ctx"] = {}
    st.session_state["executor"] = PipelineExecutor(st.session_state["ctx"])
    st.session_state["jogos_gerados"] = []

init_state()

# =============================================================================
# CARREGA BASE
# =============================================================================
df       = carregar_base()
sorteios = extrair_sorteios(df)
n_conc   = len(sorteios)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("⚙️ V5 Control")
    st.success(f"✅ Base: {n_conc} concursos | C{df['Concurso'].max()}")

    st.markdown("---")
    st.subheader("📐 Parâmetros")
    janela    = st.slider("Janela de análise", 100, n_conc, min(500, n_conc), 50)
    ref_idx   = st.number_input("Índice referência (0=último)", 0, n_conc-1, n_conc-1)
    threshold = st.slider("Threshold N+1/N+2 (%)", 50, 100, 80)

    h = get_param_hash(janela, ref_idx, threshold)
    if st.session_state["param_hash"] != h:
        resetar_pipeline()
        st.session_state["param_hash"] = h

    st.markdown("---")
    st.subheader("🧪 Experimentais")
    exp_on = st.toggle("Habilitar etapas experimentais", value=False)
    st.session_state["exp_habilitados"] = exp_on

    st.markdown("---")
    st.subheader("📊 Progresso")
    ex = st.session_state["executor"]
    conc, total = ex.progresso(incluir_experimentais=False)
    st.progress(conc / total if total > 0 else 0,
                text=f"{conc}/{total} etapas core")

    st.markdown("---")
    if st.button("⚡ EXECUTAR PROTOCOLO COMPLETO", type="primary", use_container_width=True):
        with st.spinner("Executando pipeline V5..."):
            try:
                executar_etapa_termica(ex, sorteios, janela)
                executar_etapa_atraso(ex, sorteios)
                executar_etapas_estatisticas(ex, sorteios, janela, ref_idx, threshold)
                executar_etapas_heuristicas(ex, sorteios, janela, exp_on)
                st.success("✅ Pipeline concluído!")
            except Exception as e:
                st.error(f"Erro: {e}")
        st.rerun()

    if st.button("🔄 Resetar pipeline", use_container_width=True):
        resetar_pipeline()
        st.rerun()

    st.markdown("---")
    st.subheader("🎰 Backtesting manual")
    j1 = st.text_input("Jogo 1 (15 números separados por espaço)")
    j2 = st.text_input("Jogo 2")

def parse_jogo(s):
    try:
        ns = [int(x) for x in s.strip().split()]
        if len(ns) == 15 and all(1 <= x <= 25 for x in ns):
            return sorted(ns)
    except Exception:
        pass
    return None

jogos_bt = [j for j in [parse_jogo(j1), parse_jogo(j2)] if j]

# =============================================================================
# TÍTULO
# =============================================================================
st.title("🛡️ Protocolo Lotofácil V5")
st.caption(
    f"Base: **{n_conc} concursos** | "
    f"C{df['Concurso'].min()}–C{df['Concurso'].max()} | "
    f"Janela: **{janela}** | "
    f"Ref: C{df.iloc[ref_idx]['Concurso']}"
)
conc, total = ex.progresso(False)
st.progress(conc/total if total else 0,
            text=f"Pipeline: {conc}/{total} etapas {'✅ Pronto para gerar!' if conc==total else ''}")

# =============================================================================
# ABAS
# =============================================================================
tabs = st.tabs([
    "🌡️ Térmica","⏱️ Atraso","📊 Análises",
    "🏗️ Pipeline","🎰 Gerar","📋 Relatório","🔬 Backtesting",
])

# ── ABA 1: TÉRMICA ────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("🌡️ Etapa Térmica")
    if st.button("▶️ Executar Térmica", type="primary"):
        with st.spinner("Calculando temperatura..."):
            executar_etapa_termica(ex, sorteios, janela)
        st.rerun()

    t_res = get_termica(ex)
    if not t_res:
        st.info("Execute a etapa térmica para ver os resultados.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Candidatas", len(t_res.get("candidatas",[])))
        c2.metric("Vencidas pressão", len(t_res.get("vencidas",[])))
        c3.metric("Quentes", len([d for d in DEZENAS if t_res["classificacoes"][d]["nivel"]>=5]))
        c4.metric("Frias",   len([d for d in DEZENAS if t_res["classificacoes"][d]["nivel"]<=2]))

        st.subheader("📊 Tabela Térmica")
        st.dataframe(t_res["df_tabela"], use_container_width=True, hide_index=True)

        st.subheader("🗺️ Grade 5×5")
        cols_grade = st.columns(5)
        for i, d in enumerate(DEZENAS):
            with cols_grade[i%5]:
                cl = t_res["classificacoes"][d]
                css = "dez-hot" if cl["nivel"]>=5 else "dez-warm" if cl["nivel"]>=3 else "dez-cold"
                st.markdown(f'<div class="dez-box {css}">{d:02d}</div>', unsafe_allow_html=True)

        col_f, col_q = st.columns(2)
        with col_f:
            st.subheader("Por Faixa")
            rows_f = [{"Faixa":k,"Freq%":round(v["media_freq"]*100,1)} for k,v in t_res["temp_faixas"].items()]
            st.dataframe(pd.DataFrame(rows_f), hide_index=True, use_container_width=True)
        with col_q:
            st.subheader("Por Quadrante")
            rows_q = [{"Quadrante":k,"Freq%":round(v["media_freq"]*100,1)} for k,v in t_res["temp_quadrantes"].items()]
            st.dataframe(pd.DataFrame(rows_q), hide_index=True, use_container_width=True)

        st.success(f"✅ Candidatas: {' '.join(f'{d:02d}' for d in t_res.get('candidatas',[]))}")
        if t_res.get("bloqueios"):
            st.warning(f"⛔ Bloqueios: {' '.join(f'{d:02d}' for d in t_res['bloqueios'])}")

# ── ABA 2: ATRASO ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("⏱️ Atraso & Ciclo")
    if st.button("▶️ Executar Atraso", type="primary",
                 disabled=not ex.concluido("frequencia_simples")):
        with st.spinner("Calculando atrasos..."):
            executar_etapa_atraso(ex, sorteios)
        st.rerun()

    if not ex.concluido("atraso_atual"):
        st.info("Execute a etapa térmica primeiro.")
    else:
        at_res = get_atraso(ex)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🔴 Vencidas",  len(at_res.get("vencidas",[])))
        c2.metric("🟠 Atrasadas", len(at_res.get("atrasadas",[])))
        c3.metric("🟡 No ciclo",  len(at_res.get("no_ciclo",[])))
        c4.metric("🟢 OK",        len(at_res.get("ok",[])))

        if at_res.get("vencidas"):   st.error(f"🔴 VENCIDAS: {' '.join(f'{d:02d}' for d in at_res['vencidas'])}")
        if at_res.get("no_ciclo"):   st.warning(f"🟡 NO CICLO: {' '.join(f'{d:02d}' for d in at_res['no_ciclo'])}")
        if at_res.get("atrasadas"):  st.warning(f"🟠 ATRASADAS: {' '.join(f'{d:02d}' for d in at_res['atrasadas'])}")

        st.dataframe(at_res["df_tabela"], use_container_width=True, hide_index=True)

        col_p, col_f2 = st.columns(2)
        with col_p:
            st.subheader("📐 Percentis Históricos")
            st.dataframe(at_res["df_percentis"], hide_index=True, use_container_width=True)
        with col_f2:
            st.subheader("📍 Atraso por Faixa")
            rows_af = [{"Faixa":nome,"Atraso Médio":v["atraso_medio"],
                        "Maior":v["atraso_max"],"Dezena":v["dezena_mais_atrasada"]}
                       for nome,v in at_res.get("atraso_faixa",{}).items()]
            st.dataframe(pd.DataFrame(rows_af), hide_index=True, use_container_width=True)

        st.info(f"🎯 Prioridade: {' '.join(f'{d:02d}' for d in at_res.get('prioridade',[]))}")

# ── ABA 3: ANÁLISES ───────────────────────────────────────────────────────────
with tabs[2]:
    st.header("📊 Análises")
    col_e, col_h = st.columns(2)
    with col_e:
        if st.button("▶️ Executar Estatísticas", type="primary",
                     disabled=not ex.concluido("atraso_atual")):
            with st.spinner("Calculando..."):
                executar_etapas_estatisticas(ex, sorteios, janela, ref_idx, threshold)
            st.rerun()
    with col_h:
        if st.button("▶️ Executar Heurísticas", type="secondary",
                     disabled=not ex.concluido("backtesting")):
            with st.spinner("Calculando..."):
                executar_etapas_heuristicas(ex, sorteios, janela,
                                             st.session_state["exp_habilitados"])
            st.rerun()

    if not ex.concluido("qui_quadrado"):
        st.info("Execute as etapas anteriores primeiro.")
    else:
        chi2_r = ex.ctx["resultados"].get("qui_quadrado",{})
        c1,c2,c3 = st.columns(3)
        c1.metric("χ²", chi2_r.get("chi2","—"))
        c2.metric("p-valor", chi2_r.get("p_valor","—"))
        c3.metric("Base OK","✅" if chi2_r.get("base_ok") else "⚠️")

        sc = ex.score_consolidado()
        ranking = sorted(DEZENAS, key=lambda d: -sc.get(d,0))
        st.subheader("🏆 Score Consolidado")
        st.dataframe(pd.DataFrame([{"Rank":i+1,"Dezena":d,"Score":round(sc[d],4)}
                                    for i,d in enumerate(ranking)]),
                     hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame({"Score":{d:sc[d] for d in ranking}}), color="#00FFCC")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📈 Tendência Rolling")
            tr = ex.ctx["resultados"].get("tendencia_rolling",{})
            if tr:
                if tr.get("altas"):  st.success(f"🔼 Alta: {' '.join(f'{d:02d}' for d in tr['altas'])}")
                if tr.get("baixas"): st.warning(f"🔽 Baixa: {' '.join(f'{d:02d}' for d in tr['baixas'])}")
        with col_b:
            st.subheader("🔗 N+1/N+2")
            n12 = ex.ctx["resultados"].get("n1_n2",{})
            if n12:
                st.write(f"Similares: **{n12.get('n_similares',0)}**")
                st.success(f"🔥 Fortes: {' '.join(f'{d:02d}' for d in n12.get('fortes',[]))}")
                st.info(f"🤝 Apoio: {' '.join(f'{d:02d}' for d in n12.get('apoio',[]))}")

        bt_r = ex.ctx["resultados"].get("backtesting",{})
        if bt_r:
            st.subheader("📊 Backtesting")
            c1,c2,c3 = st.columns(3)
            c1.metric("Média", bt_r.get("media_acertos"))
            c2.metric("11+%", f"{bt_r.get('pct_11',0)}%")
            c3.metric("13+%", f"{bt_r.get('pct_13',0)}%")

# ── ABA 4: PIPELINE ───────────────────────────────────────────────────────────
with tabs[3]:
    st.header("🏗️ Pipeline")
    df_status = ex.relatorio_status()

    col_f1,col_f2,col_f3 = st.columns(3)
    with col_f1:
        filtro_grupo = st.selectbox("Grupo",["Todos"]+sorted(df_status["Grupo"].unique().tolist()))
    with col_f2:
        filtro_status = st.selectbox("Status",["Todos","concluido","pendente","erro"])
    with col_f3:
        mostrar_exp = st.checkbox("Mostrar experimentais", value=False)

    df_view = df_status.copy()
    if filtro_grupo  != "Todos": df_view = df_view[df_view["Grupo"]==filtro_grupo]
    if filtro_status != "Todos": df_view = df_view[df_view["Status"]==filtro_status]
    if not mostrar_exp:          df_view = df_view[df_view["Experimental"]==""]

    def color_status(val):
        if val=="concluido": return "background-color:#0d3320;color:#00FF88"
        if val=="erro":      return "background-color:#3a0000;color:#FF4444"
        if val=="pendente":  return "color:#888888"
        return ""

    st.dataframe(df_view.style.map(color_status, subset=["Status"]),
                 use_container_width=True, hide_index=True, height=500)

    conc,total = ex.progresso(False)
    conc_e,total_e = ex.progresso(True)
    c1,c2,c3 = st.columns(3)
    c1.metric("Core",f"{conc}/{total}")
    c2.metric("Com experimentais",f"{conc_e}/{total_e}")
    c3.metric("Steps ativos", len([s for s in STEP_REGISTRY if ex.concluido(s["id"]) and s["influencia_gerador"]]))

    if ex.ctx.get("log"):
        with st.expander("📋 Log"):
            for linha in ex.ctx["log"][-50:]:
                st.code(linha)

# ── ABA 5: GERAR ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🎰 Gerador V5")
    conc,total = ex.progresso(False)
    pipeline_ok = conc >= total//2

    if not pipeline_ok:
        st.warning(f"Execute ao menos {total//2} etapas. Atual: {conc}/{total}")
        if st.button("⚡ Executar pipeline completo", type="primary"):
            with st.spinner("Executando..."):
                executar_etapa_termica(ex, sorteios, janela)
                executar_etapa_atraso(ex, sorteios)
                executar_etapas_estatisticas(ex, sorteios, janela, ref_idx, threshold)
                executar_etapas_heuristicas(ex, sorteios, janela, False)
            st.rerun()
    else:
        col_g1,col_g2,col_g3 = st.columns(3)
        with col_g1:
            n_jogos = st.number_input("Quantos jogos?",1,20,5)
            perfil  = st.selectbox("Perfil",["Híbrida","Agressiva","Equilibrada"])
        with col_g2:
            seed      = st.number_input("Seed (0=aleatório)",0,99999,0)
            max_comum = st.slider("Máx. em comum entre jogos",5,14,11)
        with col_g3:
            fixas_inp = st.text_input("Dezenas fixas (ex: 05 13 22)")
            n_cand    = st.slider("Candidatos por estratégia",20,150,60)

        fixas = []
        if fixas_inp.strip():
            try: fixas=[int(x) for x in fixas_inp.strip().split() if 1<=int(x)<=25]
            except: pass

        st.info(f"💰 {n_jogos} × R$3,50 = **R${n_jogos*3.5:.2f}**")

        if st.button("🎯 GERAR COMBINAÇÕES", type="primary", use_container_width=True):
            with st.spinner(f"Gerando {n_jogos} jogo(s)..."):
                try:
                    resultado = gerar_jogos_v5(
                        executor=ex, sorteios=sorteios, n_jogos=n_jogos,
                        fixas=fixas, perfil=perfil, seed=seed,
                        max_comum=max_comum, n_candidatos_por_estrategia=n_cand,
                    )
                    st.session_state["jogos_gerados"] = resultado
                except Exception as e:
                    st.error(f"Erro: {e}")
                    import traceback; st.code(traceback.format_exc())

        res = st.session_state.get("jogos_gerados")
        if res and isinstance(res,dict) and res.get("jogos_finais"):
            jogos_finais = res["jogos_finais"]
            vencidas_g   = res.get("vencidas",[])
            fortes_n12   = res.get("fortes_n12",[])

            st.success(f"✅ {len(jogos_finais)} jogo(s) | {res['n_candidatos']} candidatos")
            if vencidas_g: st.error(f"🔴 Vencidas: {' '.join(f'{d:02d}' for d in vencidas_g)}")

            st.subheader("📋 Comparação")
            st.dataframe(comparar_jogos(jogos_finais), hide_index=True, use_container_width=True)

            for i,av in enumerate(jogos_finais):
                jogo = av["jogo"]
                with st.expander(f"🃏 J{i+1} — Score {av['score_total']} | Soma {av['soma']} | 11+:{av['pct_11']}%",
                                  expanded=(i==0)):
                    html = "".join(f'<span class="dez-box {"dez-venc" if d in vencidas_g else "dez-hot" if d in fortes_n12 else ""}">{d:02d}</span>' for d in jogo)
                    st.markdown(html, unsafe_allow_html=True)
                    st.markdown("---")
                    c1,c2,c3,c4,c5,c6 = st.columns(6)
                    c1.metric("Score",av["score_total"]); c2.metric("Soma",av["soma"])
                    c3.metric("Pares",av["pares"]);       c4.metric("Seq max",av["max_seq"])
                    c5.metric("11+%",f"{av['pct_11']}%"); c6.metric("13+%",f"{av['pct_13']}%")
                    st.write(f"**Faixas:** {av['faixas']} | **Amplitude:** {av['amplitude']}")
                    st.write(f"**Vencidas:** {av['cob_vencidas']}/{len(vencidas_g)} | **N+1/N+2 fortes:** {av['cob_fortes_n12']}")
                    st.subheader("🔬 Breakdown")
                    st.dataframe(breakdown_jogo(av,ex), hide_index=True, use_container_width=True)

            if len(jogos_finais)>1:
                st.subheader("📐 Hamming")
                st.dataframe(pd.DataFrame(res["hamming"]), hide_index=True, use_container_width=True)

            with st.expander("📄 Relatório"):
                st.text(res["relatorio"])

# ── ABA 6: RELATÓRIO ──────────────────────────────────────────────────────────
with tabs[5]:
    st.header("📋 Relatório Final")
    conc,total = ex.progresso(False)
    st.progress(conc/total if total else 0)
    st.markdown(f"**{conc}/{total} etapas** | **{n_conc} concursos** | Janela: **{janela}**")

    col_r1,col_r2 = st.columns(2)
    with col_r1:
        st.subheader("🌡️ Térmica")
        t = get_termica(ex)
        if t:
            st.write(f"**Top 10:** {' '.join(f'{d:02d}' for d in t['ranking'][:10])}")
            st.write(f"**Candidatas:** {' '.join(f'{d:02d}' for d in t['candidatas'])}")
        else: st.info("Não executada.")
    with col_r2:
        st.subheader("⏱️ Atraso")
        at = get_atraso(ex)
        if at:
            if at.get("vencidas"):  st.error(f"🔴 {' '.join(f'{d:02d}' for d in at['vencidas'])}")
            if at.get("atrasadas"): st.warning(f"🟠 {' '.join(f'{d:02d}' for d in at['atrasadas'])}")
            if at.get("no_ciclo"):  st.info(f"🟡 {' '.join(f'{d:02d}' for d in at['no_ciclo'])}")
        else: st.info("Não executada.")

    sc = ex.score_consolidado()
    if any(v>0 for v in sc.values()):
        st.subheader("🏆 Score Final")
        ranking = sorted(DEZENAS, key=lambda d: -sc.get(d,0))
        col_sc1,col_sc2 = st.columns(2)
        with col_sc1:
            at = get_atraso(ex)
            n12 = ex.ctx["resultados"].get("n1_n2",{})
            for i,d in enumerate(ranking[:13]):
                tags = ""
                if at and d in at.get("vencidas",[]): tags+=" 🔴"
                if at and d in at.get("atrasadas",[]): tags+=" 🟠"
                if d in n12.get("fortes",[]): tags+=" 🔥"
                st.write(f"#{i+1} **{d:02d}** — {round(sc[d],4)}{tags}")
        with col_sc2:
            st.bar_chart(pd.DataFrame({"Score":{d:sc[d] for d in ranking[:15]}}), color="#00FFCC")

    res = st.session_state.get("jogos_gerados")
    if res and isinstance(res,dict) and res.get("jogos_finais"):
        st.subheader("🎰 Jogos Gerados")
        for i,av in enumerate(res["jogos_finais"]):
            erros = filtrar_jogo(av["jogo"], calibrar_filtros(sorteios))
            st.write(f"**J{i+1}:** {' '.join(f'{d:02d}' for d in av['jogo'])} | Score:{av['score_total']} | Soma:{av['soma']} | 11+:{av['pct_11']}% {'✅' if not erros else '⚠️'}")

    st.subheader("📊 Etapas por Grupo")
    grupos = {}
    for step in STEP_REGISTRY:
        g = step["grupo"]
        if g not in grupos: grupos[g]={"total":0,"concluidas":0}
        grupos[g]["total"]+=1
        if ex.concluido(step["id"]): grupos[g]["concluidas"]+=1
    st.dataframe(pd.DataFrame([{"Grupo":g,"Concluídas":v["concluidas"],"Total":v["total"],
                                  "Status":"✅" if v["concluidas"]==v["total"] else f"{v['concluidas']}/{v['total']}"}
                                 for g,v in grupos.items()]),
                 hide_index=True, use_container_width=True)

# ── ABA 7: BACKTESTING ────────────────────────────────────────────────────────
with tabs[6]:
    st.header("🔬 Backtesting Manual")
    if not jogos_bt:
        st.info("Insira até 2 jogos no menu lateral.")
    else:
        cfg_bt = calibrar_filtros(sorteios)
        for i,jogo in enumerate(jogos_bt):
            st.subheader(f"Jogo {i+1}: {' · '.join(f'{d:02d}' for d in jogo)}")
            acertos = [len(set(jogo)&set(s)) for s in sorteios[-200:]]
            dist = {}
            for a in acertos: dist[a]=dist.get(a,0)+1
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Média",round(np.mean(acertos),2))
            c2.metric("11+%",round(sum(1 for a in acertos if a>=11)/len(acertos)*100,1))
            c3.metric("13+%",round(sum(1 for a in acertos if a>=13)/len(acertos)*100,1))
            c4.metric("Máx",max(acertos))
            st.bar_chart(pd.DataFrame(sorted(dist.items()),columns=["Acertos","Freq"]).set_index("Acertos"))
            erros = filtrar_jogo(jogo,cfg_bt)
            if erros:
                for e in erros: st.warning(f"⚠️ {e}")
            else:
                st.success("✅ Passou em todos os filtros!")
            st.write(f"**Soma:** {sum(jogo)} | **Pares:** {sum(1 for n in jogo if n%2==0)} | **Primos:** {len(set(jogo)&PRIMOS)}")
            sc = ex.score_consolidado()
            if any(v>0 for v in sc.values()):
                st.metric("Score V5", round(sum(sc.get(d,0) for d in jogo)/15,4))
            st.markdown("---")
