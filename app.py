import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import time

# =============================================================================
# MOTOR DE ENGENHARIA V5
# =============================================================================

class ProtocoloV5:
    def __init__(self, df):
        # Converte para inteiro puro para evitar o erro do np.int64 na tela
        self.df = df.astype(int)
        self.dezenas = list(range(1, 26))
        self.context = {
            "termica": {},
            "atraso": {},
            "scores_etapas": pd.DataFrame(index=self.dezenas)
        }

    def executar_termica(self):
        scores = {}
        df_recente = self.df.head(10)
        df_antigo = self.df.iloc[10:30]
        for d in self.dezenas:
            f_atual = (df_recente == d).any(axis=1).sum() / 10
            f_base = (df_antigo == d).any(axis=1).sum() / 20
            tendencia = f_atual - f_base
            pressao = 1.0 + (f_atual * 2.0)
            
            status = "MORNA"
            cor = "#FFA500" # Laranja
            if f_atual >= 0.7: 
                status, cor = "QUENTE", "#FF4B4B" # Vermelho
            elif f_atual <= 0.3: 
                status, cor = "FRIA", "#00BFFF" # Azul
            
            self.context["termica"][d] = {"status": status, "cor": cor, "pressao": pressao}
            scores[d] = (f_atual * 0.5) + (tendencia * 0.5)
        return pd.Series(scores)

    def executar_atraso(self):
        scores = {}
        for d in self.dezenas:
            atraso = 0
            for _, row in self.df.iterrows():
                if d in row.values: break
                atraso += 1
            presenca = self.df.apply(lambda x: d in x.values, axis=1)
            indices = presenca[presenca].index
            ciclo = np.mean(np.diff(indices)) if len(indices) > 1 else 2.5
            razao = atraso / ciclo
            
            p = 0.7
            if 1.0 <= razao < 1.35: p = 1.4
            elif 1.35 <= razao < 2.0: p = 2.0
            elif razao >= 2.0: p = 0.4
            
            # Cascata: Térmica influencia o Atraso
            pressao_t = self.context["termica"].get(d, {}).get("pressao", 1.0)
            scores[d] = p * pressao_t
            self.context["atraso"][d] = {"atraso": atraso, "razao": razao}
        return pd.Series(scores)

    def gerar_candidatos_v5(self):
        final_s = self.context["scores_etapas"].sum(axis=1)
        # Normaliza para probabilidade
        prob = (final_s - final_s.min()) / (final_s.max() - final_s.min() + 1e-9)
        prob = prob / prob.sum()
        
        candidatos = []
        for _ in range(1000):
            # Gera combinação e converte para lista de inteiros normais (Python int)
            jogo = np.random.choice(self.dezenas, 15, replace=False, p=prob)
            jogo = sorted([int(x) for x in jogo]) 
            
            pares = len([d for d in jogo if d % 2 == 0])
            soma = sum(jogo)
            score_jogo = sum([final_s[d] for d in jogo])
            
            if 7 <= pares <= 9: score_jogo *= 1.3
            if 180 <= soma <= 210: score_jogo *= 1.1
            
            candidatos.append({"jogo": jogo, "score": score_jogo, "pares": pares, "soma": soma})
        return sorted(candidatos, key=lambda x: x['score'], reverse=True)[:10]

# =============================================================================
# INTERFACE VISUAL (DASHBOARD V5)
# =============================================================================

def main():
    st.set_page_config(page_title="V5 Protocol", layout="wide")
    
    # CSS para deixar as "coisinhas" bonitas
    st.markdown("""
        <style>
        .dezena-box { display: inline-block; padding: 5px 10px; margin: 2px; border-radius: 5px; background: #262730; border: 1px solid #464b5d; font-weight: bold; color: #00FF00; }
        .metric-card { background: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Protocolo Lotofácil V5")
    st.caption("Refatoração Estrutural | Motor em Cascata | Layer B Filter")

    try:
        df = pd.read_csv("base_lotofacil.csv").iloc[:, :15]
    except:
        df = pd.DataFrame(np.random.randint(1, 26, size=(100, 15)))

    p = ProtocoloV5(df)

    # Sidebar com cara de "Master Control"
    with st.sidebar:
        st.header("🎮 Comandos")
        if st.button("🔥 EXECUTAR PROTOCOLO INTEGRAL", use_container_width=True):
            with st.spinner("Processando Camadas..."):
                p.context["scores_etapas"]["Termica"] = p.executar_termica()
                p.context["scores_etapas"]["Atraso"] = p.executar_atraso()
                st.session_state["p_v5"] = p
        
        st.divider()
        st.markdown("**Status do Sistema:**")
        if "p_v5" in st.session_state:
            st.success("Protocolo Ativo")
        else:
            st.warning("Aguardando Execução")

    if "p_v5" in st.session_state:
        obj = st.session_state["p_v5"]
        
        t1, t2, t3 = st.tabs(["📈 PERFORMANCE", "🧬 AUDITORIA GENÉTICA", "💎 JOGOS DE ELITE"])
        
        with t1:
            st.subheader("Ranking de Força das Dezenas")
            resumo = obj.context["scores_etapas"].sum(axis=1).sort_values(ascending=False)
            st.bar_chart(resumo, color="#FF4B4B")

        with t2:
            st.subheader("Leitura Térmica e de Atraso (Real-time)")
            cols = st.columns(5)
            for i, d in enumerate(range(1, 26)):
                with cols[i % 5]:
                    t_info = obj.context["termica"][d]
                    st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {t_info['cor']}">
                        <b>Dezena {d:02d}</b><br>
                        <small>{t_info['status']}</small><br>
                        <small>Razão: {obj.context['atraso'][d]['razao']:.2f}</small>
                    </div><br>
                    """, unsafe_allow_html=True)

        with t3:
            st.subheader("Seleção Layer B (Melhores entre 1000 candidatos)")
            jogos = obj.gerar_candidatos_v5()
            for idx, item in enumerate(jogos):
                with st.expander(f"⭐ CANDIDATO #{idx+1} | Pontuação: {item['score']:.2f}"):
                    # Renderiza dezenas como "coisinhas" bonitas
                    html_dezenas = "".join([f'<span class="dezena-box">{d:02d}</span>' for d in item['jogo']])
                    st.markdown(html_dezenas, unsafe_allow_html=True)
                    st.markdown(f"---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pares", item['pares'])
                    c2.metric("Soma", item['soma'])
                    c3.metric("Aderência", f"{idx+1}º Lugar")

if __name__ == "__main__":
    main()
