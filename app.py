import streamlit as st
import pandas as pd
import numpy as np
import time

# =============================================================================
# MOTOR DE ENGENHARIA V5 (CORRIGIDO)
# =============================================================================

class ProtocoloV5:
    def __init__(self, df):
        # Converte para inteiro e limpa índices
        self.df = df.astype(int)
        self.dezenas = list(range(1, 26))
        self.context = {
            "termica": {},
            "atraso": {},
            "scores_etapas": pd.DataFrame(index=self.dezenas)
        }

    def executar_termica(self):
        scores = {}
        # Janelas para análise de temperatura
        df_recente = self.df.head(10)
        df_antigo = self.df.iloc[10:30]
        
        for d in self.dezenas:
            f_atual = (df_recente == d).any(axis=1).sum() / 10
            f_base = (df_antigo == d).any(axis=1).sum() / 20
            tendencia = f_atual - f_base
            pressao = 1.0 + (f_atual * 2.0)
            
            # Definição visual por status
            if f_atual >= 0.7: 
                status, cor = "QUENTE", "#FF4B4B"
            elif f_atual <= 0.3: 
                status, cor = "FRIA", "#00BFFF"
            else:
                status, cor = "MORNA", "#FFA500"
            
            self.context["termica"][d] = {"status": status, "cor": cor, "pressao": pressao}
            scores[d] = max(0.01, (f_atual * 0.5) + (tendencia * 0.5) + 0.1)
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
            ciclo = np.mean(np.diff(indices)) if len(indices) > 1 else 3.0
            razao = atraso / ciclo
            
            # Lógica de peso Protocolo V5
            if razao < 1.0: p = 0.8
            elif razao < 1.35: p = 1.5
            elif razao < 2.0: p = 2.2 # Pico de probabilidade por atraso
            else: p = 0.4
            
            pressao_t = self.context["termica"].get(d, {}).get("pressao", 1.0)
            scores[d] = max(0.01, p * pressao_t)
            self.context["atraso"][d] = {"atraso": atraso, "razao": razao}
        return pd.Series(scores)

    def gerar_candidatos_v5(self):
        # Consolida e limpa scores para evitar ValueError no sorteio
        final_s = self.context["scores_etapas"].sum(axis=1)
        # Proteção: impede que valores negativos ou zerados quebrem o p do np.random.choice
        final_s = final_s.apply(lambda x: max(0.001, x)) 
        prob = final_s / final_s.sum()
        
        candidatos = []
        # Camada A: Geração Massiva
        for _ in range(1000):
            jogo = np.random.choice(self.dezenas, 15, replace=False, p=prob.values)
            jogo = sorted([int(x) for x in jogo])
            
            # Camada B: Filtros Estruturais
            pares = len([d for d in jogo if d % 2 == 0])
            soma = sum(jogo)
            score_jogo = sum([final_s[d] for d in jogo])
            
            if 7 <= pares <= 9: score_jogo *= 1.4
            if 180 <= soma <= 215: score_jogo *= 1.2
            
            candidatos.append({"jogo": jogo, "score": score_jogo, "pares": pares, "soma": soma})
            
        return sorted(candidatos, key=lambda x: x['score'], reverse=True)[:10]

# =============================================================================
# INTERFACE DASHBOARD (STYLING)
# =============================================================================

def main():
    st.set_page_config(page_title="V5 Protocol", layout="wide")
    
    # CSS para os blocos de dezenas e cards
    st.markdown("""
        <style>
        .dezena-box { display: inline-block; padding: 6px 12px; margin: 3px; border-radius: 8px; background: #1E1E1E; border: 2px solid #464b5d; font-family: 'Courier New', monospace; font-weight: bold; color: #00FFCC; font-size: 18px; }
        .card-auditoria { background: #111111; padding: 15px; border-radius: 12px; border-top: 4px solid #444; margin-bottom: 10px; text-align: center; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Protocolo Lotofácil V5")
    st.caption("Refatoração Estrutural | Motor em Cascata | Layer B Analysis")

    # Carregar base ou gerar mock
    try:
        df = pd.read_csv("base_lotofacil.csv")
        # Tenta detectar colunas de dezenas (ignora data/id se houver)
        if len(df.columns) > 15:
            df = df.iloc[:, -15:] # Assume as últimas 15
        else:
            df = df.iloc[:, :15]
    except:
        df = pd.DataFrame(np.random.randint(1, 26, size=(100, 15)))

    p = ProtocoloV5(df)

    with st.sidebar:
        st.header("🎮 Central de Comando")
        if st.button("🔥 EXECUTAR PROTOCOLO INTEGRAL", use_container_width=True):
            with st.spinner("Analisando Camadas..."):
                p.context["scores_etapas"]["Termica"] = p.executar_termica()
                p.context["scores_etapas"]["Atraso"] = p.executar_atraso()
                st.session_state["p_v5"] = p
        
        if "p_v5" in st.session_state:
            st.success("Protocolo V5 Ativo")
            if st.button("Limpar Cache"):
                del st.session_state["p_v5"]
                st.rerun()

    if "p_v5" in st.session_state:
        obj = st.session_state["p_v5"]
        
        t1, t2, t3 = st.tabs(["📈 TENDÊNCIAS", "🧬 AUDITORIA TÉRMICA", "💎 JOGOS DE ELITE"])
        
        with t1:
            st.subheader("Força Consolidada do Protocolo")
            resumo = obj.context["scores_etapas"].sum(axis=1).sort_values(ascending=False)
            st.bar_chart(resumo, color="#00FFCC")

        with t2:
            st.subheader("Leitura Individual por Dezena")
            cols = st.columns(5)
            for i, d in enumerate(range(1, 26)):
                with cols[i % 5]:
                    t = obj.context["termica"][d]
                    a = obj.context["atraso"][d]
                    st.markdown(f"""
                    <div class="card-auditoria" style="border-top-color: {t['cor']}">
                        <span style="font-size: 20px; color: white;">Dezena <b>{d:02d}</b></span><br>
                        <span style="color: {t['cor']}; font-weight: bold;">{t['status']}</span><br>
                        <small>Atraso: {a['atraso']} | Razão: {a['razao']:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)

        with t3:
            st.subheader("Sugestões Filtradas (Layer B)")
            jogos = obj.gerar_candidatos_v5()
            for idx, j in enumerate(jogos):
                with st.expander(f"⭐ OPÇÃO #{idx+1} | Score: {j['score']:.2f}"):
                    # Dezenas limpas e estilizadas
                    dezenas_html = "".join([f'<span class="dezena-box">{n:02d}</span>' for n in j['jogo']])
                    st.markdown(dezenas_html, unsafe_allow_html=True)
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pares", j['pares'])
                    c2.metric("Soma", j['soma'])
                    c3.write(f"**Justificativa:** Selecionado via critério de pressão térmica e equilíbrio de paridade.")

if __name__ == "__main__":
    main()
