# =============================================================================
# etapa_termica.py — Protocolo Lotofácil V5
# Etapa Térmica Completa — base central do pipeline
# =============================================================================
# Produz:
#   - frequência absoluta e relativa por 6 janelas
#   - rolling frequency
#   - tendência curta, média e longa
#   - comparação entre janelas
#   - desvio em relação ao esperado
#   - classificação térmica: QUENTE / MORNA / FRIA
#   - pressão de retorno
#   - estabilidade térmica
#   - aceleração / desaceleração
#   - temperatura por faixa e por quadrante
#   - score térmico por dezena
#   - ranking térmico
#   - candidatas térmicas
#   - bloqueios térmicos
# =============================================================================

import numpy as np
import pandas as pd
from collections import Counter
from engine_core import (
    DEZENAS, normalizar, normalizar_serie, PipelineExecutor
)

# ── Janelas de análise ────────────────────────────────────────────────────────
JANELAS = [10, 20, 50, 100, 200, 500]
ESP_FREQ = 15 / 25  # frequência esperada por concurso = 0.60

# ── Limites de classificação térmica ─────────────────────────────────────────
LIMITES_TERMICA = {
    "quente_forte":  0.75,   # freq relativa >= 75%
    "quente":        0.65,
    "morna_alta":    0.58,
    "morna_baixa":   0.45,
    "fria":          0.35,
    "fria_forte":    0.20,
}

# ── Grade 5×5 para quadrantes ─────────────────────────────────────────────────
#   01 02 03 04 05
#   06 07 08 09 10
#   11 12 13 14 15
#   16 17 18 19 20
#   21 22 23 24 25
QUADRANTES = {
    "Q1 (canto sup-esq)": [1, 2, 6, 7],
    "Q2 (canto sup-dir)": [4, 5, 9, 10],
    "Q3 (canto inf-esq)": [16, 17, 21, 22],
    "Q4 (canto inf-dir)": [19, 20, 24, 25],
    "Centro":             [8, 12, 13, 14, 18],
    "Borda sup":          [1, 2, 3, 4, 5],
    "Borda inf":          [21, 22, 23, 24, 25],
    "Borda esq":          [1, 6, 11, 16, 21],
    "Borda dir":          [5, 10, 15, 20, 25],
    "Miolo":              [7, 8, 9, 12, 13, 14, 17, 18, 19],
}

FAIXAS = {
    "F1 (01-05)": list(range(1, 6)),
    "F2 (06-10)": list(range(6, 11)),
    "F3 (11-15)": list(range(11, 16)),
    "F4 (16-20)": list(range(16, 21)),
    "F5 (21-25)": list(range(21, 26)),
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def freq_por_janela(sorteios: list, dezena: int, janela: int) -> dict:
    """Frequência absoluta e relativa de uma dezena em uma janela."""
    dados = sorteios[-janela:]
    absoluta = sum(1 for s in dados if dezena in s)
    relativa = absoluta / len(dados) if dados else 0.0
    return {"absoluta": absoluta, "relativa": round(relativa, 4)}


def todas_freqs(sorteios: list) -> dict:
    """
    Para cada dezena, calcula frequência em todas as janelas.
    Retorna: {dezena: {janela: {"absoluta": int, "relativa": float}}}
    """
    result = {}
    for d in DEZENAS:
        result[d] = {}
        for j in JANELAS:
            j_eff = min(j, len(sorteios))
            result[d][j_eff] = freq_por_janela(sorteios, d, j_eff)
    return result


def rolling_frequency(sorteios: list, dezena: int, window: int = 20) -> list:
    """
    Frequência rolling: para cada posição i, calcula freq nos últimos `window` sorteios.
    Retorna lista de floats.
    """
    serie = []
    for i in range(window, len(sorteios) + 1):
        bloco = sorteios[i - window:i]
        freq = sum(1 for s in bloco if dezena in s) / window
        serie.append(round(freq, 4))
    return serie


def calcular_tendencias(freqs: dict, dezena: int) -> dict:
    """
    Calcula tendências comparando janelas curtas vs longas.
    Retorna: {curta, media, longa, aceleracao}
    """
    j = freqs[dezena]
    f10  = j.get(10,  {}).get("relativa", ESP_FREQ)
    f20  = j.get(20,  {}).get("relativa", ESP_FREQ)
    f50  = j.get(50,  {}).get("relativa", ESP_FREQ)
    f100 = j.get(100, {}).get("relativa", ESP_FREQ)
    f200 = j.get(200, {}).get("relativa", ESP_FREQ)
    f500 = j.get(500, {}).get("relativa", ESP_FREQ)

    tendencia_curta  = round(f10  - f50,  4)   # J10  vs J50
    tendencia_media  = round(f50  - f200, 4)   # J50  vs J200
    tendencia_longa  = round(f100 - f500, 4)   # J100 vs J500
    aceleracao       = round(tendencia_curta - tendencia_media, 4)

    return {
        "curta":       tendencia_curta,
        "media":       tendencia_media,
        "longa":       tendencia_longa,
        "aceleracao":  aceleracao,
        "f10":  f10,  "f20":  f20,  "f50":  f50,
        "f100": f100, "f200": f200, "f500": f500,
    }


def desvio_esperado(freq_relativa: float) -> dict:
    """Desvio em relação à frequência esperada (0.60)."""
    desvio = round(freq_relativa - ESP_FREQ, 4)
    desvio_pct = round(desvio / ESP_FREQ * 100, 2)
    return {"desvio_abs": desvio, "desvio_pct": desvio_pct}


def classificar_temperatura(f50: float, tendencia_curta: float, aceleracao: float) -> dict:
    """
    Classificação térmica real com 6 níveis.
    Usa frequência J50 como base + tendência curta como ajuste.
    """
    f_adj = f50 + tendencia_curta * 0.3  # ajuste pela tendência

    if f_adj >= LIMITES_TERMICA["quente_forte"]:
        status = "🔴 QUENTE FORTE"
        cor    = "#FF0000"
        nivel  = 6
    elif f_adj >= LIMITES_TERMICA["quente"]:
        status = "🟠 QUENTE"
        cor    = "#FF6600"
        nivel  = 5
    elif f_adj >= LIMITES_TERMICA["morna_alta"]:
        status = "🟡 MORNA ALTA"
        cor    = "#FFD700"
        nivel  = 4
    elif f_adj >= LIMITES_TERMICA["morna_baixa"]:
        status = "⚪ MORNA"
        cor    = "#AAAAAA"
        nivel  = 3
    elif f_adj >= LIMITES_TERMICA["fria"]:
        status = "🔵 FRIA"
        cor    = "#00BFFF"
        nivel  = 2
    else:
        status = "❄️ FRIA FORTE"
        cor    = "#0000FF"
        nivel  = 1

    # Ajuste por aceleração
    if aceleracao > 0.05:
        status += " ↑"
    elif aceleracao < -0.05:
        status += " ↓"

    return {"status": status, "cor": cor, "nivel": nivel, "f_ajustada": round(f_adj, 4)}


def pressao_retorno(sorteios: list, dezena: int) -> dict:
    """
    Pressão de retorno baseada no atraso atual vs ciclo médio.
    Quanto maior a razão atraso/ciclo, maior a pressão.
    """
    aparicoes = [i for i, s in enumerate(sorteios) if dezena in s]

    if len(aparicoes) < 2:
        return {"atraso": len(sorteios), "ciclo_medio": len(sorteios),
                "razao": 1.0, "pressao": 0.5, "status_pressao": "INDEFINIDO"}

    # Ciclo médio = média dos intervalos entre aparições
    intervalos = [aparicoes[i+1] - aparicoes[i] for i in range(len(aparicoes)-1)]
    ciclo_medio = round(np.mean(intervalos), 2)
    ciclo_std   = round(np.std(intervalos), 2)

    # Atraso atual = distância do último sorteio até o fim
    ultimo_idx = aparicoes[-1]
    atraso = len(sorteios) - 1 - ultimo_idx

    razao = round(atraso / ciclo_medio if ciclo_medio > 0 else 1.0, 3)

    # Pressão: função sigmoide centrada na razão
    pressao = round(1 / (1 + np.exp(-2 * (razao - 1))), 4)

    # Status de pressão
    if razao < 1.0:
        status_pressao = "🟢 OK"
    elif razao < 1.35:
        status_pressao = "🟡 NO CICLO"
    elif razao < 2.0:
        status_pressao = "🟠 ATRASADA"
    else:
        status_pressao = "🔴 VENCIDA"

    return {
        "atraso":        atraso,
        "ciclo_medio":   ciclo_medio,
        "ciclo_std":     ciclo_std,
        "razao":         razao,
        "pressao":       pressao,
        "status_pressao": status_pressao,
        "ultimo_idx":    ultimo_idx,
        "n_aparicoes":   len(aparicoes),
    }


def estabilidade_termica(rolling: list, janela_est: int = 20) -> dict:
    """
    Estabilidade = inverso do coeficiente de variação da frequência rolling.
    Alta estabilidade = dezena consistente.
    """
    if len(rolling) < janela_est:
        return {"estabilidade": 0.5, "cv": 0.5, "variancia": 0.5}

    serie = rolling[-janela_est:]
    media = np.mean(serie)
    std   = np.std(serie)
    cv    = round(std / media if media > 0 else 1.0, 4)
    estab = round(max(0.0, 1.0 - cv), 4)

    return {
        "estabilidade": estab,
        "cv":           cv,
        "variancia":    round(float(np.var(serie)), 4),
        "media_rolling": round(float(media), 4),
    }


def temperatura_por_faixa(sorteios: list, janela: int = 100) -> dict:
    """
    Para cada faixa (F1-F5), calcula temperatura média das dezenas.
    """
    result = {}
    dados = sorteios[-janela:]
    for nome_faixa, dezenas_faixa in FAIXAS.items():
        freqs = [sum(1 for s in dados if d in s) / len(dados) for d in dezenas_faixa]
        result[nome_faixa] = {
            "media_freq": round(np.mean(freqs), 4),
            "max_freq":   round(max(freqs), 4),
            "min_freq":   round(min(freqs), 4),
            "dezenas":    dezenas_faixa,
        }
    return result


def temperatura_por_quadrante(sorteios: list, janela: int = 100) -> dict:
    """
    Para cada quadrante da grade 5×5, calcula temperatura média.
    """
    result = {}
    dados = sorteios[-janela:]
    for nome_q, dezenas_q in QUADRANTES.items():
        freqs = [sum(1 for s in dados if d in s) / len(dados) for d in dezenas_q]
        result[nome_q] = {
            "media_freq": round(np.mean(freqs), 4),
            "dezenas":    dezenas_q,
        }
    return result


def score_termico_dezena(
    freqs: dict,
    tendencias: dict,
    pressao: dict,
    estabilidade: dict,
    dezena: int,
) -> float:
    """
    Score térmico combinado para uma dezena.
    Componentes:
        40% — frequência relativa J50 (temperatura base)
        20% — tendência curta (momentum)
        20% — pressão de retorno
        10% — estabilidade
        10% — aceleração positiva
    """
    f50 = freqs[dezena].get(50, {}).get("relativa", ESP_FREQ)

    # Normaliza tendência curta para [0,1]: range esperado [-0.4, 0.4]
    tend_norm = (tendencias[dezena]["curta"] + 0.4) / 0.8
    tend_norm = max(0.0, min(1.0, tend_norm))

    pressao_val = pressao[dezena]["pressao"]
    estab_val   = estabilidade[dezena]["estabilidade"]

    # Aceleração positiva = bônus
    acel = tendencias[dezena]["aceleracao"]
    acel_norm = (acel + 0.3) / 0.6
    acel_norm = max(0.0, min(1.0, acel_norm))

    score = (
        f50         * 0.40 +
        tend_norm   * 0.20 +
        pressao_val * 0.20 +
        estab_val   * 0.10 +
        acel_norm   * 0.10
    )
    return round(score, 6)


# =============================================================================
# RUNNER PRINCIPAL DA ETAPA TÉRMICA
# =============================================================================

def run_termica(sorteios: list, janela_ref: int = 500) -> dict:
    """
    Executa a etapa térmica completa.
    Retorna dicionário com todos os resultados estruturados.

    Parâmetros:
        sorteios    : lista de listas com os sorteios históricos
        janela_ref  : janela principal de referência (default 500)
    """
    n = len(sorteios)

    # ── 1. Frequências por todas as janelas ───────────────────────────────────
    freqs = todas_freqs(sorteios)

    # ── 2. Rolling frequency (janela 20) ─────────────────────────────────────
    rollings = {d: rolling_frequency(sorteios, d, window=20) for d in DEZENAS}

    # ── 3. Tendências ─────────────────────────────────────────────────────────
    tendencias = {d: calcular_tendencias(freqs, d) for d in DEZENAS}

    # ── 4. Desvios em relação ao esperado ────────────────────────────────────
    desvios = {}
    for d in DEZENAS:
        f50 = freqs[d].get(50, {}).get("relativa", ESP_FREQ)
        desvios[d] = desvio_esperado(f50)

    # ── 5. Pressão de retorno ─────────────────────────────────────────────────
    pressoes = {d: pressao_retorno(sorteios, d) for d in DEZENAS}

    # ── 6. Estabilidade térmica ───────────────────────────────────────────────
    estabilidades = {d: estabilidade_termica(rollings[d]) for d in DEZENAS}

    # ── 7. Classificação térmica ──────────────────────────────────────────────
    classificacoes = {}
    for d in DEZENAS:
        f50  = freqs[d].get(50, {}).get("relativa", ESP_FREQ)
        tend = tendencias[d]["curta"]
        acel = tendencias[d]["aceleracao"]
        classificacoes[d] = classificar_temperatura(f50, tend, acel)

    # ── 8. Temperatura por faixa e quadrante ─────────────────────────────────
    temp_faixas     = temperatura_por_faixa(sorteios, janela=100)
    temp_quadrantes = temperatura_por_quadrante(sorteios, janela=100)

    # ── 9. Score térmico por dezena ───────────────────────────────────────────
    scores = {}
    for d in DEZENAS:
        scores[d] = score_termico_dezena(freqs, tendencias, pressoes, estabilidades, d)

    scores_norm = normalizar(scores)

    # ── 10. Ranking térmico ───────────────────────────────────────────────────
    ranking = sorted(DEZENAS, key=lambda d: -scores_norm[d])

    # ── 11. Candidatas térmicas (top 15 por score) ────────────────────────────
    candidatas = ranking[:15]

    # ── 12. Bloqueios térmicos (bottom 5 + frias fortes) ─────────────────────
    bloqueios_score = ranking[-5:]
    bloqueios_temp  = [d for d in DEZENAS
                       if classificacoes[d]["nivel"] <= 1]
    bloqueios = list(set(bloqueios_score + bloqueios_temp))

    # ── 13. Dezenas por status de pressão ─────────────────────────────────────
    vencidas  = [d for d in DEZENAS if pressoes[d]["status_pressao"] == "🔴 VENCIDA"]
    atrasadas = [d for d in DEZENAS if pressoes[d]["status_pressao"] == "🟠 ATRASADA"]
    no_ciclo  = [d for d in DEZENAS if pressoes[d]["status_pressao"] == "🟡 NO CICLO"]

    # ── 14. Tabela-resumo para UI ─────────────────────────────────────────────
    tabela = []
    for d in DEZENAS:
        f10  = freqs[d].get(10,  {}).get("relativa", 0)
        f50  = freqs[d].get(50,  {}).get("relativa", 0)
        f200 = freqs[d].get(200, {}).get("relativa", 0)
        tabela.append({
            "Dezena":       d,
            "J10%":         round(f10  * 100, 1),
            "J50%":         round(f50  * 100, 1),
            "J200%":        round(f200 * 100, 1),
            "Tendência":    tendencias[d]["curta"],
            "Aceleração":   tendencias[d]["aceleracao"],
            "Status":       classificacoes[d]["status"],
            "Pressão":      pressoes[d]["status_pressao"],
            "Razão A/C":    pressoes[d]["razao"],
            "Estabilidade": estabilidades[d]["estabilidade"],
            "Score":        round(scores_norm[d], 4),
            "Rank":         ranking.index(d) + 1,
        })
    df_tabela = pd.DataFrame(tabela).sort_values("Score", ascending=False).reset_index(drop=True)

    return {
        # Dados brutos
        "freqs":           freqs,
        "rollings":        rollings,
        "tendencias":      tendencias,
        "desvios":         desvios,
        "pressoes":        pressoes,
        "estabilidades":   estabilidades,
        "classificacoes":  classificacoes,

        # Temperatura por dimensão espacial
        "temp_faixas":     temp_faixas,
        "temp_quadrantes": temp_quadrantes,

        # Scores
        "scores_raw":      scores,
        "scores_norm":     scores_norm,

        # Outputs principais
        "ranking":         ranking,
        "candidatas":      candidatas,
        "bloqueios":       bloqueios,
        "vencidas":        vencidas,
        "atrasadas":       atrasadas,
        "no_ciclo":        no_ciclo,

        # UI
        "df_tabela":       df_tabela,

        # Meta
        "n_sorteios":      n,
        "janela_ref":      janela_ref,
        "esp_freq":        ESP_FREQ,
    }


# =============================================================================
# SCORER — produz score_dezena para o engine_core
# =============================================================================

def scorer_termica(resultado: dict) -> dict:
    """Retorna scores normalizados por dezena para uso no score consolidado."""
    return resultado["scores_norm"]


# =============================================================================
# INTEGRAÇÃO COM PIPELINE EXECUTOR
# =============================================================================

def executar_etapa_termica(
    executor: PipelineExecutor,
    sorteios: list,
    janela_ref: int = 500,
) -> dict:
    """
    Executa a etapa térmica e salva no contexto do executor.
    Chamada pelo app.py ou pelo executor automático.
    """
    step_id = "frequencia_simples"  # primeiro passo da térmica

    try:
        resultado = run_termica(sorteios, janela_ref)
        scores_dezena = scorer_termica(resultado)

        # Salva frequencia_simples
        executor.salvar_resultado(
            "frequencia_simples",
            {"freqs_j500": {d: resultado["freqs"][d].get(500, {}) for d in DEZENAS}},
            scores_dezena=scores_dezena,
        )

        # Salva frequencia_janelas
        executor.salvar_resultado(
            "frequencia_janelas",
            {"freqs_todas": resultado["freqs"],
             "rollings": resultado["rollings"],
             "tendencias": resultado["tendencias"]},
            scores_dezena=scores_dezena,
        )

        # Salva resultado completo da térmica no contexto
        executor.ctx["resultados"]["_termica_completa"] = resultado

        executor._log("[TERMICA] Etapa térmica completa executada com sucesso.")
        return resultado

    except Exception as e:
        executor.marcar_erro("frequencia_simples", str(e))
        executor.marcar_erro("frequencia_janelas", str(e))
        raise


# =============================================================================
# HELPERS PARA USO NAS ETAPAS SEGUINTES
# =============================================================================

def get_termica(executor: PipelineExecutor) -> dict:
    """Recupera resultado completo da térmica do contexto."""
    return executor.ctx["resultados"].get("_termica_completa", {})


def get_candidatas_termicas(executor: PipelineExecutor) -> list:
    """Retorna dezenas candidatas da etapa térmica."""
    t = get_termica(executor)
    return t.get("candidatas", list(range(1, 26)))


def get_pressoes(executor: PipelineExecutor) -> dict:
    """Retorna dict de pressão de retorno por dezena."""
    t = get_termica(executor)
    return t.get("pressoes", {})


def get_vencidas(executor: PipelineExecutor) -> list:
    """Retorna dezenas com status VENCIDA."""
    t = get_termica(executor)
    return t.get("vencidas", [])


def get_scores_termicos(executor: PipelineExecutor) -> dict:
    """Retorna scores térmicos normalizados."""
    t = get_termica(executor)
    return t.get("scores_norm", {n: 0.5 for n in DEZENAS})


def get_ranking_termico(executor: PipelineExecutor) -> list:
    """Retorna ranking de dezenas por score térmico."""
    t = get_termica(executor)
    return t.get("ranking", list(range(1, 26)))
