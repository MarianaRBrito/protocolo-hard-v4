# =============================================================================
# etapa_atraso.py — Protocolo Lotofácil V5
# Etapa de Atraso & Ciclo — usa térmica como insumo
# =============================================================================
# Produz:
#   - atraso atual por dezena
#   - ciclo médio individual por dezena
#   - razão atraso/ciclo
#   - status individual (baseado no comportamento da própria dezena)
#   - status global (baseado em percentis históricos reais)
#   - pressão de retorno integrada com score térmico
#   - ciclo por posição (onde na grade 5x5 a dezena costuma aparecer)
#   - padrão de ausência (sequências de ausência típicas)
#   - atraso por faixa
#   - score de atraso por dezena
#   - integração com score térmico da etapa anterior
# =============================================================================

import numpy as np
import pandas as pd
from collections import Counter
from engine_core import (
    DEZENAS, normalizar, normalizar_serie, PipelineExecutor
)
from etapa_termica import get_termica, get_scores_termicos, get_pressoes

# ── Thresholds parametrizáveis (razão atraso/ciclo) ──────────────────────────
THRESHOLDS = {
    "ok":       1.00,   # razão < 1.00  → OK
    "no_ciclo": 1.35,   # razão < 1.35  → NO CICLO
    "atrasada": 2.00,   # razão < 2.00  → ATRASADA
    # razão >= 2.00 → VENCIDA
}

# ── Percentis globais para status global ──────────────────────────────────────
PERCENTIS_GLOBAIS = {
    "p75": 75,
    "p85": 85,
    "p90": 90,
    "p95": 95,
}


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calcular_atraso_atual(sorteios: list, dezena: int) -> int:
    """Atraso atual: quantos concursos desde a última aparição."""
    for i in range(len(sorteios) - 1, -1, -1):
        if dezena in sorteios[i]:
            return len(sorteios) - 1 - i
    return len(sorteios)  # nunca apareceu


def calcular_aparicoes(sorteios: list, dezena: int) -> list:
    """Índices de todos os concursos em que a dezena apareceu."""
    return [i for i, s in enumerate(sorteios) if dezena in s]


def calcular_intervalos(aparicoes: list) -> list:
    """Intervalos entre aparições consecutivas."""
    if len(aparicoes) < 2:
        return []
    return [aparicoes[i+1] - aparicoes[i] for i in range(len(aparicoes) - 1)]


def ciclo_stats(intervalos: list) -> dict:
    """Estatísticas do ciclo de uma dezena."""
    if not intervalos:
        return {
            "ciclo_medio":   3.0,
            "ciclo_mediana": 3.0,
            "ciclo_std":     0.0,
            "ciclo_min":     1,
            "ciclo_max":     10,
            "ciclo_p75":     4.0,
            "ciclo_p90":     6.0,
            "n_intervalos":  0,
        }
    arr = np.array(intervalos)
    return {
        "ciclo_medio":   round(float(np.mean(arr)), 3),
        "ciclo_mediana": round(float(np.median(arr)), 3),
        "ciclo_std":     round(float(np.std(arr)), 3),
        "ciclo_min":     int(np.min(arr)),
        "ciclo_max":     int(np.max(arr)),
        "ciclo_p75":     round(float(np.percentile(arr, 75)), 3),
        "ciclo_p90":     round(float(np.percentile(arr, 90)), 3),
        "n_intervalos":  len(intervalos),
    }


def status_individual(razao: float) -> dict:
    """
    Status baseado no comportamento individual da dezena.
    Usa razão atraso/ciclo_medio da própria dezena.
    """
    if razao < THRESHOLDS["ok"]:
        status = "🟢 OK"
        prioridade = 1
        peso = 0.8
    elif razao < THRESHOLDS["no_ciclo"]:
        status = "🟡 NO CICLO"
        prioridade = 2
        peso = 1.5
    elif razao < THRESHOLDS["atrasada"]:
        status = "🟠 ATRASADA"
        prioridade = 3
        peso = 2.2
    else:
        status = "🔴 VENCIDA"
        prioridade = 4
        peso = 3.0

    return {"status": status, "prioridade": prioridade, "peso": peso}


def calcular_percentis_historicos(sorteios: list) -> dict:
    """
    Percentis históricos reais de todos os intervalos entre aparições.
    Sem truncar com int() — usa float real.
    """
    todos_intervalos = []
    for d in DEZENAS:
        ap = calcular_aparicoes(sorteios, d)
        todos_intervalos.extend(calcular_intervalos(ap))

    if not todos_intervalos:
        return {f"p{p}": float(p // 10) for p in PERCENTIS_GLOBAIS.values()}

    arr = np.array(todos_intervalos)
    return {
        f"p{p}": round(float(np.percentile(arr, p)), 3)
        for p in PERCENTIS_GLOBAIS.values()
    }


def status_global(atraso: int, percentis: dict) -> dict:
    """
    Status baseado nos percentis históricos globais.
    Complementar ao status individual.
    """
    if atraso >= percentis["p95"]:
        status = "🔴 VENCIDA (global)"
        nivel = 4
    elif atraso >= percentis["p85"]:
        status = "🟠 ATRASADA (global)"
        nivel = 3
    elif atraso >= percentis["p75"]:
        status = "🟡 NO CICLO (global)"
        nivel = 2
    else:
        status = "🟢 OK (global)"
        nivel = 1
    return {"status": status, "nivel": nivel}


def score_atraso_dezena(
    razao: float,
    status_ind: dict,
    pressao_termica: float,
    score_termico: float,
) -> float:
    """
    Score de atraso combinado com score térmico.

    Componentes:
        50% — peso do status individual (razão atraso/ciclo)
        30% — pressão térmica (da etapa térmica)
        20% — score térmico geral
    """
    peso_ind = status_ind["peso"]

    # Normaliza peso individual para [0,1]: range [0.8, 3.0]
    peso_norm = (peso_ind - 0.8) / (3.0 - 0.8)
    peso_norm = max(0.0, min(1.0, peso_norm))

    score = (
        peso_norm      * 0.50 +
        pressao_termica * 0.30 +
        score_termico  * 0.20
    )
    return round(score, 6)


def ciclo_por_posicao(sorteios: list, dezena: int, janela: int = 200) -> dict:
    """
    Analisa em qual posição (índice 0-14 do jogo ordenado) a dezena
    costuma aparecer nos últimos `janela` sorteios.
    Útil para entender padrões posicionais na grade.
    """
    dados = sorteios[-janela:]
    posicoes = []
    for s in dados:
        s_ord = sorted(s)
        if dezena in s_ord:
            posicoes.append(s_ord.index(dezena))

    if not posicoes:
        return {"posicao_media": 7.0, "posicao_mais_comum": 7, "frequencia_posicao": {}}

    cnt = Counter(posicoes)
    return {
        "posicao_media":      round(np.mean(posicoes), 2),
        "posicao_mais_comum": cnt.most_common(1)[0][0],
        "frequencia_posicao": dict(cnt),
        "n_aparicoes":        len(posicoes),
    }


def padrao_ausencia(intervalos: list) -> dict:
    """
    Analisa o padrão de ausência: sequências típicas de ausência.
    Detecta se a dezena tende a sumir por períodos longos ou curtos.
    """
    if len(intervalos) < 3:
        return {"tipo": "INDEFINIDO", "ausencia_tipica": 3, "ausencia_max_historico": 10}

    arr = np.array(intervalos)
    media = np.mean(arr)
    std   = np.std(arr)

    # Classifica padrão
    if std / media < 0.3:
        tipo = "REGULAR"       # ciclo muito consistente
    elif std / media < 0.6:
        tipo = "SEMI-REGULAR"  # alguma variação
    else:
        tipo = "IRREGULAR"     # muito variável

    # Ausência típica = percentil 75 dos intervalos
    ausencia_tipica = round(float(np.percentile(arr, 75)), 1)

    return {
        "tipo":                  tipo,
        "ausencia_tipica":       ausencia_tipica,
        "ausencia_max_historico": int(np.max(arr)),
        "coef_variacao":         round(float(std / media) if media > 0 else 0, 3),
    }


def atraso_por_faixa(sorteios: list, janela: int = 200) -> dict:
    """
    Para cada faixa (1-5, 6-10, 11-15, 16-20, 21-25),
    calcula o atraso médio das dezenas e identifica a faixa mais atrasada.
    """
    faixas = {
        "F1 (01-05)": list(range(1, 6)),
        "F2 (06-10)": list(range(6, 11)),
        "F3 (11-15)": list(range(11, 16)),
        "F4 (16-20)": list(range(16, 21)),
        "F5 (21-25)": list(range(21, 26)),
    }
    result = {}
    for nome, dezenas in faixas.items():
        atrasos = [calcular_atraso_atual(sorteios, d) for d in dezenas]
        result[nome] = {
            "atraso_medio": round(np.mean(atrasos), 2),
            "atraso_max":   max(atrasos),
            "dezena_mais_atrasada": dezenas[np.argmax(atrasos)],
            "dezenas": dezenas,
        }
    return result


# =============================================================================
# RUNNER PRINCIPAL DA ETAPA DE ATRASO
# =============================================================================

def run_atraso(sorteios: list, executor: PipelineExecutor) -> dict:
    """
    Executa a etapa de atraso & ciclo completa.
    Usa a étapa térmica como insumo (pressão e score térmico).

    Retorna dicionário com todos os resultados estruturados.
    """
    # ── Recupera dados da etapa térmica ──────────────────────────────────────
    scores_termicos = get_scores_termicos(executor)
    pressoes_termicas = get_pressoes(executor)

    # ── 1. Percentis históricos globais ───────────────────────────────────────
    percentis = calcular_percentis_historicos(sorteios)

    # ── 2. Análise individual por dezena ─────────────────────────────────────
    analise = {}
    for d in DEZENAS:
        aparicoes  = calcular_aparicoes(sorteios, d)
        intervalos = calcular_intervalos(aparicoes)
        atraso     = calcular_atraso_atual(sorteios, d)
        stats      = ciclo_stats(intervalos)
        ciclo_med  = stats["ciclo_medio"]

        razao      = round(atraso / ciclo_med if ciclo_med > 0 else 1.0, 3)
        razao      = min(razao, 5.0)  # cap em 5x o ciclo

        st_ind     = status_individual(razao)
        st_glob    = status_global(atraso, percentis)

        # Pressão térmica: vem da etapa térmica
        p_termica  = pressoes_termicas.get(d, {}).get("pressao", 0.5)
        sc_termico = scores_termicos.get(d, 0.5)

        # Ciclo por posição e padrão de ausência
        cp = ciclo_por_posicao(sorteios, d)
        pa = padrao_ausencia(intervalos)

        analise[d] = {
            "dezena":         d,
            "atraso":         atraso,
            "ciclo_medio":    ciclo_med,
            "ciclo_mediana":  stats["ciclo_mediana"],
            "ciclo_std":      stats["ciclo_std"],
            "ciclo_p75":      stats["ciclo_p75"],
            "ciclo_p90":      stats["ciclo_p90"],
            "razao":          razao,
            "status_ind":     st_ind["status"],
            "prioridade_ind": st_ind["prioridade"],
            "peso_ind":       st_ind["peso"],
            "status_glob":    st_glob["status"],
            "nivel_glob":     st_glob["nivel"],
            "pressao_termica": p_termica,
            "score_termico":  sc_termico,
            "n_aparicoes":    len(aparicoes),
            "ciclo_posicao":  cp,
            "padrao_ausencia": pa,
        }

    # ── 3. Score de atraso por dezena ─────────────────────────────────────────
    scores = {}
    for d in DEZENAS:
        a = analise[d]
        scores[d] = score_atraso_dezena(
            razao=a["razao"],
            status_ind={"peso": a["peso_ind"]},
            pressao_termica=a["pressao_termica"],
            score_termico=a["score_termico"],
        )
    scores_norm = normalizar(scores)

    # ── 4. Classificação por grupos ───────────────────────────────────────────
    vencidas  = [d for d in DEZENAS if analise[d]["status_ind"] == "🔴 VENCIDA"]
    atrasadas = [d for d in DEZENAS if analise[d]["status_ind"] == "🟠 ATRASADA"]
    no_ciclo  = [d for d in DEZENAS if analise[d]["status_ind"] == "🟡 NO CICLO"]
    ok        = [d for d in DEZENAS if analise[d]["status_ind"] == "🟢 OK"]

    # ── 5. Prioridade para o gerador ─────────────────────────────────────────
    # Vencidas sempre entram; depois atrasadas; depois no ciclo
    if vencidas:
        prioridade = vencidas
    elif no_ciclo:
        prioridade = no_ciclo + atrasadas
    elif atrasadas:
        prioridade = atrasadas
    else:
        # Fallback: top 5 por maior razão
        prioridade = sorted(DEZENAS, key=lambda d: -analise[d]["razao"])[:5]

    # ── 6. Ranking por score de atraso ────────────────────────────────────────
    ranking = sorted(DEZENAS, key=lambda d: -scores_norm[d])

    # ── 7. Atraso por faixa ───────────────────────────────────────────────────
    at_faixa = atraso_por_faixa(sorteios)

    # ── 8. Tabela-resumo para UI ──────────────────────────────────────────────
    tabela = []
    for d in DEZENAS:
        a = analise[d]
        tabela.append({
            "Dezena":        d,
            "Atraso":        a["atraso"],
            "Ciclo Médio":   a["ciclo_medio"],
            "Ciclo Std":     a["ciclo_std"],
            "Razão A/C":     a["razao"],
            "Status Ind.":   a["status_ind"],
            "Status Global": a["status_glob"],
            "Pressão":       round(a["pressao_termica"], 3),
            "Score":         round(scores_norm[d], 4),
            "Rank":          ranking.index(d) + 1,
            "Padrão":        a["padrao_ausencia"]["tipo"],
        })
    df_tabela = (
        pd.DataFrame(tabela)
        .sort_values("Razão A/C", ascending=False)
        .reset_index(drop=True)
    )

    # ── 9. Tabela de percentis globais ────────────────────────────────────────
    df_percentis = pd.DataFrame([
        {"Percentil": k, "Valor (concursos)": v}
        for k, v in percentis.items()
    ])

    return {
        # Análise completa
        "analise":        analise,
        "percentis":      percentis,

        # Scores
        "scores_raw":     scores,
        "scores_norm":    scores_norm,

        # Grupos
        "vencidas":       vencidas,
        "atrasadas":      atrasadas,
        "no_ciclo":       no_ciclo,
        "ok":             ok,
        "prioridade":     prioridade,

        # Ranking
        "ranking":        ranking,

        # Análise espacial
        "atraso_faixa":   at_faixa,

        # UI
        "df_tabela":      df_tabela,
        "df_percentis":   df_percentis,

        # Meta
        "n_sorteios":     len(sorteios),
        "thresholds":     THRESHOLDS,
    }


# =============================================================================
# SCORER — produz score_dezena para o engine_core
# =============================================================================

def scorer_atraso(resultado: dict) -> dict:
    """Retorna scores normalizados por dezena para uso no score consolidado."""
    return resultado["scores_norm"]


# =============================================================================
# INTEGRAÇÃO COM PIPELINE EXECUTOR
# =============================================================================

def executar_etapa_atraso(
    executor: PipelineExecutor,
    sorteios: list,
) -> dict:
    """
    Executa as etapas de atraso & ciclo e salva no contexto do executor.
    Depende de: frequencia_simples, frequencia_janelas (térmica).
    """
    try:
        resultado = run_atraso(sorteios, executor)
        scores_dezena = scorer_atraso(resultado)

        # Salva atraso_atual
        executor.salvar_resultado(
            "atraso_atual",
            {d: {"atraso": resultado["analise"][d]["atraso"]} for d in DEZENAS},
            scores_dezena=scores_dezena,
        )

        # Salva ciclo_medio
        executor.salvar_resultado(
            "ciclo_medio",
            {d: {
                "ciclo_medio":  resultado["analise"][d]["ciclo_medio"],
                "ciclo_std":    resultado["analise"][d]["ciclo_std"],
                "razao":        resultado["analise"][d]["razao"],
            } for d in DEZENAS},
            scores_dezena=scores_dezena,
        )

        # Salva percentis_atraso
        executor.salvar_resultado(
            "percentis_atraso",
            resultado["percentis"],
            scores_dezena=scores_dezena,
            score_agg=float(np.mean(list(scores_dezena.values()))),
        )

        # Salva atraso_faixa
        executor.salvar_resultado(
            "atraso_faixa",
            resultado["atraso_faixa"],
            scores_dezena=scores_dezena,
        )

        # Salva padrao_ausencia
        executor.salvar_resultado(
            "padrao_ausencia",
            {d: resultado["analise"][d]["padrao_ausencia"] for d in DEZENAS},
            scores_dezena=scores_dezena,
        )

        # Salva ciclo_posicao
        executor.salvar_resultado(
            "ciclo_posicao",
            {d: resultado["analise"][d]["ciclo_posicao"] for d in DEZENAS},
            scores_dezena=scores_dezena,
        )

        # Salva resultado completo
        executor.ctx["resultados"]["_atraso_completo"] = resultado

        executor._log("[ATRASO] Etapa de atraso & ciclo executada com sucesso.")
        return resultado

    except Exception as e:
        for sid in ["atraso_atual", "ciclo_medio", "percentis_atraso",
                    "atraso_faixa", "padrao_ausencia", "ciclo_posicao"]:
            executor.marcar_erro(sid, str(e))
        raise


# =============================================================================
# HELPERS PARA USO NAS ETAPAS SEGUINTES
# =============================================================================

def get_atraso(executor: PipelineExecutor) -> dict:
    """Recupera resultado completo da etapa de atraso."""
    return executor.ctx["resultados"].get("_atraso_completo", {})


def get_vencidas_atraso(executor: PipelineExecutor) -> list:
    """Retorna dezenas com status VENCIDA."""
    return get_atraso(executor).get("vencidas", [])


def get_prioridade_atraso(executor: PipelineExecutor) -> list:
    """Retorna dezenas de prioridade máxima para o gerador."""
    return get_atraso(executor).get("prioridade", [])


def get_scores_atraso(executor: PipelineExecutor) -> dict:
    """Retorna scores normalizados de atraso por dezena."""
    return get_atraso(executor).get("scores_norm", {n: 0.5 for n in DEZENAS})


def get_razoes(executor: PipelineExecutor) -> dict:
    """Retorna razão atraso/ciclo por dezena."""
    a = get_atraso(executor)
    analise = a.get("analise", {})
    return {d: analise.get(d, {}).get("razao", 1.0) for d in DEZENAS}
