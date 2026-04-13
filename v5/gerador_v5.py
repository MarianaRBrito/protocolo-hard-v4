# =============================================================================
# gerador_v5.py — Protocolo Lotofácil V5
# Gerador Final — Camada A (candidatos) + Camada B (avaliação e ranking)
# =============================================================================
# Camada A — Geração de candidatos com 8 estratégias:
#   1. Agressiva      — top dezenas por score consolidado
#   2. Híbrida        — mistura score + pressão de atraso
#   3. Equilibrada    — garante cobertura de todas as faixas
#   4. Cobertura      — maximiza dezenas distintas cobertas
#   5. Pressão        — prioriza vencidas e atrasadas
#   6. Estabilidade   — prioriza dezenas estáveis termicamente
#   7. Coocorrência   — prioriza pares e trios mais frequentes
#   8. Estrutural     — equilibra moldura/miolo, pares/ímpares, soma
#
# Camada B — Avaliação e ranking dos candidatos:
#   - score consolidado das dezenas
#   - aderência térmica
#   - aderência atraso/ciclo
#   - aderência estrutural
#   - aderência N+1/N+2
#   - aderência coocorrência
#   - diversidade Hamming entre jogos
#   - filtros estruturais (soma, pares, faixas, consecutivos)
#   - breakdown completo por jogo
#   - justificativa dos melhores
# =============================================================================

import numpy as np
import pandas as pd
import random
import math
from collections import Counter
from itertools import combinations

from engine_core import (
    DEZENAS, PRIMOS, FIBONACCI, MULTIPLOS_3,
    normalizar, PipelineExecutor
)
from etapa_termica import get_termica, get_scores_termicos, get_vencidas
from etapa_atraso import get_atraso, get_scores_atraso, get_vencidas_atraso, get_razoes

ESP_FREQ = 15 / 25

# =============================================================================
# FILTROS ESTRUTURAIS AUTO-CALIBRADOS
# =============================================================================

def calibrar_filtros(sorteios: list, janela: int = 500) -> dict:
    dados = sorteios[-janela:]
    somas       = [sum(s) for s in dados]
    pares_c     = [sum(1 for n in s if n % 2 == 0) for s in dados]
    consec_c    = [_max_consec(s) for s in dados]
    faixa_max_c = [max(_dist_faixas(s)) for s in dados]
    faixa_min_c = [min(_dist_faixas(s)) for s in dados]

    return {
        "soma_min":    int(np.percentile(somas, 5)),
        "soma_max":    int(np.percentile(somas, 95)),
        "pares_min":   int(np.percentile(pares_c, 5)),
        "pares_max":   int(np.percentile(pares_c, 95)),
        "consec_max":  int(np.percentile(consec_c, 95)),
        "faixa_max":   int(np.percentile(faixa_max_c, 95)),
        "faixa_min":   max(0, int(np.percentile(faixa_min_c, 5))),
    }


def _max_consec(jogo):
    jogo = sorted(jogo)
    mx = cur = 1
    for i in range(1, len(jogo)):
        if jogo[i] == jogo[i-1] + 1:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 1
    return mx


def _dist_faixas(jogo):
    return [sum(1 for n in jogo if i*5+1 <= n <= i*5+5) for i in range(5)]


def _hamming(j1, j2):
    return 15 - len(set(j1) & set(j2))


def _composicao(jogo):
    j = set(jogo)
    return {
        "primos":    len(j & PRIMOS),
        "fibonacci": len(j & FIBONACCI),
        "mult3":     len(j & MULTIPLOS_3),
        "pares":     sum(1 for n in jogo if n % 2 == 0),
        "impares":   sum(1 for n in jogo if n % 2 != 0),
    }


def filtrar_jogo(jogo: list, cfg: dict) -> list:
    """Retorna lista de erros. Lista vazia = aprovado."""
    erros = []
    soma = sum(jogo)
    if soma < cfg["soma_min"] or soma > cfg["soma_max"]:
        erros.append(f"Soma {soma} fora [{cfg['soma_min']}–{cfg['soma_max']}]")

    mx = _max_consec(jogo)
    if mx > cfg["consec_max"]:
        erros.append(f"Consecutivos {mx} > {cfg['consec_max']}")

    p = sum(1 for n in jogo if n % 2 == 0)
    if p < cfg["pares_min"] or p > cfg["pares_max"]:
        erros.append(f"Pares {p} fora [{cfg['pares_min']}–{cfg['pares_max']}]")

    faixas = _dist_faixas(jogo)
    for i, f in enumerate(faixas):
        if f == 0:
            erros.append(f"Faixa {i*5+1}-{i*5+5} vazia")
        if f > cfg["faixa_max"]:
            erros.append(f"Faixa {i*5+1}-{i*5+5} excesso ({f})")

    return erros

# =============================================================================
# CAMADA A — GERAÇÃO ORIENTADA A TEMPLATE ESTRUTURAL
# =============================================================================
# Filosofia: cada jogo é construído do zero seguindo um template
# extraído dos sorteios históricos reais. O score por dezena pondera
# qual dezena entra em cada slot — mas a ESTRUTURA vem do histórico.
# Isso elimina o viés de pool e garante jogos com perfil real.
# =============================================================================

MIOLO   = {7, 8, 9, 12, 13, 14, 17, 18, 19}
MOLDURA = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
FAIXAS_DEF = [list(range(i*5+1, i*5+6)) for i in range(5)]


def _extrair_templates(sorteios: list, janela: int = 200) -> list:
    """
    Extrai perfis estruturais dos sorteios históricos recentes.
    Cada template = distribuição de faixas + n_miolo + n_pares + soma.
    """
    dados = sorteios[-janela:]
    templates = []
    for s in dados:
        templates.append({
            "faixas":  _dist_faixas(s),
            "n_miolo": len(set(s) & MIOLO),
            "n_pares": sum(1 for n in s if n % 2 == 0),
            "soma":    sum(s),
        })
    return templates


def _construir_por_template(
    template: dict,
    scores: dict,
    fixas: list,
    cfg: dict,
    max_tentativas: int = 200,
) -> list:
    """
    Constrói um jogo seguindo exatamente o template estrutural.
    Para cada faixa, sorteia o número de dezenas definido no template,
    ponderado pelo score de cada dezena.
    """
    for _ in range(max_tentativas):
        base = list(fixas)
        ok = True

        for fi, faixa in enumerate(FAIXAS_DEF):
            n_alvo = template["faixas"][fi]
            # Quantas já temos dessa faixa pelas fixas
            ja_tem = sum(1 for d in base if d in faixa)
            faltam = n_alvo - ja_tem
            if faltam <= 0:
                continue

            candidatos = [d for d in faixa if d not in base]
            if len(candidatos) < faltam:
                ok = False
                break

            sc = np.array([max(scores.get(d, 0.01), 0.01) for d in candidatos])
            sc /= sc.sum()
            escolhidos = list(np.random.choice(candidatos, size=faltam,
                                                replace=False, p=sc))
            base.extend(escolhidos)

        if not ok or len(base) != 15:
            continue

        jogo = sorted(set(base))
        if len(jogo) == 15 and not filtrar_jogo(jogo, cfg):
            return jogo

    return []


def _sortear_ponderado(pool: list, scores: dict, n: int, fixas: list = None) -> list:
    """Sorteia n dezenas do pool ponderado pelo score — usado como fallback."""
    fixas = fixas or []
    base  = [d for d in fixas if d in pool][:n]
    rest  = [d for d in pool if d not in base]
    faltam = n - len(base)
    if faltam <= 0:
        return sorted(base[:n])
    if len(rest) < faltam:
        rest = [d for d in DEZENAS if d not in base]

    sc_arr = np.array([max(scores.get(d, 0.01), 0.01) for d in rest])
    sc_arr = sc_arr / sc_arr.sum()
    escolhidos = list(np.random.choice(rest, size=min(faltam, len(rest)),
                                        replace=False, p=sc_arr))
    return sorted(base + escolhidos)


def gerar_por_templates(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
    janela: int = 200,
    sc_ajuste: dict = None,
) -> list:
    """
    Estratégia principal: usa templates históricos reais como molde.
    Para cada jogo, sorteia um template dos últimos `janela` concursos
    e constrói o jogo seguindo aquela estrutura.
    """
    templates = _extrair_templates(sorteios, janela)
    sc_uso = sc_ajuste or scores
    jogos = []
    tentativas = 0
    max_t = n_jogos * 500

    while len(jogos) < n_jogos and tentativas < max_t:
        tentativas += 1
        tmpl = random.choice(templates)
        jogo = _construir_por_template(tmpl, sc_uso, fixas, cfg)
        if jogo and jogo not in jogos:
            jogos.append(jogo)

    return jogos


def gerar_template_atraso(
    scores: dict,
    scores_atraso: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """Template + bônus de atraso: dezenas atrasadas recebem peso maior."""
    sc_adj = {d: scores.get(d, 0.5) * 0.5 + scores_atraso.get(d, 0.5) * 0.5
              for d in DEZENAS}
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg,
                                sc_ajuste=sc_adj)


def gerar_template_miolo(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """Template com bônus explícito para dezenas do miolo."""
    sc_adj = {}
    for d in DEZENAS:
        bonus = 1.8 if d in MIOLO else 0.7
        sc_adj[d] = scores.get(d, 0.5) * bonus
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg,
                                sc_ajuste=sc_adj)


def gerar_template_cooc(
    scores: dict,
    cooc_scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """Template + coocorrência: dezenas que saem juntas ficam mais prováveis."""
    sc_adj = {d: scores.get(d, 0.5) * 0.4 + cooc_scores.get(d, 0.5) * 0.6
              for d in DEZENAS}
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg,
                                sc_ajuste=sc_adj)


def gerar_template_pressao(
    scores: dict,
    vencidas: list,
    atrasadas: list,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """Template + pressão máxima para vencidas/atrasadas."""
    sc_adj = {}
    for d in DEZENAS:
        if d in vencidas:
            sc_adj[d] = scores.get(d, 0.5) * 3.5
        elif d in atrasadas:
            sc_adj[d] = scores.get(d, 0.5) * 2.0
        else:
            sc_adj[d] = scores.get(d, 0.5)
    sc_adj = normalizar(sc_adj)

    # Garante ao menos 1 vencida nas fixas se possível
    fixas_ext = list(fixas)
    if vencidas and not any(v in fixas_ext for v in vencidas):
        fixas_ext.append(vencidas[0])

    return gerar_por_templates(scores, sorteios, fixas_ext, n_jogos, cfg,
                                sc_ajuste=sc_adj)


def gerar_template_n12(
    scores: dict,
    fortes_n12: list,
    apoio_n12: list,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """Template + bônus para dezenas projetadas pelo N+1/N+2."""
    sc_adj = {}
    for d in DEZENAS:
        if d in fortes_n12:
            sc_adj[d] = scores.get(d, 0.5) * 3.0
        elif d in apoio_n12:
            sc_adj[d] = scores.get(d, 0.5) * 1.8
        else:
            sc_adj[d] = scores.get(d, 0.5) * 0.6
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg,
                                sc_ajuste=sc_adj)


def gerar_template_recente(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """
    Template dos últimos 30 concursos — captura o momento atual
    com mais precisão que a janela longa.
    """
    sc_rec = {}
    dados_rec = sorteios[-30:]
    for d in DEZENAS:
        f_rec = sum(1 for s in dados_rec if d in s) / len(dados_rec)
        f_lon = sum(1 for s in sorteios[-200:] if d in s) / 200
        # Bônus para dezenas em alta recente
        sc_rec[d] = scores.get(d, 0.5) * 0.4 + f_rec * 0.4 + max(0, f_rec - f_lon) * 0.2
    sc_rec = normalizar(sc_rec)
    # Usa só os últimos 50 concursos como base de templates
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg,
                                janela=50, sc_ajuste=sc_rec)


def gerar_template_anti_vies(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    """
    Anti-viés: penaliza dezenas que aparecem muito no pool mas
    saem pouco nos últimos 20 concursos.
    """
    dados_rec = sorteios[-20:]
    sc_adj = {}
    for d in DEZENAS:
        f_rec = sum(1 for s in dados_rec if d in s) / len(dados_rec)
        sc_base = scores.get(d, 0.5)
        # Penaliza dezenas com score alto mas frequência recente baixa
        if sc_base > 0.6 and f_rec < 0.5:
            sc_adj[d] = sc_base * 0.4  # penaliza
        elif f_rec >= 0.6:
            sc_adj[d] = sc_base * 1.4  # bônus para quem está saindo
        else:
            sc_adj[d] = sc_base
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg,
                                sc_ajuste=sc_adj)

# =============================================================================
# CAMADA A — ORQUESTRADOR
# =============================================================================

def camada_a_gerar_candidatos(
    executor: PipelineExecutor,
    sorteios: list,
    cfg: dict,
    fixas: list = None,
    n_por_estrategia: int = 50,
    seed: int = 0,
) -> list:
    """
    Gera candidatos usando 8 estratégias baseadas em templates históricos.
    Cada estratégia usa o perfil estrutural real dos sorteios como molde
    e o score por dezena como ponderação interna — eliminando viés de pool.
    """
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)

    fixas = [d for d in (fixas or []) if 1 <= d <= 25]

    # Scores consolidados
    scores = executor.score_consolidado()

    # Scores por grupo
    scores_atraso = executor.ctx["scores"].get("percentis_atraso",
                    {d: 0.5 for d in DEZENAS})
    cooc_scores   = executor.ctx["scores"].get("coocorrencia_pares",
                    {d: 0.5 for d in DEZENAS})

    # Vencidas e atrasadas
    at_result = get_atraso(executor)
    vencidas  = at_result.get("vencidas",  [])
    atrasadas = at_result.get("atrasadas", [])

    # N+1/N+2
    n12_res   = executor.ctx["resultados"].get("n1_n2", {})
    fortes_n12 = n12_res.get("fortes", [])
    apoio_n12  = n12_res.get("apoio",  [])

    todos = []

    # 1. Template base — score consolidado puro
    todos.extend(gerar_por_templates(
        scores, sorteios, fixas, n_por_estrategia, cfg, janela=200))

    # 2. Template + atraso
    todos.extend(gerar_template_atraso(
        scores, scores_atraso, sorteios, fixas, n_por_estrategia, cfg))

    # 3. Template + miolo
    todos.extend(gerar_template_miolo(
        scores, sorteios, fixas, n_por_estrategia, cfg))

    # 4. Template + coocorrência
    todos.extend(gerar_template_cooc(
        scores, cooc_scores, sorteios, fixas, n_por_estrategia, cfg))

    # 5. Template + pressão (vencidas/atrasadas)
    todos.extend(gerar_template_pressao(
        scores, vencidas, atrasadas, sorteios, fixas, n_por_estrategia, cfg))

    # 6. Template + N+1/N+2
    todos.extend(gerar_template_n12(
        scores, fortes_n12, apoio_n12, sorteios, fixas, n_por_estrategia, cfg))

    # 7. Template recente (últimos 50 concursos)
    todos.extend(gerar_template_recente(
        scores, sorteios, fixas, n_por_estrategia, cfg))

    # 8. Template anti-viés
    todos.extend(gerar_template_anti_vies(
        scores, sorteios, fixas, n_por_estrategia, cfg))

    # Remove duplicatas mantendo ordem
    vistos = set()
    unicos = []
    for j in todos:
        chave = tuple(j)
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(j)

    return unicos

# =============================================================================
# CAMADA B — AVALIAÇÃO E RANKING DOS CANDIDATOS
# =============================================================================

def avaliar_jogo(
    jogo: list,
    executor: PipelineExecutor,
    sorteios: list,
    scores_glob: dict,
    cfg: dict,
) -> dict:
    """
    Avalia um jogo candidato por múltiplos critérios.
    Retorna dicionário com score_total e breakdown.
    """
    # ── 1. Score consolidado das dezenas ──────────────────────────────────────
    sc_dezenas = sum(scores_glob.get(d, 0) for d in jogo) / 15
    sc_dezenas = round(float(sc_dezenas), 4)

    # ── 2. Aderência térmica ──────────────────────────────────────────────────
    sc_term = executor.ctx["scores"].get("frequencia_janelas", {})
    ad_termica = round(float(np.mean([sc_term.get(d, 0.5) for d in jogo])), 4)

    # ── 3. Aderência atraso/ciclo ─────────────────────────────────────────────
    sc_at = executor.ctx["scores"].get("percentis_atraso", {})
    ad_atraso = round(float(np.mean([sc_at.get(d, 0.5) for d in jogo])), 4)

    # ── 4. Aderência N+1/N+2 ─────────────────────────────────────────────────
    n12 = executor.ctx["resultados"].get("n1_n2", {})
    fortes_n12 = n12.get("fortes", [])
    apoio_n12  = n12.get("apoio",  [])
    cob_fortes = sum(1 for d in jogo if d in fortes_n12)
    cob_apoio  = sum(1 for d in jogo if d in apoio_n12)
    ad_n12 = round((cob_fortes * 1.0 + cob_apoio * 0.5) / max(len(fortes_n12) + len(apoio_n12)*0.5, 1), 4)

    # ── 5. Aderência coocorrência ─────────────────────────────────────────────
    sc_cooc = executor.ctx["scores"].get("coocorrencia_pares", {})
    ad_cooc = round(float(np.mean([sc_cooc.get(d, 0.5) for d in jogo])), 4)

    # ── 6. Aderência estrutural ───────────────────────────────────────────────
    faixas  = _dist_faixas(jogo)
    soma    = sum(jogo)
    pares   = sum(1 for n in jogo if n % 2 == 0)
    max_seq = _max_consec(jogo)
    comp    = _composicao(jogo)

    # Penaliza por faixas vazias ou excessivas
    pen_faixa = sum(0.1 for f in faixas if f == 0) + sum(0.05 for f in faixas if f > 5)

    # Penaliza jogos com miolo sub-representado (7-9, 12-14, 17-19)
    MIOLO = {7, 8, 9, 12, 13, 14, 17, 18, 19}
    n_miolo = len(set(jogo) & MIOLO)
    # Historicamente o miolo contribui com ~5 dezenas por jogo
    pen_miolo = max(0, (4 - n_miolo) * 0.08)  # penaliza se menos de 4 do miolo

    # Penaliza concentração excessiva em dezenas altas (18-25)
    n_altas = sum(1 for d in jogo if d >= 18)
    pen_altas = max(0, (n_altas - 5) * 0.06)  # penaliza se mais de 5 dezenas >= 18

    ad_estrut = round(float(max(0, 1.0 - pen_faixa - pen_miolo - pen_altas)), 4)

    # ── 7. Soma dentro da faixa histórica ────────────────────────────────────
    soma_ok = cfg["soma_min"] <= soma <= cfg["soma_max"]
    ad_soma = 1.0 if soma_ok else max(0, 1 - abs(soma - (cfg["soma_min"] + cfg["soma_max"])/2) / 50)

    # ── 8. Vencidas cobertas ──────────────────────────────────────────────────
    at_res   = get_atraso(executor)
    vencidas = at_res.get("vencidas", [])
    cob_venc = sum(1 for d in jogo if d in vencidas)

    # ── 9. Score total ponderado — rebalanceado para reduzir viés de bordas ───
    score_total = (
        sc_dezenas * 0.25 +  # reduzido: era 0.30
        ad_termica * 0.15 +  # reduzido: era 0.20
        ad_atraso  * 0.25 +  # aumentado: era 0.20
        ad_n12     * 0.10 +
        ad_cooc    * 0.10 +
        ad_estrut  * 0.10 +  # aumentado: era 0.05
        ad_soma    * 0.05
    )
    # Bônus por cobertura de vencidas
    if vencidas:
        score_total += cob_venc / len(vencidas) * 0.10

    score_total = round(float(min(score_total, 1.0)), 4)

    # ── Backtesting rápido ────────────────────────────────────────────────────
    acertos_hist = [len(set(jogo) & set(s)) for s in sorteios[-100:]]
    media_ac = round(float(np.mean(acertos_hist)), 2)
    pct_11   = round(sum(1 for a in acertos_hist if a >= 11) / len(acertos_hist) * 100, 1)
    pct_13   = round(sum(1 for a in acertos_hist if a >= 13) / len(acertos_hist) * 100, 1)

    return {
        "jogo":          jogo,
        "score_total":   score_total,
        "sc_dezenas":    sc_dezenas,
        "ad_termica":    ad_termica,
        "ad_atraso":     ad_atraso,
        "ad_n12":        ad_n12,
        "ad_cooc":       ad_cooc,
        "ad_estrutural": ad_estrut,
        "ad_soma":       round(float(ad_soma), 4),
        "soma":          soma,
        "pares":         pares,
        "impares":       15 - pares,
        "max_seq":       max_seq,
        "faixas":        faixas,
        "composicao":    comp,
        "cob_vencidas":  cob_venc,
        "cob_fortes_n12": cob_fortes,
        "media_acertos": media_ac,
        "pct_11":        pct_11,
        "pct_13":        pct_13,
        "amplitude":     max(jogo) - min(jogo),
    }


def camada_b_selecionar(
    candidatos: list,
    executor: PipelineExecutor,
    sorteios: list,
    cfg: dict,
    n_finais: int = 5,
    max_comum: int = 11,
) -> list:
    """
    Avalia todos os candidatos e seleciona os melhores
    garantindo diversidade mínima (Hamming).
    """
    scores_glob = executor.score_consolidado()

    # Avalia todos
    avaliados = [
        avaliar_jogo(j, executor, sorteios, scores_glob, cfg)
        for j in candidatos
    ]

    # Ordena por score_total
    avaliados.sort(key=lambda x: -x["score_total"])

    # Seleciona com filtro de Hamming
    selecionados = []
    for av in avaliados:
        if len(selecionados) >= n_finais:
            break
        jogo = av["jogo"]
        # Verifica diversidade com já selecionados
        diverso = all(
            15 - len(set(jogo) & set(sel["jogo"])) >= (15 - max_comum)
            for sel in selecionados
        )
        if diverso:
            selecionados.append(av)

    return selecionados


# =============================================================================
# GERADOR COMPLETO (Camada A + B)
# =============================================================================

def gerar_jogos_v5(
    executor: PipelineExecutor,
    sorteios: list,
    n_jogos: int = 5,
    fixas: list = None,
    perfil: str = "Híbrida",
    seed: int = 0,
    max_comum: int = 11,
    n_candidatos_por_estrategia: int = 60,
) -> dict:
    """
    Ponto de entrada principal do gerador V5.

    Parâmetros:
        executor          : PipelineExecutor com etapas concluídas
        sorteios          : histórico completo
        n_jogos           : quantos jogos finais gerar
        fixas             : dezenas fixas obrigatórias
        perfil            : "Agressiva" | "Híbrida" | "Equilibrada"
        seed              : semente aleatória (0 = aleatório)
        max_comum         : máximo de dezenas em comum entre jogos finais
        n_candidatos_por_estrategia : candidatos gerados por estratégia na Camada A

    Retorna:
        {
            "jogos_finais"  : lista de dicts com breakdown completo,
            "candidatos"    : total de candidatos avaliados,
            "hamming"       : tabela de distâncias entre jogos finais,
            "relatorio"     : resumo textual,
        }
    """
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)

    fixas = [d for d in (fixas or []) if 1 <= d <= 25]

    # ── Calibração de filtros ─────────────────────────────────────────────────
    cfg = calibrar_filtros(sorteios)

    # ── Ajuste de max_comum por perfil ────────────────────────────────────────
    perfil_cfg = {
        "Agressiva":   {"max_comum": 8,  "pool_extra": 0},
        "Híbrida":     {"max_comum": 11, "pool_extra": 3},
        "Equilibrada": {"max_comum": 13, "pool_extra": 5},
    }
    pcfg = perfil_cfg.get(perfil, perfil_cfg["Híbrida"])
    max_comum = max_comum or pcfg["max_comum"]

    # ── Camada A: gera candidatos ─────────────────────────────────────────────
    candidatos = camada_a_gerar_candidatos(
        executor=executor,
        sorteios=sorteios,
        cfg=cfg,
        fixas=fixas,
        n_por_estrategia=n_candidatos_por_estrategia,
        seed=seed,
    )

    # Fallback: expande pool se poucos candidatos
    tentativas_extra = 0
    while len(candidatos) < n_jogos * 3 and tentativas_extra < 3:
        extra = camada_a_gerar_candidatos(
            executor=executor,
            sorteios=sorteios,
            cfg=cfg,
            fixas=fixas,
            n_por_estrategia=n_candidatos_por_estrategia,
            seed=seed + tentativas_extra + 1,
        )
        candidatos.extend(extra)
        vistos = set()
        unicos = []
        for j in candidatos:
            k = tuple(j)
            if k not in vistos:
                vistos.add(k)
                unicos.append(j)
        candidatos = unicos
        tentativas_extra += 1

    # ── Camada B: avalia e seleciona ──────────────────────────────────────────
    jogos_finais = camada_b_selecionar(
        candidatos=candidatos,
        executor=executor,
        sorteios=sorteios,
        cfg=cfg,
        n_finais=n_jogos,
        max_comum=max_comum,
    )

    # ── Hamming entre finais ──────────────────────────────────────────────────
    hamming_pares = []
    jogos_lista = [av["jogo"] for av in jogos_finais]
    for i in range(len(jogos_lista)):
        for j in range(i+1, len(jogos_lista)):
            d = _hamming(jogos_lista[i], jogos_lista[j])
            hamming_pares.append({
                "Par":       f"J{i+1} × J{j+1}",
                "Hamming":   d,
                "Em comum":  15 - d,
            })

    # ── Relatório textual ─────────────────────────────────────────────────────
    at_res   = get_atraso(executor)
    vencidas = at_res.get("vencidas", [])
    n12_res  = executor.ctx["resultados"].get("n1_n2", {})
    fortes   = n12_res.get("fortes", [])

    linhas_rel = [
        f"Protocolo V5 — {len(jogos_finais)} jogo(s) gerado(s)",
        f"Candidatos avaliados: {len(candidatos)}",
        f"Perfil: {perfil} | Hamming max comum: {max_comum}",
        f"Vencidas: {vencidas or 'nenhuma'}",
        f"Fortes N+1/N+2: {fortes or 'nenhum'}",
        f"Filtros: Soma [{cfg['soma_min']}–{cfg['soma_max']}] | "
        f"Pares [{cfg['pares_min']}–{cfg['pares_max']}] | "
        f"Consec max {cfg['consec_max']}",
    ]
    for i, av in enumerate(jogos_finais):
        linhas_rel.append(
            f"J{i+1}: {' '.join(f'{d:02d}' for d in av['jogo'])} | "
            f"Score {av['score_total']} | Soma {av['soma']} | "
            f"Pares {av['pares']} | 11+: {av['pct_11']}%"
        )

    return {
        "jogos_finais":    jogos_finais,
        "n_candidatos":    len(candidatos),
        "hamming":         hamming_pares,
        "relatorio":       "\n".join(linhas_rel),
        "cfg_filtros":     cfg,
        "vencidas":        vencidas,
        "fortes_n12":      fortes,
        "perfil":          perfil,
    }


# =============================================================================
# BREAKDOWN DETALHADO PARA UI
# =============================================================================

def breakdown_jogo(av: dict, executor: PipelineExecutor) -> pd.DataFrame:
    """Retorna DataFrame com contribuição de cada dezena do jogo."""
    scores_glob = executor.score_consolidado()
    rows = []
    for d in sorted(av["jogo"]):
        sc = scores_glob.get(d, 0.0)
        rank = sorted(DEZENAS, key=lambda x: -scores_glob.get(x, 0)).index(d) + 1
        at_res  = get_atraso(executor)
        analise = at_res.get("analise", {}).get(d, {})
        rows.append({
            "Dezena":       d,
            "Score":        round(sc, 4),
            "Rank Global":  rank,
            "Atraso":       analise.get("atraso", "—"),
            "Razão A/C":    analise.get("razao", "—"),
            "Status":       analise.get("status_ind", "—"),
            "Vencida":      "🔴" if d in av.get("cob_vencidas", []) else "",
            "N+1/N+2":      "🔥" if d in executor.ctx["resultados"].get("n1_n2", {}).get("fortes", []) else "",
        })
    return pd.DataFrame(rows)


def comparar_jogos(jogos_finais: list) -> pd.DataFrame:
    """Tabela comparativa dos jogos finais."""
    rows = []
    for i, av in enumerate(jogos_finais):
        rows.append({
            "Jogo":          f"J{i+1}",
            "Dezenas":       " ".join(f"{d:02d}" for d in av["jogo"]),
            "Score":         av["score_total"],
            "Soma":          av["soma"],
            "Pares":         av["pares"],
            "Ímpares":       av["impares"],
            "Seq Max":       av["max_seq"],
            "Faixas":        str(av["faixas"]),
            "Média Acertos": av["media_acertos"],
            "11+%":          av["pct_11"],
            "13+%":          av["pct_13"],
            "Vencidas":      av["cob_vencidas"],
            "N+1/N+2 Fortes": av["cob_fortes_n12"],
        })
    return pd.DataFrame(rows)
