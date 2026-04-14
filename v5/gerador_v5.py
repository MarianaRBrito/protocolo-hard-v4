# =============================================================================
# gerador_v5.py — Protocolo Lotofácil V5
# Gerador Final — Camada A (candidatos) + Camada B (avaliação e ranking)
# =============================================================================

import numpy as np
import pandas as pd
import random
from engine_core import (
    DEZENAS, PRIMOS, FIBONACCI, MULTIPLOS_3,
    normalizar, PipelineExecutor
)
from etapa_atraso import get_atraso

ESP_FREQ = 15 / 25

# =============================================================================
# FILTROS ESTRUTURAIS AUTO-CALIBRADOS
# =============================================================================

def calibrar_filtros(sorteios: list, janela: int = 500) -> dict:
    dados = sorteios[-janela:] if janela else sorteios
    if not dados:
        return {
            "soma_min": 170,
            "soma_max": 230,
            "pares_min": 5,
            "pares_max": 10,
            "consec_max": 5,
            "faixa_max": 5,
            "faixa_min": 0,
        }

    somas = [sum(s) for s in dados]
    pares_c = [sum(1 for n in s if n % 2 == 0) for s in dados]
    consec_c = [_max_consec(s) for s in dados]
    faixa_max_c = [max(_dist_faixas(s)) for s in dados]
    faixa_min_c = [min(_dist_faixas(s)) for s in dados]

    return {
        "soma_min": int(np.percentile(somas, 5)),
        "soma_max": int(np.percentile(somas, 95)),
        "pares_min": int(np.percentile(pares_c, 5)),
        "pares_max": int(np.percentile(pares_c, 95)),
        "consec_max": int(np.percentile(consec_c, 95)),
        "faixa_max": int(np.percentile(faixa_max_c, 95)),
        "faixa_min": max(0, int(np.percentile(faixa_min_c, 5))),
    }


def _max_consec(jogo):
    jogo = sorted(jogo)
    mx = cur = 1
    for i in range(1, len(jogo)):
        if jogo[i] == jogo[i - 1] + 1:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 1
    return mx


def _dist_faixas(jogo):
    return [sum(1 for n in jogo if i * 5 + 1 <= n <= i * 5 + 5) for i in range(5)]


def _hamming(j1, j2):
    return 15 - len(set(j1) & set(j2))


def _composicao(jogo):
    j = set(jogo)
    return {
        "primos": len(j & PRIMOS),
        "fibonacci": len(j & FIBONACCI),
        "mult3": len(j & MULTIPLOS_3),
        "pares": sum(1 for n in jogo if n % 2 == 0),
        "impares": sum(1 for n in jogo if n % 2 != 0),
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
        faixa_nome = f"{i * 5 + 1}-{i * 5 + 5}"
        if f < cfg["faixa_min"]:
            erros.append(f"Faixa {faixa_nome} abaixo do mínimo ({f} < {cfg['faixa_min']})")
        if f == 0:
            erros.append(f"Faixa {faixa_nome} vazia")
        if f > cfg["faixa_max"]:
            erros.append(f"Faixa {faixa_nome} excesso ({f})")

    return erros


# =============================================================================
# CAMADA A — GERAÇÃO ORIENTADA A TEMPLATE ESTRUTURAL
# =============================================================================

MIOLO = {7, 8, 9, 12, 13, 14, 17, 18, 19}
MOLDURA = {1, 2, 3, 4, 5, 6, 10, 11, 15, 16, 20, 21, 22, 23, 24, 25}
FAIXAS_DEF = [list(range(i * 5 + 1, i * 5 + 6)) for i in range(5)]


def _extrair_templates(sorteios: list, janela: int = 200) -> list:
    dados = sorteios[-janela:] if janela else sorteios
    templates = []
    for s in dados:
        templates.append({
            "faixas": _dist_faixas(s),
            "n_miolo": len(set(s) & MIOLO),
            "n_pares": sum(1 for n in s if n % 2 == 0),
            "soma": sum(s),
        })

    if not templates:
        templates = [{
            "faixas": [3, 3, 3, 3, 3],
            "n_miolo": 5,
            "n_pares": 7,
            "soma": 195,
        }]

    return templates


def _construir_por_template(
    template: dict,
    scores: dict,
    fixas: list,
    cfg: dict,
    max_tentativas: int = 200,
) -> list:
    """
    Constrói um jogo seguindo o template estrutural.
    """
    fixas = sorted(set(fixas or []))

    for _ in range(max_tentativas):
        base = list(fixas)
        ok = True

        for fi, faixa in enumerate(FAIXAS_DEF):
            n_alvo = template["faixas"][fi]
            ja_tem = sum(1 for d in base if d in faixa)

            if ja_tem > n_alvo:
                ok = False
                break

            faltam = n_alvo - ja_tem
            if faltam <= 0:
                continue

            candidatos = [d for d in faixa if d not in base]
            if len(candidatos) < faltam:
                ok = False
                break

            sc = np.array([max(scores.get(d, 0.01), 0.01) for d in candidatos], dtype=float)
            sc = sc / sc.sum()
            escolhidos = list(np.random.choice(candidatos, size=faltam, replace=False, p=sc))
            base.extend(escolhidos)

        if not ok:
            continue

        jogo = sorted(set(base))
        if len(jogo) != 15:
            continue

        if not filtrar_jogo(jogo, cfg):
            return jogo

    return []


def _sortear_ponderado(pool: list, scores: dict, n: int, fixas: list = None) -> list:
    fixas = sorted(set(fixas or []))
    base = [d for d in fixas if d in pool][:n]
    rest = [d for d in pool if d not in base]
    faltam = n - len(base)

    if faltam <= 0:
        return sorted(base[:n])

    if len(rest) < faltam:
        rest = [d for d in DEZENAS if d not in base]

    sc_arr = np.array([max(scores.get(d, 0.01), 0.01) for d in rest], dtype=float)
    sc_arr = sc_arr / sc_arr.sum()
    escolhidos = list(np.random.choice(rest, size=min(faltam, len(rest)), replace=False, p=sc_arr))
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
    templates = _extrair_templates(sorteios, janela)
    sc_uso = sc_ajuste or scores
    jogos = []
    tentativas = 0
    max_t = max(n_jogos * 500, 500)

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
    sc_adj = {
        d: scores.get(d, 0.5) * 0.5 + scores_atraso.get(d, 0.5) * 0.5
        for d in DEZENAS
    }
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg, sc_ajuste=sc_adj)


def gerar_template_miolo(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    sc_adj = {}
    for d in DEZENAS:
        bonus = 1.8 if d in MIOLO else 0.7
        sc_adj[d] = scores.get(d, 0.5) * bonus
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg, sc_ajuste=sc_adj)


def gerar_template_cooc(
    scores: dict,
    cooc_scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    sc_adj = {
        d: scores.get(d, 0.5) * 0.4 + cooc_scores.get(d, 0.5) * 0.6
        for d in DEZENAS
    }
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg, sc_ajuste=sc_adj)


def gerar_template_pressao(
    scores: dict,
    vencidas: list,
    atrasadas: list,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    sc_adj = {}
    for d in DEZENAS:
        if d in vencidas:
            sc_adj[d] = scores.get(d, 0.5) * 3.5
        elif d in atrasadas:
            sc_adj[d] = scores.get(d, 0.5) * 2.0
        else:
            sc_adj[d] = scores.get(d, 0.5)

    sc_adj = normalizar(sc_adj)

    fixas_ext = list(sorted(set(fixas or [])))
    if vencidas and not any(v in fixas_ext for v in vencidas):
        fixas_ext.append(vencidas[0])
        fixas_ext = sorted(set(fixas_ext))

    return gerar_por_templates(scores, sorteios, fixas_ext, n_jogos, cfg, sc_ajuste=sc_adj)


def gerar_template_n12(
    scores: dict,
    fortes_n12: list,
    apoio_n12: list,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    sc_adj = {}
    for d in DEZENAS:
        if d in fortes_n12:
            sc_adj[d] = scores.get(d, 0.5) * 3.0
        elif d in apoio_n12:
            sc_adj[d] = scores.get(d, 0.5) * 1.8
        else:
            sc_adj[d] = scores.get(d, 0.5) * 0.6
    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg, sc_ajuste=sc_adj)


def gerar_template_recente(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    dados_rec = sorteios[-30:] if len(sorteios) >= 30 else sorteios
    dados_long = sorteios[-200:] if len(sorteios) >= 200 else sorteios

    sc_rec = {}
    for d in DEZENAS:
        f_rec = sum(1 for s in dados_rec if d in s) / max(len(dados_rec), 1)
        f_lon = sum(1 for s in dados_long if d in s) / max(len(dados_long), 1)
        sc_rec[d] = scores.get(d, 0.5) * 0.4 + f_rec * 0.4 + max(0, f_rec - f_lon) * 0.2

    sc_rec = normalizar(sc_rec)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg, janela=50, sc_ajuste=sc_rec)


def gerar_template_anti_vies(
    scores: dict,
    sorteios: list,
    fixas: list,
    n_jogos: int,
    cfg: dict,
) -> list:
    dados_rec = sorteios[-20:] if len(sorteios) >= 20 else sorteios
    sc_adj = {}

    for d in DEZENAS:
        f_rec = sum(1 for s in dados_rec if d in s) / max(len(dados_rec), 1)
        sc_base = scores.get(d, 0.5)

        if sc_base > 0.6 and f_rec < 0.5:
            sc_adj[d] = sc_base * 0.4
        elif f_rec >= 0.6:
            sc_adj[d] = sc_base * 1.4
        else:
            sc_adj[d] = sc_base

    sc_adj = normalizar(sc_adj)
    return gerar_por_templates(scores, sorteios, fixas, n_jogos, cfg, sc_ajuste=sc_adj)


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
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)

    fixas = sorted(set(d for d in (fixas or []) if 1 <= d <= 25))

    scores = executor.score_consolidado()
    if not scores:
        scores = {d: 0.5 for d in DEZENAS}

    scores_atraso = executor.ctx.get("scores", {}).get(
        "percentis_atraso", {d: 0.5 for d in DEZENAS}
    )
    cooc_scores = executor.ctx.get("scores", {}).get(
        "coocorrencia_pares", {d: 0.5 for d in DEZENAS}
    )

    at_result = get_atraso(executor) or {}
    vencidas = at_result.get("vencidas", [])
    atrasadas = at_result.get("atrasadas", [])

    n12_res = executor.ctx.get("resultados", {}).get("n1_n2", {})
    fortes_n12 = n12_res.get("fortes", [])
    apoio_n12 = n12_res.get("apoio", [])

    todos = []

    todos.extend(gerar_por_templates(
        scores, sorteios, fixas, n_por_estrategia, cfg, janela=200
    ))
    todos.extend(gerar_template_atraso(
        scores, scores_atraso, sorteios, fixas, n_por_estrategia, cfg
    ))
    todos.extend(gerar_template_miolo(
        scores, sorteios, fixas, n_por_estrategia, cfg
    ))
    todos.extend(gerar_template_cooc(
        scores, cooc_scores, sorteios, fixas, n_por_estrategia, cfg
    ))
    todos.extend(gerar_template_pressao(
        scores, vencidas, atrasadas, sorteios, fixas, n_por_estrategia, cfg
    ))
    todos.extend(gerar_template_n12(
        scores, fortes_n12, apoio_n12, sorteios, fixas, n_por_estrategia, cfg
    ))
    todos.extend(gerar_template_recente(
        scores, sorteios, fixas, n_por_estrategia, cfg
    ))
    todos.extend(gerar_template_anti_vies(
        scores, sorteios, fixas, n_por_estrategia, cfg
    ))

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
    sc_dezenas = sum(scores_glob.get(d, 0) for d in jogo) / 15
    sc_dezenas = round(float(sc_dezenas), 4)

    sc_term = executor.ctx.get("scores", {}).get("frequencia_janelas", {})
    ad_termica = round(float(np.mean([sc_term.get(d, 0.5) for d in jogo])), 4)

    sc_at = executor.ctx.get("scores", {}).get("percentis_atraso", {})
    ad_atraso = round(float(np.mean([sc_at.get(d, 0.5) for d in jogo])), 4)

    n12 = executor.ctx.get("resultados", {}).get("n1_n2", {})
    fortes_n12 = n12.get("fortes", [])
    apoio_n12 = n12.get("apoio", [])
    cob_fortes_list = [d for d in jogo if d in fortes_n12]
    cob_apoio = sum(1 for d in jogo if d in apoio_n12)

    denom_n12 = max(len(fortes_n12) + len(apoio_n12) * 0.5, 1)
    ad_n12 = round((len(cob_fortes_list) * 1.0 + cob_apoio * 0.5) / denom_n12, 4)

    sc_cooc = executor.ctx.get("scores", {}).get("coocorrencia_pares", {})
    ad_cooc = round(float(np.mean([sc_cooc.get(d, 0.5) for d in jogo])), 4)

    faixas = _dist_faixas(jogo)
    soma = sum(jogo)
    pares = sum(1 for n in jogo if n % 2 == 0)
    max_seq = _max_consec(jogo)
    comp = _composicao(jogo)

    pen_faixa = sum(0.1 for f in faixas if f == 0) + sum(0.05 for f in faixas if f > 5)

    n_miolo = len(set(jogo) & MIOLO)
    pen_miolo = max(0, (4 - n_miolo) * 0.08)

    n_altas = sum(1 for d in jogo if d >= 18)
    pen_altas = max(0, (n_altas - 5) * 0.06)

    ad_estrut = round(float(max(0, 1.0 - pen_faixa - pen_miolo - pen_altas)), 4)

    soma_centro = (cfg["soma_min"] + cfg["soma_max"]) / 2
    soma_ok = cfg["soma_min"] <= soma <= cfg["soma_max"]
    ad_soma = 1.0 if soma_ok else max(0, 1 - abs(soma - soma_centro) / 50)

    at_res = get_atraso(executor) or {}
    vencidas = at_res.get("vencidas", [])
    cob_vencidas_list = [d for d in jogo if d in vencidas]

    score_total = (
        sc_dezenas * 0.25 +
        ad_termica * 0.15 +
        ad_atraso * 0.25 +
        ad_n12 * 0.10 +
        ad_cooc * 0.10 +
        ad_estrut * 0.10 +
        ad_soma * 0.05
    )

    if vencidas:
        score_total += (len(cob_vencidas_list) / len(vencidas)) * 0.10

    score_total = round(float(min(score_total, 1.0)), 4)

    base_bt = sorteios[-100:] if len(sorteios) >= 100 else sorteios
    acertos_hist = [len(set(jogo) & set(s)) for s in base_bt] if base_bt else [0]
    media_ac = round(float(np.mean(acertos_hist)), 2)
    pct_11 = round(sum(1 for a in acertos_hist if a >= 11) / len(acertos_hist) * 100, 1)
    pct_13 = round(sum(1 for a in acertos_hist if a >= 13) / len(acertos_hist) * 100, 1)

    return {
        "jogo": jogo,
        "score_total": score_total,
        "sc_dezenas": sc_dezenas,
        "ad_termica": ad_termica,
        "ad_atraso": ad_atraso,
        "ad_n12": ad_n12,
        "ad_cooc": ad_cooc,
        "ad_estrutural": ad_estrut,
        "ad_soma": round(float(ad_soma), 4),
        "soma": soma,
        "pares": pares,
        "impares": 15 - pares,
        "max_seq": max_seq,
        "faixas": faixas,
        "composicao": comp,
        "cob_vencidas": len(cob_vencidas_list),
        "dezenas_vencidas_cobertas": cob_vencidas_list,
        "cob_fortes_n12": len(cob_fortes_list),
        "dezenas_fortes_n12": cob_fortes_list,
        "media_acertos": media_ac,
        "pct_11": pct_11,
        "pct_13": pct_13,
        "amplitude": max(jogo) - min(jogo),
    }


def _selecionar_diversificado(avaliados: list, n_finais: int, max_comum: int) -> list:
    selecionados = []

    for av in avaliados:
        if len(selecionados) >= n_finais:
            break

        jogo = av["jogo"]
        diverso = all((15 - _hamming(jogo, sel["jogo"])) <= max_comum for sel in selecionados)

        if diverso:
            selecionados.append(av)

    return selecionados


def camada_b_selecionar(
    candidatos: list,
    executor: PipelineExecutor,
    sorteios: list,
    cfg: dict,
    n_finais: int = 5,
    max_comum: int = 11,
) -> list:
    scores_glob = executor.score_consolidado()

    avaliados = [
        avaliar_jogo(j, executor, sorteios, scores_glob, cfg)
        for j in candidatos
    ]
    avaliados.sort(key=lambda x: (-x["score_total"], -x["pct_11"], -x["media_acertos"]))

    selecionados = _selecionar_diversificado(avaliados, n_finais, max_comum)

    if len(selecionados) < n_finais:
        for limite in range(max_comum + 1, 15):
            selecionados = _selecionar_diversificado(avaliados, n_finais, limite)
            if len(selecionados) >= n_finais:
                break

    if len(selecionados) < n_finais:
        ja = {tuple(x["jogo"]) for x in selecionados}
        for av in avaliados:
            if len(selecionados) >= n_finais:
                break
            if tuple(av["jogo"]) not in ja:
                selecionados.append(av)
                ja.add(tuple(av["jogo"]))

    return selecionados[:n_finais]


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
    max_comum: int = None,
    n_candidatos_por_estrategia: int = 60,
) -> dict:
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)

    fixas = sorted(set(d for d in (fixas or []) if 1 <= d <= 25))

    cfg = calibrar_filtros(sorteios)

    perfil_cfg = {
        "Agressiva": {"max_comum": 8},
        "Híbrida": {"max_comum": 11},
        "Equilibrada": {"max_comum": 13},
    }
    pcfg = perfil_cfg.get(perfil, perfil_cfg["Híbrida"])

    if max_comum is None:
        max_comum = pcfg["max_comum"]

    candidatos = camada_a_gerar_candidatos(
        executor=executor,
        sorteios=sorteios,
        cfg=cfg,
        fixas=fixas,
        n_por_estrategia=n_candidatos_por_estrategia,
        seed=seed,
    )

    tentativas_extra = 0
    while len(candidatos) < n_jogos * 3 and tentativas_extra < 3:
        extra = camada_a_gerar_candidatos(
            executor=executor,
            sorteios=sorteios,
            cfg=cfg,
            fixas=fixas,
            n_por_estrategia=n_candidatos_por_estrategia,
            seed=(seed + tentativas_extra + 1) if seed > 0 else 0,
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

    jogos_finais = camada_b_selecionar(
        candidatos=candidatos,
        executor=executor,
        sorteios=sorteios,
        cfg=cfg,
        n_finais=n_jogos,
        max_comum=max_comum,
    )

    hamming_pares = []
    jogos_lista = [av["jogo"] for av in jogos_finais]
    for i in range(len(jogos_lista)):
        for j in range(i + 1, len(jogos_lista)):
            d = _hamming(jogos_lista[i], jogos_lista[j])
            hamming_pares.append({
                "Par": f"J{i + 1} × J{j + 1}",
                "Hamming": d,
                "Em comum": 15 - d,
            })

    at_res = get_atraso(executor) or {}
    vencidas = at_res.get("vencidas", [])
    n12_res = executor.ctx.get("resultados", {}).get("n1_n2", {})
    fortes = n12_res.get("fortes", [])

    linhas_rel = [
        f"Protocolo V5 — {len(jogos_finais)} jogo(s) gerado(s)",
        f"Candidatos avaliados: {len(candidatos)}",
        f"Perfil: {perfil} | Máx. em comum alvo: {max_comum}",
        f"Vencidas: {vencidas or 'nenhuma'}",
        f"Fortes N+1/N+2: {fortes or 'nenhum'}",
        f"Filtros: Soma [{cfg['soma_min']}–{cfg['soma_max']}] | "
        f"Pares [{cfg['pares_min']}–{cfg['pares_max']}] | "
        f"Consec max {cfg['consec_max']}",
    ]

    for i, av in enumerate(jogos_finais):
        linhas_rel.append(
            f"J{i + 1}: {' '.join(f'{d:02d}' for d in av['jogo'])} | "
            f"Score {av['score_total']} | Soma {av['soma']} | "
            f"Pares {av['pares']} | 11+: {av['pct_11']}%"
        )

    return {
        "jogos_finais": jogos_finais,
        "n_candidatos": len(candidatos),
        "hamming": hamming_pares,
        "relatorio": "\n".join(linhas_rel),
        "cfg_filtros": cfg,
        "vencidas": vencidas,
        "fortes_n12": fortes,
        "perfil": perfil,
    }


# =============================================================================
# BREAKDOWN DETALHADO PARA UI
# =============================================================================

def breakdown_jogo(av: dict, executor: PipelineExecutor) -> pd.DataFrame:
    scores_glob = executor.score_consolidado()
    ranking_global = sorted(DEZENAS, key=lambda x: -scores_glob.get(x, 0))
    at_res = get_atraso(executor) or {}
    vencidas = set(at_res.get("vencidas", []))
    fortes_n12 = set(executor.ctx.get("resultados", {}).get("n1_n2", {}).get("fortes", []))

    rows = []
    for d in sorted(av["jogo"]):
        sc = scores_glob.get(d, 0.0)
        rank = ranking_global.index(d) + 1
        analise = at_res.get("analise", {}).get(d, {})

        rows.append({
            "Dezena": d,
            "Score": round(sc, 4),
            "Rank Global": rank,
            "Atraso": analise.get("atraso", "—"),
            "Razão A/C": analise.get("razao", "—"),
            "Status": analise.get("status_ind", "—"),
            "Vencida": "🔴" if d in vencidas else "",
            "N+1/N+2": "🔥" if d in fortes_n12 else "",
        })

    return pd.DataFrame(rows)


def comparar_jogos(jogos_finais: list) -> pd.DataFrame:
    rows = []
    for i, av in enumerate(jogos_finais):
        rows.append({
            "Jogo": f"J{i + 1}",
            "Dezenas": " ".join(f"{d:02d}" for d in av["jogo"]),
            "Score": av["score_total"],
            "Soma": av["soma"],
            "Pares": av["pares"],
            "Ímpares": av["impares"],
            "Seq Max": av["max_seq"],
            "Faixas": str(av["faixas"]),
            "Média Acertos": av["media_acertos"],
            "11+%": av["pct_11"],
            "13+%": av["pct_13"],
            "Vencidas Cobertas": av["cob_vencidas"],
            "N+1/N+2 Fortes": av["cob_fortes_n12"],
        })
    return pd.DataFrame(rows)
