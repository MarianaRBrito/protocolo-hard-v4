# =============================================================================
# etapas_estatisticas.py — Protocolo Lotofácil V5
# Implementação real de todas as etapas estatísticas do pipeline
# =============================================================================
# Cobre:
#   - soma_amplitude, pares_impares, distribuicao_faixas
#   - consecutivos, gaps, paridade_faixa
#   - coocorrencia_pares, trios_recorrentes, trios_expandido
#   - n1_n2, markov_simples
#   - entropia_shannon, entropia_condicional, informacao_mutua
#   - qui_quadrado, ks_test, ljung_box, autocorrelacao
#   - skewness_curtose, pearson, anova, teste_t
#   - bootstrap, permutation_test, cusum, kl_divergencia
#   - tendencia_rolling, frequencia_relativa
#   - monte_carlo, backtesting
#   - valor_esperado, kelly, ev_rateio, ev_ajustado_rateio
#   - lift, lift_par, roc_auc
#   - correlacao_cruzada, espectral, fourier
#   - bayesiana
#   - residuo, residuo_modelo_nulo
#   - holdout, rolling_expandido
#   - score_ponderado (consolidação parcial)
# =============================================================================

import sys
import os

# Garante que o diretório v5 está no path
_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

import numpy as np
import pandas as pd
import math
import random
from collections import Counter
from itertools import combinations

try:
    from scipy import stats as sp_stats
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy import stats as sp_stats

from engine_core import (
    DEZENAS, PRIMOS, FIBONACCI, MULTIPLOS_3,
    normalizar, normalizar_serie, PipelineExecutor
)
from etapa_termica import get_termica, get_scores_termicos
from etapa_atraso import get_atraso, get_scores_atraso, get_vencidas_atraso

# =============================================================================
# HELPERS INTERNOS
# =============================================================================

def _sorteios_janela(sorteios, janela):
    return sorteios[-janela:] if janela else sorteios

def _freq_abs(sorteios, dezena):
    return sum(1 for s in sorteios if dezena in s)

def _freq_rel(sorteios, dezena):
    return _freq_abs(sorteios, dezena) / len(sorteios) if sorteios else 0.0

def _serie_temporal(sorteios, dezena):
    """Série binária: 1 se dezena saiu, 0 se não saiu."""
    return np.array([1 if dezena in s else 0 for s in sorteios], dtype=float)

def _contar_consec(jogo):
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

# =============================================================================
# 1. SOMA & AMPLITUDE
# =============================================================================

def run_soma_amplitude(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    somas = [sum(s) for s in dados]
    ampls = [max(s) - min(s) for s in dados]
    return {
        "soma_min":    int(np.percentile(somas, 5)),
        "soma_max":    int(np.percentile(somas, 95)),
        "soma_media":  round(float(np.mean(somas)), 2),
        "soma_std":    round(float(np.std(somas)), 2),
        "soma_p25":    float(np.percentile(somas, 25)),
        "soma_p75":    float(np.percentile(somas, 75)),
        "ampl_media":  round(float(np.mean(ampls)), 2),
        "ampl_std":    round(float(np.std(ampls)), 2),
        "dist_somas":  somas,
        "dist_ampls":  ampls,
    }

# =============================================================================
# 2. PARES & ÍMPARES
# =============================================================================

def run_pares_impares(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    contagens = [sum(1 for n in s if n % 2 == 0) for s in dados]
    dist = Counter(contagens)
    total = len(dados)

    # Score por dezena: par recebe bônus se pares são sub-representados e vice-versa
    media_pares = np.mean(contagens)
    scores = {}
    for d in DEZENAS:
        eh_par = 1 if d % 2 == 0 else 0
        # Se média de pares < 7.5 (esperado), pares recebem bônus
        if media_pares < 7.5:
            scores[d] = 0.6 + 0.4 * eh_par
        else:
            scores[d] = 0.6 + 0.4 * (1 - eh_par)

    return {
        "dist_pares":    {k: round(v/total*100, 1) for k, v in sorted(dist.items())},
        "media_pares":   round(float(media_pares), 2),
        "top_pares":     dist.most_common(1)[0][0],
        "scores_dezena": scores,
        "score_agg":     round(float(sp_stats.entropy(list(dist.values()))), 4),
    }

# =============================================================================
# 3. DISTRIBUIÇÃO POR FAIXAS
# =============================================================================

def run_distribuicao_faixas(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    faixas = [_dist_faixas(s) for s in dados]
    medias = [round(float(np.mean([f[i] for f in faixas])), 2) for i in range(5)]
    stds   = [round(float(np.std([f[i] for f in faixas])), 2) for i in range(5)]

    # Score por dezena baseado na frequência da sua faixa
    scores = {}
    for d in DEZENAS:
        faixa_idx = (d - 1) // 5
        freq_faixa = medias[faixa_idx] / 3.0  # esperado = 3 por faixa
        scores[d] = round(min(freq_faixa, 1.0), 4)

    return {
        "medias_faixas": medias,
        "stds_faixas":   stds,
        "faixas_raw":    faixas[:20],  # amostra para UI
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(medias)), 4),
    }

# =============================================================================
# 4. MOLDURA & MIOLO
# =============================================================================

def run_moldura_miolo(sorteios, janela=500):
    """
    Grade 5x5: moldura = borda externa (16 dezenas), miolo = interior (9 dezenas).
    Moldura: 1,2,3,4,5,10,15,20,25,24,23,22,21,16,11,6
    Miolo:   7,8,9,12,13,14,17,18,19
    """
    MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
    MIOLO   = {7,8,9,12,13,14,17,18,19}

    dados = _sorteios_janela(sorteios, janela)
    cont_moldura = [len(set(s) & MOLDURA) for s in dados]
    cont_miolo   = [len(set(s) & MIOLO)   for s in dados]

    med_mold = np.mean(cont_moldura)
    med_miolo = np.mean(cont_miolo)

    scores = {}
    for d in DEZENAS:
        if d in MOLDURA:
            scores[d] = round(med_mold / 10.0, 4)
        else:
            scores[d] = round(med_miolo / 5.0, 4)

    return {
        "media_moldura":  round(float(med_mold), 2),
        "media_miolo":    round(float(med_miolo), 2),
        "std_moldura":    round(float(np.std(cont_moldura)), 2),
        "std_miolo":      round(float(np.std(cont_miolo)), 2),
        "MOLDURA":        sorted(MOLDURA),
        "MIOLO":          sorted(MIOLO),
        "scores_dezena":  scores,
        "score_agg":      round(float(med_mold / (med_mold + med_miolo)), 4),
    }

# =============================================================================
# 5. CONSECUTIVOS
# =============================================================================

def run_consecutivos(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    max_consec = [_contar_consec(s) for s in dados]
    dist = Counter(max_consec)
    total = len(dados)
    return {
        "dist_consec":   {k: round(v/total*100, 1) for k, v in sorted(dist.items())},
        "media_consec":  round(float(np.mean(max_consec)), 2),
        "top_consec":    dist.most_common(1)[0][0],
        "score_agg":     round(float(np.std(max_consec)), 4),
    }

# =============================================================================
# 6. GAPS
# =============================================================================

def run_gaps(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    todos_gaps = []
    for s in dados:
        js = sorted(s)
        todos_gaps.extend([js[i+1]-js[i] for i in range(len(js)-1)])

    dist = Counter(todos_gaps)
    total = sum(dist.values())
    return {
        "dist_gaps":     {k: round(v/total*100, 1) for k, v in sorted(dist.items()) if k <= 15},
        "media_gap":     round(float(np.mean(todos_gaps)), 2),
        "std_gap":       round(float(np.std(todos_gaps)), 2),
        "top_gap":       dist.most_common(1)[0][0],
        "score_agg":     round(float(np.mean(todos_gaps)), 4),
    }

# =============================================================================
# 7. PARIDADE POR FAIXA
# =============================================================================

def run_paridade_faixa(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    faixas_def = [range(i*5+1, i*5+6) for i in range(5)]

    scores = {}
    resultado_faixas = {}
    for fi, faixa in enumerate(faixas_def):
        pares_faixa   = [n for n in faixa if n % 2 == 0]
        impares_faixa = [n for n in faixa if n % 2 != 0]
        freq_pares   = np.mean([sum(1 for n in s if n in pares_faixa)   for s in dados])
        freq_impares = np.mean([sum(1 for n in s if n in impares_faixa) for s in dados])
        resultado_faixas[f"F{fi+1}"] = {
            "freq_pares": round(float(freq_pares), 3),
            "freq_impares": round(float(freq_impares), 3),
        }
        for n in faixa:
            if n % 2 == 0:
                scores[n] = round(float(freq_pares / max(freq_pares + freq_impares, 1)), 4)
            else:
                scores[n] = round(float(freq_impares / max(freq_pares + freq_impares, 1)), 4)

    return {
        "faixas": resultado_faixas,
        "scores_dezena": scores,
        "score_agg": round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 8. COOCORRÊNCIA DE PARES
# =============================================================================

def run_coocorrencia_pares(sorteios, janela=500, top=30):
    dados = _sorteios_janela(sorteios, janela)
    pf = Counter()
    for s in dados:
        for a, b in combinations(sorted(s), 2):
            pf[(a, b)] += 1

    # Score por dezena = soma das coocorrências de todos os pares que a contêm
    scores = {d: 0.0 for d in DEZENAS}
    for (a, b), cnt in pf.items():
        scores[a] += cnt
        scores[b] += cnt

    top_pares = pf.most_common(top)
    return {
        "top_pares":     [(f"{a:02d}+{b:02d}", c) for (a,b), c in top_pares],
        "scores_dezena": scores,
        "counter_raw":   dict(pf.most_common(100)),
    }

# =============================================================================
# 9. TRIOS RECORRENTES
# =============================================================================

def run_trios_recorrentes(sorteios, janela=500, top=20):
    dados = _sorteios_janela(sorteios, janela)
    tf = Counter()
    for s in dados:
        for trio in combinations(sorted(s), 3):
            tf[trio] += 1

    scores = {d: 0.0 for d in DEZENAS}
    for trio, cnt in tf.most_common(100):
        for x in trio:
            scores[x] += cnt

    top_trios = tf.most_common(top)
    return {
        "top_trios":     [(f"{a:02d}+{b:02d}+{c:02d}", cnt) for (a,b,c), cnt in top_trios],
        "scores_dezena": scores,
    }

# =============================================================================
# 10. TRIOS EXPANDIDO
# =============================================================================

def run_trios_expandido(sorteios, janela=200, top=30):
    """Trios com janela mais recente — captura tendências recentes de trio."""
    dados = _sorteios_janela(sorteios, janela)
    tf = Counter()
    for s in dados:
        for trio in combinations(sorted(s), 3):
            tf[trio] += 1

    scores = {d: 0.0 for d in DEZENAS}
    for trio, cnt in tf.most_common(150):
        for x in trio:
            scores[x] += cnt * (janela / 500)  # ponderado pela janela

    return {
        "top_trios_recentes": [(f"{a:02d}+{b:02d}+{c:02d}", cnt) for (a,b,c), cnt in tf.most_common(top)],
        "scores_dezena":      scores,
        "janela":             janela,
    }

# =============================================================================
# 11. N+1/N+2 (PROJEÇÃO)
# =============================================================================

def run_n1_n2(sorteios, ref_idx, threshold_pct=80):
    """
    Encontra concursos similares ao de referência e projeta
    as dezenas mais prováveis nos concursos seguintes.
    """
    if ref_idx >= len(sorteios):
        ref_idx = len(sorteios) - 1

    alvo = set(sorteios[ref_idx])
    threshold = int(len(alvo) * threshold_pct / 100)

    similares = []
    futuros_n1 = []
    futuros_n2 = []

    for i, s in enumerate(sorteios):
        if i == ref_idx:
            continue
        if len(alvo & set(s)) >= threshold:
            similares.append(i)
            if i + 1 < len(sorteios):
                futuros_n1.extend(sorteios[i+1])
            if i + 2 < len(sorteios):
                futuros_n2.extend(sorteios[i+2])

    freq_n1 = Counter(futuros_n1)
    freq_n2 = Counter(futuros_n2)

    # Score combinado N1+N2
    scores = {d: 0.0 for d in DEZENAS}
    total_sim = max(len(similares), 1)
    for d in DEZENAS:
        scores[d] = (freq_n1.get(d, 0) * 0.6 + freq_n2.get(d, 0) * 0.4) / total_sim

    fortes = [d for d, _ in Counter({d: scores[d] for d in DEZENAS}).most_common(6)]
    apoio  = [d for d, _ in Counter({d: scores[d] for d in DEZENAS}).most_common(12)][6:]

    return {
        "similares":     similares,
        "n_similares":   len(similares),
        "fortes":        fortes,
        "apoio":         apoio,
        "freq_n1":       dict(freq_n1.most_common(10)),
        "freq_n2":       dict(freq_n2.most_common(10)),
        "scores_dezena": scores,
        "threshold_pct": threshold_pct,
        "ref_idx":       ref_idx,
    }

# =============================================================================
# 12. CADEIA DE MARKOV SIMPLES
# =============================================================================

def run_markov_simples(sorteios, ref_idx, janela=200):
    dados = _sorteios_janela(sorteios, janela)
    tr = {n: Counter() for n in DEZENAS}
    for i in range(len(dados) - 1):
        for n in set(dados[i]):
            tr[n].update(set(dados[i+1]))

    # Aplica ao último sorteio (ref)
    ultimo = set(sorteios[min(ref_idx, len(sorteios)-1)])
    scores = {d: 0.0 for d in DEZENAS}
    for x in ultimo:
        if x in tr:
            for suc, cnt in tr[x].items():
                scores[suc] = scores.get(suc, 0) + cnt

    return {
        "scores_dezena": scores,
        "matriz_dim":    len(tr),
        "ultimo_ref":    sorted(ultimo),
    }

# =============================================================================
# 13. ENTROPIA DE SHANNON
# =============================================================================

def run_entropia_shannon(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    freqs = {d: _freq_rel(dados, d) for d in DEZENAS}
    total_entr = -sum(f * math.log2(f) for f in freqs.values() if f > 0)

    # Dezenas com frequência próxima ao esperado têm maior entropia individual
    scores = {}
    for d in DEZENAS:
        f = freqs[d]
        if f <= 0:
            scores[d] = 0.0
        else:
            scores[d] = round(-(f * math.log2(f)), 6)

    return {
        "entropia_total":  round(total_entr, 4),
        "freqs_relativas": {d: round(freqs[d], 4) for d in DEZENAS},
        "scores_dezena":   scores,
        "score_agg":       round(total_entr, 4),
    }

# =============================================================================
# 14. QUI-QUADRADO
# =============================================================================

def run_qui_quadrado(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    obs = np.array([_freq_abs(dados, d) for d in DEZENAS], dtype=float)
    esp = np.full(25, len(dados) * 15 / 25)
    chi2, p = sp_stats.chisquare(obs, esp)
    return {
        "chi2":     round(float(chi2), 4),
        "p_valor":  round(float(p), 6),
        "base_ok":  p > 0.05,
        "score_agg": round(float(p), 6),
    }

# =============================================================================
# 15. TESTE KS (Kolmogorov-Smirnov)
# =============================================================================

def run_ks_test(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    somas = [sum(s) for s in dados]
    ks_stat, p = sp_stats.kstest(somas, 'norm',
                                  args=(np.mean(somas), np.std(somas)))
    return {
        "ks_stat":  round(float(ks_stat), 4),
        "p_valor":  round(float(p), 6),
        "normal":   p > 0.05,
        "score_agg": round(float(1 - ks_stat), 4),
    }

# =============================================================================
# 16. LJUNG-BOX
# =============================================================================

def run_ljung_box(sorteios, janela=200, lags=10):
    """Testa autocorrelação nas séries temporais de cada dezena."""
    dados = _sorteios_janela(sorteios, janela)
    resultados = {}
    p_valores = []
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        if len(serie) < lags + 5:
            resultados[d] = {"lb_stat": 0.0, "p_valor": 1.0}
            continue
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            res = acorr_ljungbox(serie, lags=[lags], return_df=True)
            lb_stat = float(res["lb_stat"].iloc[0])
            p_val   = float(res["lb_pvalue"].iloc[0])
        except ImportError:
            # Fallback: usa variância como proxy
            lb_stat = float(np.var(serie) * len(serie))
            p_val   = 0.5
        resultados[d] = {"lb_stat": round(lb_stat, 4), "p_valor": round(p_val, 4)}
        p_valores.append(p_val)

    return {
        "resultados":    resultados,
        "media_p_valor": round(float(np.mean(p_valores)), 4),
        "score_agg":     round(float(np.mean(p_valores)), 4),
    }

# =============================================================================
# 17. AUTOCORRELAÇÃO
# =============================================================================

def run_autocorrelacao(sorteios, janela=200, max_lag=10):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        acfs = []
        for lag in range(1, min(max_lag+1, len(serie)//2)):
            if len(serie) > lag:
                acf = float(np.corrcoef(serie[:-lag], serie[lag:])[0,1])
                acfs.append(abs(acf))
        scores[d] = round(float(np.mean(acfs)) if acfs else 0.0, 4)

    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 18. SKEWNESS & CURTOSE
# =============================================================================

def run_skewness_curtose(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    somas = [sum(s) for s in dados]
    return {
        "skewness":  round(float(sp_stats.skew(somas)), 4),
        "curtose":   round(float(sp_stats.kurtosis(somas)), 4),
        "normal_sk": abs(sp_stats.skew(somas)) < 0.5,
        "score_agg": round(float(1 - abs(sp_stats.skew(somas))), 4),
    }

# =============================================================================
# 19. PEARSON (correlação cruzada entre dezenas)
# =============================================================================

def run_pearson(sorteios, janela=200):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    ref_serie = np.array([sum(s)/15 for s in dados])
    for d in DEZENAS:
        serie_d = _serie_temporal(dados, d)
        corr, _ = sp_stats.pearsonr(serie_d, ref_serie)
        scores[d] = round(float((corr + 1) / 2), 4)  # normaliza [-1,1] → [0,1]
    return {
        "scores_dezena": scores,
    }

# =============================================================================
# 20. ANOVA
# =============================================================================

def run_anova(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    grupos = [[_freq_abs(dados[-j:], d) for d in DEZENAS]
              for j in [50, 100, 200, 500] if j <= len(dados)]
    if len(grupos) < 2:
        return {"f_stat": 0.0, "p_valor": 1.0, "score_agg": 0.5}
    f_stat, p = sp_stats.f_oneway(*grupos)
    return {
        "f_stat":    round(float(f_stat), 4),
        "p_valor":   round(float(p), 6),
        "score_agg": round(float(p), 6),
    }

# =============================================================================
# 21. TESTE T
# =============================================================================

def run_teste_t(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    # Compara frequência de cada dezena vs esperado
    esp = ESP_FREQ
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        t_stat, p = sp_stats.ttest_1samp(serie, esp)
        # Dezenas com desvio significativo positivo recebem score alto
        scores[d] = round(float(max(0, t_stat) / (abs(t_stat) + 1)), 4)
    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 22. BOOTSTRAP
# =============================================================================

def run_bootstrap(sorteios, janela=500, n_boot=500):
    dados = _sorteios_janela(sorteios, janela)
    freqs_obs = {d: _freq_rel(dados, d) for d in DEZENAS}

    # Bootstrap: reamostrar os sorteios com reposição
    boot_freqs = {d: [] for d in DEZENAS}
    for _ in range(n_boot):
        amostra = random.choices(dados, k=len(dados))
        for d in DEZENAS:
            boot_freqs[d].append(_freq_rel(amostra, d))

    scores = {}
    ic_95 = {}
    for d in DEZENAS:
        bf = np.array(boot_freqs[d])
        ic_low  = float(np.percentile(bf, 2.5))
        ic_high = float(np.percentile(bf, 97.5))
        ic_95[d] = (round(ic_low, 4), round(ic_high, 4))
        # Score: quão acima do esperado está o IC
        scores[d] = round(float(np.mean(bf > ESP_FREQ)), 4)

    return {
        "scores_dezena": scores,
        "ic_95":         ic_95,
        "freqs_obs":     {d: round(freqs_obs[d], 4) for d in DEZENAS},
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 23. PERMUTATION TEST
# =============================================================================

def run_permutation_test(sorteios, janela=500, n_perm=500):
    dados = _sorteios_janela(sorteios, janela)
    obs_chi2, obs_p = sp_stats.chisquare(
        [_freq_abs(dados, d) for d in DEZENAS],
        np.full(25, len(dados)*15/25)
    )
    # Permutação: embaralha as dezenas dentro de cada sorteio
    perm_chi2s = []
    for _ in range(n_perm):
        perm_dados = [random.sample(list(range(1,26)), 15) for _ in dados]
        obs_p2 = [sum(1 for s in perm_dados if d in s) for d in DEZENAS]
        c2, _ = sp_stats.chisquare(obs_p2, np.full(25, len(perm_dados)*15/25))
        perm_chi2s.append(c2)

    p_perm = round(float(np.mean(np.array(perm_chi2s) >= obs_chi2)), 4)
    return {
        "obs_chi2":  round(float(obs_chi2), 4),
        "p_perm":    p_perm,
        "score_agg": round(float(1 - p_perm), 4),
    }

# =============================================================================
# 24. CUSUM
# =============================================================================

def run_cusum(sorteios, janela=200):
    """CUSUM detecta mudança de média na frequência de cada dezena."""
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        media_geral = np.mean(serie)
        cusum_pos = 0.0
        cusum_neg = 0.0
        cusums = []
        for x in serie:
            cusum_pos = max(0, cusum_pos + x - media_geral - 0.5)
            cusum_neg = max(0, cusum_neg - x + media_geral - 0.5)
            cusums.append(cusum_pos - cusum_neg)
        # Score: sinal recente do CUSUM (positivo = frequência crescendo)
        sinal_recente = np.mean(cusums[-20:]) if len(cusums) >= 20 else 0.0
        scores[d] = round(float(max(0, sinal_recente)), 4)

    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 25. DIVERGÊNCIA KL
# =============================================================================

def run_kl_divergencia(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    freqs_obs = np.array([_freq_rel(dados, d) for d in DEZENAS])
    freqs_obs = freqs_obs / freqs_obs.sum()  # normaliza
    freqs_esp = np.full(25, 1/25)

    kl_total = float(sp_stats.entropy(freqs_obs, freqs_esp))

    # Score por dezena: quanto cada uma contribui para a KL
    scores = {}
    for i, d in enumerate(DEZENAS):
        p = freqs_obs[i]
        q = freqs_esp[i]
        contrib = p * math.log(p/q + 1e-10)
        # Score: dezenas super-representadas recebem bônus
        scores[d] = round(float(max(0, contrib)), 6)

    return {
        "kl_total":      round(kl_total, 6),
        "scores_dezena": scores,
        "score_agg":     round(kl_total, 6),
    }

# =============================================================================
# 26. TENDÊNCIA ROLLING
# =============================================================================

def run_tendencia_rolling(sorteios, janelas_comp=[(20, 200), (50, 500)]):
    THRESHOLD = 0.05
    scores = {}
    status = {}

    for d in DEZENAS:
        sc_total = 0.0
        for j_curta, j_longa in janelas_comp:
            j_c = min(j_curta, len(sorteios))
            j_l = min(j_longa, len(sorteios))
            f_curta = _freq_rel(sorteios[-j_c:], d)
            f_longa = _freq_rel(sorteios[-j_l:], d)
            delta = f_curta - f_longa
            sc_total += max(0, delta)
        scores[d] = round(sc_total / len(janelas_comp), 4)

        f20  = _freq_rel(sorteios[-20:],  d)
        f200 = _freq_rel(sorteios[-200:], d) if len(sorteios) >= 200 else f20
        delta_main = f20 - f200
        if delta_main > THRESHOLD:
            status[d] = "🔼 ALTA"
        elif delta_main < -THRESHOLD:
            status[d] = "🔽 BAIXA"
        elif f20 > ESP_FREQ * 1.05:
            status[d] = "🟢 ACIMA"
        elif f20 < ESP_FREQ * 0.95:
            status[d] = "🟡 ABAIXO"
        else:
            status[d] = "➡️ ESTÁVEL"

    return {
        "scores_dezena": scores,
        "status":        status,
        "altas":  [d for d in DEZENAS if "ALTA"  in status[d]],
        "baixas": [d for d in DEZENAS if "BAIXA" in status[d]],
    }

# =============================================================================
# 27. FREQUÊNCIA RELATIVA AJUSTADA
# =============================================================================

def run_frequencia_relativa(sorteios, scores_termicos, scores_atraso, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    for d in DEZENAS:
        f_rel = _freq_rel(dados, d)
        st = scores_termicos.get(d, 0.5)
        sa = scores_atraso.get(d, 0.5)
        # Frequência ajustada pelos scores de temperatura e atraso
        scores[d] = round(f_rel * 0.5 + st * 0.3 + sa * 0.2, 6)
    return {
        "scores_dezena": scores,
        "freqs_brutas":  {d: round(_freq_rel(dados, d), 4) for d in DEZENAS},
    }

# =============================================================================
# 28. MONTE CARLO
# =============================================================================

def run_monte_carlo(sorteios, scores_consolidados, n=3000):
    """
    Simula n jogos aleatórios ponderados pelo score consolidado
    e calcula frequência simulada de cada dezena.
    """
    sc = scores_consolidados
    total_sc = sum(sc.values())
    probs = [sc.get(d, 1/25) / total_sc for d in DEZENAS]

    contagens = Counter()
    for _ in range(n):
        jogo = np.random.choice(DEZENAS, 15, replace=False, p=probs)
        contagens.update(jogo)

    scores = {d: round(contagens[d] / (n * 15 / 25), 4) for d in DEZENAS}
    return {
        "scores_dezena": scores,
        "n_simulacoes":  n,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 29. BACKTESTING
# =============================================================================

def run_backtesting(sorteios, scores_dezena, ultimos=200):
    dados = sorteios[-ultimos:]
    resultados_jogo = []

    # Gera um jogo representativo com top 15 dezenas por score
    jogo_rep = sorted(DEZENAS, key=lambda d: -scores_dezena.get(d, 0))[:15]

    acertos = [len(set(jogo_rep) & set(s)) for s in dados]
    dist = Counter(acertos)

    # Score por dezena: backtest individual
    scores = {}
    for d in DEZENAS:
        serie_d = [1 if d in s else 0 for s in dados]
        scores[d] = round(float(np.mean(serie_d)), 4)

    return {
        "jogo_rep":      jogo_rep,
        "media_acertos": round(float(np.mean(acertos)), 2),
        "max_acertos":   max(acertos),
        "pct_11":        round(sum(1 for a in acertos if a >= 11) / len(acertos) * 100, 1),
        "pct_13":        round(sum(1 for a in acertos if a >= 13) / len(acertos) * 100, 1),
        "pct_15":        round(sum(1 for a in acertos if a == 15) / len(acertos) * 100, 2),
        "dist_acertos":  {k: round(v/len(acertos)*100, 1) for k, v in sorted(dist.items())},
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(acertos)), 4),
    }

# =============================================================================
# 30. VALOR ESPERADO & KELLY
# =============================================================================

PREMIOS = {15: 1_500_000, 14: 1500, 13: 25, 12: 10, 11: 5}
PROB    = {
    15: 1/3_268_760,
    14: 15/3_268_760,
    13: 105/3_268_760,
    12: 455/3_268_760,
    11: 1365/3_268_760,
}
CUSTO_JOGO = 3.50

def run_valor_esperado():
    ev = sum(PREMIOS[k] * PROB[k] for k in PREMIOS)
    ratio = ev / CUSTO_JOGO
    return {
        "ev":         round(ev, 4),
        "custo":      CUSTO_JOGO,
        "ratio":      round(ratio, 4),
        "score_agg":  round(ratio, 4),
        "detalhes":   {k: round(PREMIOS[k]*PROB[k], 4) for k in PREMIOS},
    }

def run_kelly(p_vitoria=PROB[15], premio=PREMIOS[15], custo=CUSTO_JOGO):
    """Critério de Kelly para fração ótima do bankroll."""
    b = premio / custo
    q = 1 - p_vitoria
    kelly_f = (b * p_vitoria - q) / b
    return {
        "kelly_fraction": round(float(kelly_f), 8),
        "recomendacao":   "Apostar mínimo (Kelly < 0)" if kelly_f < 0 else f"{kelly_f*100:.4f}% do bankroll",
        "score_agg":      max(0.0, round(float(kelly_f), 8)),
    }

def run_ev_rateio(n_apostadores=1_000_000):
    """EV ajustado considerando rateio com outros apostadores."""
    ev_ajustado = sum(PREMIOS[k] / n_apostadores * PROB[k] for k in PREMIOS)
    return {
        "ev_ajustado":    round(ev_ajustado, 6),
        "n_apostadores":  n_apostadores,
        "score_agg":      round(ev_ajustado / CUSTO_JOGO, 6),
    }

def run_ev_ajustado_rateio(n_apostadores=1_000_000):
    base = run_ev_rateio(n_apostadores)
    return {**base, "razao_custo": round(base["ev_ajustado"] / CUSTO_JOGO, 6)}

# =============================================================================
# 31. LIFT
# =============================================================================

def run_lift(sorteios, janela=500, top=20):
    dados = _sorteios_janela(sorteios, janela)
    pf = Counter()
    freq_ind = {d: _freq_rel(dados, d) for d in DEZENAS}
    for s in dados:
        for a, b in combinations(sorted(s), 2):
            pf[(a, b)] += 1

    scores = {d: 0.0 for d in DEZENAS}
    lifts = {}
    for (a, b), cnt in pf.most_common(top * 5):
        p_ab = cnt / len(dados)
        p_a  = freq_ind[a]
        p_b  = freq_ind[b]
        lift_val = p_ab / (p_a * p_b) if p_a * p_b > 0 else 1.0
        lifts[(a, b)] = round(lift_val, 3)
        scores[a] += lift_val
        scores[b] += lift_val

    return {
        "top_lifts":     sorted(lifts.items(), key=lambda x: -x[1])[:top],
        "scores_dezena": scores,
    }

def run_lift_par(sorteios, janela=500, top=20):
    """Lift por par específico — complementa o lift geral."""
    r = run_lift(sorteios, janela, top)
    # Score adicional para dezenas em pares de lift > 1.2
    scores = {d: 0.0 for d in DEZENAS}
    for (a, b), lv in r["top_lifts"]:
        if lv > 1.2:
            scores[a] += lv * 0.5
            scores[b] += lv * 0.5
    return {
        "scores_dezena": scores,
        "lift_base":     r["top_lifts"],
    }

# =============================================================================
# 32. ROC/AUC
# =============================================================================

def run_roc_auc(sorteios, scores_dezena, janela=200):
    """
    Avalia AUC de cada dezena: quão bem o score prevê aparição.
    """
    dados = _sorteios_janela(sorteios, janela)
    aucs = {}
    for d in DEZENAS:
        y_true = np.array([1 if d in s else 0 for s in dados])
        sc = scores_dezena.get(d, 0.5)
        # AUC simplificada: correlação entre score constante e série
        # (proxy determinístico — sem dados de treino dinâmico)
        freq = np.mean(y_true)
        aucs[d] = round(float(freq * sc + (1 - freq) * (1 - sc)), 4)

    return {
        "aucs":      aucs,
        "score_agg": round(float(np.mean(list(aucs.values()))), 4),
    }

# =============================================================================
# 33. CORRELAÇÃO CRUZADA
# =============================================================================

def run_correlacao_cruzada(sorteios, janela=200):
    dados = _sorteios_janela(sorteios, janela)
    scores = {d: 0.0 for d in DEZENAS}
    serie_soma = np.array([sum(s) for s in dados], dtype=float)
    serie_soma = (serie_soma - np.mean(serie_soma)) / (np.std(serie_soma) + 1e-9)

    for d in DEZENAS:
        serie_d = _serie_temporal(dados, d)
        serie_d = (serie_d - np.mean(serie_d)) / (np.std(serie_d) + 1e-9)
        corr = float(np.correlate(serie_d, serie_soma, mode='valid')[0]) / len(dados)
        scores[d] = round(float((corr + 1) / 2), 4)

    return {"scores_dezena": scores}

# =============================================================================
# 34. ANÁLISE ESPECTRAL & FOURIER
# =============================================================================

def run_espectral(sorteios, janela=200):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        fft_vals = np.abs(np.fft.rfft(serie))
        # Energia nas frequências baixas (periodicidade) vs total
        energia_baixa = float(np.sum(fft_vals[:5]))
        energia_total = float(np.sum(fft_vals))
        scores[d] = round(energia_baixa / (energia_total + 1e-9), 4)

    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

def run_fourier(sorteios, janela=200):
    """Fourier completo com detecção de frequências dominantes."""
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    freqs_dominantes = {}
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        fft_vals = np.fft.rfft(serie)
        magnitudes = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(serie))
        # Frequência dominante (exceto DC)
        idx_dom = np.argmax(magnitudes[1:]) + 1
        freqs_dominantes[d] = round(float(freqs[idx_dom]), 4)
        # Score = magnitude da frequência dominante normalizada
        scores[d] = round(float(magnitudes[idx_dom] / (np.sum(magnitudes) + 1e-9)), 4)

    return {
        "scores_dezena":     scores,
        "freqs_dominantes":  freqs_dominantes,
        "score_agg":         round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 35. BAYESIANA
# =============================================================================

def run_bayesiana(sorteios, scores_termicos, janela=500):
    """
    Atualização bayesiana: prior = frequência esperada (1/25),
    likelihood = frequência observada, posterior = combinação.
    """
    dados = _sorteios_janela(sorteios, janela)
    alpha_prior = 15  # hiperparâmetro prior (esperado = 15 por 25 dezenas)
    beta_prior  = 10

    scores = {}
    for d in DEZENAS:
        obs_pos = _freq_abs(dados, d)
        obs_neg = len(dados) - obs_pos
        alpha_post = alpha_prior + obs_pos
        beta_post  = beta_prior  + obs_neg
        posterior_mean = alpha_post / (alpha_post + beta_post)
        # Ajuste pelo score térmico
        st = scores_termicos.get(d, 0.5)
        scores[d] = round(posterior_mean * 0.7 + st * 0.3, 6)

    return {
        "scores_dezena": scores,
        "alpha_prior":   alpha_prior,
        "beta_prior":    beta_prior,
    }

# =============================================================================
# 36. ENTROPIA CONDICIONAL & INFORMAÇÃO MÚTUA
# =============================================================================

def run_entropia_condicional(sorteios, janela=200):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    for d in DEZENAS:
        serie = _serie_temporal(dados, d)
        # Entropia condicional H(X_t | X_{t-1})
        pares = Counter(zip(serie[:-1].astype(int), serie[1:].astype(int)))
        total = sum(pares.values())
        h_cond = 0.0
        for (x0, x1), cnt in pares.items():
            p_joint = cnt / total
            p_x0 = sum(v for (a, b), v in pares.items() if a == x0) / total
            if p_x0 > 0 and p_joint > 0:
                h_cond -= p_joint * math.log2(p_joint / p_x0 + 1e-10)
        scores[d] = round(max(0.0, h_cond), 4)

    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

def run_informacao_mutua(sorteios, janela=200):
    dados = _sorteios_janela(sorteios, janela)
    scores = {}
    for d in DEZENAS:
        serie_d = _serie_temporal(dados, d)
        # Informação mútua entre a dezena e a soma do sorteio
        soma_bins = np.array([sum(s) for s in dados])
        soma_bins = (soma_bins > np.median(soma_bins)).astype(int)  # binariza
        # MI = H(X) - H(X|Y)
        p1 = np.mean(serie_d)
        p0 = 1 - p1
        h_x = 0.0
        if p1 > 0: h_x -= p1 * math.log2(p1)
        if p0 > 0: h_x -= p0 * math.log2(p0)

        # H(X|Y) via contagens
        joint = Counter(zip(soma_bins, serie_d.astype(int)))
        total = len(serie_d)
        h_xy = 0.0
        for (y, x), cnt in joint.items():
            p_y = sum(v for (b, _), v in joint.items() if b == y) / total
            p_joint = cnt / total
            if p_y > 0 and p_joint > 0:
                h_xy -= p_joint * math.log2(p_joint / p_y + 1e-10)

        mi = max(0.0, h_x - h_xy)
        scores[d] = round(mi, 4)

    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 37. RESÍDUO vs ESPERADO
# =============================================================================

def run_residuo(sorteios, janela=500):
    dados = _sorteios_janela(sorteios, janela)
    esp = ESP_FREQ
    scores = {}
    for d in DEZENAS:
        obs = _freq_rel(dados, d)
        residuo = obs - esp
        # Dezenas com resíduo positivo (mais frequentes que esperado) recebem score alto
        scores[d] = round(float((residuo + 1) / 2), 4)  # normaliza [-1,1] → [0,1]
    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
        "residuos":      {d: round(_freq_rel(dados, d) - esp, 4) for d in DEZENAS},
    }

def run_residuo_modelo_nulo(sorteios, janela=500):
    """Resíduo relativo ao modelo nulo (uniforme)."""
    r = run_residuo(sorteios, janela)
    # Score inverso: dezenas sub-representadas recebem bônus (pressão de volta)
    scores = {d: round(1 - r["scores_dezena"][d], 4) for d in DEZENAS}
    return {
        "scores_dezena": scores,
        "score_agg":     round(float(np.mean(list(scores.values()))), 4),
    }

# =============================================================================
# 38. HOLDOUT & ROLLING EXPANDIDO
# =============================================================================

def run_holdout(sorteios, scores_dezena, holdout_pct=0.2):
    n = len(sorteios)
    split = int(n * (1 - holdout_pct))
    treino = sorteios[:split]
    teste  = sorteios[split:]

    jogo_rep = sorted(DEZENAS, key=lambda d: -scores_dezena.get(d, 0))[:15]
    acertos_teste = [len(set(jogo_rep) & set(s)) for s in teste]

    return {
        "n_treino":      len(treino),
        "n_teste":       len(teste),
        "media_acertos": round(float(np.mean(acertos_teste)), 2),
        "pct_11":        round(sum(1 for a in acertos_teste if a >= 11) / len(acertos_teste) * 100, 1),
        "score_agg":     round(float(np.mean(acertos_teste)), 4),
    }

def run_rolling_expandido(sorteios, scores_dezena, min_treino=200, step=50):
    """Rolling window validation."""
    resultados = []
    n = len(sorteios)
    jogo_rep = sorted(DEZENAS, key=lambda d: -scores_dezena.get(d, 0))[:15]

    for start in range(min_treino, n - step, step):
        teste = sorteios[start:start+step]
        acertos = [len(set(jogo_rep) & set(s)) for s in teste]
        resultados.append(round(float(np.mean(acertos)), 2))

    scores = {d: scores_dezena.get(d, 0.5) for d in DEZENAS}
    return {
        "scores_dezena":  scores,
        "media_rolling":  round(float(np.mean(resultados)), 2) if resultados else 0.0,
        "std_rolling":    round(float(np.std(resultados)), 2) if resultados else 0.0,
        "n_janelas":      len(resultados),
        "score_agg":      round(float(np.mean(resultados)), 4) if resultados else 0.0,
    }

# =============================================================================
# 39. SCORE PONDERADO FINAL (consolidação parcial)
# =============================================================================

def run_score_ponderado(executor: PipelineExecutor) -> dict:
    """
    Consolida todos os scores disponíveis no executor
    usando os pesos do STEP_REGISTRY.
    """
    sc = executor.score_consolidado()
    return {
        "scores_dezena": sc,
        "ranking":       sorted(DEZENAS, key=lambda d: -sc.get(d, 0)),
        "score_agg":     round(float(np.mean(list(sc.values()))), 4),
    }

# =============================================================================
# EXECUTOR PRINCIPAL — roda todas as etapas estatísticas em cascata
# =============================================================================

def executar_etapas_estatisticas(
    executor: PipelineExecutor,
    sorteios: list,
    janela: int = 500,
    ref_idx: int = -1,
    threshold_n12: int = 80,
) -> dict:
    """
    Executa todas as etapas estatísticas em ordem de dependência.
    Usa resultados das etapas térmica e atraso já no executor.
    """
    if ref_idx < 0:
        ref_idx = len(sorteios) - 1

    scores_termicos = get_scores_termicos(executor)
    scores_atraso   = get_scores_atraso(executor)

    resultados = {}

    def _salvar(step_id, resultado, scores_dezena=None, score_agg=None):
        executor.salvar_resultado(step_id, resultado,
                                  scores_dezena=scores_dezena,
                                  score_agg=score_agg)
        resultados[step_id] = resultado

    # ── Estrutural ────────────────────────────────────────────────────────────
    r = run_soma_amplitude(sorteios, janela)
    _salvar("soma_amplitude", r, score_agg=r["soma_media"])

    r = run_pares_impares(sorteios, janela)
    _salvar("pares_impares", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_distribuicao_faixas(sorteios, janela)
    _salvar("distribuicao_faixas", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_moldura_miolo(sorteios, janela)
    _salvar("moldura_miolo", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_consecutivos(sorteios, janela)
    _salvar("consecutivos", r, score_agg=r["score_agg"])

    r = run_gaps(sorteios, janela)
    _salvar("gaps", r, score_agg=r["score_agg"])

    r = run_paridade_faixa(sorteios, janela)
    _salvar("paridade_faixa", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r2 = run_paridade_faixa(sorteios, min(janela, 200))
    _salvar("paridade_faixa_v2", r2, scores_dezena=r2["scores_dezena"], score_agg=r2["score_agg"])

    # ── Composição numérica ───────────────────────────────────────────────────
    scores_primos = {d: (1.0 if d in PRIMOS else 0.3) for d in DEZENAS}
    _salvar("primos", {"primos": sorted(PRIMOS)}, scores_dezena=scores_primos)

    scores_fib = {d: (1.0 if d in FIBONACCI else 0.3) for d in DEZENAS}
    _salvar("fibonacci", {"fibonacci": sorted(FIBONACCI)}, scores_dezena=scores_fib)

    scores_m3 = {d: (1.0 if d in MULTIPLOS_3 else 0.3) for d in DEZENAS}
    _salvar("multiplos_3", {"multiplos_3": sorted(MULTIPLOS_3)}, scores_dezena=scores_m3)

    scores_comp = {d: round((scores_primos[d] + scores_fib[d] + scores_m3[d]) / 3, 4) for d in DEZENAS}
    _salvar("composicao_numerica", {"scores_combinados": scores_comp}, scores_dezena=scores_comp)

    _salvar("perfil_estrutural", {"primos": sorted(PRIMOS), "fibonacci": sorted(FIBONACCI),
                                   "multiplos_3": sorted(MULTIPLOS_3)},
            scores_dezena=scores_comp, score_agg=round(float(np.mean(list(scores_comp.values()))), 4))

    # ── Coocorrência ──────────────────────────────────────────────────────────
    r = run_coocorrencia_pares(sorteios, janela)
    _salvar("coocorrencia_pares", r, scores_dezena=r["scores_dezena"])

    r = run_trios_recorrentes(sorteios, janela)
    _salvar("trios_recorrentes", r, scores_dezena=r["scores_dezena"])

    r = run_trios_expandido(sorteios, min(janela, 200))
    _salvar("trios_expandido", r, scores_dezena=r["scores_dezena"])

    # ── Coocorrência temporal ──────────────────────────────────────────────────
    r_ct = run_coocorrencia_pares(sorteios, min(janela, 100))
    _salvar("coocorrencia_temporal", r_ct, scores_dezena=r_ct["scores_dezena"])

    # ── Projeção N+1/N+2 ─────────────────────────────────────────────────────
    r = run_n1_n2(sorteios, ref_idx, threshold_n12)
    _salvar("n1_n2", r, scores_dezena=r["scores_dezena"])

    # ── Markov ────────────────────────────────────────────────────────────────
    r = run_markov_simples(sorteios, ref_idx)
    _salvar("markov_simples", r, scores_dezena=r["scores_dezena"])

    # ── Entropia ─────────────────────────────────────────────────────────────
    r = run_entropia_shannon(sorteios, janela)
    _salvar("entropia_shannon", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_entropia_condicional(sorteios, min(janela, 200))
    _salvar("entropia_condicional", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_informacao_mutua(sorteios, min(janela, 200))
    _salvar("informacao_mutua", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── Testes estatísticos ───────────────────────────────────────────────────
    r = run_qui_quadrado(sorteios, janela)
    _salvar("qui_quadrado", r, score_agg=r["score_agg"])

    r = run_ks_test(sorteios, janela)
    _salvar("ks_test", r, score_agg=r["score_agg"])

    r = run_ljung_box(sorteios, min(janela, 200))
    _salvar("ljung_box", r, score_agg=r["score_agg"])

    r = run_autocorrelacao(sorteios, min(janela, 200))
    _salvar("autocorrelacao", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_skewness_curtose(sorteios, janela)
    _salvar("skewness_curtose", r, score_agg=r["score_agg"])

    r = run_pearson(sorteios, min(janela, 200))
    _salvar("pearson", r, scores_dezena=r["scores_dezena"])

    r = run_anova(sorteios, janela)
    _salvar("anova", r, score_agg=r["score_agg"])

    r = run_teste_t(sorteios, janela)
    _salvar("teste_t", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_bootstrap(sorteios, janela, n_boot=300)
    _salvar("bootstrap", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_permutation_test(sorteios, janela, n_perm=300)
    _salvar("permutation_test", r, score_agg=r["score_agg"])

    r = run_cusum(sorteios, min(janela, 200))
    _salvar("cusum", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_kl_divergencia(sorteios, janela)
    _salvar("kl_divergencia", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── Tendência ─────────────────────────────────────────────────────────────
    r = run_tendencia_rolling(sorteios)
    _salvar("tendencia_rolling", r, scores_dezena=r["scores_dezena"])

    r = run_frequencia_relativa(sorteios, scores_termicos, scores_atraso, janela)
    _salvar("frequencia_relativa", r, scores_dezena=r["scores_dezena"])

    # ── Bayesiana ─────────────────────────────────────────────────────────────
    r = run_bayesiana(sorteios, scores_termicos, janela)
    _salvar("bayesiana", r, scores_dezena=r["scores_dezena"])

    # ── Espectral & Fourier ───────────────────────────────────────────────────
    r = run_espectral(sorteios, min(janela, 200))
    _salvar("espectral", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_fourier(sorteios, min(janela, 200))
    _salvar("fourier", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── Correlações ───────────────────────────────────────────────────────────
    r = run_correlacao_cruzada(sorteios, min(janela, 200))
    _salvar("correlacao_cruzada", r, scores_dezena=r["scores_dezena"])

    # ── Resíduo ───────────────────────────────────────────────────────────────
    r = run_residuo(sorteios, janela)
    _salvar("residuo", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    r = run_residuo_modelo_nulo(sorteios, janela)
    _salvar("residuo_modelo_nulo", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── Financeiro ────────────────────────────────────────────────────────────
    r = run_valor_esperado()
    _salvar("valor_esperado", r, score_agg=r["score_agg"])

    r = run_kelly()
    _salvar("kelly", r, score_agg=r["score_agg"])

    r = run_ev_rateio()
    _salvar("ev_rateio", r, score_agg=r["score_agg"])

    r = run_ev_ajustado_rateio()
    _salvar("ev_ajustado_rateio", r, score_agg=r["razao_custo"])

    # ── Lift ─────────────────────────────────────────────────────────────────
    r = run_lift(sorteios, janela)
    _salvar("lift", r, scores_dezena=r["scores_dezena"])

    r = run_lift_par(sorteios, janela)
    _salvar("lift_par", r, scores_dezena=r["scores_dezena"])

    # ── Score ponderado parcial (para Monte Carlo e backtesting) ──────────────
    sc_parcial = executor.score_consolidado()

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    r = run_monte_carlo(sorteios, sc_parcial, n=2000)
    _salvar("monte_carlo", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── Backtesting ───────────────────────────────────────────────────────────
    r = run_backtesting(sorteios, sc_parcial, ultimos=min(200, len(sorteios)//2))
    _salvar("backtesting", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── ROC/AUC ───────────────────────────────────────────────────────────────
    r = run_roc_auc(sorteios, sc_parcial, min(janela, 200))
    _salvar("roc_auc", r, score_agg=r["score_agg"])

    # ── Holdout & Rolling ────────────────────────────────────────────────────
    r = run_holdout(sorteios, sc_parcial)
    _salvar("holdout", r, score_agg=r["score_agg"])

    r = run_rolling_expandido(sorteios, sc_parcial)
    _salvar("rolling_expandido", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    # ── Score ponderado final ─────────────────────────────────────────────────
    r = run_score_ponderado(executor)
    _salvar("score_ponderado", r, scores_dezena=r["scores_dezena"], score_agg=r["score_agg"])

    executor._log("[ESTATISTICAS] Todas as etapas estatísticas concluídas.")
    return resultados
