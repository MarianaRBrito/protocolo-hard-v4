# =============================================================================
# engine_core.py — Protocolo Lotofácil V5
# STEP_REGISTRY + Executor em Cascata + Session State Central
# =============================================================================

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
import math

# =============================================================================
# CONSTANTES GLOBAIS
# =============================================================================
DEZENAS      = list(range(1, 26))
PRIMOS       = {2, 3, 5, 7, 11, 13, 17, 19, 23}
FIBONACCI    = {1, 2, 3, 5, 8, 13, 21}
MULTIPLOS_3  = {3, 6, 9, 12, 15, 18, 21, 24}
COLS_DEZENAS = [f"bola {i}" for i in range(1, 16)]

# =============================================================================
# STEP REGISTRY — catálogo de todas as etapas do protocolo
# Cada etapa tem:
#   id                   : identificador único
#   nome                 : nome legível
#   ordem                : posição na cascata
#   dependencias         : lista de ids que devem estar concluídos antes
#   grupo                : qual grupo de análise pertence
#   tipo                 : "estatistica" | "heuristica" | "experimental"
#   gera_score_dezena    : True se produz score por dezena
#   gera_score_agregado  : True se produz score de saúde da base
#   influencia_gerador   : True se score entra no gerador final
#   peso                 : peso relativo no score consolidado
#   experimental         : se True, desativada por padrão
#   status               : "pendente" | "executando" | "concluido" | "erro" | "desativada"
# =============================================================================

STEP_REGISTRY = [
    # ── GRUPO 1: FREQUÊNCIA BASE ──────────────────────────────────────────────
    {
        "id": "frequencia_simples",
        "nome": "Frequência Simples",
        "ordem": 1,
        "dependencias": [],
        "grupo": "frequencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    {
        "id": "frequencia_janelas",
        "nome": "Frequência por Janelas",
        "ordem": 2,
        "dependencias": ["frequencia_simples"],
        "grupo": "frequencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.2,
        "experimental": False,
    },
    # ── GRUPO 2: ATRASO & CICLO ───────────────────────────────────────────────
    {
        "id": "atraso_atual",
        "nome": "Atraso Atual",
        "ordem": 3,
        "dependencias": ["frequencia_simples"],
        "grupo": "atraso",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.5,
        "experimental": False,
    },
    {
        "id": "ciclo_medio",
        "nome": "Ciclo Médio",
        "ordem": 4,
        "dependencias": ["atraso_atual"],
        "grupo": "atraso",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.5,
        "experimental": False,
    },
    {
        "id": "percentis_atraso",
        "nome": "Percentis de Atraso",
        "ordem": 5,
        "dependencias": ["ciclo_medio"],
        "grupo": "atraso",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 1.3,
        "experimental": False,
    },
    # ── GRUPO 3: ESTRUTURAL ───────────────────────────────────────────────────
    {
        "id": "soma_amplitude",
        "nome": "Soma & Amplitude",
        "ordem": 6,
        "dependencias": ["frequencia_simples"],
        "grupo": "estrutural",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "pares_impares",
        "nome": "Paridade (Pares/Ímpares)",
        "ordem": 7,
        "dependencias": ["soma_amplitude"],
        "grupo": "estrutural",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "distribuicao_faixas",
        "nome": "Distribuição por Faixas",
        "ordem": 8,
        "dependencias": ["pares_impares"],
        "grupo": "estrutural",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "moldura_miolo",
        "nome": "Moldura & Miolo",
        "ordem": 9,
        "dependencias": ["distribuicao_faixas"],
        "grupo": "estrutural",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    # ── GRUPO 4: COMPOSIÇÃO NUMÉRICA ─────────────────────────────────────────
    {
        "id": "primos",
        "nome": "Números Primos",
        "ordem": 10,
        "dependencias": ["frequencia_simples"],
        "grupo": "composicao",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "fibonacci",
        "nome": "Sequência Fibonacci",
        "ordem": 11,
        "dependencias": ["primos"],
        "grupo": "composicao",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "multiplos_3",
        "nome": "Múltiplos de 3",
        "ordem": 12,
        "dependencias": ["fibonacci"],
        "grupo": "composicao",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.5,
        "experimental": False,
    },
    # ── GRUPO 5: SEQUÊNCIAS & GAPS ────────────────────────────────────────────
    {
        "id": "consecutivos",
        "nome": "Consecutivos",
        "ordem": 13,
        "dependencias": ["frequencia_simples"],
        "grupo": "sequencias",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "gaps",
        "nome": "Gaps entre Dezenas",
        "ordem": 14,
        "dependencias": ["consecutivos"],
        "grupo": "sequencias",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "paridade_faixa",
        "nome": "Paridade por Faixa",
        "ordem": 15,
        "dependencias": ["gaps", "distribuicao_faixas"],
        "grupo": "sequencias",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    # ── GRUPO 6: COOCORRÊNCIA ─────────────────────────────────────────────────
    {
        "id": "coocorrencia_pares",
        "nome": "Coocorrência de Pares",
        "ordem": 16,
        "dependencias": ["frequencia_janelas"],
        "grupo": "coocorrencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": False,
    },
    {
        "id": "trios_recorrentes",
        "nome": "Trios Recorrentes",
        "ordem": 17,
        "dependencias": ["coocorrencia_pares"],
        "grupo": "coocorrencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    # ── GRUPO 7: PROJEÇÃO N+1/N+2 ────────────────────────────────────────────
    {
        "id": "n1_n2",
        "nome": "Projeção N+1/N+2",
        "ordem": 18,
        "dependencias": ["coocorrencia_pares", "frequencia_janelas"],
        "grupo": "projecao",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.4,
        "experimental": False,
    },
    # ── GRUPO 8: MODELOS PROBABILÍSTICOS ─────────────────────────────────────
    {
        "id": "markov_simples",
        "nome": "Cadeia de Markov Simples",
        "ordem": 19,
        "dependencias": ["frequencia_janelas"],
        "grupo": "probabilistico",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": False,
    },
    {
        "id": "entropia_shannon",
        "nome": "Entropia de Shannon",
        "ordem": 20,
        "dependencias": ["frequencia_simples"],
        "grupo": "probabilistico",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "qui_quadrado",
        "nome": "Teste Qui-Quadrado",
        "ordem": 21,
        "dependencias": ["frequencia_simples"],
        "grupo": "validacao",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    # ── GRUPO 9: SIMULAÇÃO ────────────────────────────────────────────────────
    {
        "id": "monte_carlo",
        "nome": "Monte Carlo",
        "ordem": 22,
        "dependencias": ["percentis_atraso", "n1_n2", "markov_simples"],
        "grupo": "simulacao",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 1.2,
        "experimental": False,
    },
    {
        "id": "backtesting",
        "nome": "Backtesting Histórico",
        "ordem": 23,
        "dependencias": ["monte_carlo"],
        "grupo": "validacao",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 1.3,
        "experimental": False,
    },
    # ── GRUPO 10: TENDÊNCIA ───────────────────────────────────────────────────
    {
        "id": "tendencia_rolling",
        "nome": "Tendência Rolling",
        "ordem": 24,
        "dependencias": ["frequencia_janelas"],
        "grupo": "tendencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": False,
    },
    {
        "id": "valor_esperado",
        "nome": "Valor Esperado",
        "ordem": 25,
        "dependencias": ["backtesting"],
        "grupo": "financeiro",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    # ── GRUPO 11: FILTROS & CONSOLIDAÇÃO ─────────────────────────────────────
    {
        "id": "filtros_calibrados",
        "nome": "Filtros Auto-Calibrados",
        "ordem": 26,
        "dependencias": ["soma_amplitude", "pares_impares", "distribuicao_faixas"],
        "grupo": "filtros",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.0,  # não pontua, filtra
        "experimental": False,
    },
    {
        "id": "score_ponderado",
        "nome": "Score Ponderado Final",
        "ordem": 27,
        "dependencias": [
            "frequencia_janelas", "percentis_atraso", "n1_n2",
            "markov_simples", "coocorrencia_pares", "trios_recorrentes",
            "backtesting", "tendencia_rolling"
        ],
        "grupo": "consolidacao",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 2.0,
        "experimental": False,
    },
    # ── GRUPO 12: ANÁLISE AVANÇADA ────────────────────────────────────────────
    {
        "id": "perfil_estrutural",
        "nome": "Perfil Estrutural",
        "ordem": 28,
        "dependencias": ["moldura_miolo", "distribuicao_faixas"],
        "grupo": "estrutural",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "composicao_numerica",
        "nome": "Composição Numérica",
        "ordem": 29,
        "dependencias": ["primos", "fibonacci", "multiplos_3"],
        "grupo": "composicao",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "hamming",
        "nome": "Distância de Hamming",
        "ordem": 30,
        "dependencias": ["score_ponderado"],
        "grupo": "diversidade",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.0,  # não pontua, diversifica
        "experimental": False,
    },
    # ── GRUPO 13: TESTES ESTATÍSTICOS ─────────────────────────────────────────
    {
        "id": "bootstrap",
        "nome": "Bootstrap",
        "ordem": 31,
        "dependencias": ["frequencia_janelas"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "permutation_test",
        "nome": "Permutation Test",
        "ordem": 32,
        "dependencias": ["bootstrap"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "cusum",
        "nome": "CUSUM (Mudança de Média)",
        "ordem": 33,
        "dependencias": ["tendencia_rolling"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "ks_test",
        "nome": "Teste KS",
        "ordem": 34,
        "dependencias": ["frequencia_simples"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "ljung_box",
        "nome": "Ljung-Box (Autocorrelação)",
        "ordem": 35,
        "dependencias": ["frequencia_janelas"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "kl_divergencia",
        "nome": "Divergência KL",
        "ordem": 36,
        "dependencias": ["frequencia_simples"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "skewness_curtose",
        "nome": "Skewness & Curtose",
        "ordem": 37,
        "dependencias": ["frequencia_simples"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    # ── GRUPO 14: ANÁLISE PREDITIVA ───────────────────────────────────────────
    {
        "id": "lift",
        "nome": "Lift (Regra de Associação)",
        "ordem": 38,
        "dependencias": ["coocorrencia_pares"],
        "grupo": "preditivo",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "roc_auc",
        "nome": "ROC/AUC",
        "ordem": 39,
        "dependencias": ["backtesting"],
        "grupo": "preditivo",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "kelly",
        "nome": "Critério de Kelly",
        "ordem": 40,
        "dependencias": ["valor_esperado", "backtesting"],
        "grupo": "financeiro",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "bayesiana",
        "nome": "Atualização Bayesiana",
        "ordem": 41,
        "dependencias": ["frequencia_janelas", "tendencia_rolling"],
        "grupo": "preditivo",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    # ── GRUPO 15: CLUSTERING & COBERTURA ─────────────────────────────────────
    {
        "id": "clusters",
        "nome": "Clusters de Dezenas",
        "ordem": 42,
        "dependencias": ["coocorrencia_pares", "markov_simples"],
        "grupo": "clustering",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "hamming_minima",
        "nome": "Distância Hamming Mínima",
        "ordem": 43,
        "dependencias": ["hamming"],
        "grupo": "diversidade",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "antipadroes",
        "nome": "Antipadrões",
        "ordem": 44,
        "dependencias": ["backtesting", "moldura_miolo"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "heatmap_humano",
        "nome": "Heatmap Humano (Viés Popular)",
        "ordem": 45,
        "dependencias": ["distribuicao_faixas"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "ev_rateio",
        "nome": "EV Ajustado por Rateio",
        "ordem": 46,
        "dependencias": ["valor_esperado"],
        "grupo": "financeiro",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    # ── GRUPO 16: SÉRIES TEMPORAIS ────────────────────────────────────────────
    {
        "id": "arima",
        "nome": "ARIMA",
        "ordem": 47,
        "dependencias": ["tendencia_rolling"],
        "grupo": "series_temporais",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    # ── GRUPO 17: PADRÕES HEURÍSTICOS ─────────────────────────────────────────
    {
        "id": "amplitude_faixa",
        "nome": "Amplitude por Faixa",
        "ordem": 48,
        "dependencias": ["distribuicao_faixas"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "gaps_externos",
        "nome": "Gaps Externos (bordas)",
        "ordem": 49,
        "dependencias": ["gaps"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.5,
        "experimental": False,
    },
    {
        "id": "faixas_extremas",
        "nome": "Faixas Extremas",
        "ordem": 50,
        "dependencias": ["distribuicao_faixas"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "paridade_faixa_v2",
        "nome": "Paridade por Faixa V2",
        "ordem": 51,
        "dependencias": ["paridade_faixa"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "coocorrencia_temporal",
        "nome": "Coocorrência Temporal",
        "ordem": 52,
        "dependencias": ["coocorrencia_pares", "tendencia_rolling"],
        "grupo": "coocorrencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "atraso_faixa",
        "nome": "Atraso por Faixa",
        "ordem": 53,
        "dependencias": ["percentis_atraso", "distribuicao_faixas"],
        "grupo": "atraso",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "repeticao_dezenas",
        "nome": "Repetição de Dezenas",
        "ordem": 54,
        "dependencias": ["frequencia_janelas"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "sequencias_faixa",
        "nome": "Sequências por Faixa",
        "ordem": 55,
        "dependencias": ["consecutivos", "distribuicao_faixas"],
        "grupo": "sequencias",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "somas_por_faixa",
        "nome": "Somas por Faixa",
        "ordem": 56,
        "dependencias": ["soma_amplitude", "distribuicao_faixas"],
        "grupo": "estrutural",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "padrao_ausencia",
        "nome": "Padrão de Ausência",
        "ordem": 57,
        "dependencias": ["atraso_atual", "ciclo_medio"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "ciclo_posicao",
        "nome": "Ciclo por Posição",
        "ordem": 58,
        "dependencias": ["ciclo_medio"],
        "grupo": "atraso",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "espelhamento",
        "nome": "Espelhamento (1↔25, 2↔24…)",
        "ordem": 59,
        "dependencias": ["frequencia_janelas"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.5,
        "experimental": False,
    },
    {
        "id": "bordas",
        "nome": "Bordas (1-5, 21-25)",
        "ordem": 60,
        "dependencias": ["distribuicao_faixas"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "concentracao_quadrante",
        "nome": "Concentração por Quadrante",
        "ordem": 61,
        "dependencias": ["distribuicao_faixas", "moldura_miolo"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "frequencia_relativa",
        "nome": "Frequência Relativa Ajustada",
        "ordem": 62,
        "dependencias": ["frequencia_janelas", "tendencia_rolling"],
        "grupo": "frequencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    # ── GRUPO 18: ML CLÁSSICO ─────────────────────────────────────────────────
    {
        "id": "random_forest",
        "nome": "Random Forest",
        "ordem": 63,
        "dependencias": ["score_ponderado"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.2,
        "experimental": False,
    },
    {
        "id": "xgboost",
        "nome": "XGBoost",
        "ordem": 64,
        "dependencias": ["random_forest"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.2,
        "experimental": False,
    },
    {
        "id": "pca",
        "nome": "PCA",
        "ordem": 65,
        "dependencias": ["frequencia_janelas"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "umap",
        "nome": "UMAP",
        "ordem": 66,
        "dependencias": ["pca"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": True,
    },
    {
        "id": "hmm",
        "nome": "Hidden Markov Model",
        "ordem": 67,
        "dependencias": ["markov_simples"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    # ── GRUPO 19: ASSOCIAÇÃO AVANÇADA ─────────────────────────────────────────
    {
        "id": "trios_expandido",
        "nome": "Trios Expandido",
        "ordem": 68,
        "dependencias": ["trios_recorrentes"],
        "grupo": "coocorrencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "apriori",
        "nome": "Apriori (Regras de Associação)",
        "ordem": 69,
        "dependencias": ["coocorrencia_pares", "trios_recorrentes"],
        "grupo": "coocorrencia",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    {
        "id": "cobertura_minima",
        "nome": "Cobertura Mínima",
        "ordem": 70,
        "dependencias": ["hamming_minima"],
        "grupo": "diversidade",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.0,
        "experimental": False,
    },
    # ── GRUPO 20: VALIDAÇÃO AVANÇADA ──────────────────────────────────────────
    {
        "id": "residuo",
        "nome": "Resíduo vs Esperado",
        "ordem": 71,
        "dependencias": ["frequencia_simples", "qui_quadrado"],
        "grupo": "validacao",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "holdout",
        "nome": "Holdout Validation",
        "ordem": 72,
        "dependencias": ["backtesting"],
        "grupo": "validacao",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "rolling_expandido",
        "nome": "Rolling Expandido",
        "ordem": 73,
        "dependencias": ["tendencia_rolling", "holdout"],
        "grupo": "validacao",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "lift_par",
        "nome": "Lift por Par",
        "ordem": 74,
        "dependencias": ["lift"],
        "grupo": "preditivo",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "correlacao_cruzada",
        "nome": "Correlação Cruzada",
        "ordem": 75,
        "dependencias": ["frequencia_janelas"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "espectral",
        "nome": "Análise Espectral",
        "ordem": 76,
        "dependencias": ["tendencia_rolling"],
        "grupo": "series_temporais",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "fourier",
        "nome": "Transformada de Fourier",
        "ordem": 77,
        "dependencias": ["espectral"],
        "grupo": "series_temporais",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "autocorrelacao",
        "nome": "Autocorrelação",
        "ordem": 78,
        "dependencias": ["ljung_box"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "pearson",
        "nome": "Correlação de Pearson",
        "ordem": 79,
        "dependencias": ["frequencia_janelas"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "anova",
        "nome": "ANOVA",
        "ordem": 80,
        "dependencias": ["frequencia_janelas"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "teste_t",
        "nome": "Teste t de Student",
        "ordem": 81,
        "dependencias": ["frequencia_janelas"],
        "grupo": "testes",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    # ── GRUPO 21: ML SUPERVISIONADO ───────────────────────────────────────────
    {
        "id": "regressao_logistica",
        "nome": "Regressão Logística",
        "ordem": 82,
        "dependencias": ["score_ponderado"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": False,
    },
    {
        "id": "naive_bayes",
        "nome": "Naive Bayes",
        "ordem": 83,
        "dependencias": ["bayesiana"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "knn",
        "nome": "KNN",
        "ordem": 84,
        "dependencias": ["pca"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "svm",
        "nome": "SVM",
        "ordem": 85,
        "dependencias": ["pca"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "arvore_decisao",
        "nome": "Árvore de Decisão",
        "ordem": 86,
        "dependencias": ["random_forest"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "gradient_boosting",
        "nome": "Gradient Boosting",
        "ordem": 87,
        "dependencias": ["xgboost"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": False,
    },
    {
        "id": "adaboost",
        "nome": "AdaBoost",
        "ordem": 88,
        "dependencias": ["gradient_boosting"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "lightgbm",
        "nome": "LightGBM",
        "ordem": 89,
        "dependencias": ["xgboost"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": True,
    },
    {
        "id": "catboost",
        "nome": "CatBoost",
        "ordem": 90,
        "dependencias": ["lightgbm"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "isolation_forest",
        "nome": "Isolation Forest (Anomalias)",
        "ordem": 91,
        "dependencias": ["random_forest"],
        "grupo": "ml",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "dbscan",
        "nome": "DBSCAN Clustering",
        "ordem": 92,
        "dependencias": ["clusters"],
        "grupo": "clustering",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    # ── GRUPO 22: ANÁLISE AVANÇADA DE RESÍDUOS ────────────────────────────────
    {
        "id": "residuo_modelo_nulo",
        "nome": "Resíduo vs Modelo Nulo",
        "ordem": 93,
        "dependencias": ["residuo"],
        "grupo": "validacao",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "entropia_condicional",
        "nome": "Entropia Condicional",
        "ordem": 94,
        "dependencias": ["entropia_shannon", "markov_simples"],
        "grupo": "probabilistico",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": False,
    },
    {
        "id": "informacao_mutua",
        "nome": "Informação Mútua",
        "ordem": 95,
        "dependencias": ["entropia_condicional"],
        "grupo": "probabilistico",
        "tipo": "estatistica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": False,
    },
    {
        "id": "compressibilidade_lz",
        "nome": "Compressibilidade LZ",
        "ordem": 96,
        "dependencias": ["entropia_shannon"],
        "grupo": "probabilistico",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "complexidade_lzw",
        "nome": "Complexidade LZW",
        "ordem": 97,
        "dependencias": ["compressibilidade_lz"],
        "grupo": "probabilistico",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    # ── GRUPO 23: COBERTURA GARANTIDA ────────────────────────────────────────
    {
        "id": "cobertura_hamming_garantida",
        "nome": "Cobertura Hamming Garantida",
        "ordem": 98,
        "dependencias": ["hamming_minima", "cobertura_minima"],
        "grupo": "diversidade",
        "tipo": "heuristica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "antipadroes_populares",
        "nome": "Antipadrões Populares",
        "ordem": 99,
        "dependencias": ["antipadroes", "heatmap_humano"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": False,
    },
    {
        "id": "heatmap_concentracao",
        "nome": "Heatmap de Concentração",
        "ordem": 100,
        "dependencias": ["concentracao_quadrante"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.6,
        "experimental": False,
    },
    {
        "id": "ev_ajustado_rateio",
        "nome": "EV Ajustado por Rateio Real",
        "ordem": 101,
        "dependencias": ["ev_rateio"],
        "grupo": "financeiro",
        "tipo": "estatistica",
        "gera_score_dezena": False,
        "gera_score_agregado": True,
        "influencia_gerador": False,
        "peso": 0.0,
        "experimental": False,
    },
    {
        "id": "amplitude_media_faixa",
        "nome": "Amplitude Média por Faixa",
        "ordem": 102,
        "dependencias": ["amplitude_faixa"],
        "grupo": "heuristicas",
        "tipo": "heuristica",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.5,
        "experimental": False,
    },
    # ── GRUPO 24: MODELOS AVANÇADOS DE SÉRIES TEMPORAIS (EXPERIMENTAIS) ───────
    {
        "id": "garch",
        "nome": "GARCH (Volatilidade)",
        "ordem": 103,
        "dependencias": ["arima"],
        "grupo": "series_temporais",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": True,
    },
    {
        "id": "rnn",
        "nome": "RNN",
        "ordem": 104,
        "dependencias": ["tendencia_rolling"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "lstm",
        "nome": "LSTM",
        "ordem": 105,
        "dependencias": ["rnn"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": True,
    },
    {
        "id": "cnn_5x5",
        "nome": "CNN 5×5 (Grade Lotofácil)",
        "ordem": 106,
        "dependencias": ["moldura_miolo"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "gan",
        "nome": "GAN (Geração Adversarial)",
        "ordem": 107,
        "dependencias": ["lstm"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "fractal",
        "nome": "Dimensão Fractal",
        "ordem": 108,
        "dependencias": ["espectral"],
        "grupo": "series_temporais",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": True,
        "influencia_gerador": True,
        "peso": 0.7,
        "experimental": True,
    },
    {
        "id": "fuzzy",
        "nome": "Lógica Fuzzy",
        "ordem": 109,
        "dependencias": ["score_ponderado"],
        "grupo": "experimental",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": True,
    },
    {
        "id": "pso",
        "nome": "PSO (Particle Swarm)",
        "ordem": 110,
        "dependencias": ["score_ponderado"],
        "grupo": "experimental",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.8,
        "experimental": True,
    },
    {
        "id": "geneticos",
        "nome": "Algoritmos Genéticos",
        "ordem": 111,
        "dependencias": ["pso"],
        "grupo": "experimental",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "aprendizado_reforco",
        "nome": "Aprendizado por Reforço",
        "ordem": 112,
        "dependencias": ["geneticos"],
        "grupo": "experimental",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "deep_q_network",
        "nome": "Deep Q-Network",
        "ordem": 113,
        "dependencias": ["aprendizado_reforco"],
        "grupo": "experimental",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "transformer",
        "nome": "Transformer",
        "ordem": 114,
        "dependencias": ["lstm"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.2,
        "experimental": True,
    },
    {
        "id": "autoencoder",
        "nome": "Autoencoder",
        "ordem": 115,
        "dependencias": ["pca", "lstm"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "vae",
        "nome": "VAE (Variational Autoencoder)",
        "ordem": 116,
        "dependencias": ["autoencoder"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "attention_mechanism",
        "nome": "Mecanismo de Atenção",
        "ordem": 117,
        "dependencias": ["transformer"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "graph_neural_network",
        "nome": "Graph Neural Network",
        "ordem": 118,
        "dependencias": ["clusters", "coocorrencia_pares"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    # ── GRUPO 25: MODELOS DE FORECASTING MODERNOS ─────────────────────────────
    {
        "id": "prophet",
        "nome": "Prophet",
        "ordem": 119,
        "dependencias": ["arima"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "neural_prophet",
        "nome": "NeuralProphet",
        "ordem": 120,
        "dependencias": ["prophet"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "tcn",
        "nome": "TCN (Temporal Conv Network)",
        "ordem": 121,
        "dependencias": ["lstm"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "wavenet",
        "nome": "WaveNet",
        "ordem": 122,
        "dependencias": ["tcn"],
        "grupo": "deep_learning",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "informer",
        "nome": "Informer",
        "ordem": 123,
        "dependencias": ["transformer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "patch_tst",
        "nome": "PatchTST",
        "ordem": 124,
        "dependencias": ["transformer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "times_net",
        "nome": "TimesNet",
        "ordem": 125,
        "dependencias": ["fourier"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "dlinear",
        "nome": "DLinear",
        "ordem": 126,
        "dependencias": ["tendencia_rolling"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "nlinear",
        "nome": "NLinear",
        "ordem": 127,
        "dependencias": ["dlinear"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "fits",
        "nome": "FITS",
        "ordem": 128,
        "dependencias": ["fourier"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 0.9,
        "experimental": True,
    },
    {
        "id": "time_mixer",
        "nome": "TimeMixer",
        "ordem": 129,
        "dependencias": ["times_net"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "seg_rnn",
        "nome": "SegRNN",
        "ordem": 130,
        "dependencias": ["lstm"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "crossformer",
        "nome": "Crossformer",
        "ordem": 131,
        "dependencias": ["transformer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "itransformer",
        "nome": "iTransformer",
        "ordem": 132,
        "dependencias": ["crossformer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "tide",
        "nome": "TiDE",
        "ordem": 133,
        "dependencias": ["dlinear"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "ts_mixer",
        "nome": "TSMixer",
        "ordem": 134,
        "dependencias": ["time_mixer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "mamba",
        "nome": "Mamba (SSM)",
        "ordem": 135,
        "dependencias": ["transformer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.1,
        "experimental": True,
    },
    {
        "id": "modern_tcn",
        "nome": "ModernTCN",
        "ordem": 136,
        "dependencias": ["tcn"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
    {
        "id": "softs",
        "nome": "SOFTS",
        "ordem": 137,
        "dependencias": ["mamba", "itransformer"],
        "grupo": "forecasting",
        "tipo": "experimental",
        "gera_score_dezena": True,
        "gera_score_agregado": False,
        "influencia_gerador": True,
        "peso": 1.0,
        "experimental": True,
    },
]

# =============================================================================
# ÍNDICE RÁPIDO por ID
# =============================================================================
REGISTRY_MAP = {s["id"]: s for s in STEP_REGISTRY}


# =============================================================================
# NORMALIZAÇÃO DE SCORES
# =============================================================================
def normalizar(d: dict) -> dict:
    """Normaliza um dicionário de scores para [0, 1]."""
    if not d:
        return d
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 0.5 for k in d}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}


def normalizar_serie(serie: pd.Series) -> pd.Series:
    mn, mx = serie.min(), serie.max()
    if mx == mn:
        return pd.Series(0.5, index=serie.index)
    return (serie - mn) / (mx - mn)


# =============================================================================
# EXECUTOR EM CASCATA
# =============================================================================
class PipelineExecutor:
    """
    Gerencia execução das etapas em cascata.
    - Respeita dependências
    - Salva resultados no contexto
    - Marca status de cada etapa
    - Permite execução individual ou completa
    """

    def __init__(self, context: dict):
        """
        context: dicionário compartilhado (session_state no Streamlit)
        Deve ter as chaves:
            "resultados"  : {step_id: resultado_dict}
            "status"      : {step_id: "pendente"|"executando"|"concluido"|"erro"|"desativada"}
            "scores"      : {step_id: {dezena: float}}  — scores normalizados por etapa
            "scores_agg"  : {step_id: float}             — score agregado da base
            "log"         : lista de strings
        """
        self.ctx = context
        self._garantir_estrutura()

    def _garantir_estrutura(self):
        for key, default in [
            ("resultados", {}),
            ("status", {s["id"]: "pendente" for s in STEP_REGISTRY}),
            ("scores", {}),
            ("scores_agg", {}),
            ("log", []),
        ]:
            if key not in self.ctx:
                self.ctx[key] = default

    def _log(self, msg: str):
        self.ctx["log"].append(msg)

    def status(self, step_id: str) -> str:
        return self.ctx["status"].get(step_id, "pendente")

    def concluido(self, step_id: str) -> bool:
        return self.status(step_id) == "concluido"

    def deps_ok(self, step_id: str) -> bool:
        step = REGISTRY_MAP[step_id]
        return all(self.concluido(d) for d in step["dependencias"])

    def proxima_pendente(self, incluir_experimentais: bool = False) -> str | None:
        for step in sorted(STEP_REGISTRY, key=lambda s: s["ordem"]):
            sid = step["id"]
            if step.get("experimental") and not incluir_experimentais:
                continue
            if self.status(sid) in ("pendente", "erro") and self.deps_ok(sid):
                return sid
        return None

    def etapas_disponiveis(self, incluir_experimentais: bool = False) -> list:
        """Retorna etapas cujas dependências estão satisfeitas e ainda não concluídas."""
        out = []
        for step in sorted(STEP_REGISTRY, key=lambda s: s["ordem"]):
            sid = step["id"]
            if step.get("experimental") and not incluir_experimentais:
                continue
            if self.status(sid) not in ("concluido", "desativada") and self.deps_ok(sid):
                out.append(step)
        return out

    def salvar_resultado(self, step_id: str, resultado: dict,
                         scores_dezena: dict = None, score_agg: float = None):
        """Salva resultado de uma etapa no contexto."""
        self.ctx["resultados"][step_id] = resultado
        self.ctx["status"][step_id] = "concluido"
        if scores_dezena:
            self.ctx["scores"][step_id] = normalizar(scores_dezena)
        if score_agg is not None:
            self.ctx["scores_agg"][step_id] = score_agg
        self._log(f"[OK] {step_id}")

    def marcar_erro(self, step_id: str, msg: str):
        self.ctx["status"][step_id] = "erro"
        self._log(f"[ERRO] {step_id}: {msg}")

    def desativar(self, step_id: str):
        self.ctx["status"][step_id] = "desativada"

    def resetar(self, step_id: str, cascata: bool = True):
        """Reseta uma etapa e opcionalmente todas as dependentes."""
        self.ctx["status"][step_id] = "pendente"
        self.ctx["resultados"].pop(step_id, None)
        self.ctx["scores"].pop(step_id, None)
        self.ctx["scores_agg"].pop(step_id, None)
        if cascata:
            for step in STEP_REGISTRY:
                if step_id in step["dependencias"]:
                    self.resetar(step["id"], cascata=True)

    def get_resultado(self, step_id: str) -> dict:
        return self.ctx["resultados"].get(step_id, {})

    # ── Score Consolidado ────────────────────────────────────────────────────
    def score_consolidado(self, pesos_override: dict = None) -> dict:
        """
        Combina todos os scores por dezena das etapas concluídas,
        usando os pesos definidos no REGISTRY (ou override).

        Correções anti-viés aplicadas:
        1. Floor mínimo de 0.15 — nenhuma dezena fica com score zero
        2. Equalização parcial (30%) — puxa scores para a média
           evitando que dezenas sejam sistematicamente ignoradas
        3. Penalidade para dezenas de alta frequência histórica
           que não saem nos últimos 20 concursos (ruído)
        """
        acum = {n: 0.0 for n in DEZENAS}
        peso_total = 0.0

        for step in STEP_REGISTRY:
            sid = step["id"]
            if not self.concluido(sid):
                continue
            if not step["influencia_gerador"]:
                continue
            if sid not in self.ctx["scores"]:
                continue

            peso = (pesos_override or {}).get(sid, step["peso"])
            if peso <= 0:
                continue

            scores_etapa = self.ctx["scores"][sid]
            for n in DEZENAS:
                acum[n] += scores_etapa.get(n, 0.0) * peso
            peso_total += peso

        if peso_total == 0:
            return {n: 1.0 / 25 for n in DEZENAS}

        # Score bruto médio ponderado
        raw = {n: acum[n] / peso_total for n in DEZENAS}

        # ── 1. Equalização parcial (30% puxado para a média) ──────────────────
        # Impede que dezenas fiquem com score próximo de zero apenas
        # por não aparecerem em etapas específicas
        media = sum(raw.values()) / len(raw)
        eq = {n: raw[n] * 0.70 + media * 0.30 for n in DEZENAS}

        # ── 2. Floor mínimo de 0.15 ───────────────────────────────────────────
        # Garante que toda dezena tenha ao menos 15% de chance
        # de ser sorteada no gerador — sem excluir ninguém
        floor_val = max(eq.values()) * 0.15
        floored = {n: max(eq[n], floor_val) for n in DEZENAS}

        # ── 3. Penalidade para ruído recente ──────────────────────────────────
        # Dezenas com score alto mas que não saíram nos últimos 20 concursos
        # recebem penalidade leve — reduz viés de acumulação histórica
        resultados = self.ctx.get("resultados", {})
        sorteios_rec = resultados.get("_sorteios_recentes", [])
        if sorteios_rec and len(sorteios_rec) >= 20:
            ultimos_20 = sorteios_rec[-20:]
            freq_rec = {n: sum(1 for s in ultimos_20 if n in s) / 20
                        for n in DEZENAS}
            media_sc = sum(floored.values()) / len(floored)
            for n in DEZENAS:
                # Penaliza dezenas com score > média mas freq recente < 40%
                if floored[n] > media_sc * 1.2 and freq_rec[n] < 0.40:
                    floored[n] *= 0.75

        # Normaliza final
        return normalizar(floored)

    def progresso(self, incluir_experimentais: bool = False) -> tuple:
        """Retorna (concluidas, total) para etapas não-experimentais (ou todas)."""
        steps = [s for s in STEP_REGISTRY
                 if incluir_experimentais or not s.get("experimental")]
        conc = sum(1 for s in steps if self.concluido(s["id"]))
        return conc, len(steps)

    def todas_core_ok(self) -> bool:
        """True se todas as etapas não-experimentais estão concluídas."""
        conc, total = self.progresso(incluir_experimentais=False)
        return conc == total

    # ── Relatório de Status ──────────────────────────────────────────────────
    def relatorio_status(self) -> pd.DataFrame:
        rows = []
        for step in STEP_REGISTRY:
            sid = step["id"]
            rows.append({
                "Ordem": step["ordem"],
                "ID": sid,
                "Nome": step["nome"],
                "Grupo": step["grupo"],
                "Tipo": step["tipo"],
                "Experimental": "🧪" if step.get("experimental") else "",
                "Status": self.status(sid),
                "Peso": step["peso"],
                "Score Dezena": "✅" if step["gera_score_dezena"] else "—",
                "Influencia Gerador": "✅" if step["influencia_gerador"] else "—",
            })
        return pd.DataFrame(rows)

    # ── Breakdown por Dezena ─────────────────────────────────────────────────
    def breakdown_dezena(self, dezena: int) -> pd.DataFrame:
        """Para uma dezena, mostra contribuição de cada etapa no score final."""
        rows = []
        for step in STEP_REGISTRY:
            sid = step["id"]
            if not self.concluido(sid) or sid not in self.ctx["scores"]:
                continue
            sc = self.ctx["scores"][sid].get(dezena, 0.0)
            rows.append({
                "Etapa": step["nome"],
                "Score Norm.": round(sc, 4),
                "Peso": step["peso"],
                "Contribuição": round(sc * step["peso"], 4),
                "Tipo": step["tipo"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Contribuição", ascending=False)
        return df
