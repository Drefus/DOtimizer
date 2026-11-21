# -*- coding: utf-8 -*-
"""
===============================================================================
            OTIMIZADOR DE CARTEIRA DE AÇÕES DA B3
       Teoria Moderna de Portfólio (Markowitz) + CAPM
===============================================================================

Disciplina : Administração Financeira
Tema       : Gestão de Investimentos
Integrantes: João Paulo Cardoso dos Santos - 2023001778
             Henrique Castro Caetano - 2023104941

OBJETIVO
--------
Construir, a partir de dados REAIS de mercado, carteiras de ações da B3 que
sejam "ótimas" segundo a Teoria Moderna de Portfólio de Harry Markowitz (1952)
e avaliar cada ação à luz do CAPM (Capital Asset Pricing Model, Sharpe 1964).

O programa:
  1. Baixa preços ajustados das ações (yfinance) e do Ibovespa (carteira de
     mercado), além da taxa Selic do Banco Central (taxa livre de risco).
  2. Calcula retorno, risco (volatilidade), covariância e correlação.
  3. Simula 10.000+ carteiras aleatórias (Monte Carlo) e otimiza
     numericamente (scipy) a Carteira de Mínima Variância e a Carteira de
     Máximo Índice de Sharpe (carteira tangente), traçando a fronteira
     eficiente.
  4. Calcula o beta de cada ação, o retorno exigido pelo CAPM e o alfa de
     Jensen (se a ação está acima ou abaixo da Linha do Mercado de Títulos).
  5. Sugere quanto investir, em reais e em quantidade de ações.
  6. Gera relatório em PDF, gráficos em PNG, planilha Excel e um resumo no
     terminal, tudo dentro da pasta "relatorios".

INSTRUÇÕES DE EXECUÇÃO
----------------------
  1. Instale o Python 3.10 ou superior.
  2. Instale as dependências:      pip install -r requirements.txt
  3. Execute:                      python otimizador_carteira.py
  4. Responda às perguntas no terminal (ou apenas pressione Enter para usar
     os valores padrão).
  5. Os resultados ficam na pasta "relatorios" (PDF, PNGs e XLSX).

  É necessário estar conectado à internet para baixar os dados.

AVISO: este programa tem finalidade exclusivamente didática. Retornos passados
não garantem retornos futuros e nada aqui constitui recomendação de
investimento.
===============================================================================
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================
import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# O backend "Agg" desenha os gráficos direto em arquivo, sem abrir janelas.
# Assim o programa roda do início ao fim sem precisar que alguém feche
# janelas de gráfico.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize

# Bibliotecas de captura de dados. Importamos com proteção para dar uma
# mensagem amigável caso o professor ainda não tenha instalado algo.
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    print("ERRO: biblioteca 'yfinance' não instalada. Rode: pip install -r requirements.txt")
    sys.exit(1)

try:
    from bcb import sgs
    BCB_DISPONIVEL = True
except ImportError:  # pragma: no cover
    BCB_DISPONIVEL = False

try:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
except ImportError:  # pragma: no cover
    print("ERRO: biblioteca 'fpdf2' não instalada. Rode: pip install -r requirements.txt")
    sys.exit(1)

warnings.filterwarnings("ignore")  # Esconde avisos internos das bibliotecas.
# O yfinance imprime mensagens técnicas em inglês para tickers inválidos;
# silenciamos e exibimos nossos próprios avisos, em português.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Garante que acentos apareçam corretamente no terminal do Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =============================================================================
# CONSTANTES E PARÂMETROS PADRÃO
# =============================================================================

# Número de pregões (dias úteis de negociação) em um ano na B3.
# É a convenção de mercado no Brasil: usamos 252 para ANUALIZAR medidas
# diárias. Retornos diários são pequenos e difíceis de comparar com taxas
# como a Selic, que são anuais; por isso convertemos tudo para base anual.
DIAS_UTEIS_ANO = 252

TICKERS_PADRAO = ["PETR4", "ITUB4", "VALE3", "WEGE3", "ABEV3"]
VALOR_PADRAO = 10_000.00
ANOS_PADRAO = 3  # período padrão: últimos 3 anos

# Índice usado como "carteira de mercado" no CAPM.
TICKER_MERCADO = "^BVSP"

# Série SGS do Banco Central nº 11 = "Taxa de juros - Selic" (% ao DIA).
# É a taxa efetivamente praticada no overnight, diariamente. Vamos compô-la
# ao longo do período para obter a taxa anual efetiva (base 252).
SERIE_SELIC_DIARIA = 11

# Valor usado se a API do Banco Central estiver indisponível.
SELIC_PADRAO = 0.1075  # 10,75% a.a.

# Número de carteiras aleatórias na simulação de Monte Carlo.
N_SIMULACOES = 10_000

# Semente aleatória: garante que a simulação dê o mesmo resultado sempre
# que for executada com os mesmos dados (reprodutibilidade).
SEMENTE = 42

PASTA_RELATORIOS = "relatorios"


# =============================================================================
# FUNÇÕES UTILITÁRIAS DE FORMATAÇÃO
# =============================================================================
def formatar_reais(valor: float) -> str:
    """
    Formata um número no padrão monetário brasileiro.

    Parâmetros:
        valor (float): quantia em reais.
    Retorna:
        str: texto no formato "R$ 1.234,56".
    """
    texto = f"{valor:,.2f}"  # 1,234.56 (padrão americano)
    texto = texto.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


def formatar_pct(valor: float, casas: int = 2) -> str:
    """
    Formata um número decimal como percentual brasileiro.

    Parâmetros:
        valor (float): número em forma decimal (0.1234 = 12,34%).
        casas (int): casas decimais.
    Retorna:
        str: texto no formato "12,34%".
    """
    return f"{valor * 100:.{casas}f}%".replace(".", ",")


def formatar_num(valor: float, casas: int = 2) -> str:
    """
    Formata um número com vírgula decimal (padrão brasileiro).

    Parâmetros:
        valor (float): número qualquer.
        casas (int): casas decimais.
    Retorna:
        str: número formatado, ex.: "1,23".
    """
    return f"{valor:.{casas}f}".replace(".", ",")


def linha(caractere: str = "=", tamanho: int = 78) -> None:
    """Imprime uma linha separadora no terminal (apenas estética)."""
    print(caractere * tamanho)


# =============================================================================
# 1. ENTRADA DE DADOS
# =============================================================================
def ler_entrada(mensagem: str, padrao: str) -> str:
    """
    Lê um valor digitado pelo usuário, devolvendo o padrão se ele apenas
    pressionar Enter.

    Parâmetros:
        mensagem (str): pergunta exibida.
        padrao (str): valor usado quando a resposta é vazia.
    Retorna:
        str: resposta do usuário (sem espaços nas pontas) ou o padrão.
    """
    try:
        # .strip("﻿ ...") remove espaços e o caractere invisível BOM que
        # alguns terminais (ex.: PowerShell) inserem ao redirecionar a entrada.
        resposta = input(f"{mensagem} [padrão: {padrao}]: ").strip().strip("﻿").strip()
    except EOFError:
        # Acontece se a entrada for redirecionada e acabar; usamos o padrão.
        resposta = ""
    return resposta if resposta else padrao


def converter_data(texto: str) -> datetime:
    """
    Converte uma data digitada (DD/MM/AAAA ou AAAA-MM-DD) para datetime.

    Parâmetros:
        texto (str): data digitada.
    Retorna:
        datetime: data convertida.
    Lança:
        ValueError: se o formato não for reconhecido.
    """
    for formato in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(texto, formato)
        except ValueError:
            continue
    raise ValueError(f"Data inválida: '{texto}'. Use o formato DD/MM/AAAA.")


def obter_parametros_usuario() -> dict:
    """
    Coleta pelo terminal os parâmetros da análise, oferecendo valores padrão.

    Conceito: a análise de carteira depende de (i) QUAIS ativos considerar
    (o "universo de investimento"), (ii) o PERÍODO histórico usado para
    estimar retorno e risco e (iii) o CAPITAL disponível para a alocação.

    Retorna:
        dict com as chaves:
            'tickers'  -> lista de tickers com sufixo ".SA" (padrão Yahoo);
            'inicio'   -> datetime da data inicial;
            'fim'      -> datetime da data final;
            'valor'    -> float, valor a investir em reais.
    """
    linha()
    print("   OTIMIZADOR DE CARTEIRA - MARKOWITZ + CAPM (B3)")
    linha()
    print("Pressione Enter para aceitar o valor padrão de cada pergunta.\n")

    hoje = datetime.today()
    inicio_padrao = hoje - timedelta(days=365 * ANOS_PADRAO)

    # ---- Tickers ----
    texto = ler_entrada("Tickers separados por vírgula", ", ".join(TICKERS_PADRAO))
    tickers = []
    for t in texto.replace(";", ",").replace(" ", ",").split(","):
        t = t.strip().upper()
        if not t:
            continue
        # O Yahoo Finance identifica ações da B3 com o sufixo ".SA"
        # (de "São Paulo"). Acrescentamos automaticamente.
        if not t.endswith(".SA"):
            t += ".SA"
        if t not in tickers:  # evita duplicados
            tickers.append(t)
    if len(tickers) < 2:
        print("  -> É preciso ao menos 2 ativos para diversificar. Usando a lista padrão.")
        tickers = [t + ".SA" for t in TICKERS_PADRAO]

    # ---- Datas ----
    while True:
        try:
            inicio = converter_data(ler_entrada("Data inicial (DD/MM/AAAA)",
                                                inicio_padrao.strftime("%d/%m/%Y")))
            fim = converter_data(ler_entrada("Data final   (DD/MM/AAAA)",
                                             hoje.strftime("%d/%m/%Y")))
            if fim <= inicio:
                print("  -> A data final deve ser posterior à inicial. Tente novamente.")
                continue
            if (fim - inicio).days < 180:
                print("  -> Aviso: períodos curtos (< 6 meses) geram estimativas pouco confiáveis.")
            if fim > hoje:
                fim = hoje
            break
        except ValueError as erro:
            print(f"  -> {erro}")

    # ---- Valor ----
    while True:
        texto = ler_entrada("Valor a investir (R$)", f"{VALOR_PADRAO:.2f}")
        try:
            # Aceita "10000", "10.000,00" ou "10000.50"
            if "," in texto:
                texto = texto.replace(".", "").replace(",", ".")
            valor = float(texto)
            if valor <= 0:
                raise ValueError
            break
        except ValueError:
            print("  -> Valor inválido. Digite um número positivo, ex.: 10000")

    return {"tickers": tickers, "inicio": inicio, "fim": fim, "valor": valor}


# =============================================================================
# 2. CAPTURA DE DADOS
# =============================================================================
def baixar_precos(tickers: list, inicio: datetime, fim: datetime) -> tuple:
    """
    Baixa do Yahoo Finance os preços de fechamento AJUSTADOS das ações e do
    Ibovespa, tratando tickers inválidos e dados faltantes.

    Conceito: o preço AJUSTADO corrige o histórico por dividendos, JCP,
    desdobramentos e grupamentos. Sem esse ajuste, um desdobramento (ex.: 1
    ação vira 2) pareceria uma queda de 50% no preço, distorcendo o retorno.
    Usando preço ajustado, o retorno calculado representa o ganho TOTAL do
    acionista.

    Parâmetros:
        tickers (list): tickers com sufixo ".SA".
        inicio, fim (datetime): período da análise.
    Retorna:
        tuple (precos_acoes: DataFrame, precos_ibov: Series, invalidos: list)
    Encerra o programa:
        se não houver internet / dados, ou se restarem menos de 2 ativos.
    """
    print("\n[1/6] Baixando preços ajustados no Yahoo Finance...")
    todos = tickers + [TICKER_MERCADO]
    try:
        # auto_adjust=True faz a coluna "Close" já vir ajustada.
        # O parâmetro "end" do yfinance é exclusivo, por isso somamos 1 dia.
        dados = yf.download(todos, start=inicio.strftime("%Y-%m-%d"),
                            end=(fim + timedelta(days=1)).strftime("%Y-%m-%d"),
                            auto_adjust=True, progress=False, threads=True)
    except Exception as erro:
        print(f"\nERRO ao acessar o Yahoo Finance: {erro}")
        print("Verifique sua conexão com a internet e tente novamente.")
        sys.exit(1)

    if dados is None or dados.empty:
        print("\nERRO: nenhum dado foi retornado. Possíveis causas: sem conexão "
              "com a internet, Yahoo Finance fora do ar ou período sem pregões.")
        sys.exit(1)

    precos = dados["Close"] if isinstance(dados.columns, pd.MultiIndex) else dados[["Close"]]
    if isinstance(precos, pd.Series):
        precos = precos.to_frame(todos[0])

    # Remove fuso horário do índice (o Excel não aceita datas com fuso).
    if getattr(precos.index, "tz", None) is not None:
        precos.index = precos.index.tz_localize(None)

    # ---- Ibovespa (carteira de mercado) ----
    if TICKER_MERCADO not in precos.columns or precos[TICKER_MERCADO].dropna().empty:
        print("\nERRO: não foi possível baixar o Ibovespa (^BVSP), necessário para o CAPM.")
        sys.exit(1)
    ibov = precos[TICKER_MERCADO].copy()
    precos = precos.drop(columns=[TICKER_MERCADO])

    # ---- Tickers inválidos ou com muitos dados faltantes ----
    invalidos = []
    for t in tickers:
        if t not in precos.columns:
            invalidos.append(t)
            continue
        serie = precos[t]
        # Se mais de 20% dos pregões estão vazios, o ativo é descartado:
        # estimativas de risco ficariam enviesadas.
        if serie.dropna().empty or serie.isna().mean() > 0.20:
            invalidos.append(t)
    validos = [t for t in tickers if t not in invalidos]
    if invalidos:
        print(f"  -> Aviso: ativos descartados (ticker inválido ou dados insuficientes): "
              f"{', '.join(t.replace('.SA', '') for t in invalidos)}")
    if len(validos) < 2:
        print("\nERRO: restaram menos de 2 ativos válidos. Não há como montar uma carteira.")
        sys.exit(1)

    precos = precos[validos]

    # ---- Dados faltantes pontuais ----
    # Feriados diferentes ou falhas pontuais geram buracos. Repetimos o último
    # preço conhecido (forward fill), hipótese de que o preço não mudou, e
    # depois removemos datas que continuem incompletas (início da série).
    conjunto = precos.join(ibov.rename("IBOV"), how="inner").ffill().dropna()
    precos = conjunto[validos]
    ibov = conjunto["IBOV"]

    if len(precos) < 30:
        print("\nERRO: menos de 30 pregões em comum no período. Escolha um período maior.")
        sys.exit(1)

    # Nomes mais limpos (sem ".SA") para tabelas e gráficos.
    precos.columns = [c.replace(".SA", "") for c in precos.columns]
    print(f"  -> {len(precos)} pregões baixados para {len(validos)} ativos + Ibovespa "
          f"({precos.index[0]:%d/%m/%Y} a {precos.index[-1]:%d/%m/%Y}).")
    return precos, ibov, [t.replace(".SA", "") for t in invalidos]


def obter_selic(inicio: datetime, fim: datetime) -> tuple:
    """
    Obtém a taxa Selic na API SGS do Banco Central e a converte para taxa
    ANUAL efetiva, usada como TAXA LIVRE DE RISCO (Rf).

    Conceito: a taxa livre de risco é o retorno de um investimento sem risco
    de crédito relevante. No Brasil, a referência é a Selic (taxa básica de
    juros, remunera títulos públicos como o Tesouro Selic). Qualquer ação só
    vale a pena se oferecer um PRÊMIO acima dessa taxa, pois o investidor
    poderia obtê-la sem correr risco.

    Cálculo: a série SGS 11 dá a Selic em % ao DIA. Compomos as taxas diárias
    (juros compostos) ao longo do período e anualizamos em base 252:
        Rf_anual = [ Π(1 + r_dia) ] ^ (252 / n_dias) - 1
    Assim, Rf corresponde à Selic MÉDIA efetiva do período analisado, coerente
    com os retornos históricos das ações, medidos no mesmo período.

    Parâmetros:
        inicio, fim (datetime): período da análise.
    Retorna:
        tuple (rf_anual: float, fonte: str) — fonte descreve a origem do dado.
    """
    print("\n[2/6] Consultando a taxa Selic na API do Banco Central (SGS série 11)...")
    if not BCB_DISPONIVEL:
        print(f"  -> Biblioteca python-bcb não instalada. Usando Selic padrão de "
              f"{formatar_pct(SELIC_PADRAO)} a.a.")
        return SELIC_PADRAO, "Valor padrão (python-bcb indisponível)"
    try:
        # A API do BCB limita consultas de séries diárias a 10 anos; por isso
        # buscamos em blocos de até 5 anos e juntamos.
        partes = []
        ini_bloco = inicio
        while ini_bloco <= fim:
            fim_bloco = min(ini_bloco + timedelta(days=365 * 5), fim)
            partes.append(sgs.get({"selic": SERIE_SELIC_DIARIA},
                                  start=ini_bloco.strftime("%Y-%m-%d"),
                                  end=fim_bloco.strftime("%Y-%m-%d")))
            ini_bloco = fim_bloco + timedelta(days=1)
        selic = pd.concat(partes)["selic"].dropna()
        selic = selic[~selic.index.duplicated()]
        if selic.empty:
            raise ValueError("série vazia")
        taxa_diaria = selic / 100.0                 # de % para decimal
        fator = (1 + taxa_diaria).prod()            # capitalização composta
        rf = fator ** (DIAS_UTEIS_ANO / len(taxa_diaria)) - 1
        ultima = (1 + taxa_diaria.iloc[-1]) ** DIAS_UTEIS_ANO - 1
        print(f"  -> Selic média efetiva do período: {formatar_pct(rf)} a.a. "
              f"(última: {formatar_pct(ultima)} a.a.)")
        return rf, "API do Banco Central - SGS série 11 (média do período)"
    except Exception as erro:
        print(f"  -> Aviso: falha ao consultar o BCB ({erro}).")
        print(f"  -> Usando Selic padrão de {formatar_pct(SELIC_PADRAO)} a.a.")
        return SELIC_PADRAO, "Valor padrão (falha na API do BCB)"


# =============================================================================
# 3. CÁLCULOS - MARKOWITZ
# =============================================================================
def calcular_retornos(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os retornos diários simples de cada ativo.

    Conceito: o retorno simples do dia t é  R_t = P_t / P_{t-1} - 1.
    Trabalhamos com RETORNOS (e não preços) porque eles são comparáveis entre
    ativos de preços diferentes (uma ação de R$ 5 e outra de R$ 50) e porque
    o retorno de uma carteira é a média ponderada dos retornos simples dos
    ativos, propriedade essencial ao modelo de Markowitz.

    Parâmetros:
        precos (DataFrame): preços ajustados (linhas = datas, colunas = ativos).
    Retorna:
        DataFrame: retornos diários (a primeira data é descartada).
    """
    return precos.pct_change().dropna()


def calcular_indicadores(retornos: pd.DataFrame, precos: pd.DataFrame, rf: float) -> pd.DataFrame:
    """
    Calcula, para cada ativo, retorno e risco anualizados e o Índice de Sharpe.

    Conceitos:
      * Retorno médio anualizado = média dos retornos diários × 252.
        É a estimativa de RETORNO ESPERADO usada por Markowitz.
      * Volatilidade anualizada = desvio padrão diário × √252.
        A volatilidade mede o RISCO: o quanto o retorno oscila em torno da
        média. Multiplicamos por √252 (e não por 252) porque, supondo
        retornos diários independentes, a VARIÂNCIA cresce linearmente com o
        tempo (σ²_anual = 252·σ²_diário), logo o desvio padrão cresce com a
        raiz quadrada do tempo.
      * Retorno acumulado e CAGR (taxa composta anual) são informativos:
        mostram o que de fato aconteceu com o dinheiro investido.
      * Índice de Sharpe = (Retorno − Rf) / Volatilidade: prêmio de retorno
        obtido por unidade de risco assumido.

    Parâmetros:
        retornos (DataFrame): retornos diários.
        precos (DataFrame): preços ajustados (para o retorno acumulado).
        rf (float): taxa livre de risco anual.
    Retorna:
        DataFrame: uma linha por ativo, com os indicadores em forma decimal.
    """
    retorno_anual = retornos.mean() * DIAS_UTEIS_ANO
    vol_anual = retornos.std() * np.sqrt(DIAS_UTEIS_ANO)
    acumulado = precos.iloc[-1] / precos.iloc[0] - 1
    anos = len(retornos) / DIAS_UTEIS_ANO
    cagr = (1 + acumulado) ** (1 / anos) - 1
    sharpe = (retorno_anual - rf) / vol_anual

    return pd.DataFrame({
        "Preço Atual (R$)": precos.iloc[-1],
        "Retorno Médio Anual": retorno_anual,
        "Volatilidade Anual": vol_anual,
        "Retorno Acumulado": acumulado,
        "CAGR": cagr,
        "Sharpe": sharpe,
    })


def calcular_covariancia_correlacao(retornos: pd.DataFrame) -> tuple:
    """
    Calcula a matriz de covariância (anualizada) e a matriz de correlação.

    Conceitos:
      * COVARIÂNCIA mede como dois ativos variam JUNTOS: positiva quando
        tendem a subir/cair ao mesmo tempo; negativa quando um tende a subir
        enquanto o outro cai. É a peça central de Markowitz: o risco de uma
        carteira não é a média dos riscos individuais, pois depende das
        covariâncias:   σ²_p = wᵀ Σ w.
        Anualizamos multiplicando por 252 (mesma lógica da variância).
      * CORRELAÇÃO é a covariância "padronizada" (entre −1 e +1):
            ρ_ij = Cov_ij / (σ_i · σ_j)
        Quanto MENOR a correlação entre os ativos, MAIOR o benefício da
        diversificação: as oscilações de um compensam parcialmente as do
        outro, reduzindo o risco total sem necessariamente reduzir o retorno.

    Parâmetros:
        retornos (DataFrame): retornos diários.
    Retorna:
        tuple (cov_anual: DataFrame, correlacao: DataFrame)
    """
    cov_anual = retornos.cov() * DIAS_UTEIS_ANO
    correlacao = retornos.corr()
    return cov_anual, correlacao


def desempenho_carteira(pesos: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple:
    """
    Calcula retorno, volatilidade e Sharpe de uma carteira.

    Conceito (Markowitz):
        Retorno esperado:  R_p = Σ w_i · μ_i  = wᵀ μ
        Variância:         σ²_p = wᵀ Σ w     (inclui todas as covariâncias)
        Sharpe:            (R_p − Rf) / σ_p

    Parâmetros:
        pesos (ndarray): fração do capital em cada ativo (soma 1).
        mu (ndarray): retornos esperados anuais.
        cov (ndarray): matriz de covariância anual.
        rf (float): taxa livre de risco anual.
    Retorna:
        tuple (retorno, volatilidade, sharpe)
    """
    ret = float(pesos @ mu)
    vol = float(np.sqrt(pesos @ cov @ pesos))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def simular_monte_carlo(mu: np.ndarray, cov: np.ndarray, rf: float,
                        n: int = N_SIMULACOES) -> pd.DataFrame:
    """
    Gera n carteiras com pesos aleatórios e calcula retorno, risco e Sharpe.

    Conceito: a simulação de Monte Carlo "explora" o conjunto de carteiras
    possíveis sorteando combinações de pesos. Ao plotar risco × retorno, a
    nuvem de pontos revela visualmente a forma da FRONTEIRA EFICIENTE (a borda
    superior esquerda da nuvem). É uma aproximação; a solução exata vem da
    otimização numérica (função otimizar_carteiras).

    Os pesos são sorteados por uma distribuição de Dirichlet, que gera
    vetores não negativos que somam 1 — exatamente as restrições do problema
    (sem venda a descoberto, 100% do capital investido).

    Parâmetros:
        mu (ndarray), cov (ndarray), rf (float): ver desempenho_carteira.
        n (int): número de carteiras simuladas.
    Retorna:
        DataFrame com colunas Retorno, Volatilidade, Sharpe e os pesos.
    """
    print(f"\n[3/6] Simulando {n:,} carteiras aleatórias (Monte Carlo)...".replace(",", "."))
    rng = np.random.default_rng(SEMENTE)
    pesos = rng.dirichlet(np.ones(len(mu)), size=n)
    retornos = pesos @ mu
    # Variância de cada carteira (wᵀΣw) calculada de forma vetorizada.
    vols = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov, pesos))
    sharpes = (retornos - rf) / vols
    return pd.DataFrame({"Retorno": retornos, "Volatilidade": vols,
                         "Sharpe": sharpes, "Pesos": list(pesos)})


def otimizar_carteiras(mu: np.ndarray, cov: np.ndarray, rf: float) -> tuple:
    """
    Encontra, por otimização numérica (scipy.optimize, método SLSQP), a
    Carteira de Mínima Variância e a Carteira de Máximo Sharpe.

    Conceitos:
      * Carteira de MÍNIMA VARIÂNCIA: a combinação de ativos com o MENOR
        risco possível. É o ponto mais à esquerda da fronteira eficiente.
      * Carteira de MÁXIMO SHARPE (carteira TANGENTE): a de maior retorno em
        excesso por unidade de risco. Geometricamente, é o ponto em que uma
        reta saindo de Rf (a Linha de Mercado de Capitais, CML) tangencia a
        fronteira eficiente. Pela teoria, todo investidor racional deveria
        combinar esta carteira com o ativo livre de risco, ajustando apenas
        a proporção conforme sua aversão a risco.

    Restrições (exigidas pelo trabalho):
      * Σ w_i = 1        -> todo o capital é investido;
      * 0 ≤ w_i ≤ 1      -> sem venda a descoberto (short) e sem alavancagem.

    Parâmetros:
        mu (ndarray), cov (ndarray), rf (float): ver desempenho_carteira.
    Retorna:
        tuple (pesos_min_var: ndarray, pesos_max_sharpe: ndarray)
    """
    print("\n[4/6] Otimizando carteiras com scipy.optimize (SLSQP)...")
    n = len(mu)
    inicial = np.repeat(1 / n, n)  # ponto de partida: pesos iguais
    limites = tuple((0.0, 1.0) for _ in range(n))
    restricoes = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    # (a) Mínima variância: minimizar σ²_p = wᵀΣw
    res_mv = minimize(lambda w: w @ cov @ w, inicial, method="SLSQP",
                      bounds=limites, constraints=restricoes,
                      options={"maxiter": 1000, "ftol": 1e-12})

    # (b) Máximo Sharpe: o scipy só MINIMIZA, então minimizamos o Sharpe negativo.
    def sharpe_negativo(w):
        return -desempenho_carteira(w, mu, cov, rf)[2]

    res_ms = minimize(sharpe_negativo, inicial, method="SLSQP",
                      bounds=limites, constraints=restricoes,
                      options={"maxiter": 1000, "ftol": 1e-12})

    for nome, res in (("mínima variância", res_mv), ("máximo Sharpe", res_ms)):
        if not res.success:
            print(f"  -> Aviso: otimização de {nome} não convergiu totalmente ({res.message}).")

    # Limpeza numérica: zera pesos ínfimos (ex.: 1e-17) e renormaliza.
    def limpar(w):
        w = np.where(w < 1e-6, 0.0, w)
        return w / w.sum()

    return limpar(res_mv.x), limpar(res_ms.x)


def calcular_fronteira_eficiente(mu: np.ndarray, cov: np.ndarray, ret_min: float,
                                 n_pontos: int = 60) -> pd.DataFrame:
    """
    Traça a fronteira eficiente: para cada retorno-alvo, encontra a carteira
    de MENOR risco que o atinge.

    Conceito: uma carteira é EFICIENTE quando não existe outra com o mesmo
    retorno e menor risco (ou mesmo risco e maior retorno). O conjunto
    dessas carteiras forma uma curva (hipérbole no plano risco × retorno)
    que começa na carteira de mínima variância e sobe até o ativo de maior
    retorno. Carteiras abaixo dela são "ineficientes": nenhum investidor
    racional deveria escolhê-las.

    Parâmetros:
        mu (ndarray), cov (ndarray): retornos esperados e covariância anuais.
        ret_min (float): retorno da carteira de mínima variância (ponto inicial).
        n_pontos (int): quantidade de pontos na curva.
    Retorna:
        DataFrame com colunas Retorno e Volatilidade.
    """
    n = len(mu)
    limites = tuple((0.0, 1.0) for _ in range(n))
    pontos = []
    for alvo in np.linspace(ret_min, mu.max(), n_pontos):
        restricoes = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},
                      {"type": "eq", "fun": lambda w, a=alvo: w @ mu - a})
        res = minimize(lambda w: w @ cov @ w, np.repeat(1 / n, n), method="SLSQP",
                       bounds=limites, constraints=restricoes,
                       options={"maxiter": 1000})
        if res.success:
            pontos.append((alvo, float(np.sqrt(res.fun))))
    return pd.DataFrame(pontos, columns=["Retorno", "Volatilidade"])


# =============================================================================
# 4. CÁLCULOS - CAPM
# =============================================================================
def calcular_capm(retornos: pd.DataFrame, ret_ibov: pd.Series, rf: float) -> tuple:
    """
    Calcula beta, retorno exigido pelo CAPM e alfa de Jensen de cada ação.

    Conceitos:
      * BETA (β) mede o RISCO SISTEMÁTICO: a sensibilidade da ação aos
        movimentos do mercado.
              β_i = Cov(R_i, R_m) / Var(R_m)
          - β = 1  -> a ação oscila, em média, como o Ibovespa;
          - β > 1  -> ação "agressiva": amplifica os movimentos do mercado;
          - β < 1  -> ação "defensiva": oscila menos que o mercado.
        Pelo CAPM, só o risco sistemático é remunerado, pois o risco
        específico de cada empresa pode ser eliminado pela diversificação.

      * CAPM: retorno que o investidor DEVERIA exigir dado o risco β:
              E(R_i) = Rf + β_i · (R_m − Rf)
        onde (R_m − Rf) é o PRÊMIO DE RISCO DE MERCADO. Graficamente, essa
        equação é a LINHA DO MERCADO DE TÍTULOS (SML - Security Market Line).

      * ALFA DE JENSEN (α) = Retorno histórico − Retorno exigido pelo CAPM.
          - α > 0 -> a ação ficou ACIMA da SML: rendeu mais do que o risco
            sistemático justificava (desempenho superior / "subavaliada" no
            período);
          - α < 0 -> ficou ABAIXO da SML: rendeu menos do que deveria para o
            risco que carrega.

    Parâmetros:
        retornos (DataFrame): retornos diários das ações.
        ret_ibov (Series): retornos diários do Ibovespa.
        rf (float): taxa livre de risco anual.
    Retorna:
        tuple (tabela_capm: DataFrame, rm: float) — rm é o retorno anual
        do mercado (Ibovespa).
    """
    var_mercado = ret_ibov.var()
    rm = ret_ibov.mean() * DIAS_UTEIS_ANO
    linhas = {}
    for ativo in retornos.columns:
        beta = retornos[ativo].cov(ret_ibov) / var_mercado
        ret_hist = retornos[ativo].mean() * DIAS_UTEIS_ANO
        ret_capm = rf + beta * (rm - rf)
        alfa = ret_hist - ret_capm
        linhas[ativo] = {
            "Beta": beta,
            "Retorno Histórico": ret_hist,
            "Retorno CAPM": ret_capm,
            "Alfa de Jensen": alfa,
            "Posição na SML": "Acima (α > 0)" if alfa > 0 else "Abaixo (α < 0)",
        }
    return pd.DataFrame(linhas).T.astype({"Beta": float, "Retorno Histórico": float,
                                          "Retorno CAPM": float, "Alfa de Jensen": float}), rm


def beta_carteira(pesos: np.ndarray, betas: np.ndarray) -> float:
    """
    Calcula o beta de uma carteira.

    Conceito: o beta é LINEAR nos pesos, então o beta da carteira é a média
    ponderada dos betas dos ativos:   β_p = Σ w_i · β_i.

    Parâmetros:
        pesos (ndarray): pesos da carteira.
        betas (ndarray): betas individuais.
    Retorna:
        float: beta da carteira.
    """
    return float(pesos @ betas)


# =============================================================================
# 5. ALOCAÇÃO
# =============================================================================
def calcular_alocacao(pesos: np.ndarray, ativos: list, precos_atuais: pd.Series,
                      valor: float) -> pd.DataFrame:
    """
    Converte os pesos ótimos em reais e em quantidade de ações.

    Conceito prático: a teoria fornece PROPORÇÕES; na vida real compramos
    quantidades inteiras de ações (aqui consideramos o mercado fracionário,
    em que é possível comprar a partir de 1 ação). Arredondamos para baixo
    para não ultrapassar o capital; a sobra fica em caixa (podendo ir para
    o ativo livre de risco, ex.: Tesouro Selic).

    Parâmetros:
        pesos (ndarray): pesos da carteira de máximo Sharpe.
        ativos (list): nomes dos ativos.
        precos_atuais (Series): último preço de cada ativo.
        valor (float): capital total em reais.
    Retorna:
        DataFrame com Peso, Valor Ideal, Preço, Quantidade e Valor Investido.
    """
    tabela = pd.DataFrame({"Peso": pesos}, index=ativos)
    tabela["Valor Ideal (R$)"] = tabela["Peso"] * valor
    tabela["Último Preço (R$)"] = precos_atuais[ativos].values
    tabela["Quantidade"] = np.floor(tabela["Valor Ideal (R$)"] /
                                    tabela["Último Preço (R$)"]).astype(int)
    tabela["Valor Investido (R$)"] = tabela["Quantidade"] * tabela["Último Preço (R$)"]
    return tabela[tabela["Peso"] > 0].sort_values("Peso", ascending=False)


# =============================================================================
# 6. GRÁFICOS
# =============================================================================
def grafico_fronteira(sim: pd.DataFrame, fronteira: pd.DataFrame, indicadores: pd.DataFrame,
                      carteiras: dict, rf: float, caminho: str) -> None:
    """
    Gráfico (a): carteiras simuladas coloridas pelo Sharpe, fronteira
    eficiente, ativos individuais, as duas carteiras ótimas e a Linha de
    Mercado de Capitais (CML), que liga Rf à carteira tangente.

    Parâmetros:
        sim (DataFrame): resultado do Monte Carlo.
        fronteira (DataFrame): pontos da fronteira eficiente.
        indicadores (DataFrame): indicadores por ativo.
        carteiras (dict): métricas das carteiras ótimas.
        rf (float): taxa livre de risco.
        caminho (str): arquivo PNG de saída.
    Retorna:
        None (salva o arquivo).
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    pontos = ax.scatter(sim["Volatilidade"] * 100, sim["Retorno"] * 100, c=sim["Sharpe"],
                        cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(pontos, ax=ax, label="Índice de Sharpe")
    ax.plot(fronteira["Volatilidade"] * 100, fronteira["Retorno"] * 100, color="black",
            lw=2.2, label="Fronteira eficiente")

    # Linha de Mercado de Capitais (CML): Rf + Sharpe_tangente × σ
    ms = carteiras["Máximo Sharpe"]
    x_max = max(sim["Volatilidade"].max(), indicadores["Volatilidade Anual"].max()) * 100
    xs = np.linspace(0, x_max, 50)
    ax.plot(xs, rf * 100 + ms["Sharpe"] * xs, "--", color="gray", lw=1.3,
            label="Linha de Mercado de Capitais (CML)")

    for ativo, lin in indicadores.iterrows():
        ax.scatter(lin["Volatilidade Anual"] * 100, lin["Retorno Médio Anual"] * 100,
                   marker="D", color="#444444", s=45, zorder=4)
        ax.annotate(ativo, (lin["Volatilidade Anual"] * 100, lin["Retorno Médio Anual"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)

    mv = carteiras["Mínima Variância"]
    ax.scatter(mv["Volatilidade"] * 100, mv["Retorno"] * 100, marker="*", s=420,
               color="#1f77b4", edgecolor="black", zorder=5, label="Mínima variância")
    ax.scatter(ms["Volatilidade"] * 100, ms["Retorno"] * 100, marker="*", s=420,
               color="#d62728", edgecolor="black", zorder=5, label="Máximo Sharpe (tangente)")
    ax.scatter(0, rf * 100, marker="o", color="green", s=60, zorder=5, label="Rf (Selic)")

    ax.set_xlim(left=0)
    ax.set_xlabel("Volatilidade anual (%)")
    ax.set_ylabel("Retorno esperado anual (%)")
    ax.set_title("Fronteira Eficiente de Markowitz - Simulação de Monte Carlo")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def grafico_correlacao(correlacao: pd.DataFrame, caminho: str) -> None:
    """
    Gráfico (b): mapa de calor da matriz de correlação.

    Leitura: tons vermelhos = correlação alta (pouca diversificação entre o
    par); tons azuis = correlação baixa/negativa (boa diversificação).

    Parâmetros:
        correlacao (DataFrame): matriz de correlação.
        caminho (str): arquivo PNG de saída.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"label": "Correlação"}, ax=ax)
    ax.set_title("Matriz de Correlação dos Retornos Diários")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def grafico_evolucao(precos: pd.DataFrame, ibov: pd.Series, caminho: str) -> None:
    """
    Gráfico (c): evolução de R$ 1.000 investidos em cada ativo e no Ibovespa.

    Conceito: normalizar os preços para a mesma base (R$ 1.000 no primeiro
    dia) permite comparar o desempenho acumulado de ativos com preços muito
    diferentes:  valor_t = 1000 × P_t / P_0.

    Parâmetros:
        precos (DataFrame): preços ajustados das ações.
        ibov (Series): pontos do Ibovespa.
        caminho (str): arquivo PNG de saída.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    base = 1000 * precos / precos.iloc[0]
    for ativo in base.columns:
        ax.plot(base.index, base[ativo], lw=1.4, label=ativo)
    ax.plot(ibov.index, 1000 * ibov / ibov.iloc[0], color="black", lw=2.5,
            ls="--", label="Ibovespa")
    ax.axhline(1000, color="gray", lw=0.8)
    ax.set_ylabel("Valor (R$)")
    ax.set_title("Evolução de R$ 1.000 investidos em cada ativo")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def grafico_sml(capm: pd.DataFrame, rf: float, rm: float, carteira: dict, caminho: str) -> None:
    """
    Gráfico (d): Linha do Mercado de Títulos (SML) com as ações posicionadas.

    Leitura: a reta é o retorno EXIGIDO pelo CAPM para cada nível de beta.
    Ações acima da reta tiveram alfa positivo (renderam mais que o exigido);
    abaixo, alfa negativo.

    Parâmetros:
        capm (DataFrame): tabela do CAPM.
        rf (float), rm (float): taxa livre de risco e retorno do mercado.
        carteira (dict): métricas da carteira de máximo Sharpe (inclui Beta).
        caminho (str): arquivo PNG de saída.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    b_max = max(1.6, capm["Beta"].max() + 0.2)
    b_min = min(0.0, capm["Beta"].min() - 0.1)
    betas = np.linspace(b_min, b_max, 50)
    ax.plot(betas, (rf + betas * (rm - rf)) * 100, color="black", lw=2,
            label="SML: Rf + β(Rm − Rf)")

    for ativo, lin in capm.iterrows():
        cor = "#2ca02c" if lin["Alfa de Jensen"] > 0 else "#d62728"
        ax.scatter(lin["Beta"], lin["Retorno Histórico"] * 100, color=cor, s=70, zorder=4)
        # Segmento vertical mostra o tamanho do alfa
        ax.vlines(lin["Beta"], lin["Retorno CAPM"] * 100, lin["Retorno Histórico"] * 100,
                  color=cor, ls=":", lw=1)
        ax.annotate(ativo, (lin["Beta"], lin["Retorno Histórico"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)

    ax.scatter(0, rf * 100, color="green", marker="o", s=70, label="Rf (β = 0)", zorder=5)
    ax.scatter(1, rm * 100, color="black", marker="s", s=80, label="Ibovespa (β = 1)", zorder=5)
    ax.scatter(carteira["Beta"], carteira["Retorno"] * 100, marker="*", s=380, color="#ff7f0e",
               edgecolor="black", label="Carteira Máx. Sharpe", zorder=6)
    ax.scatter([], [], color="#2ca02c", label="Acima da SML (α > 0)")
    ax.scatter([], [], color="#d62728", label="Abaixo da SML (α < 0)")
    ax.set_xlabel("Beta (risco sistemático)")
    ax.set_ylabel("Retorno anual (%)")
    ax.set_title("Linha do Mercado de Títulos (SML) - CAPM")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def grafico_pizza(pesos: np.ndarray, ativos: list, caminho: str) -> None:
    """
    Gráfico (e): composição da carteira de máximo Sharpe.

    Parâmetros:
        pesos (ndarray): pesos da carteira.
        ativos (list): nomes dos ativos.
        caminho (str): arquivo PNG de saída.
    """
    serie = pd.Series(pesos, index=ativos)
    serie = serie[serie > 0.005].sort_values(ascending=False)  # oculta pesos < 0,5%
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.pie(serie.values, labels=serie.index, autopct=lambda p: f"{p:.1f}%".replace(".", ","),
           startangle=90, colors=sns.color_palette("Set2", len(serie)),
           wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 11})
    ax.set_title("Composição da Carteira de Máximo Sharpe")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


# =============================================================================
# 7. INTERPRETAÇÃO AUTOMÁTICA
# =============================================================================
def gerar_interpretacao(indicadores: pd.DataFrame, correlacao: pd.DataFrame,
                        capm: pd.DataFrame, carteiras: dict, pesos_ms: np.ndarray,
                        rf: float, rm: float, vol_ibov: float) -> list:
    """
    Produz parágrafos em linguagem natural interpretando os resultados.

    Parâmetros:
        indicadores, correlacao, capm (DataFrame): resultados das análises.
        carteiras (dict): métricas das carteiras ótimas.
        pesos_ms (ndarray): pesos da carteira de máximo Sharpe.
        rf, rm (float): Selic e retorno do Ibovespa (anuais).
        vol_ibov (float): volatilidade anual do Ibovespa.
    Retorna:
        list[str]: parágrafos de interpretação.
    """
    textos = []
    ms, mv = carteiras["Máximo Sharpe"], carteiras["Mínima Variância"]

    melhor = indicadores["Sharpe"].idxmax()
    pior = indicadores["Sharpe"].idxmin()
    textos.append(
        f"Desempenho individual: {melhor} apresentou o melhor Índice de Sharpe "
        f"({formatar_num(indicadores.loc[melhor, 'Sharpe'])}), ou seja, a melhor relação "
        f"entre prêmio de retorno e risco no período. {pior} teve o pior Sharpe "
        f"({formatar_num(indicadores.loc[pior, 'Sharpe'])}). Sharpe negativo indica que o "
        f"ativo rendeu menos do que a Selic ({formatar_pct(rf)} a.a.): o investidor teria "
        f"ganhado mais sem correr risco."
    )

    # Par mais e menos correlacionado (ignorando a diagonal).
    corr = correlacao.where(~np.eye(len(correlacao), dtype=bool)).stack()
    par_max, par_min = corr.idxmax(), corr.idxmin()
    textos.append(
        f"Diversificação: o par mais correlacionado é {par_max[0]}/{par_max[1]} "
        f"(ρ = {formatar_num(corr.max())}) e o menos correlacionado é {par_min[0]}/{par_min[1]} "
        f"(ρ = {formatar_num(corr.min())}). Correlações abaixo de 1 permitem reduzir o risco: "
        f"a carteira de mínima variância tem volatilidade de {formatar_pct(mv['Volatilidade'])}, "
        f"contra {formatar_pct(indicadores['Volatilidade Anual'].mean())} da média dos ativos "
        f"isolados."
    )

    principais = sorted(zip(indicadores.index, pesos_ms), key=lambda x: -x[1])
    composicao = ", ".join(f"{a} ({formatar_pct(p, 1)})" for a, p in principais if p > 0.005)
    textos.append(
        f"Carteira ótima: a carteira de máximo Sharpe é composta por {composicao}. Ela tem "
        f"retorno esperado de {formatar_pct(ms['Retorno'])} a.a., volatilidade de "
        f"{formatar_pct(ms['Volatilidade'])} e Sharpe de {formatar_num(ms['Sharpe'])}. "
        f"O Ibovespa, no mesmo período, teve retorno de {formatar_pct(rm)} a.a., volatilidade "
        f"de {formatar_pct(vol_ibov)} e Sharpe de {formatar_num((rm - rf) / vol_ibov)}."
    )

    b = ms["Beta"]
    perfil = ("agressivo (amplifica os movimentos do mercado)" if b > 1.05 else
              "defensivo (oscila menos que o mercado)" if b < 0.95 else
              "neutro (acompanha o mercado)")
    textos.append(
        f"Risco sistemático: o beta da carteira de máximo Sharpe é {formatar_num(b)}, "
        f"perfil {perfil}. Se o Ibovespa variar 10%, espera-se que a carteira varie, em "
        f"média, cerca de {formatar_num(10 * b, 1)}%."
    )

    acima = capm[capm["Alfa de Jensen"] > 0].index.tolist()
    abaixo = capm[capm["Alfa de Jensen"] <= 0].index.tolist()
    textos.append(
        f"CAPM: ficaram acima da Linha do Mercado de Títulos (alfa de Jensen positivo) "
        f"{', '.join(acima) if acima else 'nenhuma ação'}; ficaram abaixo "
        f"{', '.join(abaixo) if abaixo else 'nenhuma ação'}. O prêmio de risco de mercado "
        f"(Rm - Rf) foi de {formatar_pct(rm - rf)} a.a."
        + (" Como o Ibovespa rendeu menos que a Selic, a SML ficou negativamente inclinada "
           "no período, situação atípica que reduz o retorno exigido para ações de beta alto."
           if rm < rf else "")
    )

    textos.append(
        "Limitações: os resultados usam médias e covariâncias HISTÓRICAS como estimativas do "
        "futuro, o que é uma hipótese forte; a otimização é sensível a essas estimativas e "
        "tende a concentrar a carteira nos ativos que mais subiram no passado. Custos de "
        "transação e impostos não foram considerados. Uso estritamente didático."
    )
    return textos


# =============================================================================
# 8. RELATÓRIOS (PDF, EXCEL, TERMINAL)
# =============================================================================
def texto_pdf(texto: str) -> str:
    """
    Adapta um texto às fontes padrão do PDF (Helvetica, codificação latin-1),
    trocando símbolos que elas não suportam (α, β, ρ, −, ...).

    Parâmetros:
        texto (str): texto original.
    Retorna:
        str: texto compatível com latin-1.
    """
    trocas = {"α": "alfa", "β": "beta", "ρ": "rho", "σ": "sigma", "−": "-", "–": "-",
              "—": "-", "≥": ">=", "≤": "<=", "×": "x", "Σ": "Soma", "√": "raiz", "²": "2",
              "“": '"', "”": '"', "’": "'"}
    for antigo, novo in trocas.items():
        texto = texto.replace(antigo, novo)
    return texto.encode("latin-1", "replace").decode("latin-1")


class RelatorioPDF(FPDF):
    """Classe de PDF com cabeçalho e rodapé personalizados (fpdf2)."""

    def header(self):
        """Cabeçalho repetido em todas as páginas."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(110, 110, 110)
        self.cell(0, 6, texto_pdf("Otimizador de Carteira B3 - Markowitz + CAPM"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        """Rodapé com número da página."""
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, texto_pdf(f"Página {self.page_no()}/{{nb}} - Uso didático, não é "
                                  f"recomendação de investimento"), align="C")

    def titulo_secao(self, texto: str) -> None:
        """Escreve o título de uma seção com faixa colorida."""
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(30, 60, 114)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, texto_pdf("  " + texto), fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def paragrafo(self, texto: str, tamanho: int = 10) -> None:
        """Escreve um parágrafo com quebra automática de linha."""
        self.set_font("Helvetica", "", tamanho)
        self.multi_cell(0, 5, texto_pdf(texto), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def tabela(self, cabecalho: list, linhas: list, larguras: list) -> None:
        """
        Desenha uma tabela simples.

        Parâmetros:
            cabecalho (list[str]): títulos das colunas.
            linhas (list[list[str]]): conteúdo já formatado.
            larguras (list[float]): largura de cada coluna em mm.
        """
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(220, 228, 240)
        for titulo, w in zip(cabecalho, larguras):
            self.cell(w, 7, texto_pdf(titulo), border=1, align="C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        for i, lin in enumerate(linhas):
            if self.get_y() > 270:  # quebra de página mantendo o cabeçalho
                self.add_page()
                self.tabela(cabecalho, [], larguras)
                self.set_font("Helvetica", "", 8.5)
            self.set_fill_color(245, 247, 250)
            for j, (valor, w) in enumerate(zip(lin, larguras)):
                self.cell(w, 6, texto_pdf(str(valor)), border=1,
                          align="L" if j == 0 else "C", fill=(i % 2 == 1))
            self.ln()
        self.ln(2)

    def imagem(self, caminho: str, legenda: str, largura: float = 180) -> None:
        """Insere uma imagem centralizada com legenda, quebrando a página se preciso."""
        from PIL import Image  # Pillow já é dependência do fpdf2 e do matplotlib
        with Image.open(caminho) as img:
            altura = largura * img.height / img.width
        if self.get_y() + altura + 10 > 280:
            self.add_page()
        self.image(caminho, x=(210 - largura) / 2, w=largura)
        self.set_font("Helvetica", "I", 8.5)
        self.cell(0, 5, texto_pdf(legenda), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)


def gerar_pdf(caminho: str, params: dict, rf: float, fonte_rf: str, rm: float,
              indicadores: pd.DataFrame, carteiras: dict, pesos: dict,
              alocacao: pd.DataFrame, capm: pd.DataFrame, interpretacao: list,
              graficos: dict, invalidos: list, n_pregoes: int) -> None:
    """
    Gera o relatório completo em PDF com fpdf2.

    Parâmetros:
        caminho (str): arquivo PDF de saída.
        params (dict): parâmetros da análise (tickers, datas, valor).
        rf, rm (float): Selic e retorno do Ibovespa (anuais).
        fonte_rf (str): origem do valor da Selic.
        indicadores, alocacao, capm (DataFrame): tabelas de resultado.
        carteiras (dict): métricas das carteiras ótimas.
        pesos (dict): pesos das carteiras ótimas (nome -> ndarray).
        interpretacao (list[str]): parágrafos de interpretação.
        graficos (dict): caminhos dos PNGs.
        invalidos (list): tickers descartados.
        n_pregoes (int): número de pregões analisados.
    """
    pdf = RelatorioPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Capa / parâmetros ----
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, texto_pdf("Relatório de Otimização de Carteira"), align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, texto_pdf("Teoria Moderna de Portfólio (Markowitz) e CAPM - Ações da B3"),
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, texto_pdf("Administração Financeira"), align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for integrante in ("João Paulo Cardoso dos Santos - 2023001778",
                       "Henrique Castro Caetano - 2023104941"):
        pdf.cell(0, 6, texto_pdf(integrante), align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.titulo_secao("1. Parâmetros da análise")
    ativos_txt = ", ".join(indicadores.index)
    linhas_param = [
        ["Data de geração", datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
        ["Ativos analisados", ativos_txt],
        ["Ativos descartados", ", ".join(invalidos) if invalidos else "Nenhum"],
        ["Período", f"{params['inicio']:%d/%m/%Y} a {params['fim']:%d/%m/%Y} ({n_pregoes} pregões)"],
        ["Valor a investir", formatar_reais(params["valor"])],
        ["Taxa livre de risco (Selic)", f"{formatar_pct(rf)} a.a. - {fonte_rf}"],
        ["Carteira de mercado", f"Ibovespa (^BVSP) - retorno de {formatar_pct(rm)} a.a."],
        ["Carteiras simuladas", f"{N_SIMULACOES:,}".replace(",", ".") + " (Monte Carlo)"],
        ["Fonte dos preços", "Yahoo Finance (yfinance) - fechamento ajustado"],
    ]
    pdf.tabela(["Parâmetro", "Valor"], linhas_param, [55, 135])

    # ---- Indicadores ----
    pdf.titulo_secao("2. Indicadores por ativo (anualizados, base 252 pregões)")
    linhas_ind = [[a, formatar_reais(l["Preço Atual (R$)"]), formatar_pct(l["Retorno Médio Anual"]),
                   formatar_pct(l["Volatilidade Anual"]), formatar_pct(l["Retorno Acumulado"]),
                   formatar_pct(l["CAGR"]), formatar_num(l["Sharpe"])]
                  for a, l in indicadores.iterrows()]
    pdf.tabela(["Ativo", "Preço atual", "Ret. médio", "Volatilidade", "Ret. acum.", "CAGR",
                "Sharpe"], linhas_ind, [24, 30, 26, 28, 28, 26, 28])
    pdf.paragrafo("Retorno médio = média dos retornos diários x 252; volatilidade = desvio "
                  "padrão diário x raiz(252); Sharpe = (retorno - Rf) / volatilidade.", 8.5)

    # ---- Carteiras ótimas ----
    pdf.titulo_secao("3. Carteiras ótimas (Markowitz)")
    ativos = list(indicadores.index)
    linhas_cart = [[a, formatar_pct(pesos["Mínima Variância"][i]),
                    formatar_pct(pesos["Máximo Sharpe"][i])] for i, a in enumerate(ativos)]
    for chave, fmt in (("Retorno", formatar_pct), ("Volatilidade", formatar_pct),
                       ("Sharpe", formatar_num), ("Beta", formatar_num)):
        linhas_cart.append([f"{chave} da carteira", fmt(carteiras["Mínima Variância"][chave]),
                            fmt(carteiras["Máximo Sharpe"][chave])])
    pdf.tabela(["Ativo / Métrica", "Mínima Variância", "Máximo Sharpe"], linhas_cart,
               [70, 60, 60])

    # ---- Alocação ----
    pdf.titulo_secao("4. Alocação sugerida - Carteira de Máximo Sharpe")
    linhas_aloc = [[a, formatar_pct(l["Peso"]), formatar_reais(l["Valor Ideal (R$)"]),
                    formatar_reais(l["Último Preço (R$)"]), str(l["Quantidade"]),
                    formatar_reais(l["Valor Investido (R$)"])] for a, l in alocacao.iterrows()]
    investido = alocacao["Valor Investido (R$)"].sum()
    linhas_aloc.append(["TOTAL", "100,00%", formatar_reais(params["valor"]), "-", "-",
                        formatar_reais(investido)])
    pdf.tabela(["Ativo", "Peso", "Valor ideal", "Último preço", "Qtd.", "Valor investido"],
               linhas_aloc, [24, 24, 36, 34, 22, 50])
    pdf.paragrafo(f"Sobra em caixa (por arredondamento das quantidades): "
                  f"{formatar_reais(params['valor'] - investido)}. Sugere-se aplicá-la no "
                  f"ativo livre de risco (ex.: Tesouro Selic).", 9)

    # ---- CAPM ----
    pdf.titulo_secao("5. CAPM - Beta, retorno exigido e alfa de Jensen")
    linhas_capm = [[a, formatar_num(l["Beta"]), formatar_pct(l["Retorno Histórico"]),
                    formatar_pct(l["Retorno CAPM"]), formatar_pct(l["Alfa de Jensen"]),
                    l["Posição na SML"]] for a, l in capm.iterrows()]
    pdf.tabela(["Ativo", "Beta", "Ret. histórico", "Ret. CAPM", "Alfa de Jensen",
                "Posição na SML"], linhas_capm, [24, 22, 32, 30, 34, 48])
    pdf.paragrafo(f"Retorno CAPM = Rf + beta x (Rm - Rf), com Rf = {formatar_pct(rf)} e "
                  f"Rm = {formatar_pct(rm)}. Beta = Cov(Ri, Rm) / Var(Rm). Alfa de Jensen = "
                  f"retorno histórico - retorno CAPM.", 8.5)

    # ---- Interpretação ----
    pdf.titulo_secao("6. Interpretação automática dos resultados")
    for paragrafo in interpretacao:
        pdf.paragrafo(paragrafo)

    # ---- Gráficos ----
    pdf.add_page()
    pdf.titulo_secao("7. Gráficos")
    pdf.imagem(graficos["fronteira"], "Figura 1 - Fronteira eficiente e carteiras simuladas")
    pdf.imagem(graficos["correlacao"], "Figura 2 - Mapa de calor da correlação", 140)
    pdf.imagem(graficos["evolucao"], "Figura 3 - Evolução de R$ 1.000 investidos")
    pdf.imagem(graficos["sml"], "Figura 4 - Linha do Mercado de Títulos (SML)", 170)
    pdf.imagem(graficos["pizza"], "Figura 5 - Composição da carteira de máximo Sharpe", 130)

    pdf.output(caminho)


def gerar_excel(caminho: str, precos: pd.DataFrame, ibov: pd.Series, retornos: pd.DataFrame,
                ret_ibov: pd.Series, indicadores: pd.DataFrame, cov: pd.DataFrame,
                correlacao: pd.DataFrame, carteiras: dict, pesos: dict,
                alocacao: pd.DataFrame, capm: pd.DataFrame, params: dict, rf: float,
                rm: float) -> None:
    """
    Gera a planilha Excel (openpyxl) com as abas Precos, Retornos,
    Indicadores, Carteiras e CAPM (e as abas extras Alocacao e Parametros).

    Parâmetros: os mesmos objetos calculados ao longo do programa.
    """
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Font, PatternFill

    with pd.ExcelWriter(caminho, engine="openpyxl") as escritor:
        p = precos.copy()
        p["IBOV"] = ibov
        p.index.name = "Data"
        p.to_excel(escritor, sheet_name="Precos")

        r = retornos.copy()
        r["IBOV"] = ret_ibov
        r.index.name = "Data"
        r.to_excel(escritor, sheet_name="Retornos")

        indicadores.to_excel(escritor, sheet_name="Indicadores", index_label="Ativo")
        lin0 = len(indicadores) + 3
        pd.DataFrame(["Matriz de covariância (anualizada)"]).to_excel(
            escritor, sheet_name="Indicadores", startrow=lin0, header=False, index=False)
        cov.to_excel(escritor, sheet_name="Indicadores", startrow=lin0 + 1)
        lin1 = lin0 + len(cov) + 4
        pd.DataFrame(["Matriz de correlação"]).to_excel(
            escritor, sheet_name="Indicadores", startrow=lin1, header=False, index=False)
        correlacao.to_excel(escritor, sheet_name="Indicadores", startrow=lin1 + 1)

        cart = pd.DataFrame({nome: w for nome, w in pesos.items()}, index=indicadores.index)
        metricas = pd.DataFrame(carteiras)
        pd.concat([cart, metricas]).to_excel(escritor, sheet_name="Carteiras",
                                             index_label="Ativo / Métrica")

        capm_x = capm.copy()
        capm_x.to_excel(escritor, sheet_name="CAPM", index_label="Ativo")
        extra = pd.DataFrame({"Valor": [rf, rm, rm - rf]},
                             index=["Rf (Selic)", "Rm (Ibovespa)", "Prêmio de mercado"])
        extra.to_excel(escritor, sheet_name="CAPM", startrow=len(capm) + 3)

        alocacao.to_excel(escritor, sheet_name="Alocacao", index_label="Ativo")

        pd.DataFrame({"Valor": [", ".join(params["tickers"]),
                                params["inicio"].strftime("%d/%m/%Y"),
                                params["fim"].strftime("%d/%m/%Y"),
                                params["valor"], rf,
                                datetime.now().strftime("%d/%m/%Y %H:%M:%S")]},
                     index=["Tickers", "Data inicial", "Data final", "Valor (R$)",
                            "Selic (a.a.)", "Gerado em"]).to_excel(
            escritor, sheet_name="Parametros", index_label="Parâmetro")

        # ---- Formatação: largura de colunas, negrito no cabeçalho ----
        for aba in escritor.book.worksheets:
            nome = aba.title
            for celula in aba[1]:
                celula.font = Font(bold=True, color="FFFFFF")
                celula.fill = PatternFill("solid", fgColor="1E3C72")
            for i, coluna in enumerate(aba.columns, start=1):
                largura = max((len(str(c.value)) for c in coluna if c.value is not None),
                              default=8)
                aba.column_dimensions[get_column_letter(i)].width = min(max(largura + 2, 10), 40)
            if nome in ("Precos", "Retornos"):
                aba.freeze_panes = "B2"
                for celula in aba["A"][1:]:
                    celula.number_format = "DD/MM/YYYY"
                formato = "0.00" if nome == "Precos" else "0.0000%"
                for lin in aba.iter_rows(min_row=2, min_col=2):
                    for celula in lin:
                        celula.number_format = formato


def imprimir_resumo(params: dict, rf: float, rm: float, indicadores: pd.DataFrame,
                    correlacao: pd.DataFrame, carteiras: dict, pesos: dict,
                    alocacao: pd.DataFrame, capm: pd.DataFrame) -> None:
    """
    Imprime no terminal um resumo legível dos principais resultados.

    Parâmetros: objetos calculados ao longo do programa.
    """
    print()
    linha()
    print("   RESUMO DOS RESULTADOS")
    linha()
    print(f"Período: {params['inicio']:%d/%m/%Y} a {params['fim']:%d/%m/%Y}   |   "
          f"Selic (Rf): {formatar_pct(rf)} a.a.   |   Ibovespa (Rm): {formatar_pct(rm)} a.a.")

    print("\n>> INDICADORES POR ATIVO (anualizados)")
    t = pd.DataFrame({
        "Preço": indicadores["Preço Atual (R$)"].map(lambda v: formatar_num(v)),
        "Retorno": indicadores["Retorno Médio Anual"].map(formatar_pct),
        "Volat.": indicadores["Volatilidade Anual"].map(formatar_pct),
        "Acumul.": indicadores["Retorno Acumulado"].map(formatar_pct),
        "Sharpe": indicadores["Sharpe"].map(formatar_num),
    })
    print(t.to_string())

    print("\n>> MATRIZ DE CORRELAÇÃO")
    print(correlacao.map(formatar_num).to_string())

    print("\n>> CARTEIRAS ÓTIMAS")
    t = pd.DataFrame({n: [formatar_pct(x) for x in w] for n, w in pesos.items()},
                     index=indicadores.index)
    for chave, fmt in (("Retorno", formatar_pct), ("Volatilidade", formatar_pct),
                       ("Sharpe", formatar_num), ("Beta", formatar_num)):
        t.loc[f"[{chave}]"] = [fmt(carteiras[n][chave]) for n in pesos]
    print(t.to_string())

    print("\n>> CAPM")
    t = pd.DataFrame({
        "Beta": capm["Beta"].map(formatar_num),
        "Ret.Hist.": capm["Retorno Histórico"].map(formatar_pct),
        "Ret.CAPM": capm["Retorno CAPM"].map(formatar_pct),
        "Alfa": capm["Alfa de Jensen"].map(formatar_pct),
        "SML": capm["Posição na SML"],
    })
    print(t.to_string())

    print(f"\n>> ALOCAÇÃO DE {formatar_reais(params['valor'])} (Máximo Sharpe)")
    t = pd.DataFrame({
        "Peso": alocacao["Peso"].map(formatar_pct),
        "Valor ideal": alocacao["Valor Ideal (R$)"].map(formatar_reais),
        "Preço": alocacao["Último Preço (R$)"].map(formatar_reais),
        "Qtd": alocacao["Quantidade"],
        "Investido": alocacao["Valor Investido (R$)"].map(formatar_reais),
    })
    print(t.to_string())
    investido = alocacao["Valor Investido (R$)"].sum()
    print(f"Total investido: {formatar_reais(investido)}   |   Sobra em caixa: "
          f"{formatar_reais(params['valor'] - investido)}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================
def main() -> None:
    """
    Orquestra todas as etapas: entrada -> captura -> Markowitz -> CAPM ->
    alocação -> gráficos -> relatórios.
    """
    params = obter_parametros_usuario()

    # ---------------- Captura ----------------
    precos, ibov, invalidos = baixar_precos(params["tickers"], params["inicio"], params["fim"])
    rf, fonte_rf = obter_selic(params["inicio"], params["fim"])

    # ---------------- Markowitz ----------------
    retornos = calcular_retornos(precos)
    ret_ibov = ibov.pct_change().dropna().loc[retornos.index]
    indicadores = calcular_indicadores(retornos, precos, rf)
    cov, correlacao = calcular_covariancia_correlacao(retornos)

    ativos = list(precos.columns)
    mu = indicadores["Retorno Médio Anual"].values
    cov_np = cov.values

    simulacao = simular_monte_carlo(mu, cov_np, rf)
    w_mv, w_ms = otimizar_carteiras(mu, cov_np, rf)
    ret_mv = desempenho_carteira(w_mv, mu, cov_np, rf)[0]
    fronteira = calcular_fronteira_eficiente(mu, cov_np, ret_mv)

    # ---------------- CAPM ----------------
    print("\n[5/6] Calculando betas e CAPM em relação ao Ibovespa...")
    capm, rm = calcular_capm(retornos, ret_ibov, rf)
    vol_ibov = ret_ibov.std() * np.sqrt(DIAS_UTEIS_ANO)
    betas = capm["Beta"].values

    pesos = {"Mínima Variância": w_mv, "Máximo Sharpe": w_ms}
    carteiras = {}
    for nome, w in pesos.items():
        r, v, s = desempenho_carteira(w, mu, cov_np, rf)
        carteiras[nome] = {"Retorno": r, "Volatilidade": v, "Sharpe": s,
                           "Beta": beta_carteira(w, betas)}

    # ---------------- Alocação ----------------
    alocacao = calcular_alocacao(w_ms, ativos, indicadores["Preço Atual (R$)"], params["valor"])

    # ---------------- Relatórios ----------------
    print("\n[6/6] Gerando gráficos, PDF e Excel...")
    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    arq = lambda nome, ext: os.path.join(PASTA_RELATORIOS, f"{nome}_{carimbo}.{ext}")

    graficos = {
        "fronteira": arq("fronteira_eficiente", "png"),
        "correlacao": arq("correlacao", "png"),
        "evolucao": arq("evolucao_1000", "png"),
        "sml": arq("linha_mercado_titulos", "png"),
        "pizza": arq("pizza_max_sharpe", "png"),
    }
    grafico_fronteira(simulacao, fronteira, indicadores, carteiras, rf, graficos["fronteira"])
    grafico_correlacao(correlacao, graficos["correlacao"])
    grafico_evolucao(precos, ibov, graficos["evolucao"])
    grafico_sml(capm, rf, rm, carteiras["Máximo Sharpe"], graficos["sml"])
    grafico_pizza(w_ms, ativos, graficos["pizza"])

    interpretacao = gerar_interpretacao(indicadores, correlacao, capm, carteiras, w_ms,
                                        rf, rm, vol_ibov)

    caminho_pdf = arq("relatorio_carteira", "pdf")
    caminho_xlsx = arq("planilha_carteira", "xlsx")
    try:
        gerar_pdf(caminho_pdf, params, rf, fonte_rf, rm, indicadores, carteiras, pesos,
                  alocacao, capm, interpretacao, graficos, invalidos, len(retornos))
    except Exception as erro:
        print(f"  -> ERRO ao gerar o PDF: {erro}")
    try:
        gerar_excel(caminho_xlsx, precos, ibov, retornos, ret_ibov, indicadores, cov,
                    correlacao, carteiras, pesos, alocacao, capm, params, rf, rm)
    except PermissionError:
        print("  -> ERRO: não foi possível salvar o Excel (o arquivo está aberto?).")

    imprimir_resumo(params, rf, rm, indicadores, correlacao, carteiras, pesos, alocacao, capm)

    print("\n>> INTERPRETAÇÃO")
    for paragrafo in interpretacao:
        print(f"- {paragrafo}\n")

    linha()
    print(f"Arquivos salvos na pasta '{os.path.abspath(PASTA_RELATORIOS)}':")
    for caminho in [caminho_pdf, caminho_xlsx, *graficos.values()]:
        if os.path.exists(caminho):
            print(f"   {os.path.basename(caminho)}")
    linha()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecução interrompida pelo usuário.")
        sys.exit(130)
