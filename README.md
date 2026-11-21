# Otimizador de Carteira de Ações da B3 — Markowitz + CAPM

Trabalho da disciplina **Administração Financeira** (tema: Gestão de Investimentos).
Integrantes:
- **João Paulo Cardoso dos Santos** — 2023001778
- **Henrique Castro Caetano** — 2023104941

O programa baixa dados reais de mercado (Yahoo Finance e Banco Central) e:

- calcula retorno, volatilidade, covariância e correlação dos ativos;
- simula 10.000 carteiras (Monte Carlo) e otimiza com `scipy` as carteiras de **Mínima Variância** e de **Máximo Sharpe**, traçando a **fronteira eficiente**;
- calcula **beta**, **retorno exigido pelo CAPM** e **alfa de Jensen** de cada ação;
- sugere a alocação do capital em reais e em quantidade de ações;
- gera relatório **PDF**, gráficos **PNG** e planilha **Excel** na pasta `relatorios/`.

## Requisitos

- Python 3.10 ou superior
- Conexão com a internet

## Instalação

```bash
# (opcional) criar ambiente virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

## Execução

```bash
python otimizador_carteira.py
```

Responda às perguntas no terminal ou apenas pressione **Enter** para usar os padrões:

| Pergunta      | Padrão                              |
|---------------|-------------------------------------|
| Tickers       | PETR4, ITUB4, VALE3, WEGE3, ABEV3   |
| Data inicial  | 3 anos antes de hoje (DD/MM/AAAA)   |
| Data final    | hoje (DD/MM/AAAA)                   |
| Valor (R$)    | 10000.00                            |

O sufixo `.SA` é acrescentado automaticamente aos tickers.

## Arquivos gerados (`relatorios/`)

Todos levam data e hora no nome (ex.: `_20261121_170557`):

| Arquivo                          | Conteúdo                                                    |
|----------------------------------|-------------------------------------------------------------|
| `relatorio_carteira_*.pdf`       | Relatório completo, com tabelas, interpretação e gráficos    |
| `planilha_carteira_*.xlsx`       | Abas Precos, Retornos, Indicadores, Carteiras, CAPM (+ Alocacao, Parametros) |
| `fronteira_eficiente_*.png`      | Fronteira eficiente e carteiras simuladas                    |
| `correlacao_*.png`               | Mapa de calor da correlação                                  |
| `evolucao_1000_*.png`            | Evolução de R$ 1.000 em cada ativo e no Ibovespa             |
| `linha_mercado_titulos_*.png`    | Linha do Mercado de Títulos (SML)                            |
| `pizza_max_sharpe_*.png`         | Composição da carteira de máximo Sharpe                      |

## Fontes de dados

- **Preços:** Yahoo Finance (`yfinance`), fechamento ajustado por proventos e desdobramentos.
- **Mercado:** Ibovespa (`^BVSP`).
- **Taxa livre de risco:** Selic diária, série SGS 11 do Banco Central (`python-bcb`), composta e anualizada em base 252. Se a API falhar, usa-se 10,75% a.a. e o usuário é avisado.

## Exemplo de execução (dados reais, 21/11/2026)

```
==============================================================================
   OTIMIZADOR DE CARTEIRA - MARKOWITZ + CAPM (B3)
==============================================================================
Pressione Enter para aceitar o valor padrão de cada pergunta.

Tickers separados por vírgula [padrão: PETR4, ITUB4, VALE3, WEGE3, ABEV3]:
Data inicial (DD/MM/AAAA) [padrão: 22/11/2023]:
Data final   (DD/MM/AAAA) [padrão: 21/11/2026]:
Valor a investir (R$) [padrão: 10000.00]:

[1/6] Baixando preços ajustados no Yahoo Finance...
  -> 748 pregões baixados para 5 ativos + Ibovespa (22/11/2023 a 19/11/2026).

[2/6] Consultando a taxa Selic na API do Banco Central (SGS série 11)...
  -> Selic média efetiva do período: 13,04% a.a. (última: 13,65% a.a.)

[3/6] Simulando 10.000 carteiras aleatórias (Monte Carlo)...

[4/6] Otimizando carteiras com scipy.optimize (SLSQP)...

[5/6] Calculando betas e CAPM em relação ao Ibovespa...

[6/6] Gerando gráficos, PDF e Excel...

==============================================================================
   RESUMO DOS RESULTADOS
==============================================================================
Período: 22/11/2023 a 21/11/2026   |   Selic (Rf): 13,04% a.a.   |   Ibovespa (Rm): 11,01% a.a.

>> INDICADORES POR ATIVO (anualizados)
       Preço Retorno  Volat. Acumul. Sharpe
PETR4  42,67  22,68%  26,69%  76,14%   0,36
ITUB4  40,07  23,51%  23,73%  84,58%   0,44
VALE3  58,48   2,83%  24,76%  -0,68%  -0,41
WEGE3  51,13  20,58%  29,04%  62,37%   0,26
ABEV3  16,08  14,63%  23,34%  42,46%   0,07

>> MATRIZ DE CORRELAÇÃO
       PETR4  ITUB4  VALE3  WEGE3  ABEV3
PETR4   1,00   0,18   0,09   0,02   0,08
ITUB4   0,18   1,00   0,21   0,18   0,40
VALE3   0,09   0,21   1,00   0,21   0,17
WEGE3   0,02   0,18   0,21   1,00   0,10
ABEV3   0,08   0,40   0,17   0,10   1,00

>> CARTEIRAS ÓTIMAS
               Mínima Variância Máximo Sharpe
PETR4                    23,19%        33,75%
ITUB4                    14,51%        46,00%
VALE3                    21,07%         0,00%
WEGE3                    16,64%        20,25%
ABEV3                    24,59%         0,00%
[Retorno]                16,29%        22,64%
[Volatilidade]           14,47%        17,21%
[Sharpe]                   0,22          0,56
[Beta]                     0,65          0,80

>> CAPM
       Beta Ret.Hist. Ret.CAPM    Alfa             SML
PETR4  0,63    22,68%   11,77%  10,91%   Acima (α > 0)
ITUB4  1,09    23,51%   10,84%  12,67%   Acima (α > 0)
VALE3  0,56     2,83%   11,90%  -9,07%  Abaixo (α < 0)
WEGE3  0,43    20,58%   12,17%   8,42%   Acima (α > 0)
ABEV3  0,63    14,63%   11,76%   2,87%   Acima (α > 0)

>> ALOCAÇÃO DE R$ 10.000,00 (Máximo Sharpe)
         Peso  Valor ideal     Preço  Qtd    Investido
ITUB4  46,00%  R$ 4.599,61  R$ 40,07  114  R$ 4.567,98
PETR4  33,75%  R$ 3.375,22  R$ 42,67   79  R$ 3.370,93
WEGE3  20,25%  R$ 2.025,17  R$ 51,13   39  R$ 1.994,07
Total investido: R$ 9.932,98   |   Sobra em caixa: R$ 67,02

>> INTERPRETAÇÃO
- Desempenho individual: ITUB4 apresentou o melhor Índice de Sharpe (0,44) ...
- Diversificação: o par mais correlacionado é ITUB4/ABEV3 (ρ = 0,40) ...
- Carteira ótima: a carteira de máximo Sharpe é composta por ITUB4 (46,0%), PETR4 (33,8%), WEGE3 (20,3%) ...
- Risco sistemático: o beta da carteira de máximo Sharpe é 0,80, perfil defensivo ...
- CAPM: ficaram acima da Linha do Mercado de Títulos ... PETR4, ITUB4, WEGE3, ABEV3; ficaram abaixo VALE3 ...
- Limitações: os resultados usam médias e covariâncias HISTÓRICAS ...

==============================================================================
Arquivos salvos na pasta '...\relatorios':
   relatorio_carteira_20261121_170557.pdf
   planilha_carteira_20261121_170557.xlsx
   fronteira_eficiente_20261121_170557.png
   correlacao_20261121_170557.png
   evolucao_1000_20261121_170557.png
   linha_mercado_titulos_20261121_170557.png
   pizza_max_sharpe_20261121_170557.png
==============================================================================
```

> Os números mudam a cada execução, porque os dados de mercado são baixados na hora.

## Aviso

Projeto exclusivamente didático. Retornos passados não garantem retornos futuros, e nada aqui é recomendação de investimento.
