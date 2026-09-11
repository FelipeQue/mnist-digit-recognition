# Cernere - Reconhecimento de Dígitos Manuscritos

Projeto de visão computacional com o desenvolvimento, comparação e avaliação de algoritmos de aprendizado de máquina e redes neurais no reconhecimento de dígitos manuscritos (MNIST). Projeto desenvolvido para o curso Desenvolvimento de IA para Análise Preditiva do programa SCTEC.

O nome do projeto é a palavra "cernere" (pronuncia-se "quernere") do latim, que significa "distinguir".

## Bibliotecas e Dependências:

- Python 3.12.10 (O TensorFlow não funciona com uma versão superior)
- Pandas 3.0.5
- Numpy 2.5.3
- Matplotlib 3.11.1
- Seaborn 0.13.2
- Scikit-learn 1.9.0
- XGBoost 3.4.1
- TensorFlow 2.21.0
- Keras 3.15.1
- Keras-tuner 1.4.8

- Foi utilizado um ambiente virtual para uso controlado das dependências do projeto.

## Estrutura de Pastas:

O projeto foi desenvolvido de maneira modularizada, com a seguinte estrutura de pastas:

```
├── data
│   ├── raw
│   ├── custom
├── notebooks
│   ├── cernere.ipynb
├── outputs
│   ├── images
├── src
│   ├── config.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── models.py
│   ├── ood.py
│   ├── plots.py
│   ├── preprocessing.py
```

## Etapas do projeto:

Fase 1: Exploração e análise do dataset MNIST, incluindo visualização de amostras de dígitos e análise da distribuição de classes.

Fase 2: Pré-processamento dos dados, envolvendo normalização e divisão em conjuntos de treino, validação e teste.

Fase 3: Desenvolvimento e treinamento de modelos de aprendizado de máquina, incluindo KNN e XGBoost, com otimização de hiperparâmetros.

Fase 4: Avaliação dos modelos treinados, incluindo métricas de desempenho, matrizes de confusão e relatórios de classificação.

Fase 5.1:

Fase 5.2:

Fase 5.3: Inferência com Imagens Manuscritas Próprias

## Resultados da inferência com imagens manuscritas próprias:

Foi feito um teste de inferência com imagens manuscritas próprias, utilizando o melhor modelo treinado (MLP). Escrevi digitalmente os numerais 4 e 9 em imagens PNG com fundo branco e 400 pixels de largura e altura.

<p align="center">
  <img src="data/custom/custom-4.png" alt="4 manuscrito" width="45%" />
  <img src="data/custom/custom-9.png" alt="9 manuscrito" width="45%" />
</p>

As imagens foram processadas e normalizadas para o formato esperado pelo modelo conforme o seguinte pipeline:

- Conversão em escala de cinza ;
- Inversão caso o fundo seja claro;
- Recorte da imagem em um quadrado de 20x20;
- Centralização da imagem num quadrado de 28x28 (mantendo um padding ao redor do desenho do numeral);
- Normalização dos valores numéricos de cor dos pixels do intervalo `[0, 255]` para o intervalo continuo `[0.0, 1.0]`;
- Alinhamento pelo Centro de Massa: deslocamento (*shift*) espacial dos eixos da matriz usando a biblioteca `scipy.ndimage` para alinhar o centro de gravidade do traço ao centro geométrico da imagem (`14.0, 14.0`);
- Estruturação em Tabela (DataFrame): conversão da matriz processada final em um formato bidimensional `28x28` para inspeção tabular dos pixels.

No notebook essa tabela passa por um achatamento (`1x784`) para enfim ser submetida à predição pela rede neural.

![Predição do Dígito 4](../outputs/images/prediction_custom-4.png)
![Predição do Dígito 9](../outputs/images/prediction_custom-9.png)

As inferências do modelo MLP nas amostras customizadas parecem refletir o impacto direto do estilo de escrita na distribuição de probabilidades

Dígito 4 (Confiança: 96,0%): A grafia em estilo "barco à vela" (com topo fechado, linhas retas e intersecção proeminente) tem uma assinatura geométrica marcante que preveniu a confusão clássica do MNIST entre 4 e 9 (comum em traços com topo aberto), garantindo alta certeza preditiva.

Dígito 9 (Confiança: 76,6% vs. 20,0% para o Dígito 3): A curvatura da cabeça e o alinhamento da haste geraram uma sobreposição parcial no espaço de características com o dígito 3. Embora o modelo tenha classificado a imagem corretamente, a distribuição de probabilidades capturou com precisão a ambiguidade morfológica do traço.

## Idioma do código:

- Nomes de funções em inglês;
- Nomes de variáveis em inglês;
- Nomes de arquivos em inglês;
- Nomes de branches em inglês;
- Mensagens de commit em inglês;
- Comentários em português;
- Docstrings em português;
- Discussão e explicações no notebook em português.

A ideia aqui foi deixar o código em si acessível para a comunidade desenvolvedora, mas com comentários e explicações voltados para o contexto avaliativo do SCTEC.
