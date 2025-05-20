# Video Analysis with Facial Recognition, Emotion Detection, and Activity Recognition

## Descrição

Este projeto realiza a análise de vídeos em três etapas principais:

1. **Reconhecimento Facial:** Detecta e marca rostos presentes em cada frame do vídeo.  
2. **Análise de Expressões Emocionais:** Identifica a emoção predominante em cada rosto detectado.  
3. **Detecção de Atividades:** Classifica a atividade predominante presente no vídeo.

Além disso, o sistema realiza a detecção de anomalias com base em regras definidas, como a presença de mais de dois rostos em um mesmo frame.

## Tecnologias e Bibliotecas Utilizadas

- Python 3.9.13  
- [OpenCV](https://opencv.org/) (para captura e manipulação de vídeo e imagens)  
- [MediaPipe](https://mediapipe.dev/) (para reconhecimento facial e análise de expressões)  
- [NumPy](https://numpy.org/) (para operações numéricas)  
- [Pandas](https://pandas.pydata.org/) (para manipulação de dados e geração de relatórios)  
- [tqdm](https://github.com/tqdm/tqdm) (para barra de progresso na análise de frames)

## Instalação

```bash
pip install opencv-python mediapipe numpy pandas tqdm


## Uso

O script principal (`main.py`) processa um vídeo definido em `VIDEO_PATH`, realiza as análises e gera um relatório com as informações de rostos detectados, emoções predominantes, atividades e possíveis anomalias.

```bash
python main.py

Antes de executar o modulo main.py, tenha certeza que definiu o caminho do vídeo .mp4 na mesma pasta que o script se localiza

## Metodologia

O projeto realiza a análise do vídeo em três etapas principais, que são aplicadas para cada frame do vídeo:

| Etapa                      | Descrição                                                                                                   |
|----------------------------|-------------------------------------------------------------------------------------------------------------|
| **1. Reconhecimento Facial** | Detecta e marca todos os rostos presentes no frame utilizando a biblioteca MediaPipe.                        |
| **2. Análise de Expressões Emocionais** | Para cada rosto detectado, identifica a expressão emocional predominante (exemplo: feliz, triste, neutro).|
| **3. Detecção de Atividades**   | Analisa o conteúdo do frame para classificar a atividade que está sendo realizada (exemplo: caminhando, sentado). |

### Detecção de Anomalias

As anomalias são definidas a partir de regras específicas para o contexto da análise. Um exemplo implementado é:

- **Anomalia:** mais de 2 rostos detectados em um mesmo frame, o que pode indicar situações inesperadas conforme o cenário de uso (ex.: monitoramento de ambiente com limite máximo de pessoas).

O sistema conta com uma lógica de detecção de anomalias baseada em regras simples, porém eficazes, para identificar comportamentos inesperados durante a análise de vídeo. Essas anomalias podem indicar situações fora do padrão, falhas no monitoramento ou alterações comportamentais relevantes. As principais regras implementadas são:

Múltiplos rostos no mesmo frame:
Quando o número de rostos detectados em um frame excede um limite pré-definido, o sistema registra uma anomalia. Isso é útil, por exemplo, em cenários de monitoramento onde há uma quantidade máxima esperada de pessoas em determinado ambiente.

Ausência prolongada de rostos:
Caso nenhum rosto seja detectado por uma sequência contínua de frames, o sistema interpreta isso como uma possível falha no monitoramento (ex: câmera obstruída) ou comportamento atípico (ex: local momentaneamente vazio em horários inusitados).

Mudança brusca de emoção:
O sistema acompanha a emoção predominante de cada rosto ao longo do tempo. Se uma mudança súbita de emoção for detectada (por exemplo, de "feliz" para "triste" em poucos segundos), isso pode indicar uma reação anormal ou evento relevante, sendo registrado como anomalia para investigação posterior.

Todas as anomalias são associadas ao número do frame em que ocorreram, permitindo uma análise temporal detalhada. Essa abordagem torna o sistema mais robusto para aplicações de segurança, bem-estar emocional e controle de ambiente.

## Relatórios Gerados

Ao final da análise do vídeo, o sistema gera um relatório com as seguintes informações:

- **Total de frames analisados:** Quantidade total de frames processados no vídeo.
- **Anomalias detectadas:** Número total de frames onde as anomalias foram identificadas, de acordo com a regra definida (ex.: mais de 2 rostos no frame).
- **Frames com anomalias:** Lista dos IDs dos frames onde anomalias foram detectadas.
- **Expressões emocionais mais comuns:** As emoções identificadas ao longo do vídeo, junto com suas frequências.
- **Atividades mais comuns:** As categorias de atividades mais detectadas, com contagem de ocorrência.
- **Quantidade de frames por expressão emocional:** Número total de frames onde cada tipo de emoção foi identificada.

Esses dados permitem uma análise quantitativa e qualitativa do conteúdo do vídeo, facilitando a identificação de padrões e eventos relevantes.
