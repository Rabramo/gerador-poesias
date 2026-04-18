# Gerador de Poesias — Álvaro de Campos

Modelo generativo treinado para criar poemas no estilo de Álvaro de Campos, heterônimo futurista e modernista de Fernando Pessoa, utilizando Hugging Face Transformers com fine-tuning via LoRA no Google Colab (GPU) e aplicação pelo Streamlit no Hugging Face Spaces.

## Demo

**HF Space:** https://huggingface.co/spaces/Rabramo/gerador-poesias

A interface foi desenvolvida com **Streamlit** (`hf-space/app.py`). O modelo é carregado uma vez com `@st.cache_resource` e mantido em memória durante a sessão.

O **deploy foi feito no Hugging Face Spaces**, pois o Streamlit Cloud oferece apenas 1 GB de RAM, memória insuficiente para carregar o modelo. O Hugging Face Spaces oferece gratuitamente 16 GB de RAM e suporta o SDK Streamlit nativamente, o que permite carregar o modelo via `transformers` e `peft` sem custo adicional.

---

## Sobre o projeto

O usuário digita um verso inicial e o modelo gera um poema completo no estilo de Álvaro de Campos, com versos livres e visão ao mesmo tempo deslumbrada e angustiada da modernidade industrial.

A aplicação carrega o modelo fine-tuned `Rabramo/gerador-poesias-alvaro-campos` diretamente via Hugging Face Transformers, sem depender da Serverless Inference API.

> *"Não sou nada. Nunca serei nada. Não posso querer ser nada.*
> *À parte isso, tenho em mim todos os sonhos do mundo."*
>
> — Álvaro de Campos, Tabacaria

---

## Stack

| Componente | Tecnologia |
|---|---|
| Modelo base | `meta-llama/Llama-3.2-1B` |
| Modelo fine-tuned | `Rabramo/gerador-poesias-alvaro-campos` |
| Fine-tuning | LoRA (Low-Rank Adaptation) via Google Colab (GPU T4/A100) |
| Notebook de treino | `notebooks/gerador_poesias_colab.ipynb` |
| Biblioteca principal | Hugging Face Transformers |
| Adaptador LoRA | Hugging Face PEFT |
| Limpeza do dataset | spaCy `pt_core_news_sm` (NER em português de Portugal) |
| Interface | Streamlit (hospedado em HF Spaces) |
| Linguagem | Python 3.x |

---

## Sobre Álvaro de Campos

Álvaro de Campos é o heterônimo mais radical de Fernando Pessoa. Engenheiro naval formado em Glasgow, de origem portuguesa e alma cosmopolita, representa a vertente futurista e sensacionista da obra pessoana. Seus temas centrais são as máquinas, a velocidade, o tédio existencial, o mar e a crise de identidade moderna.

---

## Domínio público

As obras de Fernando Pessoa e seus heterônimos são domínio público desde 2006, 70 anos após sua morte em 1935, conforme a legislação brasileira (Lei 9.610/98) e portuguesa. O uso das obras neste projeto é legalmente livre.

---

## Dataset

O dataset é composto por 314 poemas autênticos de Álvaro de Campos em português, coletados do [Arquivo Pessoa](http://arquivopessoa.net) via `src/scrape_alvaro_campos.py`. 

O scraper percorre o acervo do site, filtra os textos pelo autor e extrai título e corpo de cada poema, gerando dois arquivos em `data/`:

- `poemas_alvaro_campos.json`: arquivo com metadados completos de cada poema, incluindo título, autor, URL de origem e texto integral. Serve como fonte primária e registro auditável do corpus coletado.
- `dataset_alvaro_campos.csv`: versão simplificada com apenas a coluna `texto`, formato diretamente compatível com `src/finetune.py`. Contém somente os poemas em português, filtrados pelo script `src/json_to_csv.py`.

O `src/json_to_csv.py` lê o JSON, detecta o idioma de cada poema via `langdetect` e grava no CSV apenas os poemas em português. Os títulos ignorados são listados no terminal para rastreabilidade. A filtragem evita que palavras de outros idiomas contaminem o estilo aprendido pelo modelo.

Cada entrada é um poema completo em texto livre, sem separação entre prompt e resposta, formato ideal para fine-tuning de modelos decoder-only como o Llama. O modelo aprende o estilo lendo os poemas como texto contínuo e, ao receber um verso inicial, replica a voz, o ritmo e os temas característicos de Álvaro de Campos.

O script `src/finetune.py` valida que o CSV contenha a coluna `texto` e ignora linhas vazias ou inválidas antes de iniciar o treinamento.

---

## Estrutura do projeto

```
gerador-poesias/
├── data/                               # dataset scrapeado (não versionado)
│   ├── poemas_alvaro_campos.json      # poemas com metadados completos
│   └── dataset_alvaro_campos.csv      # formato pronto para o notebook de treino
├── hf-space/                           # clone do HF Space (repositório separado, não versionado)
│   ├── app.py                         # interface Streamlit (entry point do Space)
│   ├── requirements.txt               # dependências do Space
│   ├── Dockerfile                     # configuração de container para o Space
│   └── README.md                      # metadados do Space (YAML frontmatter)
├── notebooks/
│   └── gerador_poesias_colab.ipynb    # notebook de fine-tuning (executa no Google Colab)
├── src/
│   ├── scrape_alvaro_campos.py        # scraping de poemas do arquivopessoa.net
│   ├── finetune.py                    # fine-tuning local (referência; substituído pelo notebook)
│   └── merge_and_push.py              # merge do LoRA e push para Hugging Face Hub
├── LICENSE
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Como rodar localmente

Clone o repositório e acesse a pasta:

```bash
git clone https://github.com/Rabramo/gerador-poesias.git
cd gerador-poesias
```

Crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Faça login no Hugging Face (necessário para baixar o Llama):

```bash
hf auth login
```

Configure o token do Hugging Face para o Streamlit, por exemplo em `~/.streamlit/secrets.toml`:

```toml
HF_TOKEN = "seu_token_aqui"
```

Execute o scraping para coletar os poemas:

```bash
python3 src/scrape_alvaro_campos.py
```

Gere o CSV filtrado por idioma a partir do JSON coletado:

```bash
python3 src/json_to_csv.py
```

Execute o fine-tuning via notebook no Google Colab:

1. Abra `notebooks/gerador_poesias_colab.ipynb` no [Google Colab](https://colab.research.google.com)
2. Selecione *Ambiente de execução → GPU (T4 ou A100)*
3. Execute as células em sequência

O notebook carrega o dataset `data/dataset_alvaro_campos.csv`, limpa os textos com spaCy, aplica LoRA e salva o adapter no Google Drive. O fine-tuning local via `src/finetune.py` é mantido como referência, mas o notebook é a forma recomendada por utilizar GPU.

> **Por que Colab?** O treino local em Apple Silicon (MPS) levou 4h43min para 5 épocas com velocidades de até 210 segundos por passo devido a thermal throttling. No Colab T4, o mesmo treino completa em ~20 minutos.

Rode o playground localmente:

```bash
streamlit run src/app.py
```

O app carrega o modelo `Rabramo/gerador-poesias-alvaro-campos` diretamente via Hugging Face Transformers e requer `HF_TOKEN` em `~/.streamlit/secrets.toml`.

Opcionalmente, use `src/merge_and_push.py` para mesclar o adapter LoRA e enviar o modelo ao Hugging Face Hub:

```bash
python3 src/merge_and_push.py
```

---

## Deploy no Hugging Face Space

O deploy em produção é feito via **HF Space** (Streamlit SDK), que carrega o modelo diretamente, sem depender da Serverless Inference API.

### 1. Criar o Space

Acesse [huggingface.co/new-space](https://huggingface.co/new-space) e configure:

| Campo | Valor |
|---|---|
| Owner | Rabramo |
| Space name | gerador-poesias |
| SDK | Streamlit |
| Visibility | Public |

### 2. Enviar os arquivos

```bash
# Clone o Space recém-criado
git clone https://huggingface.co/spaces/Rabramo/gerador-poesias hf-space
cd hf-space

# Copie os arquivos do diretório space/ deste repositório
cp /caminho/para/gerador-poesias/space/* .

git add .
git commit -m "deploy inicial"
git push
```

### 3. Configurar o token

No Space, acesse **Settings → Variables and secrets → New secret**:

```
Name:  HF_TOKEN
Value: seu_token_aqui
```

O Space baixa e inicializa o modelo na primeira requisição (2 a 3 min). As seguintes são instantâneas graças ao cache do Streamlit.

---

## Avaliação do Modelo

### Configuração do treino

O fine-tuning foi executado no Google Colab com GPU **Tesla T4** (15.6 GB), dtype bfloat16. O dataset foi dividido em 90/10: **282 poemas para treino** e **32 para validação**, separados antes do treino para evitar contaminação da avaliação.

| Hiperparâmetro | Valor |
|---|---|
| Épocas | 3 |
| Batch efetivo | 8 (batch=2 × grad_accum=4) |
| Learning rate | 2e-4 |
| LoRA rank / alpha | 16 / 32 |
| Módulos LoRA | q, k, v, o proj + gate, up, down proj |
| Parâmetros treináveis | 11.272.192 de 1.247.086.592 (**0,90%**) |
| Gradient clipping | max_grad_norm = 1.0 |
| Avaliação | ao final de cada época (`eval_strategy='epoch'`) |

O melhor checkpoint foi selecionado automaticamente pelo menor `eval_loss` (`load_best_model_at_end=True`).

---

### Métrica objetiva — Perplexidade

A perplexidade (*perplexity*, PPL) foi calculada ao final do treino sobre o conjunto de validação. Mede quão bem o modelo prevê o próximo token: **valores menores indicam maior aderência ao estilo**. A fórmula é `PPL = exp(eval_loss)`.

```
A avaliar no conjunto de validação...
Eval Loss    : 3.3596
Perplexidade : 28.78
```

| Perplexidade | Interpretação |
|---|---|
| < 15 | Excelente aderência ao estilo |
| 15 – 20 | Boa aderência |
| 20 – 30 | Aderência moderada — geração aceitável |
| > 30 | Pouco ajuste ao estilo |

O resultado de **28.78** situa-se na faixa de aderência moderada. Para referência, o treino local anterior (Apple Silicon MPS, 5 épocas) obteve 27.85 — valores muito próximos, indicando que a qualidade de ajuste foi equivalente entre os dois ambientes apesar das diferenças de hardware e configuração.

#### Por que 28.78 é um resultado esperado e saudável

A perplexidade não deve ser interpretada isoladamente: ela depende diretamente do tipo de texto, do tamanho do corpus e da natureza da tarefa.

**O corpus é pequeno e altamente específico.** Com apenas 314 poemas de um único autor, o modelo não tem volume suficiente para reduzir a perplexidade para faixas típicas de modelos treinados em milhões de documentos. A tabela abaixo compara contextos distintos para situar o resultado:

| Contexto | PPL típica | Observação |
|---|---|---|
| GPT-2 em texto jornalístico (WikiText-103) | 10 – 20 | Corpus com bilhões de tokens, domínio amplo |
| LLM fine-tuned em prosa literária (corpus grande) | 15 – 25 | Domínio restrito, mas volume alto |
| **Este modelo (314 poemas, domínio único)** | **28.78** | Corpus pequeno, estilo altamente idiossincrático |
| Modelo sem fine-tuning (Llama base no mesmo corpus) | > 60 | Sem adaptação ao estilo |

**A poesia de Álvaro de Campos é estruturalmente imprevisível.** Diferentemente de prosa ou de poesia formal com métrica e rima regulares, os poemas de Campos são compostos de versos livres de extensão variável, quebras sintáticas abruptas, interjeições, listas e inversões inesperadas. Cada verso abre um espaço de possibilidades muito amplo para o próximo token — o que matematicamente resulta em distribuições menos concentradas e, portanto, maior entropia e perplexidade.

**Um loss muito baixo seria um sinal negativo.** Se a perplexidade caísse abaixo de 10 neste corpus, o mais provável seria que o modelo tivesse memorizado os poemas em vez de aprender o estilo. O modelo reproduziria trechos literais do dataset ao invés de gerar texto novo — comportamento indesejável para uma aplicação generativa.

O resultado de 28.78 indica que o modelo aprendeu padrões estilísticos suficientes para gerar texto coerente com a voz de Álvaro de Campos, sem colapsar para memorização.

---

### Avaliação qualitativa — Parâmetros de geração

A qualidade das gerações foi avaliada testando combinações de parâmetros e observando coerência estilística, aderência aos temas de Álvaro de Campos e ausência de repetições.

| Temperatura | Top-p | Observação |
|---|---|---|
| 0.5 | 0.80 | Texto previsível, pouca variação vocabular, frases repetidas |
| 0.7 | 0.85 | Coerente, mas conservador; perde fluidez poética |
| **0.75** | **0.90** | **Melhor equilíbrio: criativo, coerente e fiel ao estilo** |
| 1.0 | 0.95 | Criativo em excesso; perde coerência semântica |
| 1.5 | 1.00 | Saída incoerente, geração caótica |

Os parâmetros `repetition_penalty=1.3` e `no_repeat_ngram_size=3` foram adotados para suprimir repetição de versos e n-gramas, problema comum em modelos treinados em corpus pequeno. `top_k=40` limita o vocabulário a candidatos mais prováveis por passo.

Os valores **temperatura 0.75** e **top-p 0.90** foram selecionados como padrão por produzirem poemas com maior aderência ao estilo sensacionista e futurista de Álvaro de Campos: versos longos, vocabulário variado e tensão emocional característica.

---

## Como funciona o LoRA

O LoRA congela os pesos originais do modelo e treina apenas matrizes de baixo rank inseridas nas camadas de atenção e FFN, permitindo fine-tuning eficiente com poucos recursos. Neste projeto, **0,90% dos parâmetros do Llama são treinados** (11,2M de 1,24B), cobrindo os módulos `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` e `down_proj`.

---

## Contexto acadêmico

Projeto desenvolvido como prova substitutiva da Fase 4 do curso de pós-graduação *stricto sensu* em Machine Learning Engineering oferecido no Centro de Pós-Graduação da FIAP, Centro Universitário.

O objetivo é demonstrar o ciclo completo de desenvolvimento de um modelo generativo utilizando a biblioteca Hugging Face Transformers: definição do tema, curadoria do dataset, fine-tuning com LoRA e deploy de uma aplicação playground em produção via Streamlit.

---

## AI como ferramenta de apoio

O desenvolvimento contou com o auxílio do assistente de IA Claude (Anthropic, modelo Claude Sonnet 4.6) para tarefas de suporte como scaffolding de código, resolução de erros de compatibilidade de bibliotecas e sugestão e revisão de textos. A definição da arquitetura, escolha do modelo, coleta e curadoria do dataset, depuração, testes e validação dos resultados foram realizados pelo autor.

## Licença

MIT. As obras poéticas utilizadas no dataset são de domínio público.
