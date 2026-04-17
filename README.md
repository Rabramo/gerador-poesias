# Gerador de Poesias — Álvaro de Campos

Modelo generativo treinado para criar poemas no estilo de Álvaro de Campos, heterônimo futurista e modernista de Fernando Pessoa, utilizando Hugging Face Transformers com fine-tuning via LoRA.

## Demo do app no Streamlit

URL: https://geradorpoesiasalvarodecampos.streamlit.app/
---

## Sobre o projeto

O usuário digita um verso inicial e o modelo gera um poema completo no estilo de Álvaro de Campos — com sua voz sensacionista, versos longos e livres, e visão ao mesmo tempo deslumbrada e angustiada da modernidade industrial.

O app Streamlit consome o modelo fine-tuned `Rabramo/gerador-poesias-alvaro-campos` via Hugging Face Inference API.

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
| Fine-tuning | LoRA (Low-Rank Adaptation) |
| Biblioteca principal | Hugging Face Transformers |
| Adaptador LoRA | Hugging Face PEFT |
| Interface | Streamlit |
| Linguagem | Python 3.x |

---

## Sobre Álvaro de Campos

Álvaro de Campos é o heterônimo mais radical de Fernando Pessoa. Engenheiro naval formado em Glasgow, de origem portuguesa e alma cosmopolita, representa a vertente futurista e sensacionista da obra pessoana. Seus temas centrais são as máquinas, a velocidade, o tédio existencial, o mar e a crise de identidade moderna.

---

## Domínio público

As obras de Fernando Pessoa e seus heterônimos são domínio público desde 2006 — 70 anos após sua morte em 1935 — conforme a legislação brasileira (Lei 9.610/98) e portuguesa. O uso das obras neste projeto é legalmente livre.

---

## Dataset

O dataset foi criado especificamente para este projeto e contém 30 poemas originais escritos no estilo de Álvaro de Campos, cobrindo seus temas centrais: as máquinas e a modernidade industrial, o mar e as viagens, o tédio existencial, a solidão urbana, Lisboa e Portugal, e a multiplicidade do eu sensacionista.

Cada entrada é um poema completo em texto livre, formato ideal para fine-tuning de modelos generativos do tipo decoder-only como o Llama. O modelo aprende o estilo lendo os poemas como texto contínuo — sem separação entre prompt e resposta — e ao receber um verso inicial replica a voz, o ritmo e os temas característicos de Álvaro de Campos.

O script `src/finetune.py` valida que o CSV contenha a coluna `texto` e ignora linhas vazias ou inválidas antes de iniciar o treinamento. O modelo treinado é salvo localmente em `modelo_poesia/`.

O dataset foi gerado com auxílio de inteligência artificial e revisado manualmente para garantir aderência ao estilo pessoano. 

---

## Estrutura do projeto

```
gerador-poesias/
├── dataset/
│   └── dataset_poesias.csv       # poemas no estilo de Álvaro de Campos
├── modelo_poesia/                 # modelo treinado (não versionado)
├── src/
│   ├── finetune.py               # script de fine-tuning com LoRA
│   └── app.py                    # interface Streamlit
├── merge_and_push.py             # merge do LoRA e push para Hugging Face Hub
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Como rodar localmente

Clone o repositório e acesse a pasta:

```bash
git clone https://github.com/SEU_USUARIO/gerador-poesias.git
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

Execute o fine-tuning:

```bash
python3 src/finetune.py
```

O `src/finetune.py` carrega o dataset `dataset/dataset_poesias.csv`, valida a coluna `texto`, ignora linhas vazias e salva o adapter LoRA em `modelo_poesia/`.

Rode o playground localmente:

```bash
streamlit run src/app.py
```

O app Streamlit usa o modelo fine-tuned `Rabramo/gerador-poesias-alvaro-campos` via Hugging Face Inference API e requer `HF_TOKEN` em `~/.streamlit/secrets.toml`.

Opcionalmente, use `merge_and_push.py` para mesclar o adapter LoRA e enviar o modelo ao Hugging Face Hub:

```bash
python3 merge_and_push.py
```

---

## Como funciona o LoRA

O LoRA congela os pesos originais do modelo e treina apenas uma camada adicional de baixo rank — permitindo fine-tuning eficiente de modelos grandes com poucos recursos computacionais. Neste projeto, apenas 0,17% dos parâmetros do Llama são treinados.

---

## Contexto acadêmico

Projeto desenvolvido como prova substitutiva da Fase 4 do curso de pós-graduação *stricto sensu* em Machine Learning Engineering oferecido no Centro de Pós-Graduação da FIAP — Centro Universitário.

O objetivo é demonstrar o ciclo completo de desenvolvimento de um modelo generativo utilizando a biblioteca Hugging Face Transformers: definição do tema, curadoria do dataset, fine-tuning com LoRA e deploy de uma aplicação playground em produção via Streamlit.

---
## AI como ferramenta de apoio

O desenvolvimento contou com o auxílio do assistente de IA Claude (Anthropic, modelo Claude Sonnet 4.6) para tarefas de suporte como geração inicial do dataset, scaffolding de código, resolução de erros de compatibilidade de bibliotecas e sugestão inicial e revisão de textos. A definição da arquitetura, escolha do modelo, revisão do dataset, depuração, testes e validação dos resultados foram realizados pelo autor.

## Licença

MIT. As obras poéticas utilizadas no dataset são de domínio público.
