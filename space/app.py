import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------
# 1. CONFIGURAÇÕES
# -------------------------------------------------------------
FINE_TUNED_MODEL = "Rabramo/gerador-poesias-alvaro-campos"

# -------------------------------------------------------------
# 2. CARREGAMENTO DO MODELO
# Carrega uma vez e mantém em cache durante a sessão.
# Aceita token via st.secrets (local / Streamlit Cloud) ou
# variável de ambiente (HF Spaces).
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo... (pode levar alguns minutos na primeira vez)")
def load_model():
    token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        st.error(
            "Token do Hugging Face não encontrado. "
            "Configure HF_TOKEN em `.streamlit/secrets.toml` ou como variável de ambiente."
        )
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        FINE_TUNED_MODEL,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        token=token,
    )
    model.eval()
    return tokenizer, model

# -------------------------------------------------------------
# 3. GERAÇÃO DE POESIA
# -------------------------------------------------------------
def gerar_poesia(verso_inicial, max_tokens, temperatura, top_p):
    tokenizer, model = load_model()

    inputs = tokenizer(verso_inicial, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperatura,
            top_p=top_p,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------------------------------------
# 4. INTERFACE STREAMLIT
# -------------------------------------------------------------

st.set_page_config(
    page_title="Gerador de Poesias — Álvaro de Campos",
    page_icon="✍️",
    layout="centered",
)

# --- Cabeçalho ---
st.title("Gerador de Poesias")
st.subheader("Estilo Álvaro de Campos — Fernando Pessoa")
st.markdown(
    """
    > *"Não sou nada. Nunca serei nada. Não posso querer ser nada.
    > À parte isso, tenho em mim todos os sonhos do mundo."*
    >
    > — Álvaro de Campos, *Tabacaria*
    """
)
st.divider()

# --- Sidebar ---
st.sidebar.header("Parâmetros de Geração")

max_tokens = st.sidebar.slider(
    "Tamanho do poema (tokens)",
    min_value=50,
    max_value=300,
    value=150,
    step=10,
    help="Controla o tamanho do poema gerado.",
)

temperatura = st.sidebar.slider(
    "Criatividade (temperatura)",
    min_value=0.5,
    max_value=1.5,
    value=0.85,
    step=0.05,
    help="Valores maiores = mais criativo. Menores = mais conservador.",
)

top_p = st.sidebar.slider(
    "Diversidade (top-p)",
    min_value=0.5,
    max_value=1.0,
    value=0.92,
    step=0.01,
    help="Controla a diversidade do vocabulário usado.",
)

st.sidebar.divider()
st.sidebar.markdown("**Sobre o modelo**")
st.sidebar.markdown(
    """
    - **Modelo:** Rabramo/gerador-poesias-alvaro-campos
    - **Base:** Llama 3.2-1B
    - **Técnica:** LoRA Fine-tuning
    - **Dataset:** Poesias de Álvaro de Campos
    - **Biblioteca:** Hugging Face Transformers
    """
)

# --- Estado inicial do verso ---
if "verso_inicial" not in st.session_state:
    st.session_state.verso_inicial = ""

# --- Área principal ---
st.markdown("### Digite um verso inicial")

verso_inicial = st.text_area(
    label="Verso inicial",
    value=st.session_state.verso_inicial,
    placeholder="Ex: Estou cansado de tudo e de nada,",
    height=80,
    label_visibility="collapsed",
)

# Exemplos rápidos
st.markdown("**Ou escolha um exemplo:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Mar e solidão"):
        st.session_state.verso_inicial = "Ó mar imenso e indiferente,"
        st.rerun()

with col2:
    if st.button("Máquinas"):
        st.session_state.verso_inicial = "As máquinas trabalham enquanto eu"
        st.rerun()

with col3:
    if st.button("Tédio"):
        st.session_state.verso_inicial = "Estou cansado de tudo,"
        st.rerun()

# --- Botão gerar ---
st.markdown("")
gerar = st.button("Gerar Poema", type="primary", use_container_width=True)

# --- Resultado ---
if gerar:
    if not verso_inicial.strip():
        st.warning("Digite um verso inicial antes de gerar.")
    else:
        with st.spinner("Gerando poema no estilo de Álvaro de Campos..."):
            try:
                poema = gerar_poesia(verso_inicial, max_tokens, temperatura, top_p)

                st.divider()
                st.markdown("### Poema Gerado")
                st.code(poema, language=None)
                st.caption("Use o botão de cópia acima para copiar o poema.")

            except Exception as e:
                if "StopException" in type(e).__name__:
                    raise
                msg = str(e) or type(e).__name__
                st.error(f"Erro ao gerar poema: {msg}")

# --- Rodapé ---
st.divider()
st.caption(
    "Desenvolvido com Hugging Face Transformers + LoRA | "
    "Inspirado em Álvaro de Campos (Fernando Pessoa) | "
    "Obras em domínio público desde 2006"
)
