import streamlit as st
from huggingface_hub import InferenceClient

BASE_MODEL = "meta-llama/Llama-3.2-1B"

@st.cache_resource
def get_client():
    return InferenceClient(
        model=BASE_MODEL,
        token=st.secrets["HF_TOKEN"],
    )

def gerar_poesia(verso_inicial, max_tokens, temperatura, top_p):
    client = get_client()
    resposta = client.text_generation(
        prompt=verso_inicial,
        max_new_tokens=max_tokens,
        temperature=temperatura,
        top_p=top_p,
        repetition_penalty=1.2,
        do_sample=True,
    )
    return verso_inicial + resposta

st.set_page_config(page_title="Gerador de Poesias — Álvaro de Campos", page_icon="✍️", layout="centered")
st.title("Gerador de Poesias")
st.subheader("Estilo Álvaro de Campos — Fernando Pessoa")
st.markdown("> *"Não sou nada. Nunca serei nada."* — Álvaro de Campos, *Tabacaria*")
st.divider()

st.sidebar.header("Parâmetros de Geração")
max_tokens = st.sidebar.slider("Tamanho do poema (tokens)", 50, 300, 150, 10)
temperatura = st.sidebar.slider("Criatividade (temperatura)", 0.5, 1.5, 0.85, 0.05)
top_p = st.sidebar.slider("Diversidade (top-p)", 0.5, 1.0, 0.92, 0.01)
st.sidebar.divider()
st.sidebar.markdown("**Sobre o modelo**\n- **Modelo:** Llama 3.2-1B\n- **Técnica:** LoRA Fine-tuning\n- **Biblioteca:** Hugging Face Transformers")

st.markdown("### Digite um verso inicial")
verso_inicial = st.text_area(label="Verso inicial", placeholder="Ex: Estou cansado de tudo e de nada,", height=80, label_visibility="collapsed")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Mar e solidão"): verso_inicial = "Ó mar imenso e indiferente,"
with col2:
    if st.button("Máquinas"): verso_inicial = "As máquinas trabalham enquanto eu"
with col3:
    if st.button("Tédio"): verso_inicial = "Estou cansado de tudo,"

st.markdown("")
gerar = st.button("Gerar Poema", type="primary", use_container_width=True)

if gerar:
    if not verso_inicial.strip():
        st.warning("Digite um verso inicial antes de gerar.")
    else:
        with st.spinner("Gerando poema no estilo de Álvaro de Campos..."):
            try:
                poema = gerar_poesia(verso_inicial, max_tokens, temperatura, top_p)
                st.divider()
                st.markdown("### Poema Gerado")
                st.markdown(f'<div style="background-color:#1e1e2e;border-left:4px solid #cba6f7;padding:20px 24px;border-radius:8px;font-family:Georgia,serif;font-size:16px;line-height:1.8;color:#cdd6f4;white-space:pre-wrap;">{poema}</div>', unsafe_allow_html=True)
                st.code(poema, language=None)
                st.caption("Use o botão de cópia acima para copiar o poema.")
            except Exception as e:
                st.error(f"Erro ao gerar poema: {e}")

st.divider()
st.caption("Desenvolvido com Hugging Face Transformers + LoRA | Inspirado em Álvaro de Campos (Fernando Pessoa) | Obras em domínio público desde 2006")
