from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ATENÇÃO: este script mescla o adapter LoRA nos pesos da base e faz push do
# modelo completo para ADAPTER_REPO. Após o push, o repositório deixará de
# conter o adapter isolado — apenas o modelo já mesclado estará disponível.
# Requer autenticação prévia: huggingface-cli login
BASE_MODEL   = "meta-llama/Llama-3.2-1B"
ADAPTER_REPO = "Rabramo/gerador-poesias-alvaro-campos"


def main():
    try:
        print("Carregando modelo base...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        print("Carregando adapter LoRA...")
        model = PeftModel.from_pretrained(base, ADAPTER_REPO)

        print("Mesclando pesos...")
        model = model.merge_and_unload()

        print("Fazendo push...")
        model.push_to_hub(ADAPTER_REPO)
        tokenizer.push_to_hub(ADAPTER_REPO)

        print("Pronto!")

    except Exception as exc:
        print("Erro durante o merge ou push do modelo:", exc)
        raise


if __name__ == "__main__":
    main()