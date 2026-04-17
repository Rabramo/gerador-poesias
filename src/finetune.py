# =============================================================
#  GERADOR DE POESIAS — ESTILO ÁLVARO DE CAMPOS
#  Modelo : meta-llama/Llama-3.2-1B
#  Técnica: LoRA (Low-Rank Adaptation)
# =============================================================

import csv
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------------------------------------
# 1. CAMINHOS
# -------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "dataset_poesias.csv"
OUTPUT_DIR   = str(BASE_DIR / "modelo_poesia")

# -------------------------------------------------------------
# 2. CONFIGURAÇÕES
# -------------------------------------------------------------
MODEL_NAME    = "meta-llama/Llama-3.2-1B"
MAX_LENGTH    = 512
BATCH_SIZE    = 2
EPOCHS        = 5
LEARNING_RATE = 2e-4

# Configuração do LoRA
LORA_R        = 8      
LORA_ALPHA    = 16     
LORA_DROPOUT  = 0.05

# -------------------------------------------------------------
# 3. CARREGAMENTO DO DATASET
# -------------------------------------------------------------
def carregar_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"\n❌ Dataset não encontrado em: {path}\n")

    textos = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "texto" not in reader.fieldnames:
            raise ValueError(
                f"\n❌ O dataset deve conter a coluna 'texto'. Campos encontrados: {reader.fieldnames}\n"
            )

        for row in reader:
            texto = row.get("texto", "")
            if texto is None:
                texto = ""

            texto = texto.strip()
            if not texto:
                skipped += 1
                continue

            textos.append(texto)

    print(f"✅ Dataset carregado: {len(textos)} poemas")
    if skipped:
        print(f"⚠️ Linhas ignoradas: {skipped} entradas vazias ou inválidas")
    return textos


# -------------------------------------------------------------
# 4. TOKENIZAÇÃO
# -------------------------------------------------------------
def tokenizar(textos, tokenizer):
    def tokenize_fn(exemplos):
        tokens = tokenizer(
            exemplos["texto"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="longest",
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_dict({"texto": textos})
    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["texto"])
    return dataset


# -------------------------------------------------------------
# 5. CONFIGURAÇÃO DO LoRA
# -------------------------------------------------------------
def aplicar_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   # modelo generativo
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        # Camadas alvo do Llama
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model


# -------------------------------------------------------------
# 6. FINE-TUNING
# -------------------------------------------------------------
def treinar():
    print("\n🚀 Iniciando fine-tuning com Llama 3.2-1B + LoRA...\n")
    print(f"📂 Dataset:      {DATASET_PATH}")
    print(f"📂 Modelo salvo: {OUTPUT_DIR}\n")

    # --- Tokenizer ---
    print(f"📥 Carregando tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Llama não tem pad_token nativo

    # --- Modelo ---
    print(f"📥 Carregando modelo: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": "cpu"},          
    )
    model.config.use_cache = False  

    # --- Aplica LoRA ---
    print("\n🔧 Aplicando LoRA ao modelo...")
    model = aplicar_lora(model)

    # --- Dataset ---
    textos  = carregar_dataset(DATASET_PATH)
    dataset = tokenizar(textos, tokenizer)

    # Divisão treino/validação 90/10
    split       = dataset.train_test_split(test_size=0.1, seed=42)
    train_data  = split["train"]
    val_data    = split["test"]
    print(f"\n📊 Treino: {len(train_data)} | Validação: {len(val_data)}\n")

    # --- Argumentos de treino ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,   # simula batch maior sem mais memória
        learning_rate=LEARNING_RATE,
        warmup_steps=20,
        weight_decay=0.01,
        
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # float16 só com GPU
        report_to="none",                # desativa W&B
        optim="adamw_torch",
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Treinando o modelo...\n")
    trainer.train()

    # --- Salvar modelo + tokenizer ---
    print(f"\nSalvando modelo em: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuning concluído com sucesso!\n")


# -------------------------------------------------------------
# 7. TESTE RÁPIDO PÓS-TREINO
# -------------------------------------------------------------
def gerar_poesia(verso_inicial: str, max_new_tokens: int = 200) -> str:
    """
    Carrega o modelo treinado e continua um verso inicial
    no estilo de Álvaro de Campos.
    """
    from peft import PeftModel

    print(f"\n📥 Carregando modelo salvo de: {OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    inputs = tokenizer(verso_inicial, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,        
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.2,  
            pad_token_id=tokenizer.eos_token_id,
        )

    poema = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return poema


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    # Treina
    treinar()

    # Testa com 3 versos iniciais
    print("=" * 55)
    print("🧪 TESTE DO MODELO TREINADO")
    print("=" * 55)

    versos_teste = [
        "Estou cansado de tudo,",
        "Ó mar imenso e indiferente,",
        "As máquinas trabalham enquanto eu",
    ]

    for verso in versos_teste:
        print(f"\nVerso inicial : {verso}")
        print(f"Poema gerado  :\n")
        print(gerar_poesia(verso))
        print("\n" + "-" * 55)
