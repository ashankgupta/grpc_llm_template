from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str):
    print(f"[LLM] Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    print("[LLM] Model loaded")
    return tokenizer, model
