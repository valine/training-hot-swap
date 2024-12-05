# inference.py
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MistralForCausalLM
import huggingface_hub.utils as hf_hub_utils
from transformers import logging as hf_logging



def get_model():
    """Get model either from global context or load it fresh"""
    global model  # Reference the global model variable

    try:
        # Check if model exists in global scope
        model
    except NameError:
        print("Global model not found, loading fresh model...")
        model = AutoModelForCausalLM.from_pretrained(
            "/models/latent-descent/base",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("Using existing model from global scope")

    return model


def get_tokenizer():
    """Get tokenizer either from global context or load fresh"""
    global tokenizer  # Reference the global tokenizer variable

    try:
        # Check if tokenizer exists in global scope
        tokenizer
    except NameError:
        print("Global tokenizer not found, loading fresh tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("/models/latent-descent/base")
    else:
        print("Using existing tokenizer from global scope")

    return tokenizer


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module='transformers.*')
    hf_logging.set_verbosity_error()

    print("Starting inference...")

    # Get model and tokenizer
    local_model = get_model()
    local_tokenizer = get_tokenizer()

    # Example prompt
    prompt = "Write a story about a dragon that terrorizes a village."

    # Tokenize input
    inputs = local_tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False).to(local_model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    num_tokens_to_generate = 1000
    # Generate text
    print("\nGenerating response...")

    outputs = local_model.generate(
        input_ids,
        max_length=num_tokens_to_generate + input_ids.shape[-1],
        temperature=1.0,
        repetition_penalty=1.1,
        do_sample=False,
        attention_mask=attention_mask,
        use_cache=True,
        eos_token_id=local_tokenizer.eos_token_id,
    )

    # Decode and print the generated text
    generated_text = local_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{generated_text}")


if __name__ == "__main__":
    main()