import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model_name = "./flan-tuned"  # or use "google/flan-t5-small" for base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def ask(instruction):
    prompt = instruction.strip().lower().rstrip("?")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        num_beams=4,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    if (
        not response
        or response.lower() == prompt.lower()
        or len(response) < 2
        or all(word in prompt.lower() for word in response.lower().split())
    ):
        return "No data found"

    return response

# Read from command-line argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_flan.py \"your question here\"")
    else:
        instruction = sys.argv[1]
        print("Q:", instruction)
        print("A:", ask(instruction))