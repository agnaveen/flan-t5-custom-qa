# 🧠 FLAN-T5 Fine-Tuning: Custom QA Model

This project fine-tunes the `google/flan-t5-small` model using a small Alpaca-style instruction dataset to answer custom questions.


## ✅ Requirements

- Python 3.8+
- pip

---

## ⚙️ Installation

### 🔹 Option 1: Standard Installation

```bash
git clone [https://github.com/agnaveen/flan-t5-custom-qa.git](https://github.com/agnaveen/flan-t5-custom-qa.git)
cd flan-t5-custom-qa

# Install dependencies
pip install -r requirements.txt

```
### 🔹 Option 2: Using a Virtual Environment

```bash
# Create virtual environment
python3 -m venv train-model

# Activate the environment
source train-model/bin/activate   # On Mac/Linux
# .\train-model\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

### 🏋️‍♀️ Train the Model
python train_flan.py

This will:
Load data from data/train.json
Fine-tune the model
Save it to the folder flan-tuned/
```

### 💬 Run Inference
```bash
python test_flan.py "hyderabad's number 1 engineering college ?"

# Expected Output
Q: hyderabad's number 1 engineering college ?
A: Guru Nanak College of Engineering & Technology

If the model doesn’t know the answer:
A: No data found
```

### ✍️ Customize the Dataset
```bash
Edit the data/train.json file to include your own Q&A pairs.

Format:
{
  "instruction": "your question here",
  "input": "optional extra context or empty string",
  "output": "expected answer"
}

Add minimum 10-100 examples for every context/question.

then run this command,
python train_flan.py
```

### 🔁 Reset Training
```bash
rm -rf flan-tuned
python train_flan.py
```


### Helpful commands
```bash
git clone https://github.com/ggml-org/llama.cpp

cd llama.cpp

pip install -r requirements.txt

python3 convert_hf_to_gguf.py \
  --outfile ./flan-custom.gguf \
  --outtype f16 \
  ../flan-alpaca-clean/flan-tuned


flan-custom-model.txt:
FROM flan-custom.gguf

Then run:
ollama create flanc -f flan-custom-model.txt

ollama run flanc


find ~/.cache/huggingface/hub/ -name "spiece.model"

pip install torch torchvision torchaudio

brew install cmake pkg-config coreutils

pip install sentencepiece






