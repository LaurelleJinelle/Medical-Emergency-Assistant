# Medical Emergency Assistant  
### Fine-Tuned TinyLlama for Emergency Response Guidance

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Model](https://img.shields.io/badge/Model-TinyLlama-orange)
![Status](https://img.shields.io/badge/Status-Fine--Tuned-success)

---

## Demo Video - ![Here](https://drive.google.com/file/d/1gy1Ag7xMAy6F2pj6nyP_bflXS8V1n6qm/view?usp=sharing)
---

## Project Overview

This project presents a **fine-tuned TinyLlama (1.1B)** model designed to act as a **Medical Emergency Assistant**.

The objective is to generate structured, clear, and safety-aware responses to emergency medical questions using **Supervised Fine-Tuning (SFT)**.

The fine-tuned model improves:

- Instruction-following capability  
- Structured response formatting  
- Domain-specific emergency vocabulary  
- Response relevance  

---

## Base Model

We fine-tuned:

```
TinyLlama-1.1B
```

### Why TinyLlama?

- Lightweight (runs on limited GPU)
- Faster experimentation
- Suitable for instruction tuning
- Efficient memory usage

---

## 📊 Dataset

The dataset consists of **instruction–response pairs** formatted as:

```text
### Instruction:
<Question>

### Response:
<Medical guidance answer>
```

### Dataset Schema

| Column   | Description |
|----------|-------------|
| question | Emergency-related user query |
| answer   | Ground-truth structured response |
| metadata | Includes difficulty, reasoning, topic |

---

### Example Data Entry

```text
### Instruction:
Someone is unconscious after an accident. What should I do?

### Response:
Call emergency services immediately. Check for breathing and pulse. Avoid moving the person unless necessary.
```

The dataset focuses on:

- First aid guidance
- Emergency procedures
- Safety-first responses
- Clear step-by-step instructions

---

## Fine-Tuning Methodology

We used **Supervised Fine-Tuning (SFT)**.

### Training Prompt Template

```python
def format_example(example):
    return f"""### Instruction:
{example['question']}

### Response:
{example['answer']}"""
```

---

### Training Configuration

| Parameter       | Value |
|----------------|--------|
| Model          | TinyLlama-1.1B |
| Training Type  | Supervised Fine-Tuning |
| Optimizer      | AdamW |
| Max Tokens     | 512 |
| Evaluation     | ROUGE + BLEU |
| Hardware       | GPU |

---

## Evaluation Metrics

After training, the model was evaluated on a validation subset using:

- **Loss**
- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**
- **BLEU**

---

## Results

| Metric      | Value     |
|------------|----------|
| Loss       | 1.125900 |
| ROUGE-1    | 0.111369 |
| ROUGE-2    | 0.048803 |
| ROUGE-L    | 0.077551 |
| BLEU Score | 0.000022 |

---

## Interpretation of Results

### Loss (1.1259)
Indicates successful learning and convergence.

### ROUGE Scores
- ROUGE-1 → Captures unigram overlap
- ROUGE-2 → Captures bigram overlap
- ROUGE-L → Measures sequence similarity

These show the model captures key emergency-related terms but generates varied wording.

### BLEU Score (0.000022)
BLEU is strict for generative models.  
Low BLEU suggests semantic similarity but lexical variation.

---

## How to Run the Project

---

### 1️ Clone the Repository

```bash
git clone [<your_repository_url>](https://github.com/LaurelleJinelle/Medical-Emergency-Assistant/)
cd medical-emergency-assistant
```

---

### 2️ Install Dependencies
Required libraries:

```
transformers
datasets
torch
evaluate
gradio
```

---

### 3️ Run Inference

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="your_finetuned_model")

prompt = """### Instruction:
Someone is choking. What should I do?

### Response:
"""

output = pipe(prompt, max_new_tokens=150)
print(output[0]["generated_text"])
```

---

### 4️ Launch Gradio Interface

```python
import gradio as gr

def chatbot(question):
    prompt = f"""### Instruction:
{question}

### Response:
"""
    output = pipe(prompt, max_new_tokens=150)[0]["generated_text"]
    return output.split("### Response:")[-1].strip()

gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=3, placeholder="Describe the emergency..."),
    outputs=gr.Textbox(lines=12),
    title="Medical Emergency Assistant",
    description="AI-powered emergency response assistant"
).launch()
```

---

## Limitations

- Not a substitute for professional medical care
- BLEU not ideal for generative evaluation
- Limited dataset size
- Limited context window

---

## Future Improvements

- Larger and more diverse dataset
- RLHF (Reinforcement Learning from Human Feedback)
- Safety alignment layers
- Human evaluation study
- Hyperparameter tuning

---

## Impact of Fine-Tuning

Compared to the base TinyLlama model, the fine-tuned model:

- Produces more structured responses
- Focuses on emergency-specific language
- Reduces irrelevant outputs
- Follows instruction format consistently

---

## Conclusion

This project demonstrates how a lightweight LLM like TinyLlama can be adapted to a specialized domain through supervised fine-tuning.

Even with limited computational resources, domain-specific alignment is achievable and measurable using evaluation metrics.

---

## Author

Your Name  
GitHub: (https://github.com/LaurelleJinelle) 
Project: Medical Emergency Assistant  

---
