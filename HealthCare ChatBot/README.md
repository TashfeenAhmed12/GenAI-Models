# 🧠 Fine-Tuning a Conversational LLM for Psychological Dialogue Support

This project fine-tunes a **Large Language Model (LLM)** to simulate natural, empathetic, and context-aware conversations between a **psychologist** and a **patient**.  

The goal is to create an AI-driven conversational assistant that demonstrates **reflective listening**, **emotional validation**, and **supportive dialogue** — helping users explore thoughts and feelings safely while receiving psychoeducational guidance.

---

## 🎯 Objective

Traditional chatbots are designed for open-domain Q&A and casual conversation.  
This project focuses on **mental-wellness dialogue**, teaching the model to:

- Recognize and mirror emotional tone  
- Provide thoughtful, non-judgmental responses  
- Encourage constructive self-reflection and healthy reframing  
- Offer gentle, educational guidance (without giving medical advice)

The end product is a **wellness-focused AI companion** — **not** a diagnostic or therapeutic system — meant for early exploration of mental-health topics and emotional awareness.

---

## 💡 Business Value

- **Mental-Wellness Platforms:** Embed the model into digital counseling or self-help apps to simulate compassionate conversation.  
- **Patient Engagement Tools:** Offer pre-session journaling or reflection support.  
- **Training & Research:** Create realistic conversational simulations for psychology and communication studies.  

By combining **ethical AI design** with **human-centered communication**, this project showcases how conversational models can improve access to empathetic guidance in a safe, scalable way.

---

## 🧰 Methodology (High-Level)

1. **Data Preparation**  
   - Custom dataset of short dialogues where users ask questions and psychologists reply empathetically.  
   - Cleaned, stripped of null values, and split into training and evaluation subsets.

2. **Model Selection**  
   - Started with lightweight, instruction-tuned backbones (`Phi-3 Mini` and `DialoGPT-medium`) for approachable fine-tuning on limited hardware.  
   - Configured tokenizer to handle dialogue padding and end-of-sequence tokens properly.

3. **Fine-Tuning**  
   - Implemented with Hugging Face **Transformers**, **Trainer API**, and **PyTorch**.  
   - Supervised fine-tuning (SFT) objective using pairs of `(Context → Response)` turns.  
   - Trained for a few epochs with modest hyperparameters for rapid experimentation.

4. **Evaluation**  
   - Generated sample dialogues to test coherence, empathy, and tone alignment.  
   - Adjusted hyperparameters and tokenization strategy to balance fluency and emotional tone.

---

## 🧩 Repository Structure


