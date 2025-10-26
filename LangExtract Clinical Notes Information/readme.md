**🧠 Clinical Entity Extraction using LangExtract + Ollama**

**🔍 Project Overview**

This project extracts key clinical information such as patient age, department, diagnosis, medication, dosage, route, and frequency from unstructured medical text. It uses LangExtract, a lightweight NLP framework, together with Ollama, which runs local large language models (LLMs) such as Gemma 2 (2B).

The project demonstrates how local language models can turn free-text medical notes into structured, machine-readable data and visualize the results interactively.

**🎯 Business Objective**

Healthcare organizations often store data in unstructured clinical notes, making it difficult to analyze or track key metrics. Manually reviewing these notes is time-consuming and inefficient.

This project’s goal is to automatically extract structured information from text to:

Save time and reduce manual data entry.

Improve accuracy in patient data extraction.

Enable analytics teams to measure metrics like diagnosis frequency, medication trends, and departmental workload.

Build the foundation for data-driven healthcare decision-making.

**⚙️ How It Works**

Input – Clinical text examples such as outpatient or surgical notes.

Prompt Design – Defines what to extract (age, department, problem, duration, medication, dosage, route, frequency).

Example Guidance – Uses few-shot examples to teach the model how to extract relevant entities.

Model Inference – Uses Ollama running a local model (Gemma 2B) to perform the extraction.

Output – Generates structured extraction results saved in a JSONL file.

Visualization – Creates an interactive HTML file highlighting all extracted entities within the original text.

**🧩 Example Output**

Input Text:
A 62-year-old woman was admitted to the Surgery department for postoperative wound infection that had been present for five days. The physician started her on cefazolin 1 g intravenous every 8 hours for infection control.

Extracted Entities:

Patient age: 62-year-old

Department: Surgery department

Problem: Postoperative wound infection

Duration: Five days

Medication: Cefazolin

Dosage: 1 g

Route: Intravenous

Frequency: Every 8 hours

**🧠 Visualization**

The project produces an interactive HTML visualization (visualization.html) that displays the text with highlighted entities. You can open it locally or host it through GitHub Pages so others can interact with it directly in the browser.
