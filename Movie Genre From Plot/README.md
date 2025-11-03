# Movie Genre Classification from Plot Summaries

A comparative machine learning study that predicts movie genres from plot summaries by evaluating traditional ML approaches (Naive Bayes, SVM, Logistic Regression) against a fine-tuned DistilBERT transformer model to determine which method offers the best balance of accuracy, speed, and resource efficiency.

## 🎯 Business Objective

The primary goal of this project is to **automatically classify movies into genres based on their plot summaries**. This has several practical applications:

- **Content Recommendation Systems**: Help streaming platforms recommend movies to users based on genre preferences
- **Content Organization**: Automatically tag and categorize large movie databases
- **Market Analysis**: Understand genre trends and distributions in the film industry
- **Metadata Enhancement**: Enrich movie databases with accurate genre information

## 📊 Dataset

The project uses a movie dataset containing:
- **Movie titles**
- **Plot summaries** 
- **Genre labels** 

### Genre Categories
The dataset includes 27 different genres:
```
Action, Adult, Adventure, Animation, Biography, Comedy, Crime, Documentary, 
Drama, Family, Fantasy, Game-Show, History, Horror, Music, Musical, Mystery, 
News, Reality-TV, Romance, Sci-Fi, Short, Sport, Talk-Show, Thriller, War, Western
```

### Preprocessing Steps
1. **Language Detection**: Filtered dataset to include only English movies
2. **Data Cleaning**: Removed genres with zero samples (e.g., "Lifestyle")
3. **Train-Test Split**: 70-30 stratified split to maintain genre distribution

## 🔧 Technical Overview

This project implements a **comparative analysis** of multiple machine learning approaches:

### Approach 1: Traditional Machine Learning
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Models**: Naive Bayes, Linear SVM, Logistic Regression
- **Optimization**: GridSearchCV for hyperparameter tuning
- **Configuration**: OneVsRest multi-class classification strategy

### Approach 2: Deep Learning (Transfer Learning)
- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning**: Trained on movie plot data for genre classification
- **Framework**: Hugging Face Transformers with PyTorch

### 4. Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **Cross-Validation Score**: 3-fold CV for traditional models


## 📈 Results & Comparison

### Model Performance Comparison

| Model | Test Accuracy |
|-------|---------------|
| **DistilBERT (Fine-tuned)** | **33.33%** |
| **Linear SVM** | 32.42% |
| **Logistic Regression** | 32.42% |
| **Naive Bayes** | 30.14% |

**The fine-tuned DistilBERT model performed best**, achieving 33.33% accuracy on the test set. While the improvement over traditional ML models is modest (0.91 percentage points over Linear SVM), it demonstrates that transfer learning with pre-trained language models can capture semantic relationships in movie plots more effectively than TF-IDF-based approaches. However, the relatively low overall accuracy (~30-33%) across all models indicates that classifying movies into 27 distinct genres based solely on plot text is inherently challenging, as genre boundaries are often ambiguous and movies can legitimately span multiple categories.

### Key Insights

#### Traditional ML Advantages:
✅ **Faster Training**: Minutes vs hours  
✅ **Smaller Model Size**: ~10MB vs ~250MB  
✅ **Interpretability**: TF-IDF weights are human-readable  
✅ **Lower Resource Requirements**: CPU-friendly  

#### Deep Learning Advantages:
✅ **Better Semantic Understanding**: Captures context and nuance  
✅ **Transfer Learning**: Leverages pre-trained knowledge  
✅ **Potential for Higher Accuracy**: With sufficient data  
✅ **Handles Complex Patterns**: Non-linear relationships  


