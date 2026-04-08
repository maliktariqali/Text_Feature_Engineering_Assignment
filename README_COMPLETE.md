# Text Feature Engineering Assignment

A complete Text Processing Pipeline implementing One Hot Encoding, Bag of Words (CountVectorizer), and TF-IDF for real-world text data analysis.

## Overview

This project builds a text classification system that converts reviews into numerical features for machine learning models. It includes:

- **Data Collection & Preprocessing**: Text cleaning, tokenization, lemmatization
- **Vocabulary Creation**: Manual vocabulary building with frequency analysis
- **Feature Engineering**: OHE, BoW, and TF-IDF implementations
- **Comparison Analysis**: Side-by-side method comparison
- **Sparse Matrix Analysis**: Memory efficiency and computational considerations
- **Real-world Q&A**: Industry best practices and limitations
- **Sentiment Classification**: Using Logistic Regression and Naive Bayes

## Project Structure

```
.
├── text_feature_engineering.ipynb   # Main Jupyter notebook with complete pipeline
├── requirements.txt                  # Required Python packages
├── README.md                         # This file
├── main.py                          # Standalone Python script version
└── pyproject.toml                   # Project configuration
```

## Installation

1. **Clone/Setup the project**:
   ```bash
   cd "Text_Feature_Engineering_Assignment"
   ```

2. **Create virtual environment (optional but recommended)**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run Jupyter Notebook (Interactive)
```bash
jupyter notebook text_feature_engineering.ipynb
```

### Option 2: Run Python Script
```bash
python main.py
```

## Key Sections

### 1. Text Preprocessing
- Convert text to lowercase
- Remove punctuation and tokenization
- Remove stopwords
- Apply lemmatization
- Complete preprocessing pipeline implemented

### 2. Vocabulary Creation
- Build vocabulary from documents
- Identify top frequent words
- Analyze word frequency distribution
- Visualize word frequency with plots

### 3. Feature Extraction Methods

| Method | Type | Use Case |
|--------|------|----------|
| **One Hot Encoding** | Binary (0/1) | Multi-label classification |
| **Bag of Words** | Integer counts | Frequency-based analysis |
| **TF-IDF** | Weighted scores | ML models, importance ranking |

### 4. Sparse Matrix Analysis
- Calculate sparsity percentages
- Compare memory usage
- Explain inefficiency for large-scale systems
- Show computational implications

### 5. Real-world Questions Answered
- Why BoW fails at semantic understanding
- When to use BoW vs TF-IDF in industry
- Limitations of TF-IDF in production systems

### 6. Sentiment Classification
- Train models with BoW and TF-IDF features
- Use Logistic Regression and Naive Bayes
- Compare model performance metrics
- Generate classification reports

## Sample Dataset

The notebook includes 20 pre-classified reviews (positive/negative sentiment). For real-world use:

**To add your own data:**
1. Create `reviews.csv` with columns: `review_text`, `sentiment`
2. Replace the data loading section in the notebook
3. Run the complete pipeline

**Example CSV format:**
```csv
review_text,sentiment
"This product is amazing!",1
"Terrible quality and poor service",0
```

## Key Findings

### TF-IDF Results in Sentiment Classification
- **Logistic Regression + TF-IDF**: Best performing combination
- **Naive Bayes + BoW**: Fast baseline alternative
- **Sparsity**: 98%+ zeros in matrices (memory critical)

### Common Words vs Rare Words
- Common words: "the", "a", "and" → Low TF-IDF scores
- Specific words: "amazing", "terrible", "excellent" → High TF-IDF scores
- This helps models focus on sentiment-bearing terms

## Dependencies

```
pandas                  # Data manipulation
numpy                   # Numerical operations
scikit-learn           # ML algorithms and vectorizers
matplotlib             # Plotting
seaborn                # Statistical visualization
nltk                   # Natural Language Toolkit
```

## Performance Metrics

The notebook demonstrates:
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True positives / All predicted positives
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall

## Real-world Applications

✓ Email spam detection  
✓ Product review analysis  
✓ Customer sentiment monitoring  
✓ Document classification  
✓ Search engine ranking  
✓ Recommendation systems  

## Industry Best Practices

1. **Start Simple**: Begin with TF-IDF + Logistic Regression
2. **Validate Properly**: Use train/test split and cross-validation
3. **Analyze Errors**: Understand misclassifications
4. **Track Metrics**: Monitor performance in production
5. **Iterate**: Gradually improve model complexity
6. **Document**: Keep preprocessing reproducible

## Limitations & Next Steps

### Current Limitations:
- Ignores semantic meaning (synonyms treated as different)
- No word ordering preserved
- Cannot handle polysemy (word ambiguity)
- Fixed vocabulary (unknown words ignored)

### Future Improvements:
- Use Word2Vec/GloVe embeddings
- Apply BERT/Transformer models
- Implement deep learning (LSTM, CNN)
- Use transfer learning approaches
- Build ensemble methods

## Comparison with Modern Methods

| Aspect | TF-IDF | BERT/Transformers |
|--------|---------|-------------------|
| Speed | Fast | Slower |
| Accuracy | Good | Excellent |
| Interpretability | High | Lower |
| Setup | Simple | Complex |
| Production | Proven | Emerging |

## Troubleshooting

**Issue**: Package installation fails
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Issue**: NLTK data not found
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**Issue**: Memory issues with large datasets
→ Use sparse matrix formats (CSR, CSC)
→ Reduce max_features in vectorizers
→ Process data in batches

## Files Description

- **text_feature_engineering.ipynb**: Complete interactive notebook with all tasks
- **requirements.txt**: Python package dependencies
- **README.md**: This documentation file
- **main.py**: Standalone Python script with core functionality
- **pyproject.toml**: Project metadata and configuration

## Learning Outcomes

After completing this assignment, you'll understand:

✓ How to preprocess and clean text data  
✓ Difference between OHE, BoW, and TF-IDF  
✓ When to use each feature extraction method  
✓ How to handle sparse matrices efficiently  
✓ Real-world applications of text analysis  
✓ How to build and evaluate ML models  
✓ Industry best practices for NLP  
✓ Limitations and future directions of these methods  

## References

- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Natural Language Toolkit (NLTK)](https://www.nltk.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Bag of Words Model](https://en.wikipedia.org/wiki/Bag-of-words_model)

## Author

Generated for Krish Naik's Gen AI Bootcamp - Text Feature Engineering Assignment

## License

This project is for educational purposes.

---

**Last Updated**: April 2026  
**Status**: Complete and ready to use
