# Movie Review System

## Problem Statement

The problem addressed by this Movie Review System is the sentiment analysis of movie reviews. The goal is to classify reviews as either positive or negative based on their textual content. This system uses a machine learning approach, specifically the BERT (Bidirectional Encoder Representations from Transformers) model, to achieve high accuracy in sentiment classification.

## Libraries Used

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `torch`: For working with PyTorch, an open-source machine learning library.
- `transformers`: For using pre-trained transformer models like BERT from Hugging Face.
- `scikit-learn`: For data splitting and performance evaluation metrics.
- `nltk`: For natural language processing tasks.

## Functionality of Each Function

1. **load_data(file_path)**
   - Loads text data from a file and organizes it into a pandas DataFrame with two columns: text and label.

2. **preprocess_text(text)**
   - Preprocesses text by converting it to lowercase and removing punctuation.

3. **tokenize_texts(texts)**
   - Tokenizes the text data using the BERT tokenizer and converts it to input tensors suitable for model training and inference.

4. **evaluate_model(model, input_ids, attention_masks, labels)**
   - Evaluates the performance of the trained BERT model on the test set using various metrics.

5. **classify_text(input_text)**
   - Classifies individual text inputs as positive or negative using the trained BERT model.

## Metrics Used

- **Accuracy:** Measures the overall correctness of the model.
- **Precision:** Measures the proportion of true positive predictions among all positive predictions.
- **Recall:** Measures the proportion of true positive predictions among all actual positives.
- **F1 Score:** The harmonic mean of precision and recall, providing a balance between the two.

## Performance

The model performs well on the test dataset, achieving high accuracy, precision, recall, and F1 score. The example provided demonstrates an accuracy of 94%, precision of approximately 92%, recall of approximately 97%, and an F1 score of approximately 94%.

## Challenges

- **Data Preprocessing:** Ensuring that text data is appropriately cleaned and preprocessed for the BERT model.
- **Model Fine-tuning:** Properly fine-tuning the BERT model to avoid overfitting and ensure generalization to new data.
- **Performance Metrics:** Balancing different performance metrics to achieve a well-rounded model.
- **Resource Intensive:** Training transformer models like BERT can be computationally intensive and time-consuming.

