# Movie Review System README

## Overview

This Movie Review System is a sentiment analysis tool designed to classify movie reviews as either positive or negative. The model leverages the BERT (Bidirectional Encoder Representations from Transformers) architecture for sequence classification. By fine-tuning BERT on a labeled dataset of movie reviews, the system achieves high accuracy in predicting the sentiment of new reviews.

## Problem Statement

The main goal of this project is to develop a machine learning model that can automatically classify movie reviews as positive or negative based on their text content. This helps in understanding the overall sentiment expressed in movie reviews, providing valuable insights for movie producers, marketers, and potential viewers.

## Libraries Used

- **numpy**: For numerical operations and handling arrays.
- **pandas**: For data manipulation and analysis.
- **torch**: For PyTorch, used in building and training neural networks.
- **transformers**: For BERT model and tokenizer.
- **scikit-learn**: For model evaluation and data preprocessing.
- **nltk**: For natural language processing tasks.

## Functionality of Each Function

1. **load_data(file_path)**: Loads the dataset from a specified file path and returns a pandas DataFrame with text and label columns.
2. **preprocess_text(text)**: Preprocesses the input text by converting it to lowercase and removing punctuation.
3. **tokenize_texts(texts)**: Tokenizes the input texts using BERT tokenizer and converts them to input tensors.
4. **evaluate_model(model, input_ids, attention_masks, labels)**: Evaluates the trained model on the input data and returns accuracy, precision, recall, and F1 score.
5. **classify_text(input_text)**: Classifies individual text inputs as positive or negative using the trained model.

## Metrics Used

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive predictions to the total positive predictions.
- **Recall**: The ratio of true positive predictions to the total actual positives.
- **F1 Score**: The harmonic mean of precision and recall.

## Performance

The model's performance on the test dataset is evaluated using the following metrics:

- **Accuracy**: 94%
- **Precision**: 92.17%
- **Recall**: 97.25%
- **F1 Score**: 94.64%

## Challenges

- **Data Preprocessing**: Ensuring that text data is clean and formatted correctly for input into the BERT model.
- **Model Fine-tuning**: Finding the optimal hyperparameters for fine-tuning BERT on the specific task of sentiment classification.
- **Computational Resources**: Training large models like BERT requires significant computational power and memory.

## Conclusion

The Movie Review System effectively classifies movie reviews as positive or negative with high accuracy. The use of BERT for sequence classification provides robust performance, making it a valuable tool for sentiment analysis tasks. Continuous improvements and more extensive training data can further enhance the model's accuracy and generalization capabilities.

## Contributors

- **Chode Nithin** (@Chode-Nithin)

## Feedback and Support

For any issues or suggestions, please contact nithinchode@gmail.com. We appreciate your feedback!
