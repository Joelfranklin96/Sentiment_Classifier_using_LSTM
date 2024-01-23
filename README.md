# Emojifier-V1 and Emojifier-V2: Sentiment Classification using GloVe Embeddings and LSTM Networks

## Project Overview
This project focuses on building two versions of a sentiment classifier, Emojifier-V1 and Emojifier-V2, using pre-trained GloVe word embeddings and LSTM networks. The aim is to predict emojis for sentences based on their sentiment. Emojifier-V1 is a baseline model that averages GloVe embeddings, while Emojifier-V2 is a more sophisticated model utilizing LSTM networks.

## Dataset
- **Training Data**: 132 sentences (`X_train`) with corresponding emoji labels (`Y_train`).
- **Test Data**: 56 sentences (`X_test`) with labels (`Y_test`).
- **Labels**: Range from 0 to 4, representing different emojis.

## Emojifier-V1: Baseline Model
- **Average Embedding**: Each sentence is converted to an average of its word embeddings.
- **Model Architecture**: The averaged embedding is passed through a dense layer followed by a softmax activation.
- **Challenges**: This model doesn't account for word order, affecting its predictive performance.

## Emojifier-V2: LSTM Model
- **Preprocessing**: Sentences are converted to indices and padded for uniform length.
- **Embedding Layer**: Utilizes GloVe embeddings as pre-trained weights.
- **LSTM Layers**: Two LSTM layers capture sequence information.
  - First LSTM with `return_sequences=True`.
  - Second LSTM with `return_sequences=False`.
- **Regularization**: Dropout layers with a rate of 0.5 after each LSTM layer.
- **Output Layer**: A dense layer with softmax activation produces the final predictions.
- **Training**: The model is compiled with categorical cross-entropy loss and optimized using Adam.

## Training and Evaluation
- The models are trained on the training dataset and evaluated on the test dataset.
- Emojifier-V1 provides insights into the importance of sequence modeling.
- Emojifier-V2 demonstrates the effectiveness of LSTMs in capturing sequential dependencies.

## Key Takeaways
- Sequence padding is crucial for mini-batch processing in LSTM networks.
- Pre-trained word embeddings can be fine-tuned to adapt to specific tasks.
- Dropout is a simple yet effective regularization technique for LSTMs.

## How to Run
- Run each cell in the provided Jupyter Notebook to replicate the training and evaluation process.
- Modify the input sentences in the test section to see the model predictions.

## Future Work
- Experiment with deeper LSTM architectures or different types of RNNs.
- Explore the impact of different word embeddings on model performance.
- Extend the model to classify a broader range of sentiments and emojis.
