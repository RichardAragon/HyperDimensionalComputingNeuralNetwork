# Hyperdimensional Computing Neural Network (HDCNN)

This repository contains a novel implementation of a `Hyperdimensional Computing Neural Network (HDCNN)` applied to the AG News dataset for multi-class text classification. This approach leverages hyperdimensional computing principles to encode text sequences into high-dimensional vectors and applies a simple neural network for classification tasks. Additionally, a Naive Bayes classifier is used as a baseline comparison.

## Key Features:
- **Hyperdimensional Computing (HD)**: Utilizes high-dimensional vectors (hypervectors) for representing data, offering a robust method for binding and superposition operations.
- **Custom AG News Dataset Processing**: Implements a custom dataset class that tokenizes and encodes text data into hypervectors for training the neural network.
- **Neural Network Architecture**: Employs a feedforward neural network (`HDCNNClassifier`) with ReLU activations and dropout layers for classification.
- **Naive Bayes Baseline**: A traditional Naive Bayes classifier is included as a baseline to compare performance against the HDCNN.
- **Early Stopping**: Training includes early stopping to prevent overfitting, based on validation loss.
- **Reproducibility**: The random seed is set for NumPy, PyTorch, and Python's `random` to ensure reproducible results.

### Parameters:
- `dim`: The dimension of the hypervectors used for encoding (default: 5000).
- `num_epochs`: Number of training epochs (default: 5).
- `batch_size`: Batch size for training (default: 128).
- `learning_rate`: Learning rate for the optimizer (default: 0.001).
- `max_vocab_size`: Maximum size of the vocabulary (default: 5000).
- `max_seq_len`: Maximum sequence length for tokenized input (default: 50).

### Output:
- **Training & Validation**: The model will display training loss and validation loss after each epoch.
- **Naive Bayes Baseline**: Displays the accuracy of the Naive Bayes model.
- **Best Model**: The best-performing model based on validation loss is saved as `best_model.pth`.

### Testing the Model
After training, the script evaluates the model on the test set and prints the final test loss and accuracy.

## Results

- **Naive Bayes Baseline Accuracy**: The Naive Bayes classifier is trained and evaluated to provide a baseline accuracy for comparison with the HDCNN model.
- **HDCNN Model Performance**: The final test accuracy and loss are reported after training and evaluation on the AG News dataset.

## Future Work:
- **Hyperdimensional Encoding Extensions**: Experiment with different binding and superposition techniques.
- **Dataset Generalization**: Test the HDCNN on other datasets to explore its generalization capability.
- **Optimization**: Improve the neural network architecture for better accuracy and faster convergence.

## License:
This project is licensed under the MIT License.

Feel free to contribute by submitting issues or pull requests!
