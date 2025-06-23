# week-3-assignment-AI

Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

Primary Differences:

Computation Graphs:
TensorFlow (prior to 2.0): Used static computation graphs. You would define the entire graph first, then execute it. This allowed for more optimizations during deployment and distributed training. However, it made debugging more challenging as you couldn't inspect intermediate values easily.
PyTorch: Uses dynamic computation graphs (also known as "define-by-run"). The graph is built on the fly as operations are executed, making it more intuitive for debugging (similar to regular Python code) and for handling variable-length inputs.
TensorFlow 2.0+: Introduced "Eager Execution" which brings dynamic graphs as the default, significantly bridging this gap with PyTorch. It still supports static graphs (using @tf.function) for performance optimization and deployment.
Ease of Use & Pythonic Nature:
PyTorch: Often considered more "Pythonic" and intuitive for Python developers due to its object-oriented design and dynamic nature, making it easier to learn for those already familiar with Python.
TensorFlow: Historically had a steeper learning curve due to its static graph paradigm. However, with TensorFlow 2.0 and Keras as its high-level API, it has become much more user-friendly and comparable to PyTorch in terms of ease of use.
Deployment and Production Readiness:
TensorFlow: Historically held an advantage in production deployment due to its comprehensive ecosystem (TensorFlow Serving, TensorFlow Lite, TFX) designed for scalable deployment on various platforms (mobile, web, edge devices).
PyTorch: Has significantly improved its production capabilities with tools like TorchServe and ONNX export, making it increasingly viable for production environments.
Community and Adoption:
TensorFlow: Backed by Google, it has broad industry adoption, especially in large-scale enterprise applications.
PyTorch: Has gained significant momentum, particularly in the research community and academia, due to its flexibility and ease of experimentation.
When to choose one over the other:

Choose PyTorch when:
Research and rapid prototyping: Its dynamic graphs and Pythonic nature make it excellent for experimentation, quick iterations, and debugging complex models.
Academic projects: Popular in research due to its flexibility and clear API.
Smaller to medium-scale projects: Where ease of development and iteration speed are priorities.
Choose TensorFlow when:
Large-scale production deployment: Its mature ecosystem (TensorFlow Serving, TF Lite, TFX) provides robust solutions for deploying models at scale across various platforms.
Enterprise-level applications: Often preferred for its established stability, comprehensive tooling, and strong Google backing.
Distributed training on large datasets: TensorFlow has strong built-in support for distributed computing with TPUs and GPUs.
Mobile and embedded device deployment: TensorFlow Lite is specifically designed for this.
Q2: Describe two use cases for Jupyter Notebooks in AI development.

Exploratory Data Analysis (EDA) and Data Preprocessing:

Jupyter Notebooks provide an interactive environment where data scientists can load, inspect, clean, and transform datasets. Cells allow for step-by-step execution, visualizing data distributions, identifying outliers, handling missing values, and performing feature engineering. This iterative process is crucial for understanding the data before building a model. For example, you can load a CSV, display its head, plot histograms of numerical features, and then apply scaling or encoding, all within the same notebook, seeing the results immediately.
Model Prototyping, Training, and Evaluation:

Jupyter Notebooks are ideal for quickly prototyping different AI models. You can define model architectures, train them on small subsets of data, and evaluate their performance with various metrics, all in separate, executable cells. This allows for rapid experimentation with different algorithms, hyperparameters, and network configurations. Visualizing training progress (e.g., loss curves, accuracy plots) and displaying prediction examples directly within the notebook helps in understanding model behavior and iterating efficiently. For instance, you could train a linear regression model, then a neural network, compare their performance side-by-side, and visualize their predictions on sample data.

Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

spaCy significantly enhances NLP tasks compared to basic Python string operations by providing:

Linguistic Annotation and Richer Context:

Basic string operations (like split(), find(), replace()) treat text as a sequence of characters without understanding linguistic structure or meaning.
spaCy, on the other hand, processes text into a Doc object, which is a rich data structure containing token-level annotations. These annotations include: 
Tokenization: Intelligent splitting of text into meaningful units (words, punctuation, etc.), handling contractions and special cases more accurately than simple splitting by space.
Part-of-Speech (POS) Tagging: Assigning grammatical categories (noun, verb, adjective, etc.) to each token.
Dependency Parsing: Identifying grammatical relationships between words (e.g., subject-verb, verb-object).
Lemmatization: Reducing words to their base or dictionary form (e.g., "running" -> "run", "better" -> "good"), which is crucial for reducing vocabulary size and improving model generalization.
Named Entity Recognition (NER): Identifying and classifying named entities (persons, organizations, locations, dates, etc.) in text.
These linguistic annotations provide a much deeper understanding of the text's structure and semantics, enabling more sophisticated NLP applications that go far beyond simple keyword matching.
Efficiency and Production-Readiness:

spaCy is designed for efficiency and production use. It's written in Cython, making it very fast compared to operations on raw Python strings, especially for large volumes of text.
It offers pre-trained statistical models optimized for various languages, allowing for quick setup and high-quality results without extensive manual rule creation.
Basic string operations often require writing complex regex patterns or manual parsing logic, which can be error-prone, difficult to maintain, and inefficient for complex linguistic tasks. spaCy abstracts away much of this complexity, providing robust and performant functionalities out-of-the-box.
2. Comparative Analysis
Compare Scikit-learn and TensorFlow in terms of:

Target applications (e.g., classical ML vs. deep learning):

Scikit-learn: Primarily targets classical machine learning algorithms. It's excellent for tasks like:

Supervised Learning: Classification (e.g., Logistic Regression, SVMs, Decision Trees, Random Forests, Gradient Boosting) and Regression (e.g., Linear Regression, Ridge, Lasso).
Unsupervised Learning: Clustering (e.g., K-Means, DBSCAN), Dimensionality Reduction (e.g., PCA, t-SNE), and Anomaly Detection.
Model Selection and Preprocessing: Tools for cross-validation, hyperparameter tuning, feature scaling, and encoding.
Best suited for: Tabular data, smaller to medium-sized datasets, and problems where traditional ML models are sufficient or provide good baselines.
TensorFlow: Primarily targets deep learning and neural networks. It's designed for:

Complex Neural Network Architectures: Building and training various types of neural networks, including Feedforward Neural Networks (FNNs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers, Generative Adversarial Networks (GANs), etc.
Large-scale Data and High-Dimensional Data: Excels in tasks involving images, text, audio, and other unstructured data, often requiring vast amounts of data and computational resources.
Distributed Training: Built to scale training across multiple GPUs, TPUs, and distributed systems.
Best suited for: Computer Vision, Natural Language Processing, Reinforcement Learning, and other domains where deep learning models achieve state-of-the-art performance. While it can implement classical ML algorithms, it's generally overkill for simpler tasks.
Ease of use for beginners:

Scikit-learn: Generally considered easier for beginners to get started with classical machine learning.

High-level API: Provides a consistent and intuitive API for all its models (.fit(), .predict(), .transform()).
Less boilerplate: Requires minimal code to implement and train common algorithms.
Focus on concepts: Allows beginners to focus on understanding ML concepts rather than intricate implementation details of neural networks.
TensorFlow: Historically had a steeper learning curve for beginners, but has significantly improved with TensorFlow 2.0 and Keras.

TensorFlow 1.x: Required defining static graphs, which was less intuitive and harder to debug.
TensorFlow 2.0 with Keras: Keras, now integrated as TensorFlow's high-level API, makes building and training neural networks much simpler and more user-friendly. It abstracts away much of the low-level complexity.
Still more complex than Scikit-learn: Even with Keras, deep learning concepts (e.g., backpropagation, activation functions, optimizers, complex network architectures) are inherently more complex than classical ML algorithms, requiring a deeper understanding of mathematical foundations. Setting up custom layers, loss functions, or distributed training still demands more expertise.
Community support:

Scikit-learn: Has a mature and active community with extensive documentation, numerous tutorials, and a strong user base.

Well-established: Being one of the oldest and most widely used ML libraries in Python, it has a vast amount of online resources, Stack Overflow answers, and academic papers utilizing it.
Strong support for classical ML: Experts in classical ML actively contribute and maintain the library.
TensorFlow: Has an enormous and highly active community, backed by Google, and is at the forefront of deep learning research and development.

Constantly evolving: Rapid development, frequent updates, and new features driven by cutting-edge research.
Vast resources: Excellent official documentation, numerous Google-backed tutorials, Coursera specializations, and a massive community on platforms like GitHub and Stack Overflow.
Industry and research leader: Being a dominant framework in deep learning, it attracts a huge number of developers and researchers.




# classical_ml_iris.py (or iris_classification.ipynb)

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target) # Numerical labels by default for iris.target

# Optional: If you want to work with string labels and then encode them
# y_labels = pd.Series(iris.target_names[iris.target])

# 2. Data Preprocessing
# No missing values in Iris dataset, so no imputation needed.

# Encode labels (if starting with string labels, otherwise iris.target is already numerical)
# If y was like ['setosa', 'versicolor', 'virginica'], you would do:
# le = LabelEncoder()
# y_encoded = le.fit_transform(y_labels)
# For this dataset, iris.target is already 0, 1, 2, so direct use is fine.

# 3. Split Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# stratify=y ensures that the proportion of target labels is the same in train and test sets

# 4. Train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted') # Use 'weighted' for imbalanced classes or 'macro' for unweighted mean
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Model: Decision Tree Classifier")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plt.figure(figsize=(15,10))
# plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names.tolist())
# plt.show()


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical # Import to_categorical
import numpy as np
import matplotlib.pyplot as plt # For visualization

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Fix 1: Normalize pixel values and reshape for CNN input
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, -1) # Add a channel dimension
X_test = np.expand_dims(X_test, -1)

# Fix 2: One-hot encode labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'), # Added an extra Conv layer for better accuracy
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'), # Fix 3: Added a hidden Dense layer
    Dense(num_classes, activation='softmax')
])

# Fix 4: Correct loss function for one-hot encoded labels
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Fix 5: Add validation split to monitor overfitting
history = model.fit(X_train, y_train,
                    epochs=10, # Increased epochs for better accuracy
                    batch_size=64,
                    validation_split=0.1) # Use 10% of training data for validation

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Example visualization (from Task 2, good for debugging too)
predictions = model.predict(X_test)
plt.figure(figsize=(10, 8))
for i in range(5):
    idx = np.random.randint(0, len(X_test))
    image = X_test[idx]
    true_label = np.argmax(y_test[idx])
    predicted_label = np.argmax(predictions[idx])

    plt.subplot(1, 5, i + 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {true_label}\nPred: {predicted_label}")
    plt.axis('off')
plt.suptitle('MNIST Sample Predictions (Fixed Code)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
