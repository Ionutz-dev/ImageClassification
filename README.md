# Image Classification

## Overview

**ImageClassification** is a C++ machine learning application with a Qt-based GUI for classifying images using **K-Nearest Neighbors (KNN)** and **Naive Bayes** algorithms. It uses the popular **MNIST dataset** in CSV format and displays results, including a **confusion matrix**, in a clean user interface.

## Features

- 🧠 **Machine Learning Classifiers**:
  - **KNNClassifier** – Distance-based classification.
  - **BayesClassifier** – Probability-based classification.
- 📊 **Confusion Matrix Widget** – Visual representation of model accuracy.
- 📂 **CSV Data Handling** – Uses `mnist_train.csv` and `mnist_test.csv`.
- 🖥️ **Qt GUI Interface** – Built with Qt Designer (`.ui` file).
- 📈 **Modular Design** – Supports new classifier integration via `Classifier.h` interface.

## Technologies Used

- **C++**
- **Qt Framework (Qt Widgets)**
- **Visual Studio** (Project Files Included)
- **MNIST Dataset (CSV Format)**

## Project Structure

```
ImageClassification/
│
├── main.cpp                           # Application entry point
├── MainWindow.cpp/.h                  # GUI logic and event handling
├── BayesClassifier.cpp/.h            # Naive Bayes implementation
├── KNNClassifier.cpp/.h              # K-Nearest Neighbors implementation
├── Metric.cpp/.h                     # Distance metric calculations
├── ConfusionMatrixWidget.cpp/.h      # GUI component for confusion matrix
├── Classifier.h                      # Abstract classifier interface
├── T.cpp/.h                          # Utility or template logic
│
├── mnist_train.csv                   # Training data
├── mnist_test.csv                    # Test data
│
├── ImageClassification.ui            # GUI layout (Qt Designer)
├── ImageClassification.qrc           # Qt resource file
├── ImageClassification.vcxproj       # Visual Studio project file
└── *.filters/.user                   # VS config files
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ionutz-dev/ImageClassification.git
   cd ImageClassification
   ```

2. **Open in Visual Studio with Qt Installed**:
   - Double-click `ImageClassification.vcxproj`.
   - Ensure Qt tools are properly integrated into Visual Studio.

3. **Build & Run**:
   - Press `Ctrl + Shift + B` to build the project.
   - Press `F5` to run.

## Sample Usage

- Load training data (`mnist_train.csv`).
- Load test data (`mnist_test.csv`).
- Select classifier: KNN or Bayes.
- Run classification.
- View the **confusion matrix** and accuracy metrics in the GUI.
