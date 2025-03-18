#include "MainWindow.h"
#include "T.h"
#include "Metric.h"
#include "KNNClassifier.h"
#include "BayesClassifier.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), classifier(nullptr) {
    setWindowTitle("Image Classification");

    trainButton = new QPushButton("Train Classifier", this);
    predictButton = new QPushButton("Predict", this);
    evaluateButton = new QPushButton("Evaluate", this);
    saveButton = new QPushButton("Save Classifier", this);
    loadButton = new QPushButton("Load Classifier", this);

    trainPathEdit = new QLineEdit(this);
    testPathEdit = new QLineEdit(this);

    classifierComboBox = new QComboBox(this);
    classifierComboBox->addItem("K-Nearest Neighbors");
    classifierComboBox->addItem("Naive Bayes");

    kSpinBox = new QSpinBox(this);
    kSpinBox->setRange(1, 100);
    kSpinBox->setValue(3);
    kSpinBox->setEnabled(false); 

    resultTable = new QTableWidget(this);
    resultTable->setColumnCount(2);
    resultTable->setHorizontalHeaderLabels({ "True Label", "Predicted Label" });

    confusionMatrixWidget = new ConfusionMatrixWidget(this);
    confusionMatrixWidget->setMinimumSize(500, 500); 

    statusLabel = new QLabel(this);

    QVBoxLayout* inputFormLayout = new QVBoxLayout;
    QHBoxLayout* trainLayout = new QHBoxLayout;
    QHBoxLayout* testLayout = new QHBoxLayout;

    trainLayout->addWidget(new QLabel("Train Path:"));
    trainLayout->addWidget(trainPathEdit);
    trainLayout->addWidget(trainButton);

    testLayout->addWidget(new QLabel("Test Path:"));
    testLayout->addWidget(testPathEdit);
    testLayout->addWidget(predictButton);
    testLayout->addWidget(evaluateButton);

    inputFormLayout->addLayout(trainLayout);
    inputFormLayout->addLayout(testLayout);
    inputFormLayout->addWidget(new QLabel("Classifier:"));
    inputFormLayout->addWidget(classifierComboBox);
    inputFormLayout->addWidget(new QLabel("K value for KNN:"));
    inputFormLayout->addWidget(kSpinBox);
    inputFormLayout->addWidget(saveButton);
    inputFormLayout->addWidget(loadButton);
    inputFormLayout->addWidget(statusLabel);
    inputFormLayout->addWidget(resultTable);

    QHBoxLayout* mainLayout = new QHBoxLayout;
    mainLayout->addLayout(inputFormLayout);
    mainLayout->addWidget(confusionMatrixWidget, 1); 

    QWidget* centralWidget = new QWidget(this);
    centralWidget->setLayout(mainLayout);
    setCentralWidget(centralWidget);

    connect(trainButton, &QPushButton::clicked, this, &MainWindow::onTrainClicked);
    connect(predictButton, &QPushButton::clicked, this, &MainWindow::onPredictClicked);
    connect(evaluateButton, &QPushButton::clicked, this, &MainWindow::onEvaluateClicked);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::onSaveClicked);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadClicked);
    connect(classifierComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onClassifierChanged);

    onClassifierChanged(classifierComboBox->currentIndex());
}

MainWindow::~MainWindow() {
    delete classifier;
}

void MainWindow::onClassifierChanged(int index) {
    if (index == 0) { 
        kSpinBox->setEnabled(true);
    }
    else { 
        kSpinBox->setEnabled(false);
    }
}

void MainWindow::onTrainClicked() {
    QString trainPath = QFileDialog::getOpenFileName(this, "Select Train Dataset");
    if (trainPath.isEmpty()) {
        return;
    }
    trainPathEdit->setText(trainPath);

    try {
        T trainData(trainPath.toStdString());
        if (classifierComboBox->currentIndex() == 0) {
            classifier = new KNNClassifier(kSpinBox->value());
        }
        else {
            classifier = new BayesClassifier();
        }
        classifier->fit(trainData);
        statusLabel->setText("Training completed successfully.");
    }
    catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", e.what());
    }
}

void MainWindow::onPredictClicked() {
    QString testPath = QFileDialog::getOpenFileName(this, "Select Test Dataset");
    if (testPath.isEmpty()) {
        return;
    }
    testPathEdit->setText(testPath);

    try {
        T testData(testPath.toStdString());
        std::vector<int> predictions = classifier->predict(testData);
        const auto& testLabels = testData.getLabels();

        resultTable->setRowCount(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            resultTable->setItem(i, 0, new QTableWidgetItem(QString::number(testLabels[i])));
            resultTable->setItem(i, 1, new QTableWidgetItem(QString::number(predictions[i])));
        }
    }
    catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", e.what());
    }
}

void MainWindow::onEvaluateClicked() {
    QString testPath = testPathEdit->text();
    if (testPath.isEmpty()) {
        QMessageBox::warning(this, "Error", "Please select a test dataset.");
        return;
    }

    try {
        T testData(testPath.toStdString());
        std::vector<int> predictions = classifier->predict(testData);
        const auto& testLabels = testData.getLabels();

        ConfusionMatrix confusionMatrix;
        auto matrix = confusionMatrix.computeMatrix(testLabels, predictions);
        displayConfusionMatrix(matrix);

        double accuracy = confusionMatrix.compute(testLabels, predictions);
        statusLabel->setText("Accuracy: " + QString::number(accuracy));
    }
    catch (const std::exception& e) {
        QMessageBox::warning(this, "Error", e.what());
    }
}

void MainWindow::onSaveClicked() {
    QString savePath = QFileDialog::getSaveFileName(this, "Save Classifier");
    if (savePath.isEmpty()) {
        return;
    }
    if (classifier && classifier->save(savePath.toStdString())) {
        QMessageBox::information(this, "Success", "Classifier saved successfully.");
    }
    else {
        QMessageBox::warning(this, "Error", "Failed to save classifier.");
    }
}

void MainWindow::onLoadClicked() {
    QString loadPath = QFileDialog::getOpenFileName(this, "Load Classifier");
    if (loadPath.isEmpty()) {
        return;
    }
    if (classifierComboBox->currentIndex() == 0) {
        classifier = new KNNClassifier(kSpinBox->value());
    }
    else {
        classifier = new BayesClassifier();
    }
    if (classifier->load(loadPath.toStdString())) {
        QMessageBox::information(this, "Success", "Classifier loaded successfully.");
    }
    else {
        QMessageBox::warning(this, "Error", "Failed to load classifier.");
    }
}

void MainWindow::displayConfusionMatrix(const std::vector<std::vector<int>>& matrix) {
    confusionMatrixWidget->setConfusionMatrix(matrix);
}