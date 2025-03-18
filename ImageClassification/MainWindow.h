#pragma once
#include "Classifier.h"
#include "confusionmatrixwidget.h"
#include <QMainWindow>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QTableWidget>
#include <QSpinBox>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void onTrainClicked();
    void onPredictClicked();
    void onEvaluateClicked();
    void onSaveClicked();
    void onLoadClicked();
    void onClassifierChanged(int index);

private:
    QPushButton* trainButton;
    QPushButton* predictButton;
    QPushButton* evaluateButton;
    QPushButton* saveButton;
    QPushButton* loadButton;
    QLabel* statusLabel;
    QLineEdit* trainPathEdit;
    QLineEdit* testPathEdit;
    QComboBox* classifierComboBox;
    QSpinBox* kSpinBox; 
    QTableWidget* resultTable;
    ConfusionMatrixWidget* confusionMatrixWidget;
    Classifier* classifier;

    void displayConfusionMatrix(const std::vector<std::vector<int>>& matrix);
};


