#pragma once
#include <QWidget>
#include <vector>

class ConfusionMatrixWidget : public QWidget {
    Q_OBJECT

public:
    explicit ConfusionMatrixWidget(QWidget* parent = nullptr);
    void setConfusionMatrix(const std::vector<std::vector<int>>& matrix);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<std::vector<int>> confusionMatrix;
    int maxCount = 1; // Used for scaling colors
};

