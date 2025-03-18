#include "ConfusionMatrixWidget.h"
#include <QPainter>
#include <QColor>
#include <algorithm>

ConfusionMatrixWidget::ConfusionMatrixWidget(QWidget* parent) : QWidget(parent) {
}

void ConfusionMatrixWidget::setConfusionMatrix(const std::vector<std::vector<int>>& matrix) {
    confusionMatrix = matrix;
    maxCount = 1;
    for (const auto& row : matrix) {
        maxCount = std::max(maxCount, *std::max_element(row.begin(), row.end()));
    }
    update();
}

void ConfusionMatrixWidget::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);

    if (confusionMatrix.empty()) return;

    QPainter painter(this);
    int numClasses = confusionMatrix.size();
    int margin = 40; 
    int labelOffset = -40; 
    int cellWidth = (width() - margin - labelOffset) / (numClasses + 1);
    int cellHeight = (height() - margin - labelOffset) / (numClasses + 1);
    int cellPadding = 2; 

    for (int i = 0; i < numClasses; ++i) {
        for (int j = 0; j < numClasses; ++j) {
            QRect rect(margin + labelOffset + (j + 1) * cellWidth + cellPadding / 2, margin + labelOffset + (i + 1) * cellHeight + cellPadding / 2, cellWidth - cellPadding, cellHeight - cellPadding);

            int value = confusionMatrix[i][j];
            double intensity = static_cast<double>(value) / maxCount;
            QColor color = QColor::fromHsv(0, static_cast<int>(255 * intensity), 255);  
            painter.fillRect(rect, color);

            painter.setPen(Qt::black);
            painter.drawRect(rect);

            QString text = QString::number(value);
            painter.drawText(rect, Qt::AlignCenter, text);
        }
    }

    painter.setPen(Qt::white);
    QFont font = painter.font();
    font.setBold(true);
    painter.setFont(font);
    for (int i = 0; i < numClasses; ++i) {
        painter.drawText(margin + labelOffset + (i + 1) * cellWidth, margin / 2, cellWidth, margin / 2, Qt::AlignCenter, QString::number(i));  // X axis
        painter.drawText(margin / 2, margin + labelOffset + (i + 1) * cellHeight, margin / 2, cellHeight, Qt::AlignCenter, QString::number(i));  // Y axis
    }

    painter.drawText(margin + labelOffset + (numClasses + 1) * cellWidth / 2, margin / 2, "Predicted Class");
    painter.save();
    painter.rotate(-90);
    painter.drawText(-height() / 2, margin / 2, "True Class");
    painter.restore();
}