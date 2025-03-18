#include "Metric.h"
#include <unordered_map>
#include <algorithm>
#include <numeric>

double Accuracy::compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    int correct = 0;
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (trueLabels[i] == predictedLabels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / trueLabels.size();
}

double Precision::compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    std::unordered_map<int, int> truePositive, falsePositive;
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (predictedLabels[i] == trueLabels[i]) {
            ++truePositive[trueLabels[i]];
        }
        else {
            ++falsePositive[predictedLabels[i]];
        }
    }
    double precisionSum = 0.0;
    for (const auto& [label, tp] : truePositive) {
        precisionSum += static_cast<double>(tp) / (tp + falsePositive[label]);
    }
    return precisionSum / truePositive.size();
}

double Recall::compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    std::unordered_map<int, int> truePositive, falseNegative;
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (predictedLabels[i] == trueLabels[i]) {
            ++truePositive[trueLabels[i]];
        }
        else {
            ++falseNegative[trueLabels[i]];
        }
    }
    double recallSum = 0.0;
    for (const auto& [label, tp] : truePositive) {
        recallSum += static_cast<double>(tp) / (tp + falseNegative[label]);
    }
    return recallSum / truePositive.size();
}

std::vector<std::vector<int>> ConfusionMatrix::computeMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    int numClasses = *std::max_element(trueLabels.begin(), trueLabels.end()) + 1;
    std::vector<std::vector<int>> matrix(numClasses, std::vector<int>(numClasses, 0));
    for (size_t i = 0; i < trueLabels.size(); ++i) {
        matrix[trueLabels[i]][predictedLabels[i]]++;
    }
    return matrix;
}

double ConfusionMatrix::compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    auto matrix = computeMatrix(trueLabels, predictedLabels);
    int correct = 0, total = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
        correct += matrix[i][i];
        total += std::accumulate(matrix[i].begin(), matrix[i].end(), 0);
    }
    return static_cast<double>(correct) / total;
}