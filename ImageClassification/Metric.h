#pragma once
#include <vector>

class Metric {
public:
    virtual double compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) = 0;
    virtual ~Metric() {}
};

class Accuracy : public Metric {
public:
    double compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) override;
};

class Precision : public Metric {
public:
    double compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) override;
};

class Recall : public Metric {
public:
    double compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) override;
};

class ConfusionMatrix : public Metric {
public:
    std::vector<std::vector<int>> computeMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels);
    double compute(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) override;
};

