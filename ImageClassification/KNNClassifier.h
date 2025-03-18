#pragma once
#include "T.h"
#include "Classifier.h"
#include <vector>

class KNNClassifier : public Classifier {
public:
    KNNClassifier(int k = 3);
    void fit(const T& trainData) override;
    std::vector<int> predict(const T& testData) override;
    bool save(const std::string& filepath) override;
    bool load(const std::string& filepath) override;
    double eval(const T& testData) override;

private:
    int k;
    std::vector<std::vector<int>> trainImages;
    std::vector<int> trainLabels;
    int getDistance(const std::vector<int>& img1, const std::vector<int>& img2);
};

