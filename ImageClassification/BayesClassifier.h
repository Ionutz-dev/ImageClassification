#pragma once
#include "T.h"
#include "Classifier.h"
#include <vector>

class BayesClassifier : public Classifier {
public:
    void fit(const T& trainData) override;
    std::vector<int> predict(const T& testData) override;
    bool save(const std::string& filepath) override;
    bool load(const std::string& filepath) override;
    double eval(const T& testData) override;

private:
    std::vector<double> priors;
    std::vector<std::vector<double>> likelihoods;
    void calculatePriors(const std::vector<int>& trainLabels);
    void calculateLikelihoods(const std::vector<std::vector<int>>& trainImages, const std::vector<int>& trainLabels);
};

