#include "BayesClassifier.h"
#include <cmath>
#include <fstream>
#include <unordered_map>

void BayesClassifier::fit(const T& trainData) {
    const auto& trainImages = trainData.getImages();
    const auto& trainLabels = trainData.getLabels();
    calculatePriors(trainLabels);
    calculateLikelihoods(trainImages, trainLabels);
}

std::vector<int> BayesClassifier::predict(const T& testData) {
    const auto& testImages = testData.getImages();
    std::vector<int> predictions;
    for (const auto& image : testImages) {
        std::vector<double> logPosteriors(priors.size(), 0.0);
        for (size_t i = 0; i < priors.size(); ++i) {
            logPosteriors[i] = std::log(priors[i]);
            for (size_t j = 0; j < image.size(); ++j) {
                logPosteriors[i] += std::log(image[j] == 255 ? likelihoods[i][j] : 1 - likelihoods[i][j]);
            }
        }
        int predictedClass = std::distance(logPosteriors.begin(), std::max_element(logPosteriors.begin(), logPosteriors.end()));
        predictions.push_back(predictedClass);
    }
    return predictions;
}

bool BayesClassifier::save(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    size_t numClasses = priors.size();
    size_t numFeatures = likelihoods[0].size();
    file.write(reinterpret_cast<const char*>(&numClasses), sizeof(numClasses));
    file.write(reinterpret_cast<const char*>(&numFeatures), sizeof(numFeatures));
    for (double prior : priors) {
        file.write(reinterpret_cast<const char*>(&prior), sizeof(prior));
    }
    for (const auto& likelihood : likelihoods) {
        for (double value : likelihood) {
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    }
    file.close();
    return true;
}

bool BayesClassifier::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    size_t numClasses, numFeatures;
    file.read(reinterpret_cast<char*>(&numClasses), sizeof(numClasses));
    file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures));
    priors.resize(numClasses);
    likelihoods.resize(numClasses, std::vector<double>(numFeatures));
    for (double& prior : priors) {
        file.read(reinterpret_cast<char*>(&prior), sizeof(prior));
    }
    for (auto& likelihood : likelihoods) {
        for (double& value : likelihood) {
            file.read(reinterpret_cast<char*>(&value), sizeof(value));
        }
    }
    file.close();
    return true;
}

double BayesClassifier::eval(const T& testData) {
    const auto& testImages = testData.getImages();
    const auto& testLabels = testData.getLabels();
    std::vector<int> predictions = predict(testData);
    int correct = 0;
    for (size_t i = 0; i < testLabels.size(); ++i) {
        if (predictions[i] == testLabels[i]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / testLabels.size();
}

void BayesClassifier::calculatePriors(const std::vector<int>& trainLabels) {
    std::unordered_map<int, int> classCounts;
    for (int label : trainLabels) {
        ++classCounts[label];
    }
    priors.resize(classCounts.size());
    int totalSamples = trainLabels.size();
    for (const auto& [label, count] : classCounts) {
        priors[label] = static_cast<double>(count) / totalSamples;
    }
}

void BayesClassifier::calculateLikelihoods(const std::vector<std::vector<int>>& trainImages, const std::vector<int>& trainLabels) {
    size_t numClasses = priors.size();
    size_t numFeatures = trainImages[0].size();
    likelihoods.resize(numClasses, std::vector<double>(numFeatures, 1e-5));
    std::vector<std::unordered_map<int, int>> featureCounts(numClasses, std::unordered_map<int, int>());
    for (size_t i = 0; i < trainImages.size(); ++i) {
        int label = trainLabels[i];
        for (size_t j = 0; j < trainImages[i].size(); ++j) {
            if (trainImages[i][j] == 255) {
                ++featureCounts[label][j];
            }
        }
    }
    for (size_t i = 0; i < numClasses; ++i) {
        for (size_t j = 0; j < numFeatures; ++j) {
            likelihoods[i][j] = static_cast<double>(featureCounts[i][j] + 1) / (std::count(trainLabels.begin(), trainLabels.end(), i) + numClasses);
        }
    }
}
