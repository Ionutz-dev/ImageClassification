#include "KNNClassifier.h"
#include <algorithm>
#include <cmath>
#include <fstream>

KNNClassifier::KNNClassifier(int k) : k(k) {}

void KNNClassifier::fit(const T& trainData) {
    trainImages = trainData.getImages();
    trainLabels = trainData.getLabels();
}

std::vector<int> KNNClassifier::predict(const T& testData) {
    const auto& testImages = testData.getImages();
    std::vector<int> predictions;
    for (const auto& image : testImages) {
        std::vector<std::pair<int, int>> distances;
        for (size_t i = 0; i < trainImages.size(); ++i) {
            int distance = getDistance(image, trainImages[i]);
            distances.push_back({ distance, trainLabels[i] });
        }
        std::sort(distances.begin(), distances.end());
        std::vector<int> kLabels(k);
        for (int i = 0; i < k; ++i) {
            kLabels[i] = distances[i].second;
        }
        std::sort(kLabels.begin(), kLabels.end());
        int majorityLabel = kLabels[k / 2];
        predictions.push_back(majorityLabel);
    }
    return predictions;
}

bool KNNClassifier::save(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    size_t trainSize = trainImages.size();
    file.write(reinterpret_cast<const char*>(&trainSize), sizeof(trainSize));
    for (const auto& img : trainImages) {
        for (int pixel : img) {
            file.write(reinterpret_cast<const char*>(&pixel), sizeof(pixel));
        }
    }
    for (int label : trainLabels) {
        file.write(reinterpret_cast<const char*>(&label), sizeof(label));
    }
    file.close();
    return true;
}

bool KNNClassifier::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    size_t trainSize;
    file.read(reinterpret_cast<char*>(&trainSize), sizeof(trainSize));
    trainImages.resize(trainSize, std::vector<int>(784));
    trainLabels.resize(trainSize);
    for (auto& img : trainImages) {
        for (int& pixel : img) {
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
        }
    }
    for (int& label : trainLabels) {
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
    }
    file.close();
    return true;
}

double KNNClassifier::eval(const T& testData) {
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

int KNNClassifier::getDistance(const std::vector<int>& img1, const std::vector<int>& img2) {
    int distance = 0;
    for (size_t i = 0; i < img1.size(); ++i) {
        distance += std::pow(img1[i] - img2[i], 2);
    }
    return std::sqrt(distance);
}