#include "T.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

T::T(const std::string& filepath) {
    loadCSV(filepath);
    binarizeImages();
}

const std::vector<std::vector<int>>& T::getImages() const {
    return images;
}

const std::vector<int>& T::getLabels() const {
    return labels;
}

void T::loadCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string line;
    int lineNumber = 0;

    if (std::getline(file, line)) {
        lineNumber++;
        if (line.find("label") == std::string::npos) {
            throw std::runtime_error("Expected header line not found");
        }
    }

    while (std::getline(file, line)) {
        lineNumber++;
        std::stringstream ss(line);
        std::string value;
        std::vector<int> image;
        int label;

        if (!std::getline(ss, value, ',')) {
            throw std::runtime_error("Missing label in line " + std::to_string(lineNumber));
        }

        try {
            label = std::stoi(value);
        }
        catch (const std::invalid_argument& e) {
            throw std::runtime_error("Invalid label value in line " + std::to_string(lineNumber) + ": " + value);
        }

        labels.push_back(label);

        while (std::getline(ss, value, ',')) {
            try {
                image.push_back(std::stoi(value));
            }
            catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid pixel value in line " + std::to_string(lineNumber) + ": " + value);
            }
        }

        if (image.size() != 784) {
            throw std::runtime_error("Invalid number of pixels in line " + std::to_string(lineNumber) + ": expected 784, got " + std::to_string(image.size()));
        }

        images.push_back(image);
    }

    file.close();
}

void T::binarizeImages() {
    for (auto& image : images) {
        for (auto& pixel : image) {
            pixel = (pixel > 127) ? 255 : 0;
        }
    }
}
