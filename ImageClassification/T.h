#pragma once
#include <vector>
#include <string>

class T {
public:
    T(const std::string& filepath);
    const std::vector<std::vector<int>>& getImages() const;
    const std::vector<int>& getLabels() const;

private:
    std::vector<std::vector<int>> images;
    std::vector<int> labels;
    void loadCSV(const std::string& filepath);
    void binarizeImages();
};

