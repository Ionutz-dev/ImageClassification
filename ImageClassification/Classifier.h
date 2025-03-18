#pragma once
#include "T.h"
#include <vector>
#include <string>

class Classifier {
public:
    virtual void fit(const T& trainData) = 0;
    virtual std::vector<int> predict(const T& testData) = 0;
    virtual bool save(const std::string& filepath) = 0;
    virtual bool load(const std::string& filepath) = 0;
    virtual double eval(const T& testData) = 0;
    virtual ~Classifier() {}
};

