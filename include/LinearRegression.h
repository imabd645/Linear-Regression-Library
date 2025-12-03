#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <string>
#include <cmath>

class LinearRegression {
private:
    float slope;
    float intercept;
    bool valid;

public:
    LinearRegression();
    LinearRegression(float slope, float intercept, bool valid);

    void fit(const float X[], const float Y[], int size);
    float predict(float x) const;
    void show() const;
    float R_2(float X[], float Y[], int size);
    float MSE(float X[], float Y[], int size);
    float Slope() const;
    float Intercept() const;
    bool Valid() const;
    void save(const std::string& name);
};

#endif // LINEARREGRESSION_H
