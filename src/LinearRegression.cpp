#include "LinearRegression.h"
#include <iostream>
#include <fstream>

using namespace std;

LinearRegression::LinearRegression() : slope(0.0f), intercept(0.0f), valid(false) {}
LinearRegression::LinearRegression(float slope, float intercept, bool valid) 
    : slope(slope), intercept(intercept), valid(valid) {}

void LinearRegression::fit(const float X[], const float Y[], int size) {
    if (size < 2) {
        cout << "Error! Not enough data points." << endl;
        valid = false;
        return;
    }

    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < size; i++) {
        sum_x += X[i];
        sum_y += Y[i];
        sum_xy += X[i] * Y[i];
        sum_xx += X[i] * X[i];
    }

    float numerator = (size * sum_xy) - (sum_x * sum_y);
    float denominator = (size * sum_xx) - (sum_x * sum_x);

    if (fabs(denominator) < 1e-10) {
        cout << "Error! Cannot fit line (Insufficient variance in X)." << endl;
        valid = false;
        return;
    }

    slope = numerator / denominator;
    intercept = (sum_y - slope * sum_x) / size;
    valid = true;
    cout << "Model Trained Successfully!" << endl;
}

float LinearRegression::predict(float x) const {
    if (!valid) {
        cout << "Error! Model not trained or invalid." << endl;
        return NAN;
    }
    return slope * x + intercept;
}

void LinearRegression::show() const {
    if (valid)
        cout << "Slope: " << slope << ", Intercept: " << intercept << endl;
    else
        cout << "Model is invalid or not trained yet." << endl;
}

float LinearRegression::R_2(float X[], float Y[], int size) {
    float difference = 0, sum_y = 0;
    for (int i = 0; i < size; i++) {
        float error = Y[i] - (slope * X[i] + intercept);
        difference += error * error;
        sum_y += Y[i];
    }

    float mean = sum_y / size;
    float var = 0;
    for (int i = 0; i < size; i++)
        var += (Y[i] - mean) * (Y[i] - mean);

    return 1 - (difference / var);
}

float LinearRegression::MSE(float X[], float Y[], int size) {
    float difference = 0;
    for (int i = 0; i < size; i++) {
        float error = Y[i] - (X[i] * slope + intercept);
        difference += error * error;
    }
    return difference / size;
}

float LinearRegression::Slope() const { return slope; }
float LinearRegression::Intercept() const { return intercept; }
bool LinearRegression::Valid() const { return valid; }

void LinearRegression::save(const string& name) {
    if (!valid) {
        cout << "Error! Model is invalid or not trained. Cannot save." << endl;
        return;
    }

    ofstream file(name);
    if (file.is_open()) {
        file << "Slope: " << slope << endl;
        file << "Intercept: " << intercept << endl;
        file.close();
        cout << "Model saved successfully!" << endl;
    } else {
        cout << "Error! Could not open file to save model." << endl;
    }
}
