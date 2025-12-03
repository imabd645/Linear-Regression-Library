#include <iostream>
#include "LinearRegression.h"

using namespace std;

int main() {
    float X[] = {1, 2, 3, 4, 5};
    float Y[] = {2, 4, 5, 4, 5};
    int size = 5;

    LinearRegression lr;
    lr.fit(X, Y, size);
    lr.show();

    cout << "Predict for x = 6: " << lr.predict(6) << endl;
    cout << "R^2: " << lr.R_2(X, Y, size) << endl;
    cout << "MSE: " << lr.MSE(X, Y, size) << endl;

    lr.save("model.txt");
    return 0;
}
