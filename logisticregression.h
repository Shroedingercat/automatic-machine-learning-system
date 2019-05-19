#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H
#include <iostream>
#include <cmath>
#include "dataframe.h"
#include "mllibrary.h"


class LogisticRegression
{
public:
    LogisticRegression();

    void fit(const DataFrame<double> &X, const DataFrame<double> &y, double Alpha = 0.0001, double Delta = 0.1);
    void fit_multi_class(const DataFrame<double> &X_train, const DataFrame<double> &y_train, int class_num, double Alpha = 0.0001, double Delta = 0.1);

    DataFrame<double> sigmoid(const DataFrame<double> &X, const DataFrame<double> & W) const;

    std::vector<double> predict(const DataFrame<double> &X) const;

    std::vector<double> bin_y(const DataFrame<double> &y, double target);

    std::vector<double> predict_multi_class(const DataFrame<double> &X) const;

private:
    DataFrame<double> weight;
    std::vector<std::vector<double>> Weights;
    std::vector<double> targets;
    double alpha, delta;
    int iter_num;
};

#endif // LOGISTICREGRESSION_H
