#ifndef LINEARREG_H
#define LINEARREG_H
#include "dataframe.h"
#include "mllibrary.h"
#include "iostream"
#include "mllibrary.h"


class LinearReg
{
public:
    //конструктор по умолчанию
    LinearReg();

    //конструктор инициализации
    LinearReg(double Alpha, double W_delta, int Iter_num);


    void fit(const DataFrame<double>& X_train, const DataFrame<double>& y_train);
    std::vector<double> predict(const DataFrame<double> &X) const;

    DataFrame<double> weight;

private:
    //learning rate
    double alpha;
    double w_delta;
    // number of iterations
    int iter_num;
    // features importans

};

#endif // LINEARREG_H
