#ifndef SVM_H
#define SVM_H
#include "dataframe.h"
#include "mllibrary.h"
#include <iostream>


class SVM
{
public:
    //конструктор по умолчанию
    SVM();

    //конструктор инициализации
    SVM(double Alpha, double W_delta, int Iter_num);

    void fit(const DataFrame<double>& X_train, const DataFrame<double>& y_train);
    std::vector<double> predict(const DataFrame<double> &X) const;

private:
    //learning rate
    double alpha;

    double w_delta;
    // number of iterations
    int iter_num;
    DataFrame<double> weight;
};

#endif // SVM_H
