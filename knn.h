#ifndef KNN_H
#define KNN_H
#include "dataframe.h"
#include "mllibrary.h"
#include<limits>
#include <vector>

class Knn
{

public:
    //конструктор по умолчанию
    Knn();

    // конструктор копирования
    Knn(const Knn& other);

    DataFrame<double> get_X_train()const{return X_train;}
    DataFrame<double> get_y_train()const{return y_train;}

    // train knn
    void fit(DataFrame<double> X, DataFrame<double> y, int k); // where k is number of neighbors

    std::vector<double> predict(DataFrame<double> X);

private:
    //train data
    DataFrame<double> X_train;
    DataFrame<double> y_train;

    //number of neighbors
    int neighbors;
};


#endif // KNN_H
