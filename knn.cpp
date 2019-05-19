#include "knn.h"

Knn::Knn()
{
    X_train = DataFrame<double>();
    y_train = DataFrame<double>();
    neighbors = 1;
}

Knn::Knn(const Knn &other)
{
    X_train = get_X_train();
    y_train = get_y_train();
}

void Knn::fit(DataFrame<double> X, DataFrame<double> y, int k)
{
    X_train = X;
    y_train = y;
    neighbors = k;
}


std::vector<double> Knn::predict(DataFrame<double> X)
{
    std::vector<double> pred;
    double dist;
    double min = std::numeric_limits<double>::max();
    int index_min = 0;

    for(int i = 0; i < X.get_row(); i++)
    {
        min = std::numeric_limits<double>::max();
        for(int j = 0; j < X_train.get_row(); j++)
        {
            dist = DataFrame<double>::dist(X_train[j], X[i]);
            if(dist < min)
            {

                index_min = j;
                 min = dist;
            }

        }
        pred.push_back(y_train[0][index_min]);


    }
    return pred;
}
