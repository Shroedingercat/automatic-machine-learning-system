#include "svm.h"

//конструктор по умолчанию
SVM::SVM()
{
    alpha = 0.001;
    w_delta = 0.1;
    iter_num = 500;
    weight = DataFrame<double>();
}

//конструктор инициализации
SVM::SVM(double Alpha, double W_delta, int Iter_num)
{
    alpha = Alpha;
    w_delta = W_delta;
    iter_num = Iter_num;
    weight = DataFrame<double>();
}

//train svm
void SVM::fit(const DataFrame<double> &X_train, const DataFrame<double> &y_train)
{
    double loss, new_loss;
    std::vector<double> tmp;
    std::vector<std::vector<double>> gr;
    gr.push_back(tmp);
    DataFrame<double> grad = DataFrame<double>(gr);
    DataFrame<double> X_train_bias = X_train;
    std::vector<std::vector<double>> weight2 = MlLibrary::random(X_train.get_column() + 1);
    weight2[0][1] = 0.1;
    weight = DataFrame<double>(weight2);

    for (int i = 0; i < X_train.get_row(); i++)
    {
        X_train_bias[i].push_back(1);
    }

    for(int i = 0; i < iter_num; i++){
        loss = MlLibrary::hinge_loss(X_train_bias, y_train, weight);
        std::cout << "loss: " << loss << std::endl;
        gr = {{}};
        weight2[0] = weight[0];
        for (int j = 0; j < weight.get_column(); j++)
        {
            weight2[0][j] = weight2[0][j] + w_delta;
            new_loss = MlLibrary::hinge_loss(X_train_bias, y_train, weight2);
            gr[0].push_back((new_loss - loss) / (w_delta));
            weight2[0][j] -= w_delta;
        }
        DataFrame<double> grad = DataFrame<double>(gr);
        weight = weight - (grad*alpha);

    }
}

//predict
std::vector<double> SVM::predict(const DataFrame<double> &X) const
{
    std::vector<std::vector<double>> pred;
    std::vector<double> tmp;
    pred.push_back(tmp);
    double pred_number;
    DataFrame<double> X_test_bias = X;
    for (int i = 0; i < X_test_bias.get_row(); i++)
    {
        X_test_bias[i].push_back(1);
    }
    for (int i = 0; i < X_test_bias.get_row(); i++)
    {
        pred_number = 0;

        for(int j = 0; j < X_test_bias.get_column(); j++)
        {
            pred_number += X_test_bias[i][j] * weight[0][j];
        }

        if(pred_number >= 0)
        {
            pred[0].push_back(1);
        }
        else
        {
            pred[0].push_back(-1);
        }
    }
    return pred[0];
}


