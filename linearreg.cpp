#include "linearreg.h"

//конструктор по умолчанию
LinearReg::LinearReg()
{
    alpha = 0.001;
    w_delta = 0.1;
    iter_num = 10000;
    weight = DataFrame<double>();
}

//конструктор инициализации
LinearReg::LinearReg(double Alpha, double W_delta, int Iter_num)
{
    alpha = Alpha;
    w_delta = W_delta;
    iter_num = Iter_num;
    weight = DataFrame<double>();
}

// train linreg
void LinearReg::fit(const DataFrame<double> &X_train, const DataFrame<double> &y_train)
{
    double loss, new_loss;
    std::vector<double> tmp;
    std::vector<std::vector<double>> gr;
    gr.push_back(tmp);
    DataFrame<double> grad = DataFrame<double>(gr);
    std::vector<std::vector<double>> X_train_bias;
    std::vector<std::vector<double>> weight2 = MlLibrary::random(X_train.get_column() + 1);
    weight = DataFrame<double>(weight2);
    for (int i = 0; i < X_train.get_row(); i++)
    {
        X_train_bias.push_back(tmp);

        for (int j = 0; j < X_train.get_column(); j++)
        {
            X_train_bias[i].push_back(X_train[i][j]);
        }
        X_train_bias[i].push_back(1);
    }



    for(int i = 0; i < iter_num; i++)
    {
        loss = MlLibrary::MSE(X_train_bias, y_train, weight);
        std::cout << "loss: " << loss << std::endl;
        gr = {{}};
        weight2[0] = weight[0];
        DataFrame<double> X_df = X_train_bias;
        double f, fun;
        for (int l = 0; l < X_df.get_column(); l++){
            double f = 0, fun = 0;
            for (int j = 0; j < X_df.get_row(); j++) {

                for (int k = 0; k < X_df.get_column() ;k++) {
                    f+=weight[0][k]*X_df[j][k];
                }
                fun += (y_train[0][j] - f)*(X_df[j][l]/X_df.get_row());
            }
            gr[0].push_back(fun);
        }
        DataFrame<double> grad = DataFrame<double>(gr);
        weight = weight + (grad*alpha);

    }
}

// predict new data
std::vector<double> LinearReg::predict(const DataFrame<double> &X) const
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

        pred[0].push_back(pred_number);

    }
    return pred[0];
}
