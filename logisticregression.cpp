#include "logisticregression.h"

LogisticRegression::LogisticRegression()
{
    alpha=0.001;
    iter_num=10000;
    delta=0.000001;
}

DataFrame<double> LogisticRegression::sigmoid(const DataFrame<double> &X, const DataFrame<double> & W) const{
    std::vector<double> ans;
    DataFrame<double> tmp;
    for(int j = 0; j < X.get_row(); j++){
        double f = 0;

        for (int k = 0; k < X.get_column(); k++) {
            f += W[0][k]*X[j][k];
        }
        ans.push_back(1/(1+exp(-f)));
    }
    tmp.append(ans);
    return tmp;
}

void LogisticRegression::fit(const DataFrame<double> &X_train, const DataFrame<double> &y_train, double Alpha, double Delta){
    alpha = Alpha;
    delta = Delta;
    double loss;
    double f = 0, fun = 0;
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
        loss = MlLibrary::log_loss(sigmoid(X_train_bias, weight), y_train);
        std::cout << "loss: " << loss << std::endl;
        gr = {{}};
        weight2[0] = weight[0];
        DataFrame<double> X_df = X_train_bias;

        for (int l = 0; l < X_df.get_column(); l++){
            fun = 0;
            for (int j = 0; j < X_df.get_row(); j++) {
                f = 0;
                for (int k = 0; k < X_df.get_column() ;k++) {
                    f+=weight[0][k]*X_df[j][k];
                }

                if(j != X_df.get_column()-1){
                    fun -= X_df[j][l]*((y_train[0][j] * exp(-y_train[0][j]*f) )/(1 + exp(-y_train[0][j]*f)));
                    //fun += (delta*2*weight[0][j]);
                }
                else {
                    fun -= X_df[j][l]*((y_train[0][j] * exp(-y_train[0][j]*f) )/(1 + exp(-y_train[0][j]*f)));
                }
            }
            gr[0].push_back(fun);
        }
        DataFrame<double> grad = DataFrame<double>(gr);
        weight = weight - (grad*alpha);

    }
}

std::vector<double> LogisticRegression::predict(const DataFrame<double> &X) const{
   DataFrame<double> prob;
   std::vector<double> tmp;
   std::vector<std::vector<double>> X_train_bias;

   for (int i = 0; i < X.get_row(); i++)
   {
       X_train_bias.push_back(tmp);

       for (int j = 0; j < X.get_column(); j++)
       {
           X_train_bias[i].push_back(X[i][j]);
       }
       X_train_bias[i].push_back(1);
   }

   prob =  sigmoid(X_train_bias, weight);

   for (int i = 0; i < prob.get_column(); i++) {

       if(prob[0][i] > 0.5){
           tmp.push_back(1);
       }
       else{
           tmp.push_back(-1);
       }
   }

   return tmp;
}

void LogisticRegression::fit_multi_class(const DataFrame<double> &X_train, const DataFrame<double> &y_train,int class_num, double Alpha, double Delta){
    alpha = Alpha;
    delta = Delta;
    double loss;
    double f = 0, fun = 0;
    std::vector<double> tmp;
    std::vector<std::vector<double>> gr;
    gr.push_back(tmp);
    DataFrame<double> grad = DataFrame<double>(gr);
    std::vector<std::vector<double>> X_train_bias;
    std::vector<std::vector<double>> weight2 = MlLibrary::random(X_train.get_column() + 1);
    std::vector<std::vector<double>> weights;
    targets = DataFrame<double>::unique(y_train[0]);

    for (int i = 0; i < class_num; i++) {
        weights.push_back(MlLibrary::random(X_train.get_column() + 1)[0]);
    }

    for (int i = 0; i < X_train.get_row(); i++)
    {
        X_train_bias.push_back(tmp);

        for (int j = 0; j < X_train.get_column(); j++)
        {
            X_train_bias[i].push_back(X_train[i][j]);
        }
        X_train_bias[i].push_back(1);
    }


    for(int index = 0; index < class_num; index++){
        std::vector<double> y_bin = bin_y(y_train, targets[index]);

        weight = DataFrame<double>();
        weight.append(weights[index]);

        for(int i = 0; i < iter_num; i++)
        {
            DataFrame<double> tmpp;
            tmpp.append(y_bin);
            loss = MlLibrary::log_loss(sigmoid(X_train_bias, weight), tmpp);
            std::cout << "loss: " << loss << std::endl;
            gr = {{}};

            DataFrame<double> X_df = X_train_bias;

            for (int l = 0; l < X_df.get_column(); l++){
                fun = 0;
                for (int j = 0; j < X_df.get_row(); j++) {
                    f = 0;
                    for (int k = 0; k < X_df.get_column() ;k++) {
                        f += weight[0][k]*X_df[j][k];
                    }

                    if(j != X_df.get_column()-1){
                        fun -= X_df[j][l]*((y_bin[j] * exp(-y_bin[j]*f) )/(1 + exp(-y_bin[j]*f)));
                        //fun += (delta*2*weight[0][j]);
                    }
                    else {
                        fun -= X_df[j][l]*((y_bin[j] * exp(-y_bin[j]*f) )/(1 + exp(-y_bin[j]*f)));
                    }
                }
                gr[0].push_back(fun);
            }
            DataFrame<double> grad = DataFrame<double>(gr);
            weight = weight - (grad*alpha);

        }
        weights[index] = weight[0];
    }
    Weights = weights;
}

std::vector<double> LogisticRegression::bin_y(const DataFrame<double> &y, double target){
    std::vector<double> new_y;
    for (int i = 0; i < y.get_column(); i++) {

        if (y[0][i] == target){
            new_y.push_back(1);
        }
        else {
            new_y.push_back(-1);
        }
    }
    return new_y;
}

std::vector<double> LogisticRegression::predict_multi_class(const DataFrame<double> &X) const{
    std::vector<double> pred;
    DataFrame<double> tmpp;
    std::vector<DataFrame<double>> prop;
    std::vector<std::vector<double>> X_train_bias;
    std::vector<double> tmp;
    double max_prop, index_prop;

    for (int i = 0; i < X.get_row(); i++)
    {
        X_train_bias.push_back(tmp);

        for (int j = 0; j < X.get_column(); j++)
        {
            X_train_bias[i].push_back(X[i][j]);
        }
        X_train_bias[i].push_back(1);
    }

    for (int index = 0; index < Weights.size(); index++) {
        tmpp = DataFrame<double>();
        tmpp.append(Weights[index]);
        prop.push_back(sigmoid(X_train_bias, tmpp));
    }

    for (int i = 0; i < X.get_row(); i++) {
        max_prop = 0;
        index_prop = 0;

        for (int index = 0; index < Weights.size(); index++) {

            if (prop[index][0][i] > max_prop){
                index_prop = index;
                max_prop = prop[index][0][i];
            }
        }
        pred.push_back(targets[index_prop]);
    }
    return pred;
}

