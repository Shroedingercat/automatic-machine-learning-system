#include "randomforestclassifier.h"

RandomForestClassifier::RandomForestClassifier()
{
    max_depth = 1000;
    min_samples_split = 2;
    n_features = NULL;
    min_gain=0.0001;
    n_base_models = 100;
    models = new DecisionTree[100];
}

RandomForestClassifier::RandomForestClassifier(int N_base_models, int Min_samples_split, int Max_depth, int N_features,   double Min_gain){
    max_depth = Max_depth;
    min_samples_split = Min_samples_split;
    min_gain = Min_gain;
    n_features = N_features;
    n_base_models = N_base_models;
    models = new DecisionTree[n_base_models];
}

void RandomForestClassifier::train(DataFrame<double> X, std::vector<double> y, int Max_depth, int N_base_models, int N_features){
    DataFrame<double> Bootstrap_sample;

    if (n_features == NULL){
        n_features = int(std::sqrt(X.get_column()));
    }

    max_depth = Max_depth;
    n_base_models = N_base_models;
    n_features = N_features;

    models = new DecisionTree[n_base_models];

    for (int i = 0; i < n_base_models; ++i) {
        std::vector<DataFrame<double>> tmp;
        tmp = create_sample(X, y);
        Bootstrap_sample = tmp[0];
        models[i].train(Bootstrap_sample, tmp[1][0], std::vector<double>(), X.get_column() -  n_features, min_samples_split, max_depth, min_gain);
    }

}

std::vector<double> RandomForestClassifier::predict(DataFrame<double> X){
    std::vector<double> pred, preds, unique, count;
    std::map<double, double> dic;
    int max_i;
    double max_el;

    for (int i = 0; i < X.get_row(); ++i) {

        dic = std::map<double, double>();

        preds = std::vector<double>();

        for (int j = 0; j < n_base_models; ++j){
            preds.push_back(models[j].predict_row(X[i]));
        }

        unique = DataFrame<double>::unique(preds);

        for (int j = 0; j < unique.size(); ++j) {
            dic[unique[i]] = 0;
        }

        for (int j = 0; j < preds.size(); ++j){
            dic[preds[i]]++;
        }

        max_el = dic[unique[0]];
        max_i = unique[0];

        for (int j = 1; j < unique.size(); ++j) {

            if(max_el < dic[unique[j]]){
                max_el = dic[unique[j]];
                max_i = unique[j];
            }
        }

        pred.push_back(max_i);

    }
    return pred;
}

std::vector<DataFrame<double>> RandomForestClassifier::create_sample(const DataFrame<double> &X, const std::vector<double> &y) const{
    DataFrame<double> Bootstrap_sample;
    std::vector<DataFrame<double>> tmp;
    DataFrame<double> Bootstrap_y;
    std::vector<double> y_new;
    int rand_i = 0;


    srand(time(NULL));

    for (int i = 0; i < X.get_row(); ++i) {
        rand_i = rand() % (X.get_row());
        Bootstrap_sample.append(X[rand_i]);
        y_new.push_back(y[rand_i]);
    }

    Bootstrap_y.append(y_new);
    tmp.push_back(Bootstrap_sample);
    tmp.push_back(Bootstrap_y);

    return tmp;
}





