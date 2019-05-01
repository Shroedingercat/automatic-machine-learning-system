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

RandomForestClassifier::RandomForestClassifier(int N_base_models, int Min_samples_split, int Max_depth, int N_features, double Min_gain){
    max_depth = Max_depth;
    min_samples_split = Min_samples_split;
    min_gain = Min_gain;
    n_features = N_features;
    n_base_models = N_base_models;
    models = new DecisionTree[n_base_models];
}

void RandomForestClassifier::train(DataFrame<double> X, std::vector<int> y){
    DataFrame<double> Bootstrap_sample;

    if (n_features == NULL){
        n_features = int(std::sqrt(X.get_column()));
    }

    for (int i = 0; i < n_base_models; ++i) {
        Bootstrap_sample = create_sample(X);
        models[i].train(Bootstrap_sample, y, std::vector<int>(), X.get_column() -  n_features, min_samples_split, max_depth, min_gain);
    }

}

std::vector<int> RandomForestClassifier::predict(DataFrame<double> X){
    std::vector<int> pred, preds, unique, count;
    std::map<int, int> dic;
    int max_i;
    int max_el;

    for (int i = 0; i < X.get_row(); ++i) {

        dic = std::map<int, int>();

        preds = std::vector<int>();

        for (int j = 0; j < n_base_models; ++j){
            preds.push_back(models[j].predict_row(X[i]));
        }
        
        unique = DataFrame<int>::unique(preds);

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

DataFrame<double> RandomForestClassifier::create_sample(DataFrame<double> X) const{
    DataFrame<double> Bootstrap_sample;
    int rand_i;

    srand(time(NULL));

    for (int i = 0; i < X.get_row(); ++i) {
        rand_i = rand() % (X.get_row());
        Bootstrap_sample.append(X[rand_i]);
    }

    return Bootstrap_sample;
}





