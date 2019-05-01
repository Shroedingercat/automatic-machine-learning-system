#ifndef RANDOMFORESTCLASSIFIER_H
#define RANDOMFORESTCLASSIFIER_H
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <map>
#include "dataframe.h"
#include "mllibrary.h"
#include "decisiontree.h"



class RandomForestClassifier
{
public:
    RandomForestClassifier();

    RandomForestClassifier(int N_base_models, int Min_samples_split, int Max_depth, int N_features, double Min_gain);

    ~RandomForestClassifier(){
        for (int i = 0; i < n_base_models; ++i) {
            models[i] = DecisionTree();
        }
    }

    void train(DataFrame<double> X, std::vector<int> y);
    std::vector<int> predict(DataFrame<double> X);
    
private:
    DataFrame<double> create_sample(DataFrame<double> X) const;

    int max_features, min_samples_split, max_depth, n_features, n_base_models;
    double min_gain;
    DecisionTree *models;


};

#endif // RANDOMFORESTCLASSIFIER_H
