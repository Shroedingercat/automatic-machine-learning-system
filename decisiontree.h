#pragma once
#include "mllibrary.h"
#include "dataframe.h"
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>

class DecisionTree
{
public:
    DecisionTree(){
        gain = 0;
        size = 0;
        column_index = 0;
        threshold =NULL;
        outcome = NULL;
        outcome_proba = std::vector<double>();;

        left_child = nullptr;
        right_child = nullptr;
    }

    ~DecisionTree(){
        delete left_child;
        delete  right_child;
    }

    void train(DataFrame<double> X, std::vector<int> y, std::vector<int> unique_targets = std::vector<int>(), int max_features = NULL, int min_samples_split=2, int max_depth=1000, double min_gain=0.0001){
        std::vector<double> best_split;
        std::vector<int> left_y, right_y;
        DataFrame<double> left_X, right_X;
        DataFrame<int> tmp_y;
        std::vector<DataFrame<double>> tmp_X;
        int column;
        double value, loc_gain = NULL;
        bool flag = false;

        if (unique_targets.size() == 0){
            unique_targets = DataFrame<int>::unique(y);
            std::sort(unique_targets.begin(),  unique_targets.end());
        }

        size = X.get_row();
        if (max_features == NULL){
            max_features = X.get_column();
        }

        if(max_depth > 0 && X.get_row() >= min_samples_split){

            best_split = find_best_split(X, y, max_features);
            column = int(best_split[0]);
            value = best_split[1];
            loc_gain = best_split[2];

            if (loc_gain != NULL && loc_gain > min_gain){

                column_index = column;
                threshold = value;
                //Split dataset
                tmp_X = MlLibrary::split_dataset_X(X, column, value);
                tmp_y = MlLibrary::split_dataset_y(X, y, column, value);
                left_X = tmp_X[0];
                right_X = tmp_X[1];
                left_y = tmp_y[0];
                right_y = tmp_y[1];
                // Grow left and right child
                left_child = new DecisionTree();
                right_child = new DecisionTree();
                left_child->train(left_X, left_y, unique_targets, max_features, min_samples_split, max_depth - 1, min_gain);
                right_child->train(right_X, right_y, unique_targets, max_features, min_samples_split, max_depth - 1, min_gain);
            }
            else {
                flag = true;
            }
        }
        else {
            flag = true;
        }

        if(flag){
            calculate_leaf_value(y, unique_targets);
            left_child = nullptr;
            right_child = nullptr;
        }
    }

    int predict_row(std::vector<double> row){
        if(left_child != nullptr && right_child != nullptr){

            if (row[column_index] < threshold){
                return left_child->predict_row(row);
            }
            else {
                return right_child->predict_row(row);
            }
        }
        return outcome;
    }

    std::vector<double> predict_proba_row(std::vector<double> row) const{
        if(left_child != nullptr && right_child != nullptr){

            if (row[column_index] < threshold){
                return left_child->predict_proba_row(row);
            }
            else {
                return right_child->predict_proba_row(row);
            }
        }
        return outcome_proba;
    }

    std::vector<int> predict(DataFrame<double> X){
        std::vector<int> y_pred;

        for (int i = 0; i < X.get_row(); ++i) {
            y_pred.push_back(predict_row(X[i]));
        }
        return y_pred;
    }

    DataFrame<double> predict_proba(const DataFrame<double> &X) const{
        DataFrame<double> pred;
        for (int i = 0; i < X.get_row(); ++i) {
            pred.append(predict_proba_row(X[i]));
        }

        return pred;
    }

private:
    std::vector<double> outcome_proba;
    double gain, threshold;
    int size, column_index, outcome;
    DecisionTree *left_child, *right_child;

    std::vector<double> find_splits(std::vector<double> X){
        std::set<double> split_values;
        std::vector<double> values;
        std::vector<double> x_unique;
        double average;
        x_unique = DataFrame<double>::unique(X);

        for (int i = 1; i < x_unique.size(); ++i) {
            average = (x_unique[i - 1] + x_unique[i]) / 2.0;
            split_values.insert(average);
        }

        for (auto value: split_values) {
            values.push_back(value);
        }

        return values;
    }

    std::vector<double> find_best_split(const DataFrame<double> &X, const std::vector<int> &y, int max_features = NULL){
        std::vector<int> subset, rand_vec;
        std::vector<double> split_values, ans;
        DataFrame<int> splits;
        double gain;
        int rand_col, a, b, max_col = NULL,  max, rand_i, min;
        double max_col_double, max_gain = NULL, max_val = NULL;

        srand(time(NULL));
        for (int i = 0; i < max_features; ++i) {
            rand_vec.push_back(i);
        }

        if (max_features == NULL || max_features == X.get_column()){
            max_features = X.get_column();
            subset = rand_vec;
        }
        else {
            max = max_features;
            min = 0;
            for (int i = 0; i < max_features; ++i) {
                rand_i = rand() % (rand_vec.size());

                for (int j = 0; j < rand_vec.size(); ++j) {

                    if (rand_vec[j] == rand_i){
                        rand_vec.erase(rand_vec.begin() + j);
                        break;
                    }
                }
                subset.push_back(rand_i);
            }

        }

        for (auto i: subset) {
            split_values = find_splits(X.get_vector_column(i));

            for (auto value: split_values) {
                splits = MlLibrary::split(X, y, value, i);
                gain = MlLibrary::information_gain(y, splits);

                 if(max_gain == NULL || gain > max_gain){
                    max_col = i;
                    max_val = value;
                    max_gain = gain;
                } //error, can be
            }
        }
        max_col_double = double(max_col);
        ans.push_back(max_col);
        ans.push_back(max_val);
        ans.push_back(max_gain);
        return ans;
    }

    void calculate_leaf_value(std::vector<int> y, std::vector<int> unique_targets){
        std::vector<int>::iterator max = std::max_element(y.begin(), y.end());
        std::vector<int> count, uniques;
        std::vector<double> counts_normed;
        std::map<int, double> probs;
        double sum, p;
        int max2, number;

        uniques = DataFrame<int>::unique(y);
        for (int i = 0; i <= *max; ++i) {
            count.push_back(0);
        }

        for (int i = 0; i < y.size(); ++i) {
            count[y[i]]++;
        }

        max2 = count[0];
        number = 0;
        for (int i = 1; i < count.size(); ++i) {

            if (max2 < count[i]){
                max2 = count[i];
                number = i;
            }
        }
        outcome = number;
        //Outcome probabilities
        sum = std::accumulate(count.begin(), count.end(), 0.0);
        for (auto item: count) {
            counts_normed.push_back(double(item)/sum);
        }

        for (int i = 0; i < uniques.size(); ++i) {
            probs.insert(std::pair<int, double>(uniques[i], counts_normed[i]));
        }

        outcome_proba = std::vector<double>();
        for(auto unique_target: unique_targets){

            if(std::find(uniques.begin(), uniques.end(), unique_target) != uniques.end()){
                p = probs[unique_target];
                outcome_proba.push_back(p);
            }
        }
    }
};


