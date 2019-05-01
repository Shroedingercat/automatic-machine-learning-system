#ifndef MLLIBRARY_H
#define MLLIBRARY_H
#include <vector>
#include <algorithm>
#include <cmath>
#include "dataframe.h"
//template <typename data_type>


class MlLibrary
{
public:
    MlLibrary();

    /* Calculate information entropy for list of
   target values (entropy = - sum_i(p_i * log(p_i)))*/

    static double f_entropy(std::vector<int> target){
        double entropy = 0;
        double prop = 0;
        std::vector<double> probability;
        std::vector<int>::iterator max = std::max_element(target.begin(), target.end());
        for (int i = 0; i <= *max; ++i) {
            probability.push_back(0);
        }
        prop =  1/double(target.size());
        for (int i = 0; i < target.size(); i++) {
             probability[target[i]] += prop;
        }

        for (int i = 0; i < probability.size(); i++){
            if(probability[i] > 0){
                entropy -= probability[i]*log2(probability[i]);
            }
        }
        return entropy;
    }

    //Calculate the information gain for a split of 'y' to 'splits' (gain = f(y) - (n_1/n)*f(split_1) - (n_2/n)*f(split_2) ...)
    static double information_gain(std::vector<int> y, DataFrame<int> splits) {
        double splits_entropy = 0;
        for (int i = 0; i < splits.get_row(); i++) {
            splits_entropy += f_entropy(splits[i])*(splits[i].size());
            splits_entropy /= splits.get_row();
        }
        return f_entropy(y) - splits_entropy;
    }

    //Make a binary split of (X, y) using the threshold
    static DataFrame<int> split(DataFrame<double> X, std::vector<int> y, double threshold, int n_feture){
        std::vector<int> left_mask, right_mask;
        DataFrame<int> splits;
        for (int i = 0; i < y.size(); ++i) {

            if(X[i][n_feture] >= threshold){
                right_mask.push_back(y[i]);
            }
            else{
                left_mask.push_back(y[i]);
            }
        }
        splits.append(left_mask);
        splits.append(right_mask);
        return splits;
    }

    static std::vector<DataFrame<double>> split_dataset_X(DataFrame<double> X, int column, double value){
        DataFrame<double> left_X, right_X;
        std::vector<DataFrame<double>> ans;

        for (int i = 0; i < X.get_row(); ++i) {

            if(X[i][column] < value){
                left_X.append(X[i]);
            }
            else {
                right_X.append(X[i]);
            }
        }
        ans.push_back(left_X);
        ans.push_back(right_X);
        return ans;
    }

    static DataFrame<int> split_dataset_y(DataFrame<double> X, std::vector<int> y, int column, double value){
        std::vector<int> left_y, right_y;
        DataFrame<int> ans;

        for (int i = 0; i < X.get_row(); ++i) {

            if(X[i][column] < value){
                left_y.push_back(y[i]);//error, can be
            }
            else {
                right_y.push_back(y[i]);
            }
        }
        ans.append(left_y);
        ans.append(right_y);
        return ans;
    }

    //accuracy:
    static double accuracy(std::vector<int> y_true, std::vector<int> y_pred)
    {
        double true_positive = 0;
        for (int i = 0; i < y_true.size(); i++)
        {

            if(y_true[i] == y_pred[i])
            {

                true_positive += 1;
            }

        }

        return true_positive / (double)(y_true.size());
    }
};

#endif // MLLIBRARY_H
