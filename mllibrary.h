#ifndef MLLIBRARY_H
#define MLLIBRARY_H
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "dataframe.h"
//template <typename data_type>


class MlLibrary
{
public:
    MlLibrary();

    /* Calculate information entropy for list of
   target values (entropy = - sum_i(p_i * log(p_i)))*/

    static  double f_entropy(std::vector<double> target){
         double entropy = 0;
         double prop = 0;
        std::vector<double> probability;
        if(target.size() != 0){
            std::vector<double>::iterator max = std::max_element(target.begin(), target.end());


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
        }
        else{
            entropy = 0;
        }
        return -entropy;
    }

    //Calculate the information gain for a split of 'y' to 'splits' (gain = f(y) - (n_1/n)*f(split_1) - (n_2/n)*f(split_2) ...)
    static  double information_gain(std::vector<double> y, DataFrame<double> splits) {
        double splits_entropy = 0;
        for (int i = 0; i < splits.get_row(); i++) {
            splits_entropy += f_entropy(splits[i])*(splits[i].size());
            splits_entropy /= splits.get_row();
        }
        return f_entropy(y) - splits_entropy;
    }

    //Make a binary split of (X, y) using the threshold
    static DataFrame<double> split(DataFrame<double> X, std::vector<double> y,  double threshold, int n_feture){
        std::vector<double> left_mask, right_mask;
        DataFrame<double> splits;
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

    static std::vector<DataFrame< double>> split_dataset_X(DataFrame< double> X, int column,  double value){
        DataFrame< double> left_X, right_X;
        std::vector<DataFrame< double>> ans;

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

    static DataFrame<double> split_dataset_y(DataFrame< double> X, std::vector<double> y, int column,  double value){
        std::vector<double> left_y, right_y;
        DataFrame<double> ans;

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
    static  double accuracy(std::vector<double> y_true, std::vector<double> y_pred)
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

    //hinge loss:
    static double hinge_loss(const DataFrame<double> &X, const DataFrame<double> & y, const DataFrame<double> & W)
    {
        double pred = 0;
        double pred_X = 0;
        for(int i = 0; i < X.get_row() ; i++)
        {

            for(int j = 0; j < X.get_column(); j++)
            {
                pred_X += W[0][j]*X[i][j];
            }
            pred_X = 1 - y[0][i]*pred_X;

            if(pred_X < 0)
            {
                pred += 0;
            }
            else
            {
                pred += pred_X;
            }
        }
        return pred / X.get_row();
    }

    //gen random weights(mostly)
    static std::vector<std::vector<double>> random(int size)
    {
        srand(time(NULL));
        std::vector<double> ans;
        double number;

        for(int i = 0; i < size; i++)
        {
            number = double(rand()%100)/100;
            ans.push_back(number);
        }
        std::vector<std::vector<double>> rep;
        rep.push_back(ans);
        return rep;
    }

    //Mean squre error:
    static double MSE(const DataFrame<double> &X,const DataFrame<double> & y, const DataFrame<double> & W)
    {
        double pred = 0;
        double pred_X = 0;
        for(int i = 0; i < X.get_row() ; i++)
        {
            pred_X = 0;
            for(int j = 0; j < X[0].size(); j++)
            {
                pred_X += W[0][j]*X[i][j];
            }
            pred_X = pow(y[0][i] - pred_X, 2)/(2 * X.get_row());
            pred += pred_X;

        }
        return pred  ;
    }

    static double MSE2(const DataFrame<double> &pred,const DataFrame<double> & y)
    {
        double pred_X = 0;
        for(int i = 0; i < pred.get_column() ; i++)
        {

            pred_X += pow(y[0][i] - pred[0][i], 2)/(2*pred.get_column());


        }
        return pred_X  ;
    }

    //log loss:
    static double log_loss(DataFrame<double> prob, DataFrame<double> y_true)
    {
        double ans = 0;
        for (int i = 0; i < y_true.get_column(); i++){
            ans += log(1 + exp(-y_true[0][i]*prob[0][i]));
        }

        return ans;
    }

};

#endif // MLLIBRARY_H
