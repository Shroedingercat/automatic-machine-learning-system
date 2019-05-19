#ifndef DIFFERENTIALEVOLUTION_H
#define DIFFERENTIALEVOLUTION_H
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "dataframe.h"
#include "decisiontree.h"
#include "mllibrary.h"
#include "logisticregression.h"
#include "randomforestclassifier.h"
template <typename classifier>


class DifferentialEvolution
{
public:
    DifferentialEvolution(){
        model = DecisionTree();
        start_num = 5;
        iter = 100;
    }
    DifferentialEvolution(int Start_num = 5, int Iter = 100){
        model = classifier();
        start_num = Start_num;
        iter = Iter;
    }

     std::vector<double> fit(const DataFrame<double>& X, const std::vector<double>& y, DataFrame<double> X_test, std::vector<double> y_test){
        if(std::is_same<classifier, DecisionTree>::value){
            DecisionTree tree;
            DataFrame<double> value;
            srand(time(NULL));
            for (int i = 0; i < start_num; i++) {
                double tmp = double((rand()%1000))/1000;
                value.append({double(rand()%20), double(rand()%500), tmp});
            }

            for (int i = 0; i < iter; i++) {
                std::vector<double> acc;

                for (int j = 0; j < start_num; j++) {
                    tree = DecisionTree();
                    tree.train(X, y, std::vector<double>(), NULL, value[j][0], value[j][1], value[j][2]);
                    acc.push_back(MlLibrary::accuracy(tree.predict(X_test), y_test));
                }
                value = new_pop(value, acc);
            }

            int max_i = 0;
            double max = 0;
            for (int j = 0; j < start_num; j++) {
                double acc;
                tree = DecisionTree();
                tree.train(X, y, std::vector<double>(), NULL, value[j][0], value[j][1], value[j][2]);
                acc = MlLibrary::accuracy(tree.predict(X_test), y_test);
                if(acc > max){
                    max_i = j;
                    max = acc;
                }
            }
            return value[max_i];
        }


        else if(std::is_same<classifier, RandomForestClassifier>::value){
            RandomForestClassifier model;
            DataFrame<double> value;

            for (int i = 0; i < start_num; i++) {
                double tmp = double((rand()%1000))/10000, tmp2 = double((rand()%1000))/10000;
                value.append({double(rand()%500), double(rand()%500), double(rand()%X.get_column()) + 1});
            }
            for (int i = 0; i < iter; i++) {
                std::vector<double> acc;

                for (int j = 0; j < start_num; j++) {
                    model = RandomForestClassifier();
                    DataFrame<double> df;
                    df.append(y);
                    model.train(X, df[0], value[j][0], value[j][1], value[j][2]);
                    acc.push_back(MlLibrary::accuracy(model.predict(X_test), y_test));
                }
                value = new_pop(value, acc);
            }

            int max_i = 0;
            double max = 0;
            for (int j = 0; j < start_num; j++) {
                double acc;
                model = RandomForestClassifier();
                DataFrame<double> df;
                df.append(y);
                model.train(X, df[0], value[j][0], value[j][1], value[j][2]);
                acc = MlLibrary::accuracy(model.predict(X_test), y_test);
                if(acc > max){
                    max_i = j;
                    max = acc;
                }
            }
            return value[max_i];
        }

        else if(std::is_same<classifier, LogisticRegression>::value){
            LogisticRegression model;
            DataFrame<double> value;

            for (int i = 0; i < start_num; i++) {
                double tmp = double((rand()%1000))/10000, tmp2 = double((rand()%1000))/10000;
                value.append({tmp, tmp2});
            }
            for (int i = 0; i < iter; i++) {
                std::vector<double> acc;

                for (int j = 0; j < start_num; j++) {
                    model = LogisticRegression();
                    DataFrame<double> df;
                    df.append(y);
                    model.fit(X, df, value[j][0], value[j][1]);
                    acc.push_back(MlLibrary::accuracy(model.predict(X_test), y_test));
                }
                value = new_pop(value, acc);
            }

            int max_i = 0;
            double max = 0;
            for (int j = 0; j < start_num; j++) {
                double acc;
                model = LogisticRegression();
                DataFrame<double> df;
                df.append(y);
                model.fit(X, df, value[j][0], value[j][1]);
                acc = MlLibrary::accuracy(model.predict(X_test), y_test);
                if(acc > max){
                    max_i = j;
                    max = acc;
                }
            }
            return value[max_i];
        }
    }

    DataFrame<double> new_pop(const DataFrame<double> &prev_pop, const std::vector< double>& acc){
        std::vector<std::vector<double>> arr, new_pop;
        srand(time(NULL));

        for (int i = 0; i < prev_pop.get_column(); i++) {
            arr.push_back({});
        }
         double sum = 0;
        for (auto item: acc) {
            sum += item;
        }
        for (int i = 0; i < acc.size(); i++) {

            for (int j = 0; j < prev_pop[j].size(); j++) {

                for(int num = 0; num < int(10*acc[i]/sum); num++){
                    arr[j].push_back(prev_pop[i][j]);
                }
            }
        }

        for (int i = 0; i < prev_pop.get_row(); i++) {
            new_pop.push_back({});

            for (int j = 0; j < prev_pop[i].size(); j++) {
                int index = rand() % arr[j].size();
                new_pop[i].push_back(arr[j][index]);
            }
        }
        return DataFrame<double>(new_pop);
    }



private:
    classifier model;
    int start_num;
    int iter;
};

#endif // DIFFERENTIALEVOLUTION_H
