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
template <typename classifier>


class DifferentialEvolution
{
public:
    DifferentialEvolution();
    DifferentialEvolution(const classifier &Model, int Start_num = 5, int Iter = 100){
        model = Model;
        start_num = Start_num;
        iter = Iter;
    }

    double fit(const DataFrame<double>& X, const std::vector<int>& y, DataFrame<double> X_test, std::vector<int> y_test){
        if(std::is_same<classifier, DecisionTree>::value){
            DecisionTree tree;
            DataFrame<double> value;
            srand(time(NULL));
            for (int i = 0; i < iter; i++) {
                double tmp = double((rand()%1000))/1000;
                value.append({double(rand()%20), double(rand()%500), tmp});
            }

            for (int i = 0; i < iter; i++) {
                std::vector<double> acc;

                for (int j = 0; j < start_num; j++) {
                    tree = DecisionTree();
                    tree.train(X, y, std::vector<int>(), NULL, value[j][0], value[j][1], value[j][2]);
                    acc.push_back(MlLibrary::accuracy(tree.predict(X_test), y_test));
                }
                value = new_pop(value, acc);
            }

            int max_i = 0;
            double max = 0;
            for (int j = 0; j < start_num; j++) {
                double acc;
                tree = DecisionTree();
                tree.train(X, y, std::vector<int>(), NULL, value[j][0], value[j][1], value[j][2]);
                acc = MlLibrary::accuracy(tree.predict(X_test), y_test);
                if(acc > max){
                    max_i = j;
                }
            }
            return max;
        }
    }

    DataFrame<double> new_pop(const DataFrame<double> &prev_pop, const std::vector<double>& acc){
        DataFrame<double> new_pop;
        std::vector<std::vector<double>> arr;
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
            new_pop.append({});

            for (int j = 0; j < prev_pop[i].size(); j++) {
                int index = rand() % arr[j].size();
                new_pop[i].push_back(arr[j][index]);
            }
        }
        return new_pop;
    }

private:
    classifier model;
    int start_num;
    int iter;
};

#endif // DIFFERENTIALEVOLUTION_H
