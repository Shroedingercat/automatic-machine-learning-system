#include <iostream>
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
#include "differentialevolution.h"

using namespace std;

int main() {
    DecisionTree tree;
    DataFrame<double> X, X_test;
    DataFrame<int> y, y_test;
    DifferentialEvolution<DecisionTree> gen(tree);

    X.read_csv("test2.txt");
    y.read_csv("test2_y.txt");
    X_test.read_csv("real_test.txt");
    y_test.read_csv("testy.txt");
    y.T();
    y_test.T();
    tree.train(X, y[0], std::vector<int>(), NULL, 2, -1);

    cout << MlLibrary::accuracy(y_test[0], tree.predict(X_test));

    gen.fit(X,y[0], X_test, y_test[0]);


   return 0;
}
