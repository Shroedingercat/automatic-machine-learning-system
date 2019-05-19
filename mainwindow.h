#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <string>
#include <sstream>
#include <fstream>
#include <QDesktopServices>
#include <QUrl>
#include <QString>
#include "differentialevolution.h"
#include "dialog.h"
#include "dataframe.h"
#include "decisiontree.h"
#include "mllibrary.h"
#include "svm.h"
#include "knn.h"
#include "linearreg.h"
#include "logisticregression.h"
#include "randomforestclassifier.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QMenu *menu= new QMenu("File");
    QMenu *learning_menu = new QMenu("learning");
    QMenu *genetic_menu = new QMenu("genetic");

    DataFrame<double> X;
    DataFrame<double> y;
    std::vector<double> param;
    Dialog dge;
    std::string path_X, path_y;
    DecisionTree tree;
    SVM svm_model;
    Knn knn_model;
    LinearReg lin_model;
    LogisticRegression log_model;
    RandomForestClassifier forest_model;
    int pop_size = 5;
    int iter_num = 10;
    bool gen_flag = false;

private slots:
    void load();
    void open();
    void start();
    void on_off();
    void set_pop();
    void set_iter();
    void predict();
};

#endif // MAINWINDOW_H
