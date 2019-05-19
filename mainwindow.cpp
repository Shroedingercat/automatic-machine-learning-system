#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    menu->addAction("load", this, SLOT(load()));
    menu->addAction("open", this, SLOT(open()));
    ui->menuBar->addMenu(menu);
    learning_menu->addAction("Start", this, SLOT(start()));
    learning_menu->addAction("use gen algorithm", this, SLOT(on_off()));
    learning_menu->addAction("Predict", this, SLOT(predict()));
    ui->menuBar->addMenu(learning_menu);
    genetic_menu->addAction("Set population size", this, SLOT(set_pop()));
    genetic_menu->addAction("Set iter", this, SLOT(set_iter()));
    ui->menuBar->addMenu(genetic_menu);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::load(){
    QString tmp = QFileDialog::getOpenFileName(0, tr("load train data"));
    path_X = tmp.toStdString();
    X.read_csv(path_X);
    tmp = QFileDialog::getOpenFileName(0, tr("load target"));
    path_y = tmp.toStdString();
    y.read_csv(path_y);
    y.T();
}

void MainWindow::open(){
    QString url = QFileDialog::getOpenFileName(0, tr("open"));
    QDesktopServices::openUrl(QUrl(url));
}

void MainWindow::start(){
    dge.setModal(true);
    dge.exec();
    if (y.get_row() <= 0){
        load();
    }

    if(dge.ans == "Decision Tree"){

        if (gen_flag || ui->radioButton->isChecked()){
            DifferentialEvolution<DecisionTree> gen(pop_size,iter_num);
            param = gen.fit(X, y[0], X, y[0]);
            tree.train(X, y[0], std::vector<double>(), NULL, param[0], param[1], param[2]);
            double tmp = MlLibrary::accuracy(y[0], tree.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }

        else {
            tree.train(X, y[0]);
            double tmp = MlLibrary::accuracy(y[0], tree.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");

        }
    }

    else if(dge.ans == "SVM"){

        if (gen_flag || ui->radioButton->isChecked()){
            DifferentialEvolution<SVM> gen(pop_size,iter_num);
            param = gen.fit(X, y[0], X, y[0]);
            svm_model.fit(X, y);
            double tmp = MlLibrary::accuracy(y[0], svm_model.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }

        else{
            svm_model.fit(X, y);
            double tmp = MlLibrary::accuracy(y[0], svm_model.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }
    }

    else if(dge.ans == "KNN"){

        if(gen_flag || ui->radioButton->isChecked()){
            DifferentialEvolution<Knn> gen(pop_size, iter_num);
            param = gen.fit(X, y[0], X, y[0]);
            knn_model.fit(X, y, 1);
            double tmp = MlLibrary::accuracy(y[0], knn_model.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }

        else{
            knn_model.fit(X, y, 1);
            double tmp = MlLibrary::accuracy(y[0], knn_model.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");

        }
    }

    else if(dge.ans == "lin reg"){

        if(gen_flag || ui->radioButton->isChecked()){
            DifferentialEvolution<LinearReg> gen(pop_size,iter_num);
            param = gen.fit(X, y[0], X, y[0]);
            lin_model.fit(X, y);
            double tmp = MlLibrary::accuracy(y[0], lin_model.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }

        else{
            lin_model.fit(X, y);
            DataFrame<double> pred;
            pred.append(lin_model.predict(X));
            double tmp = MlLibrary::MSE2(pred, y);
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("MSE:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }
    }
    else if(dge.ans == "log reg"){

        if(gen_flag || ui->radioButton->isChecked()){
            DifferentialEvolution<LogisticRegression> gen(pop_size,iter_num);
            param = gen.fit(X, y[0], X, y[0]);
            double tmp;
            if (DataFrame<double>::unique(y[0]).size() <= 2) {
                log_model.fit(X, y, param[0], param[1]);
                tmp = MlLibrary::accuracy(y[0], log_model.predict(X));
            }
            else {
                log_model.fit_multi_class(X, y, DataFrame<double>::unique(y[0]).size(), param[0], param[1]);
                tmp = MlLibrary::accuracy(y[0], log_model.predict_multi_class(X));
            }

            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }

        else{
            double tmp;
            if (DataFrame<double>::unique(y[0]).size() <= 2) {
                log_model.fit(X, y);
                tmp = MlLibrary::accuracy(y[0], log_model.predict(X));
            }
            else {
                log_model.fit_multi_class(X, y, DataFrame<double>::unique(y[0]).size());
                tmp = MlLibrary::accuracy(y[0], log_model.predict_multi_class(X));
            }
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }
    }
    else if(dge.ans == "Random forest"){

        if(gen_flag || ui->radioButton->isChecked()){
            DifferentialEvolution<RandomForestClassifier> gen(pop_size,iter_num);
            param = gen.fit(X, y[0], X, y[0]);
            forest_model.train(X, y[0], int(param[0]), int(param[1]), int(param[2]));
            double tmp = MlLibrary::accuracy(y[0], forest_model.predict(X));
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }

        else{
            forest_model.train(X, y[0]);
            DataFrame<double> pred;
            pred.append(forest_model.predict(X));
            double tmp = MlLibrary::accuracy(pred[0], y[0]);
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText("Accuracy:");
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText(" ");
        }
    }
}

void MainWindow::on_off(){
    if(!gen_flag){
        gen_flag = true;
        QMessageBox::information(0, "Load bar", "Genetic algorithm enabled!");
    }
    else{
        gen_flag = false;
        QMessageBox::information(0, "Load bar", "Genetic algorithm is off!");
    }
}

void MainWindow::set_pop(){
    pop_size = QInputDialog::getInt(this, "Input size", "size");
}

void MainWindow::set_iter(){
    iter_num = QInputDialog::getInt(this, "Input size", "size");
}

void MainWindow::predict(){
    DataFrame<double> X_test;
    std::string path;
    std::vector<double> pred;

    QString tmp = QFileDialog::getOpenFileName(0, tr("load train data"));
    path = tmp.toStdString();
    X_test.read_csv(path);

    if(dge.ans == "Decision Tree"){

            pred = tree.predict(X_test);
            for(int i = 0; i <pred.size(); i++){
                double tmp = pred[i];
                std::string str = std::to_string(tmp);
                ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
                ui->plainTextEdit->insertPlainText("\n");
            }
    }

    else if(dge.ans == "SVM"){

        pred = svm_model.predict(X_test);
        for(int i = 0; i <pred.size(); i++){
            double tmp = pred[i];
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText("\n");
        }
    }

    else if(dge.ans == "KNN"){

        pred = knn_model.predict(X_test);
        for(int i = 0; i <pred.size(); i++){
            double tmp = pred[i];
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText("\n");
        }
    }

    else if(dge.ans == "lin reg"){

        pred = lin_model.predict(X_test);
        for(int i = 0; i <pred.size(); i++){
            double tmp = pred[i];
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText("\n");
        }
    }

    else if(dge.ans == "log reg"){
        if (DataFrame<double>::unique(y[0]).size() > 2) {
            pred = log_model.predict_multi_class(X_test);
        }
        else{
            pred = log_model.predict(X_test);
        }

        for(int i = 0; i <pred.size(); i++){
            double tmp = pred[i];
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText("\n");
        }
    }
    else if(dge.ans == "Random forest"){
        pred = forest_model.predict(X_test);
        for(int i = 0; i <pred.size(); i++){
            double tmp = pred[i];
            std::string str = std::to_string(tmp);
            ui->plainTextEdit->insertPlainText(QString::fromStdString(str));
            ui->plainTextEdit->insertPlainText("\n");
        }
    }

}



