#include "dialog.h"
#include "ui_dialog.h"

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
    ui->list->addItem("Decision Tree");
    ui->list->addItem("Random forest");
    ui->list->addItem("SVM");
    ui->list->addItem("log reg");
    ui->list->addItem("lin reg");
    ui->list->addItem("KNN");
    QObject::connect(ui->list, SIGNAL(doubleClicked( QModelIndex )),
                     this, SLOT(add(QModelIndex)));
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::add( const QModelIndex& index){
    ans = ui->list->item(index.row())->text();
    close();
}
