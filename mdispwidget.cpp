#include "mdispwidget.h"
#include "ui_mdispwidget.h"

MDispWidget::MDispWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MDispWidget)
{
    ui->setupUi(this);
}

MDispWidget::~MDispWidget()
{
    delete ui;
}
