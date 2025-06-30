#include "dispwidget.h"
#include "ui_dispwidget.h"

DispWidget::DispWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DispWidget)
{
    ui->setupUi(this);
    QWidget::setAcceptDrops(true);
}

DispWidget::~DispWidget()
{
    delete ui;
}
