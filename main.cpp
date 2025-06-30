#include "yolo.h"

#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    Yolo w;
    w.show();
    return a.exec();
}
