#ifndef DISPWIDGET_H
#define DISPWIDGET_H

#include "qpainter.h"
#include <QWidget>

namespace Ui {
class DispWidget;
}

class DispWidget : public QWidget {
    Q_OBJECT

public:
    explicit DispWidget(QWidget* parent = nullptr);
    ~DispWidget();
    std::shared_ptr<QPixmap> m_pixmap;

protected:
    // class DispWidget : public QWidget
    void paintEvent(QPaintEvent* event) override
    {
        QPainter painter(this);
        if (m_pixmap) {
            painter.drawPixmap(0, 0, m_pixmap->scaled(size().width(), size().height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

private:
    Ui::DispWidget* ui;
};

#endif // DISPWIDGET_H
