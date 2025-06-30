#ifndef YOLO_H
#define YOLO_H

#include "dispwidget.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntimethread.h"
#include <QMainWindow>
#include <QPainter>
#include <QThread>
#include <opencv2/opencv.hpp>
QT_BEGIN_NAMESPACE
namespace Ui {
class Yolo;
}
QT_END_NAMESPACE

class Yolo : public QMainWindow {
    Q_OBJECT

public:
    Yolo(QWidget* parent = nullptr);
    ~Yolo();
    void predictYOLOv8Seg(cv::dnn::Net& net, cv::Mat& frame);
    void predictYOLOv8Seg2(cv::dnn::Net& net, const cv::Mat& frame);

    void init_onnxRuntime(QString onnx_path);
    void predictYOLOv8Seg(Ort::Session& session, const cv::Mat& frame);
    void run_onnxRuntime();
    void proc_output(cv::Mat& frame, const cv::Mat& output);
    void proc_output_seg(cv::Mat& frame, const cv::Mat& output, const cv::Mat& output_mask);
    void applySegmentationMask(cv::Mat& frame,
        const cv::Rect& box,
        const cv::Mat& mask);

private slots:

    void on_playTimeout();
    void on_onnxRTGetFrame();
    void updataPixmap(cv::Mat);

    void on_pushButtonRuntime_clicked();

    void on_pushButtonDnn_clicked();

    void on_pushButtonOpenVideo_clicked();

private:
    Ui::Yolo* ui;
    cv::dnn::Net net;
    int init_detect(QString onnx_path);
    QTimer* playFrameTimer;
    QPixmap cvMatToQPixmap(const cv::Mat& mat);

    cv::VideoCapture cap;
    DispWidget* m_dispWidget;

    OnnxRuntimeThread* m_onnxRuntimeThread;

    cv::Mat frame;
    QString videoPath;

protected:
    void paintEvent(QPaintEvent*) override;
};
#endif // YOLO_H
