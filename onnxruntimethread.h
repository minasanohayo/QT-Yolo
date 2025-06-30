#ifndef ONNXRUNTIMETHREAD_H
#define ONNXRUNTIMETHREAD_H

#include "onnxruntime_cxx_api.h"
#include <QElapsedTimer>
#include <QThread>
#include <opencv2/opencv.hpp>

class Yolo;
class OnnxRuntimeThread : public QThread {
    Q_OBJECT
public:
    explicit OnnxRuntimeThread(QObject* parent = nullptr);
    using CallbackFuncPro = std::function<void(cv::Mat&, const cv::Mat&)>;
    using CallbackFuncProSeg = std::function<void(cv::Mat&, const cv::Mat&, const cv::Mat&)>;
    void storeFrame(cv::Mat frame)
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        sharedFrame = frame.clone(); // 深拷贝确保数据安全
        hasNewFrame = true;
        cond_.notify_one(); // 通知子线程有新帧

    }
    bool frameEmpty()
    {
        std::lock_guard<std::mutex> lock(frame_mutex);
        return !hasNewFrame; // 返回是否有新帧
    }
    void setStop(bool b)
    {
        _stop = b;
        cond_.notify_one(); // 确保线程能退出
    }
    bool isStop()
    {

        return _stop;
    }
    void setCallback_proc_output(CallbackFuncPro cb) { m_callback_proc_output = cb; }
    void setCallback_proc_output_seg(CallbackFuncProSeg cb) { m_callback_proc_output_seg = cb; }

protected:
    void run() override;

private:
    //***********************************************************
    std::atomic_flag flag = ATOMIC_FLAG_INIT; // 原子标志位
    std::atomic<bool> _stop = false;
    std::mutex frame_mutex;
    std::condition_variable cond_;
    bool hasNewFrame = false; // 是否有新帧标志
    cv::Mat sharedFrame;
    //***********************************************************

    CallbackFuncPro m_callback_proc_output = nullptr;
    CallbackFuncProSeg m_callback_proc_output_seg = nullptr;
signals:
    void ableToUpdate(cv::Mat frame);
    void needNewFrame();
};

#endif // ONNXRUNTIMETHREAD_H
