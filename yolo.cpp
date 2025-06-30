#include "Yolo.h"
#include "ui_yolo.h"
#include <QElapsedTimer>
#include <QFileDialog>
#include <QTimer>
#include <windows.h>
const float confThreshold = 0.5f;
const float iouThreshold = 0.45f;
const int inputWidth = 640;
const int inputHeight = 640;
Yolo::Yolo(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::Yolo)
{
    ui->setupUi(this);

    playFrameTimer = new QTimer(this);
    m_onnxRuntimeThread = new OnnxRuntimeThread(this);

    connect(m_onnxRuntimeThread, &OnnxRuntimeThread::needNewFrame,
        this, &Yolo::on_onnxRTGetFrame, Qt::UniqueConnection);
    connect(m_onnxRuntimeThread, &OnnxRuntimeThread::ableToUpdate,
        this, &Yolo::updataPixmap);

    // 绑定成员函数，支持引用参数
    m_onnxRuntimeThread->setCallback_proc_output(std::bind(&Yolo::proc_output, this, std::placeholders::_1, std::placeholders::_2));
    m_onnxRuntimeThread->setCallback_proc_output_seg(std::bind(&Yolo::proc_output_seg, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

Yolo::~Yolo()
{
    m_onnxRuntimeThread->setStop(true);
    m_onnxRuntimeThread->wait();
    delete ui;
}

void Yolo::predictYOLOv8Seg(cv::dnn::Net& net, cv::Mat& frame)
{

    // Step 1: 图像预处理
    cv::Mat blob;
    cv::Mat resized;

    // BGR -> RGB，归一化到 [0,1]，不做均值减法
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
    net.setInput(blob);

    QElapsedTimer timer;
    timer.start();

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    qint64 elapsedMs = timer.elapsed();
    qDebug() << "forward time:" << elapsedMs << "ms";

    for (int i = 0; i < outputs.size(); ++i) {
        std::cout << "Output[" << i << "] shape = ";
        for (int j = 0; j < outputs[i].dims; ++j) {
            std::cout << outputs[i].size[j] << " ";
        }
        std::cout << std::endl;
    }

    cv::Mat dets = outputs[0]; // outputs[0] shape = 1x(4+1+num_class+32)*8400
    cv::Mat mask_protos = outputs[1]; // 32x160x160 (对应原图640x640)

    proc_output(frame, dets);

    //    qDebug() << "indices.size = " << indices.size();
    //    for (int idx : indices) {
    //        // std::cout << "index:" << idx << " spot:" << boxes[idx] << std::endl;
    //        // cv::Rect box = boxes[idx];

    //        int ss = mask_protos.size[0]; // mask_protos = output[1]

    //        // 确保mask_protos正确reshape
    //        // 1*32*160*160
    //        cv::Mat mask_proto_flat = mask_protos.reshape(1, 32); // mask_protos.size[1]==32

    //        // cv::Mat mask_proto_flat = mask_protos.reshape(1, {32, 160, 160}); // 1x32x(160*160) -> 32x(160*160)

    //        // 确保矩阵连续且类型正确
    //        mask_proto_flat = mask_proto_flat.clone(); // 保证内存连续
    //        mask_proto_flat.convertTo(mask_proto_flat, CV_32F); // 确保浮点类型

    //        cv::Mat mask_vector = masks[idx]; // 1x32
    //        mask_vector = mask_vector.reshape(1, 1).clone(); // 强制转为1x32连续矩阵
    //        mask_vector.convertTo(mask_vector, CV_32F); // 确保浮点类型

    //        // 检查维度是否匹配
    //        if (mask_vector.cols != mask_proto_flat.rows) {
    //            std::cerr << "dim no match: "
    //                      << mask_vector.size() << " vs "
    //                      << mask_proto_flat.size() << std::endl;
    //            continue;
    //        }
    //        /*
    //         * outPut[0]：每个框都带有32个系数
    //         * outPut[1]：32张 160×160 的图
    //         *
    //         * outPut[1]的每个[160x160]的图乘上outPut[0]的对应系数，加在一起
    //         * 就得到了该框的 160×160 的mask
    //         *

    //        */
    //        // 安全矩阵乘法
    //        cv::Mat mask_flat;
    //        cv::gemm(mask_vector, mask_proto_flat, 1.0, cv::Mat(), 0.0, mask_flat); // 1x(160*160)

    //        // 后续处理保持不变...
    //        cv::Mat mask = mask_flat.reshape(0, 160); // 160x160

    //        //********//********//********//********
    //        // sigmoid
    //        cv::exp(-mask, mask);
    //        mask = 1.0 / (1.0 + mask);
    //        // resize到原图
    //        cv::Mat full_mask;
    //        cv::resize(mask, full_mask, frame.size());

    //        cv::Mat cropped_mask = full_mask(boxes[idx]) > 0.6;

    //        cv::Scalar mask_color(rand() % 256, rand() % 256, rand() % 256); // 随机颜色

    //        // 修改后的掩膜应用部分：
    //        cv::Mat color_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
    //        color_mask.setTo(cv::Scalar(0, 0, 255), full_mask > 0.5);

    //        // 仅将框内满足阈值的区域复制到原图
    //        cv::Mat roi = frame(boxes[idx]);
    //        color_mask(boxes[idx]).copyTo(roi, cropped_mask);
    //        // resize到原图
    //        cv::Mat full_mask;
    //        cv::resize(mask, full_mask, frame.size());

    //        // 裁剪 box 对应的部分
    //        cv::Mat cropped_mask = full_mask(boxes[idx]) > 0.6;

    //        cv::Mat color_roi = cropped_mask(boxes[idx]);
    //        color_roi.setTo(full_mask, cropped_mask);

    //        // 将彩色掩膜叠加到原图（带透明度）
    //        double alpha = 0.5; // 掩膜透明度
    //        cv::addWeighted(full_mask, alpha, frame, 1.0 - alpha, 0, frame);

    //        // 可视化
    //        //rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
    //        //putText(frame, std::to_string(class_ids[idx]), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);

    //        // 叠加 mask
    //        //cv::Mat color(frame.rows, frame.cols, CV_8UC3, cv::Scalar(0, 0, 255));
    //        //color.copyTo(frame, full_mask > 0.6);
    // }

    qDebug() << "Proceses time:" << elapsedMs << "ms";
}

void Yolo::predictYOLOv8Seg2(cv::dnn::Net& net, const cv::Mat& frame)
{

    // Step 1: 图像预处理
    cv::Mat blob;
    cv::Mat resized;
    // cv::resize(frame, resized, cv::Size(inputWidth, inputHeight));

    // BGR -> RGB，归一化到 [0,1]，不做均值减法
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false, CV_32F);

    // Step 2: 输入网络
    net.setInput(blob);

    // Step 3: 推理
    QElapsedTimer timer;
    timer.start();

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    qint64 elapsedMs = timer.elapsed();
    qDebug() << "forward time:" << elapsedMs << "ms";

    // Initialize vectors to hold respective outputs while unwrapping     detections.
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];

    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);

    float* data = (float*)outputs[0].data;

    // Resizing factor.
    float x_factor = frame.cols / 640;
    float y_factor = frame.rows / 640;

    // Iterate through  detections.
    // cout << "num detections  : " << rows << " " << dimensions << endl;
    for (int i = 0; i < rows; ++i) {
        float* classes_scores = data + 4;

        cv::Mat scores(1, 80, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > 0.45) // SCORE_THRESHOLD
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);

            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }
    // Perform Non-Maximum Suppression and draw predictions.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.45, 0.5, indices); // SCORE_THRESHOLD, NMS_THRESHOLD
    // std::cout<<"detect numb :"<<indices.size()<<std::endl;

    // 6. 绘制检测结果
    for (int idx : indices) {

        // std::cout << "index:" << idx << " spot:" << boxes[idx] << std::endl;
        cv::Rect box = boxes[idx];
        float confidence = confidences[idx];
        int class_id = class_ids[idx];

        // 随机生成鲜艳颜色
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

        cv::rectangle(frame, box, color, 2);

        // 6.4 在框右下角显示类别ID
        std::string idText = cv::format("ID:%d", class_id);
        cv::putText(frame, idText,
            cv::Point(box.x + box.width - 50, box.y + box.height - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void Yolo::init_onnxRuntime(QString onnx_path)
{
    //    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "my-onnx");
    //    Ort::SessionOptions session_options;
    //    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    //    session_options.SetIntraOpNumThreads(4); //    // 线程数:4
    //    std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
}

void Yolo::predictYOLOv8Seg(Ort::Session& session, const cv::Mat& frame)
{

    // 可 reshape 为 cv::Mat 或者 OpenCV MatVec 来进行后处理
}

void Yolo::run_onnxRuntime()
{
    // 创建 VideoCapture 对象并打开视频文件

    cap.open(videoPath.toStdString());
    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

#define OTHERTHREAD

#ifndef OTHERTHREAD
    //  ********************************************************************************
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "my-onnx");

    qDebug() << " Ort::Env env ";
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    session_options.SetIntraOpNumThreads(8); //    // 线程数:4

    std::string onnxpath = "G:/yolo11n-seg-nodynamic.onnx";
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    //    // 加载模型并创建会话
    Ort::Session session(env, modelPath.c_str(), session_options);

    // ******************* 4.获取模型输入输出信息 *******************
    int input_nodes_num = session.GetInputCount(); // 输入节点输
    int output_nodes_num = session.GetOutputCount(); // 输出节点数
    std::vector<std::string> input_node_names; // 输入节点名称
    std::vector<std::string> output_node_names; // 输出节点名称
    Ort::AllocatorWithDefaultOptions allocator; // 创建默认配置的分配器实例,用来分配和释放内存
    // 输入图像尺寸
    int input_h = 0;
    int input_w = 0;
    // 获取模型输入信息
    for (int i = 0; i < input_nodes_num; i++) {
        // 获得输入节点的名称并存储
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());

        // 显示输入图像的形状
        auto inputShapeInfo = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        int ch = inputShapeInfo[1];
        input_h = inputShapeInfo[2];
        input_w = inputShapeInfo[3];
        std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
    }

    // 获取模型输出信息
    int num = 0;
    int nc = 0;
    for (int i = 0; i < output_nodes_num; i++) {
        // 获得输出节点的名称并存储
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(output_name.get());
        // 显示输出结果的形状
        auto outShapeInfo = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        num = outShapeInfo[0];
        nc = outShapeInfo[1];
        QString outstr = "output format";
        outstr += "[";
        outstr += std::to_string(i);
        outstr += "]: ";
        for (auto out : outShapeInfo) {
            outstr += std::to_string(out);
            outstr += 'x';
        }
        outstr.chop(1);
        std::cout << outstr.toStdString() << std::endl;
    }

    // ******************* 6.推理准备 *******************
    // 占用内存大小,后续计算是总像素*数据类型大小
    size_t tpixels = 3 * input_h * input_w;
    std::array<int64_t, 4> input_shape_info { 1, 3, input_h, input_w };
    // 准备数据输入
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    // const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    //  声明所有输出节点
    std::vector<const char*> outNames;
    for (auto& name : output_node_names) {
        outNames.push_back(name.c_str());
    }
    // **************************************************

#else

#endif

    while (true) {

        QElapsedTimer timer;
        timer.start();
        {

            cv::Mat tmp;
            cap >> tmp;
            // frame.release(); 不需要
            // std::lock_guard<std::mutex> lock(frame_mutex);
            frame = std::move(tmp);
            // 如果帧为空，说明视频已经结束
            if (frame.empty()) {
                //_stop = true;
                // if(cv::waitKey(1) == 27)
                // std::cout << "按下了ESC键，停止播放视频";
                break;
            }
        }
#ifndef OTHERTHREAD
        // 1. 预处理图像（归一化、尺寸变换）
        cv::Mat resized, float_img;
        cv::resize(frame, resized, cv::Size(inputWidth, inputHeight));
        resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

        // NCHW 格式
        std::vector<float> input_tensor_values;
        input_tensor_values.resize(3 * inputHeight * inputWidth);

        // 通道分离（HWC -> CHW）
        std::vector<cv::Mat> chw(3);
        for (int i = 0; i < 3; ++i)
            chw[i] = cv::Mat(inputHeight, inputWidth, CV_32F, &input_tensor_values[i * inputHeight * inputWidth]);
        cv::split(float_img, chw);

        // 2. 创建输入张量
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::array<int64_t, 4> input_shape = { 1, 3, inputHeight, inputWidth };

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size());

        // 3. 推理
        auto ort_outputs = session.Run(
            Ort::RunOptions { nullptr },
            inputNames.data(),
            &input_tensor,
            1,
            outNames.data(),
            outNames.size());

        // 创建与 OpenCV DNN 相同格式的 cv::Mat 输出
        std::vector<cv::Mat> outputs;

        // 处理第一个输出 (1, 116, 8400)
        {
            float* data = ort_outputs[0].GetTensorMutableData<float>();
            auto shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

            // 创建 3D cv::Mat: [1, 116, 8400]
            std::vector<int> sizes(shape.begin(), shape.end());
            cv::Mat output0(static_cast<int>(sizes.size()), sizes.data(), CV_32F, data);
            outputs.push_back(output0);
        }

        // 处理第二个输出 (1, 32, 160, 160)
        {
            float* data = ort_outputs[1].GetTensorMutableData<float>();
            auto shape = ort_outputs[1].GetTensorTypeAndShapeInfo().GetShape();

            // 创建 4D cv::Mat: [1, 32, 160, 160]
            std::vector<int> sizes(shape.begin(), shape.end());
            cv::Mat output1(static_cast<int>(sizes.size()), sizes.data(), CV_32F, data);
            outputs.push_back(output1);
        }

        //        std::cout << "Output[" << 0 << "] shape = ";
        //        for (int j = 0; j < outputs[0].dims; ++j) {
        //            std::cout << outputs[0].size[j] << " ";
        //        }
        //        std::cout << std::endl;

        //        std::cout << "Output[" << 1 << "] shape = ";
        //        for (int j = 0; j < outputs[1].dims; ++j) {
        //            std::cout << outputs[1].size[j] << " ";
        //        }
        //        std::cout << std::endl;

        proc_output(frame, outputs[0]);
        // ****************imshow("Video", frame);******************************************

        qint64 elapsedMs = timer.elapsed();
        qDebug() << "forward time:" << elapsedMs << "ms";

#endif
    }
}

/*
 * 处理目标检测输出
 */
void Yolo::proc_output(cv::Mat& frame, const cv::Mat& output)
{
    // 解析预测结果
    int numDet1 = output.size[1]; //  84
    int numDet2 = output.size[2]; //  8400

    int dimensions = output.size[1];
    cv::Mat dets = output.reshape(1, dimensions); // dimensions == 116
    cv::transpose(dets, dets);

    int num_classes = dets.cols - 4 - 0 - 32; // 80

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Mat> masks;
    // qDebug()<<"dets.rows = "<<dets.rows;

    for (int i = 0; i < dets.rows; ++i) {

        // 直接提取类别分数
        cv::Mat scores = dets.row(i).colRange(4, 4 + 80); // [4,84)

        //  找到最高类别分数
        cv::Point classIdPoint;
        double maxScore;
        cv::minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

        // 过滤低置信度检测
        if (maxScore < confThreshold)
            continue;

        // std::cout<<"maxScore"<<maxScore<<" ";

        // Box: cx, cy, w, h

        float cx = dets.at<float>(i, 0); // 中心x
        float cy = dets.at<float>(i, 1); // 中心y
        float w = dets.at<float>(i, 2); // 宽度
        float h = dets.at<float>(i, 3); // 高度
        // printf("%.1f,%.1f,%.1f,%.1f\n",cx,cy,w,h);

        // 转换为原图坐标
        int left = static_cast<int>((cx - w / 2.0f) * frame.cols / inputWidth);
        int top = static_cast<int>((cy - h / 2.0f) * frame.rows / inputHeight);
        int width = static_cast<int>(w * frame.cols / inputWidth);
        int height = static_cast<int>(h * frame.rows / inputHeight);

        boxes.emplace_back(left, top, width, height);
        // std::cout << " spot:" << boxes.back() << std::endl;
        confidences.push_back(static_cast<float>(maxScore));
        class_ids.push_back(classIdPoint.x);

        // 取掩膜向量
        cv::Mat mask_feat = dets.row(i).colRange(4 + num_classes, dets.cols); // [85,116)
        masks.push_back(mask_feat.clone()); // 1x32
    }

    // NMS
    std::vector<int> indices;
    // 非极大值抑制NMS的函数，常用于目标检测任务中，以去除重叠度较高的框，保留检测得分最高的框。
    //  boxes格式为 (x, y, width, height)?
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);
    // std::cout<<"detect numb :"<<indices.size()<<std::endl;

    // 6. 绘制检测结果
    for (int idx : indices) {

        // std::cout << "index:" << idx << " spot:" << boxes[idx] << std::endl;
        cv::Rect box = boxes[idx];
        float confidence = confidences[idx];
        int class_id = class_ids[idx];

        // 随机生成鲜艳颜色
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

        // 绘制矩形框（带2像素边框）

        cv::rectangle(frame, box, color, 2);

        //  绘制填充背景的标签
        // std::string label = cv::format("%s: %.2f", classNames[class_id].c_str(), confidence);

        int baseline;
        // cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 标签背景矩形
        // cv::Rect labelRect(box.x, box.y - labelSize.height - 5,
        //                   labelSize.width, labelSize.height + 5);

        // cv::rectangle(frame, labelRect, color, cv::FILLED);
        // cv::rectangle(frame, labelRect, cv::Scalar(0, 0, 0), 1); // 黑色边框

        //   绘制文本
        // cv::putText(frame, label,
        //            cv::Point(box.x, box.y - 5),
        //            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        //  在框右下角显示类别ID
        std::string idText = cv::format("ID:%d", class_id);
        cv::putText(frame, idText,
            cv::Point(box.x + box.width - 50, box.y + box.height - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // ui->dispWidget->m_pixmap = std::make_unique<QPixmap>(cvMatToQPixmap(frame));
    // ui->dispWidget->update();
}
/*
 * 处理实例分割输出
 */
void Yolo::proc_output_seg(cv::Mat& frame, const cv::Mat& output, const cv::Mat& output_mask)
{ /**
     cv::Mat &frame, // 原始图像
         cv::Rect box, // (left, top, width, height)
         cv::Mat mask, //[160,160]. 已经由output[1]:[1,32,160,160]->[32,160*160]->与output[0]锚框的权重系数 [1,32]相乘得到[160,160]
 **/
    /*
     * output : dims=3, 1×116×8400
     * ->>> 8400×116
     * 遍历8400个锚框
     *  ->> 每个锚框 80 + 32 个参数
     *
     * output_mask : dims=4, 1×32×160×160
     * ->>> 32×160×160
     *
     *  output_mask是共用的
     *  拿锚框的116个参数其中的32个，跟output_mask乘：
     *  [1×32] × [32×[160*160]] = [1×[160*160]]
     */
    // 解析预测结果

    int dimensions0 = output.size[0];
    int dimensions1 = output.size[1];
    int dimensions2 = output.size[2];

    cv::Mat dets = output.reshape(1, output.size[1]); // dimensions == 116
    cv::transpose(dets, dets);

    int num_classes = dets.cols - 4 - 0 - 32; // 80

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Mat> masks;
    // qDebug()<<"dets.rows = "<<dets.rows;

    for (int i = 0; i < dets.rows; ++i) {

        // 直接提取类别分数
        cv::Mat scores = dets.row(i).colRange(4, 4 + 80); // [4,84)

        //  找到最高类别分数
        cv::Point classIdPoint;
        double maxScore;
        cv::minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

        // 过滤低置信度检测
        if (maxScore < confThreshold)
            continue;

        // std::cout<<"maxScore"<<maxScore<<" ";

        // Box: cx, cy, w, h

        float cx = dets.at<float>(i, 0); // 中心x
        float cy = dets.at<float>(i, 1); // 中心y
        float w = dets.at<float>(i, 2); // 宽度
        float h = dets.at<float>(i, 3); // 高度
        // printf("%.1f,%.1f,%.1f,%.1f\n",cx,cy,w,h);

        // 转换为原图坐标
        float k_w = static_cast<float>(frame.cols) / inputWidth;
        float k_h = static_cast<float>(frame.rows) / inputHeight;

        //        int left = static_cast<int>((cx - w / 2.0f) * k_w;
        //        int top = static_cast<int>((cy - h / 2.0f) * frame.rows / inputHeight);
        //        int width = static_cast<int>(w * frame.cols / inputWidth);
        //        int height = static_cast<int>(h * frame.rows / inputHeight);

        int left = std::round((cx - w / 2.0f) * k_w);
        int top = std::round((cy - h / 2.0f) * k_h);
        int width = std::round(w * k_w);
        int height = std::round(h * k_h);

        // 修正为不越界
        left = std::clamp(left, 0, frame.cols - 1);
        top = std::clamp(top, 0, frame.rows - 1);
        if (left + width > frame.cols)
            width = frame.cols - left;
        if (top + height > frame.rows)
            height = frame.rows - top;

        boxes.emplace_back(left, top, width, height);
        // std::cout << " spot:" << boxes.back() << std::endl;
        confidences.push_back(static_cast<float>(maxScore));
        class_ids.push_back(classIdPoint.x);

        // 取掩膜向量
        cv::Mat mask_feat = dets.row(i).colRange(4 + num_classes, dets.cols); // [85,116)
        masks.push_back(mask_feat.clone()); // 1x32
    }

    // NMS
    std::vector<int> indices;
    // 非极大值抑制NMS的函数，常用于目标检测任务中，以去除重叠度较高的框，保留检测得分最高的框。
    //  boxes格式为 (x, y, width, height)?
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);
    // std::cout<<"detect numb :"<<indices.size()<<std::endl;

    // 6. 绘制检测结果
    QElapsedTimer timer;
    timer.start();

    for (int idx : indices) {

        // std::cout << "index:" << idx << " spot:" << boxes[idx] << std::endl;
        cv::Rect box = boxes[idx];
        float confidence = confidences[idx];
        int class_id = class_ids[idx];

        /** **********************************
         * ******掩膜部分************
         ********************************/

        // 将其 reshape 成 2D 的 32 x 25600
        // 第一个参数 1 表示每个通道保持为1个通道（不合并通道），第二个参数为新的行数

        cv::Mat mask_proto_flat = output_mask.reshape(1, 32); // 32行，自动计算出列数为160*160=25600
        //        std::cout << "output_mask shape = ";
        //        for (int j = 0; j < output_mask.dims; ++j) {
        //            std::cout << output_mask.size[j] << " ";
        //        }
        //        std::cout << std::endl;

        // 检查维度是否匹配

        if (masks[idx].cols != mask_proto_flat.rows) {
            std::cerr << "dim no match: "
                      << masks[idx].size() << " vs "
                      << mask_proto_flat.size() << std::endl;
            continue;
        }

        cv::Mat mask_flat; // 水平展开
        cv::gemm(masks[idx], mask_proto_flat, 1.0, cv::Mat(), 0.0, mask_flat); // 1x(160*160) [25600 x 1]
        // std::cerr << " mask_flat " << mask_flat.size() << std::endl;

        cv::Mat mask = mask_flat.reshape(0, 160); // 160x160
        // std::cerr << " mask_flat " << mask_flat.size() << std::endl;
        /*
       cv::exp(-mask, mask);
       mask = 1.0 / (1.0 + mask);

       cv::resize(mask, mask, frame.size()); // resize到原图大小
       std::cerr << " mask " << mask.size() << std::endl;
       cv::Mat cropped_mask = mask(boxes[idx]) > 0.6;
       std::cerr << " cropped_mask " << cropped_mask.size() << std::endl;

       cv::Scalar mask_color(rand() % 256, rand() % 256, rand() % 256); // 随机颜色

       cv::Mat color_mask = cv::Mat::zeros(frame.size(), CV_8UC3);
       cv::Mat roi_color_mask = color_mask(boxes[idx]);
       roi_color_mask.setTo(cv::Scalar(0, 0, 255), cropped_mask);

       color_mask.setTo(cv::Scalar(0, 0, 255), roi_color_mask > 0.5);

       cv::Mat roi = frame(boxes[idx]);
       color_mask(boxes[idx]).copyTo(roi, cropped_mask);

       //        // resize到原图
       //        cv::Mat full_mask;
       //        cv::resize(mask, full_mask, frame.size());

       //        // 裁剪 box 对应的部分
       //        cv::Mat cropped_mask = full_mask(boxes[idx]) > 0.6;

       //        cv::Mat color_roi = cropped_mask(boxes[idx]);
       //        color_roi.setTo(full_mask, cropped_mask);

       //        // 将彩色掩膜叠加到原图（带透明度）
       //        double alpha = 0.5; // 掩膜透明度
       //        cv::addWeighted(full_mask, alpha, frame, 1.0 - alpha, 0, frame);
*/

        // applySegmentationMask(frame, boxes[idx], mask);

        /** **********************************
         * ***********************************
         ********************************/
        cv::Scalar color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        cv::rectangle(frame, box, color, 2);

        //  绘制填充背景的标签
        // std::string label = cv::format("%s: %.2f", classNames[class_id].c_str(), confidence);

        int baseline;
        // cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // 标签背景矩形
        // cv::Rect labelRect(box.x, box.y - labelSize.height - 5,
        //                   labelSize.width, labelSize.height + 5);

        // cv::rectangle(frame, labelRect, color, cv::FILLED);
        // cv::rectangle(frame, labelRect, cv::Scalar(0, 0, 0), 1); // 黑色边框

        //   绘制文本
        // cv::putText(frame, label,
        //            cv::Point(box.x, box.y - 5),
        //            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

        //  在框右下角显示类别ID
        std::string idText = cv::format("ID:%d", class_id);
        cv::putText(frame, idText,
            cv::Point(box.x + box.width - 50, box.y + box.height - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    qint64 paintTime = timer.elapsed();
    qDebug() << "paintTime time =" << paintTime << "ms,";
    // ui->dispWidget->m_pixmap = std::make_unique<QPixmap>(cvMatToQPixmap(frame));
    // ui->dispWidget->update();
}

void Yolo::applySegmentationMask(cv::Mat& frame, const cv::Rect& box, const cv::Mat& mask)
{
#ifdef true

    // 1. 将160x160掩膜缩放到原图尺寸
    //
    cv::Mat fullsize_mask;
    cv::resize(mask, fullsize_mask, frame.size()); //

    // 2. 裁剪出检测框区域的掩膜
    cv::Mat roi_mask;
    try {
        roi_mask = fullsize_mask(box);
    } catch (const cv::Exception& e) {
        std::cout << frame.size << box << std::endl;

        return;
    }

    // 3. 二值化处理 (仅检测框内区域)
    cv::Mat binary_roi_mask = roi_mask > 0.6; // 得到检测框内的二值掩膜

    // 4. 创建彩色掩膜 (仅检测框大小)
    cv::Mat color_roi_mask = cv::Mat::zeros(box.size(), CV_8UC3);

    // 5. 生成随机颜色
    // cv::Scalar mask_color(rand() % 256, rand() % 256, rand() % 256);
    cv::Scalar mask_color(0, 0, 255);

    // 6. 在检测框区域应用颜色
    color_roi_mask.setTo(mask_color, binary_roi_mask);

    // 7. 获取原图的检测框区域
    cv::Mat frame_roi = frame(box);

    // 8. 半透明叠加 (仅在检测框内)
    double alpha = 0.5; // 掩膜透明度
    cv::Mat blended_roi;
    cv::addWeighted(frame_roi, 1.0, color_roi_mask, alpha, 0.0, blended_roi);

    // 9. 将处理后的区域复制回原图
    blended_roi.copyTo(frame_roi);

    // 10. 绘制检测框边界 (可选)
    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
#else
    // 1. 边界检查 (确保box在frame范围内)
    cv::Rect valid_box = box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (valid_box.empty()) {
        std::cout << "Invalid box: " << box << " on frame " << frame.size() << std::endl;
        return;
    }

    // 2. 计算原始mask中对应的ROI区域 (避免全图缩放)
    cv::Rect mask_roi(
        cvRound(valid_box.x * mask.cols / static_cast<float>(frame.cols)),
        cvRound(valid_box.y * mask.rows / static_cast<float>(frame.rows)),
        cvRound(valid_box.width * mask.cols / static_cast<float>(frame.cols)),
        cvRound(valid_box.height * mask.rows / static_cast<float>(frame.rows)));

    // 确保mask_roi在mask范围内
    mask_roi &= cv::Rect(0, 0, mask.cols, mask.rows);
    if (mask_roi.empty())
        return;

    // 3. 提取并缩放mask的对应区域
    cv::Mat roi_mask;
    cv::resize(mask(mask_roi), roi_mask, valid_box.size());

    // 4. 二值化处理
    cv::Mat binary_roi_mask = roi_mask > 0.6;

    // 5. 创建彩色掩膜
    cv::Scalar mask_color(0, 0, 255);
    cv::Mat color_roi_mask(valid_box.size(), CV_8UC3, mask_color);
    color_roi_mask.setTo(cv::Scalar(0, 0, 0), ~binary_roi_mask);

    // 6. 获取原图检测框区域
    cv::Mat frame_roi = frame(valid_box);

    // 7. 高效混合
    double alpha = 0.5;
    double beta = 1.0 - alpha;
    for (int y = 0; y < frame_roi.rows; ++y) {
        uchar* f = frame_roi.ptr<uchar>(y);
        const uchar* m = color_roi_mask.ptr<uchar>(y);
        for (int x = 0; x < frame_roi.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                int idx = 3 * x + c;
                f[idx] = cv::saturate_cast<uchar>(f[idx] * beta + m[idx] * alpha);
            }
        }
    }

    // 8. 绘制检测框边界
    cv::rectangle(frame, valid_box, cv::Scalar(0, 255, 0), 2);
#endif
}

int Yolo::init_detect(QString onnx_path)
{

    /**
     * 1、有可用的gpu设备 getCudaEnabledDeviceCount>0
     * 2、判断当前gpu是否兼容opencv的cuda info.isCompatible()
     */
    std::cout << "getCudaEnabledDeviceCount :" << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
    std::cout << "getDevice :" << cv::cuda::getDevice() << std::endl;

    net = cv::dnn::readNetFromONNX(onnx_path.toStdString());

    QString dllName = "cudnn64_8.dll";
    HMODULE handle = LoadLibraryA(dllName.toStdString().c_str());
    if (handle) {
        FreeLibrary(handle);
        std::cout << "cudnn64_8 findout" << std::endl;
    }

    if (cv::cuda::getCudaEnabledDeviceCount() != 0) { // 返回非0就是可用
        qDebug() << "CudaEnabled, Device :" << cv::cuda::getDevice();
        cv::cuda::DeviceInfo info(0); // 获取当前 CUDA 设备的信息，包括名称、内存、是否支持特性等。
        qDebug() << "Supports CUDA: " << info.isCompatible(); // 判断当前设备是否兼容 OpenCV 的 CUDA 功能
        if (info.isCompatible()) {

            try {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

                cv::Mat input = cv::Mat::ones(1, 3 * 640 * 640, CV_32F);
                input = input.reshape(1, { 1, 3, 640, 640 });
                net.setInput(input);
                cv::Mat out = net.forward();
                qWarning() << "CUDA DNN forward succeeded";
                return 0;

            } catch (const cv::Exception& e) {
                qWarning() << "CUDA DNN not working: " << e.what();
            }
        } else {
            qWarning() << "DNN USE CPU.. ";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            cv::Mat input = cv::Mat::ones(1, 3 * 640 * 640, CV_32F);
            input = input.reshape(1, { 1, 3, 640, 640 });
            net.setInput(input);
            cv::Mat out = net.forward();

            qWarning() << "...SUCCESS. ";
        }
    } else
        return -1;

    return 0;
}

void Yolo::paintEvent(QPaintEvent* e)
{

    // if(m_pixmap->isNull()) return ;
    // QPainter painter(ui->graphicsView);

    // painter.setRenderHint(QPainter::Antialiasing);

    // painter.drawPixmap(0, 0, m_pixmap->scaled(size().width(), size().height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

QPixmap Yolo::cvMatToQPixmap(const cv::Mat& input)
{
    // 1. 检查图像是否为空
    if (input.empty()) {
        return QPixmap();
    }

    // 2. 区分图像通道数
    QImage image;
    switch (input.type()) {
    case CV_8UC1: { // 灰度图
        image = QImage(input.data, input.cols, input.rows, input.step, QImage::Format_Grayscale8);
        break;
    }
    case CV_8UC3: { // BGR 彩图，需转为 RGB

        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        image = QImage(input.data, input.cols, input.rows, input.step, QImage::Format_RGB888);

        break;
    }
    case CV_8UC4: { // BGRA，Qt 支持 ARGB32 格式

        image = QImage(input.data, input.cols, input.rows, input.step, QImage::Format_ARGB32);
        break;
    }
    default:
        qWarning("Unsupported cv::Mat image type for conversion to QPixmap.");
        return QPixmap();
    }

    // 3. 拷贝数据，防止原始 mat 释放后指针失效
    return QPixmap::fromImage(image.copy());
}

void Yolo::on_pushButtonDnn_clicked()
{
    // 创建 VideoCapture 对象并打开视频文件

    cap.open(videoPath.toStdString());
    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

    int ret = init_detect("G:\\yolo11n-seg-nodynamic.onnx");
    std::cout << "init_detect :" << ret << std::endl;
    playFrameTimer->start(30);
    connect(playFrameTimer, &QTimer::timeout, this, &Yolo::on_playTimeout);
}

void Yolo::on_playTimeout()
{
    cv::Mat frame;
    {
        // 读取下一帧
        cap >> frame;

        // 如果帧为空，说明视频已经结束
        if (frame.empty()) {
            // if(cv::waitKey(1) == 27)
            // std::cout << "按下了ESC键，停止播放视频";

            playFrameTimer->stop();
            disconnect(playFrameTimer, &QTimer::timeout, this, &Yolo::on_playTimeout);
        }
        // 处理frame
        predictYOLOv8Seg(net, frame);
        // imshow("Video", frame);

        ui->dispWidget->m_pixmap = std::make_unique<QPixmap>(cvMatToQPixmap(frame));

        ui->dispWidget->update();
    }
}

void Yolo::on_onnxRTGetFrame()
{
    if (!m_onnxRuntimeThread->frameEmpty()) {
        playFrameTimer->start(10);
        return;
    }
    cv::Mat tmp;
    cap >> tmp;
    // frame.release(); 不需要
    // std::lock_guard<std::mutex> lock(frame_mutex);
    // frame = std::move(tmp);

    // 如果帧为空，说明视频已经结束
    if (tmp.empty() || tmp.size().height <= 0) {

        m_onnxRuntimeThread->setStop(true);
        m_onnxRuntimeThread->wait();

        // if(cv::waitKey(1) == 27)
        // std::cout << "按下了ESC键，停止播放视频";
        playFrameTimer->stop();

        disconnect(playFrameTimer, &QTimer::timeout, this, &Yolo::on_onnxRTGetFrame);
    } else {
        playFrameTimer->start(30);
        // qDebug() << tmp.size().height;
        m_onnxRuntimeThread->storeFrame(tmp);
    }
}

void Yolo::updataPixmap(cv::Mat frame)
{
    // qDebug() << "updataPixmap";
    ui->dispWidget->m_pixmap = std::make_unique<QPixmap>(cvMatToQPixmap(frame));
    update();
}

void Yolo::on_pushButtonRuntime_clicked()
{
    // 创建 VideoCapture 对象并打开视频文件
    if (m_onnxRuntimeThread->isRunning()) {
        m_onnxRuntimeThread->setStop(true);
        m_onnxRuntimeThread->wait();
    }
    cap.open(videoPath.toStdString());
    // 检查视频是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

    m_onnxRuntimeThread->setStop(false);

    m_onnxRuntimeThread->start();

    playFrameTimer->start(30);
    connect(playFrameTimer, &QTimer::timeout, this, &Yolo::on_onnxRTGetFrame);
}

void Yolo::on_pushButtonOpenVideo_clicked()
{

    QString fileName = QFileDialog::getOpenFileName(this, "open video", nullptr, "video (*.mp4 *.mkv *.ts);;所有文件 (*.*)");
    if (!fileName.isNull()) {
        videoPath = fileName;
    }
}
