#include "onnxruntimethread.h"
#include "qdebug.h"

const float confThreshold = 0.7f;
const float iouThreshold = 0.45f;
const int inputWidth = 640;
const int inputHeight = 640;

OnnxRuntimeThread::OnnxRuntimeThread(QObject* parent)
    : QThread { parent }
{

    qDebug() << "OnnxRuntimeThread's parent is " << parent;
}

void OnnxRuntimeThread::run()
{
    /*******************************************************************
     * 初始化onnx组件
     * ***************************************************************** */
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

    // ******************* 推理准备 *******************
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
    /*******************************************************************
     * 初始化onnx组件 结束
     * ***************************************************************** */
    std::cout << "..init onnxRuntime success..." << std::endl;

    while (true) {
        cv::Mat localFrame;

        QElapsedTimer timer;
        timer.start();

        // 条件变量阻塞直到-有新帧/结束
        //  等待新帧或停止信号
        {
            std::unique_lock<std::mutex> lock(frame_mutex);
            cond_.wait(lock, [this] {
                return hasNewFrame || _stop; // 等待条件：有新帧或停止信号
            });

            if (_stop)
                break; // 检查停止信号

            // 获取帧并重置状态
            localFrame = sharedFrame.clone();
            sharedFrame.release();
            hasNewFrame = false;
        }
        // 通知主线程需要新帧

        emit needNewFrame();

        /***********************************************************
         ******* 开始处理  localFrame   *
         ***********************************************************/
        // 预处理图像（归一化、尺寸变换）
        cv::Mat resized, float_img;
        cv::resize(localFrame, resized, cv::Size(inputWidth, inputHeight));
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
        qint64 elapsedM_SsessionRun = timer.elapsed();
        timer.start();
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
        auto outputs0 = outputs[0];
        auto outputs1 = outputs[1];

        m_callback_proc_output_seg(localFrame, outputs[0], outputs[1]);

        qint64 elapsedM_proc = timer.elapsed();
        qDebug() << "forward time =" << elapsedM_SsessionRun << "ms,"
                 << "process time =" << elapsedM_proc << "ms";

        // 通知主线程可以更新UI
        emit ableToUpdate(localFrame);
    }
}
