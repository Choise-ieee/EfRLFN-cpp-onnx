#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <locale>
#include <codecvt>

// Helper function to check Ort status
void CheckStatus(OrtStatus* status) {
    if (status != nullptr) {
        std::cout << Ort::GetApi().GetErrorMessage(status) << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // 1. Parse Arguments
    /*if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_path> <input_image> <output_image>" << std::endl;
        return -1;
    }*/

    std::string onnx_path = "efrlfn_x4.onnx";// argv[1];
    std::string input_path = "bus.jpg";// argv[2];
    std::string output_path = "output4.jpg";// argv[3];

    // 2. Initialize ONNX Runtime Environment
    // Use CPU Execution Provider for simplicity. For GPU, use "CUDAExecutionProvider"
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EfRLFN-Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    //enable CUDA
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0; // 指定 GPU ID，0 表示第一块显卡
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // Windows 下路径需要转换为宽字符
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wide_onnx_path = converter.from_bytes(onnx_path);

    Ort::Session session(env, wide_onnx_path.c_str(), session_options);

    // 3. Load Image using OpenCV
    cv::Mat img = cv::imread(input_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return -1;
    }

    // Original dimensions
    int img_h = img.rows;
    int img_w = img.cols;

    // 4. Preprocessing
    // OpenCV reads BGR, Model expects RGB
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    // Convert HWC (Height, Width, Channel) to NCHW (Batch, Channel, Height, Width)
    // and Normalize (0-255 -> 0-1)
    cv::Mat img_float;
    img_rgb.convertTo(img_float, CV_32F, 1.0 / 255.0);

    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);

    // Create input tensor
    // Shape: {1, 3, Height, Width}
    std::vector<int64_t> input_shape = { 1, 3, img_h, img_w };
    size_t input_tensor_size = 1 * 3 * img_h * img_w;

    std::vector<float> input_tensor_values(input_tensor_size);

    // Interleave HWC to NCHW
    // Channel 0 (R)
    std::memcpy(input_tensor_values.data(), channels[0].data, img_h * img_w * sizeof(float));
    // Channel 1 (G)
    std::memcpy(input_tensor_values.data() + img_h * img_w, channels[1].data, img_h * img_w * sizeof(float));
    // Channel 2 (B)
    std::memcpy(input_tensor_values.data() + 2 * img_h * img_w, channels[2].data, img_h * img_w * sizeof(float));

    // Create Ort Memory Info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_size,
        input_shape.data(),
        input_shape.size()
        );

    // 5. Run Inference
    const char* input_names[] = { "input" }; // Must match export script
    const char* output_names[] = { "output" };

    try {
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        // 6. Postprocessing
        // Get output tensor info
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();

        int out_h = static_cast<int>(output_shape[2]);
        int out_w = static_cast<int>(output_shape[3]);
        int out_c = static_cast<int>(output_shape[1]); // Should be 3

        // Convert NCHW back to HWC
        // We need to de-interleave
        std::vector<cv::Mat> out_channels(out_c);
        for (int c = 0; c < out_c; ++c) {
            out_channels[c] = cv::Mat(out_h, out_w, CV_32FC1, output_data + c * out_h * out_w);
        }

        cv::Mat result_float;
        cv::merge(out_channels, result_float);

        // Clamp (0, 1) and Scale (0-255)
        // Clamping manually or using OpenCV threshold
        cv::threshold(result_float, result_float, 0.0, 0.0, cv::THRESH_TOZERO);
        cv::threshold(result_float, result_float, 1.0, 0.0, cv::THRESH_TRUNC);

        result_float *= 255.0;

        cv::Mat result_uint8;
        result_float.convertTo(result_uint8, CV_8UC3);

        // RGB back to BGR for saving
        cv::Mat result_bgr;
        cv::cvtColor(result_uint8, result_bgr, cv::COLOR_RGB2BGR);

        // 7. Save Output
        cv::imwrite(output_path, result_bgr);
        std::cout << "Saved output to " << output_path << std::endl;

    }
    catch (const Ort::Exception& e) {
        std::cerr << "Error running model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
