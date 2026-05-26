#include "opencv2/opencv.hpp"
#include "onnxruntime_c_api.h"

#include <time.h>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Replace with the actual path to your model file
#define MODEL_PATH "./best.onnx"

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

#define ORT_ABORT_ON_ERROR(expr)                             \
  do {                                                       \
    OrtStatus* onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                          \
      g_ort->ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

int getPredictedClass(float* outputData) {
    // Find the index with the highest probability among 43 classes
    int numClasses = 43;
    int predictedClass = 0;

    for (int i = 1; i < numClasses; ++i) {
        if (outputData[i] > outputData[predictedClass]) {
            predictedClass = i;
        }
    }

    return predictedClass;
}

// HWC (Height-Width-Channel) to CHW (Channel-Height-Width) transpose
// OpenCV stores pixels as [R,G,B,R,G,B,...], ONNX model expects [all R, all G, all B]
static void hwc_to_chw(const float* input, size_t h, size_t w, float* output) {
  size_t stride = h * w;
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output[c * stride + i] = input[i * 3 + c];
    }
  }
}

int total = 0;
double total_time = 0.0;
int main() {
    // Specify the path to the folder containing images
    std::string folderPath = "/home/fazliddin/work/xor/german_traffic_sign_classification/gtrsb_dataset/Test";

    // Initialize ONNX Runtime environment once
    OrtEnv* env;
    ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime", &env));

    // Load the ONNX model once
    OrtSession* session;
    ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, MODEL_PATH, NULL, &session));

    // Iterate over all image files in the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        // Skip non-image files
        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png") {
            continue;
        }

        std::cout << "filename: " << entry.path().string() << '\n';
        cv::Mat image = cv::imread(entry.path().string());

        if (image.empty()) {
            std::cerr << "Error reading image: " << entry.path() << std::endl;
            continue;
        }

        int channels = 3;
        int height = 32;
        int width = 32;
        cv::resize(image, image, cv::Size(width, height));

        // Preprocess: BGR->RGB, then normalize to [-1, 1] to match Python training
        // Python uses: ToTensor() -> [0,1], then Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) -> [-1,1]
        // Equivalent: pixel * (2/255) - 1
        cv::Mat norm_image(width, height, CV_32FC3);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(norm_image, norm_image.type(), 2. / 255., -1.0);

        float* input_data = (float*)malloc(height * width * 3 * sizeof(float));
        hwc_to_chw((const float *)norm_image.ptr(), height, width, input_data);
        total += 1;

        const int64_t input_shape[] = {1, channels, height, width};

        // Create the input tensor
        OrtValue* inputTensor;
        OrtMemoryInfo* memory_info;
        ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
        OrtStatus* stat = g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data, height * width * channels * sizeof(float), input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);

        if (stat != NULL) {
            const char* error_message = g_ort->GetErrorMessage(stat);
            printf("Error creating tensor: %s\n", error_message);
            g_ort->ReleaseStatus(stat);
            free(input_data);
            continue;
        }

        // Run inference
        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};
        OrtValue* outputTensor = NULL;
        clock_t begin = clock();
        ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&inputTensor, 1, output_names, 1, &outputTensor));
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        total_time += time_spent;

        // Extract output and determine predicted class
        float* outputData = NULL;
        ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(outputTensor, (void**)&outputData));

        int predictedClass = getPredictedClass(outputData);
        printf("Predicted Class: %d\n", predictedClass);

        // Release per-image resources
        g_ort->ReleaseValue(inputTensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        g_ort->ReleaseValue(outputTensor);
        free(input_data);
    }

    // Release session and environment
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    std::cout << "total image: " << total << '\n';
    return 0;
}
