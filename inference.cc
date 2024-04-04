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
    // Implement the logic to interpret the output tensor and determine the predicted class
    // This might involve finding the index with the highest probability or applying a threshold

    // For simplicity, let's assume the output tensor is a flat array of probabilities,
    // and the predicted class is the index with the highest probability.

    int numClasses = 43; 
    int predictedClass = 0;    
    
    // for (int i = 0; i < numClasses; ++i) {
    //     std::cout << outputData[i] << i << " " << " ";
    // }

    for (int i = 1; i < numClasses; ++i) {
        if (outputData[i] > outputData[predictedClass]) {
            predictedClass = i;
        }
    }

    return predictedClass;
}

// image shape transpose
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
    std::string folderPath = "/home/fazliddin/work/xor/german_traffic_sign_classification/gtrsb_dataset/Test";  // Path to images folder

    // Iterate over all files in the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OrtStatus* status;      

        // Initialize ONNX Runtime environment
        OrtEnv* env;
        
        ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime", &env));      
       
        // Load the ONNX model
        OrtSession* session;    
        ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, MODEL_PATH, NULL, &session));       

        // Check if the file is an image (you can customize this condition)
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            // Read the image
            std::cout << "filename: " << entry.path().string() << '\n';
            cv::Mat image = cv::imread(entry.path().string());

            // Assuming the image has 3 channels (R, G, B)
            int channels = 3; //image.channels();
            int height = 32;//image.rows;
            int width = 32;//image.cols;
            cv::resize(image, image, cv::Size(width, height));
            
            // cv::imshow("Window Name", image); 
            //     // Wait for any keystroke 
            // cv::waitKey(400);

            // Check if the image was successfully loaded
            if (!image.empty()) {
                // Process the image (you can add your image processing logic here)  
                cv::Mat norm_image(width, height, CV_32FC3);
                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                image.convertTo(norm_image, norm_image.type(), 1. / 255., 0);

                float* input_data_1 = (float*)malloc(height * width * 3 * sizeof(float));
                hwc_to_chw((const float *)norm_image.ptr(), height, width, (float *) input_data_1);                
                total += 1;

                if (image.empty()) {
                    // fprintf(stderr, "Failed to load the image: %s\n", folderPath);
                    g_ort->ReleaseSession(session);
                    g_ort->ReleaseEnv(env);
                    return 1;
                }                

                // Convert image data (uchar) to float and normalize
                float* input_data = (float*)malloc(height * width * 3 * sizeof(float));
                int index = 0;
                const int64_t input_shape[] = {1, channels, height, width};
               
                // Create the input tensor
                OrtAllocator* allocator;
                ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&allocator));                

                // Allocate memory for input tensor
                float* inputData = (float*)malloc(height * width * channels * sizeof(float));

                // Create the input tensor
                OrtValue* inputTensor;
                OrtMemoryInfo* memory_info;
                ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
                OrtStatus* stat = g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_data_1, height * width * channels * sizeof(float), input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor); 
                
                if (stat != NULL) {
                const char* error_message = g_ort->GetErrorMessage(status);
                printf("Error creating tensor: %s\n", error_message);
                g_ort->ReleaseStatus(status);
                // Handle the error as needed
                return 1;
                } 

                const char* input_names[] = {"input"};
                const char* output_names[] = {"output"};
                OrtValue* outputTensor = NULL;
                clock_t begin = clock();
                ORT_ABORT_ON_ERROR(g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&inputTensor, 1, output_names, 1, &outputTensor));  
                clock_t end = clock();
                double time_spent = (double)(end - begin) / CLOCKS_PER_SEC ; 
                total_time += time_spent;
                // printf("\nSpending time: %.5f\n", time_spent);
                // printf("\nTotal time: %.5f\n", total_time);
                // printf("\nAverage time: %.5f\n", total_time /total);                

                float* outputData = NULL;
                ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(outputTensor, (void**)&outputData));

                // Determine the predicted class
                int predictedClass = getPredictedClass(outputData);

                // Print the predicted class
                printf("Predicted Class: %d\n", predictedClass);
                
                // Release the input tensor, session, and environment when done
                g_ort->ReleaseValue(inputTensor);
                g_ort->ReleaseSession(session);
                g_ort->ReleaseEnv(env);
                g_ort->ReleaseMemoryInfo(memory_info);
                g_ort->ReleaseValue(outputTensor);                
                free(input_data_1);
               
            } else {
                std::cerr << "Error reading image: " << entry.path() << std::endl;
            }
        }
    }
    std::cout << "total image: " << total << '\n'; 
    return 0;    
}