// GTSRB traffic sign classification with the ONNX Runtime C API.
//
// Usage:
//   ./run [model.onnx] [dataset_root | image_dir]
//
//   dataset_root : folder containing Test.csv (columns ClassId, Path) -> accuracy is reported
//   image_dir    : any folder of images -> predictions only
//
// Defaults: ./weights/best.onnx ./gtrsb_dataset

#include "onnxruntime_c_api.h"
#include "opencv2/opencv.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const OrtApi* g_ort = nullptr;

#define ORT_ABORT_ON_ERROR(expr)                                   \
  do {                                                             \
    OrtStatus* onnx_status = (expr);                               \
    if (onnx_status != nullptr) {                                  \
      std::fprintf(stderr, "ONNX Runtime error: %s\n",             \
                   g_ort->GetErrorMessage(onnx_status));           \
      g_ort->ReleaseStatus(onnx_status);                           \
      std::abort();                                                \
    }                                                              \
  } while (0)

// Must match utils/dataset.py (IMAGE_SIZE, MEAN, STD) and pytorch2onnx.py (input/output names).
constexpr int kChannels = 3;
constexpr int kHeight = 32;
constexpr int kWidth = 32;
constexpr float kMean = 0.5f;
constexpr float kStd = 0.5f;
constexpr const char* kInputName = "input";
constexpr const char* kOutputName = "output";

struct Sample {
  fs::path path;
  int label;  // -1 when the ground truth is unknown
};

// Same preprocessing as training: RGB, resize to 32x32, scale to [0,1], normalize ((x - mean) / std), HWC -> CHW.
static bool preprocess(const fs::path& path, std::vector<float>& chw) {
  cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) return false;

  cv::Mat resized, rgb, normalized;
  // INTER_AREA is closest to PIL's antialiased bilinear downscaling used by torchvision Resize.
  cv::resize(bgr, resized, cv::Size(kWidth, kHeight), 0, 0, cv::INTER_AREA);
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  // x_norm = (x / 255 - mean) / std
  rgb.convertTo(normalized, CV_32FC3, 1.0 / (255.0 * kStd), -kMean / kStd);
  if (!normalized.isContinuous()) normalized = normalized.clone();

  const size_t stride = static_cast<size_t>(kHeight) * kWidth;
  const float* hwc = normalized.ptr<float>();
  for (size_t i = 0; i < stride; ++i) {
    for (size_t c = 0; c < kChannels; ++c) {
      chw[c * stride + i] = hwc[i * kChannels + c];
    }
  }
  return true;
}

static std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) {
    const auto first = field.find_first_not_of(" \t\r");
    const auto last = field.find_last_not_of(" \t\r");
    fields.push_back(first == std::string::npos ? "" : field.substr(first, last - first + 1));
  }
  return fields;
}

// Reads <root>/Test.csv. Returns false if the file or the required columns are missing.
static bool load_labeled_samples(const fs::path& root, std::vector<Sample>& samples) {
  std::ifstream file(root / "Test.csv");
  if (!file) return false;

  std::string line;
  if (!std::getline(file, line)) return false;
  const auto header = split_csv_line(line);
  const auto class_it = std::find(header.begin(), header.end(), "ClassId");
  const auto path_it = std::find(header.begin(), header.end(), "Path");
  if (class_it == header.end() || path_it == header.end()) {
    std::cerr << "Test.csv must contain 'ClassId' and 'Path' columns\n";
    return false;
  }
  const size_t class_idx = class_it - header.begin();
  const size_t path_idx = path_it - header.begin();

  while (std::getline(file, line)) {
    const auto fields = split_csv_line(line);
    if (fields.size() <= std::max(class_idx, path_idx) || fields[path_idx].empty()) continue;
    samples.push_back({root / fields[path_idx], std::stoi(fields[class_idx])});
  }
  return true;
}

static void load_unlabeled_samples(const fs::path& dir, std::vector<Sample>& samples) {
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".ppm" || ext == ".bmp") {
      samples.push_back({entry.path(), -1});
    }
  }
  std::sort(samples.begin(), samples.end(),
            [](const Sample& a, const Sample& b) { return a.path < b.path; });
}

int main(int argc, char** argv) {
  const fs::path model_path = argc > 1 ? argv[1] : "./weights/best.onnx";
  const fs::path data_path = argc > 2 ? argv[2] : "./gtrsb_dataset";

  if (!fs::is_regular_file(model_path)) {
    std::cerr << "Model file not found: " << model_path << '\n';
    return 1;
  }
  if (!fs::is_directory(data_path)) {
    std::cerr << "Data folder not found: " << data_path << '\n';
    return 1;
  }

  std::vector<Sample> samples;
  const bool labeled = load_labeled_samples(data_path, samples);
  if (!labeled) load_unlabeled_samples(data_path, samples);
  if (samples.empty()) {
    std::cerr << "No images found in " << data_path << '\n';
    return 1;
  }
  std::cout << "Images: " << samples.size() << (labeled ? " (labels from Test.csv)" : " (no labels)") << '\n';

  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (g_ort == nullptr) {
    std::cerr << "Failed to initialize the ONNX Runtime API\n";
    return 1;
  }

  // Environment and session are created once and reused for every image.
  OrtEnv* env = nullptr;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "gtsrb", &env));

  OrtSessionOptions* session_options = nullptr;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
  ORT_ABORT_ON_ERROR(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL));

  OrtSession* session = nullptr;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path.c_str(), session_options, &session));

  OrtMemoryInfo* memory_info = nullptr;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  // The input tensor wraps this buffer (no copy), so it is created once and refilled per image.
  std::vector<float> input(static_cast<size_t>(kChannels) * kHeight * kWidth);
  const int64_t input_shape[] = {1, kChannels, kHeight, kWidth};
  OrtValue* input_tensor = nullptr;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
      memory_info, input.data(), input.size() * sizeof(float), input_shape, 4,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

  const char* input_names[] = {kInputName};
  const char* output_names[] = {kOutputName};

  size_t processed = 0, failed = 0, correct = 0;
  double total_ms = 0.0;

  for (const auto& sample : samples) {
    if (!preprocess(sample.path, input)) {
      std::cerr << "Error reading image: " << sample.path << '\n';
      ++failed;
      continue;
    }

    OrtValue* output_tensor = nullptr;
    const auto begin = std::chrono::steady_clock::now();
    ORT_ABORT_ON_ERROR(g_ort->Run(session, nullptr, input_names, &input_tensor, 1,
                                  output_names, 1, &output_tensor));
    const auto end = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(end - begin).count();

    float* scores = nullptr;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorMutableData(output_tensor, reinterpret_cast<void**>(&scores)));

    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    size_t num_classes = 0;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorTypeAndShape(output_tensor, &shape_info));
    ORT_ABORT_ON_ERROR(g_ort->GetTensorShapeElementCount(shape_info, &num_classes));
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    const int predicted = static_cast<int>(std::max_element(scores, scores + num_classes) - scores);
    g_ort->ReleaseValue(output_tensor);

    ++processed;
    if (sample.label >= 0) {
      correct += (predicted == sample.label);
      std::printf("%s: predicted %d, label %d\n", sample.path.string().c_str(), predicted, sample.label);
    } else {
      std::printf("%s: predicted %d\n", sample.path.string().c_str(), predicted);
    }
  }

  std::printf("\nProcessed images: %zu, failed: %zu\n", processed, failed);
  if (processed > 0) {
    std::printf("Average inference time: %.3f ms/image\n", total_ms / processed);
    if (labeled) {
      std::printf("Accuracy: %.4f (%zu/%zu)\n", static_cast<double>(correct) / processed, correct, processed);
    }
  }

  g_ort->ReleaseValue(input_tensor);
  g_ort->ReleaseMemoryInfo(memory_info);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);

  return failed == 0 ? 0 : 2;
}
