#include <string>

#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

int main() {
  std::string model_config = "model/deploy.prototxt";
  std::string model_binary = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel";

  // Load the model
  cv::dnn::Net caffe_net = cv::dnn::readNetFromCaffe(model_config, model_binary);
  if (caffe_net.empty()) {
    std::cerr << "Load caffe model failed" << std::endl;
    exit(-1);
  }

  // Read an image
  cv::Mat frame = cv::imread("image/face.jpg");
  if (frame.empty()) {
    std::cerr << "Load image failed" << std::endl;
    exit(-1);
  }

  // 模型输入
  cv::Mat input_blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300));
  caffe_net.setInput(input_blob, "data");

  // 模型推理
  cv::Mat detection = caffe_net.forward("detection_out");

  int dims = detection.dims;
  // 输出维度数
  std::cout << "Number of dimensions: " << dims << std::endl;
  // 输出每个维度的大小
  for (int i = 0; i < dims; ++i) {
      std::cout << detection.size[i] << " ";
  }
  std::cout << std::endl;
  // blob(detection) 输出为 [1, 1, x, 7]，这是一个四维数组
  // 其中 x 是最后保留的框的个数（这里模型的输出有 200，但不一定都有人脸），最后一维存放每次检测结果的数据。不同的人脸模型的输出可能是不一样的
  // 最后一维每一列的含义：
  // 列 0：物体存在的置信度
  // 列 1：包围盒的置信度
  // 列 2：检测到的人脸置信度
  // 列 3：人脸边界框的左上角坐标 x 在整张图片中的比例
  // 列 4：人脸边界框的左上角坐标 y 在整张图片中的比例
  // 列 5：人脸边界框的右下角坐标 x 在整张图片中的比例
  // 列 6：人脸边界框的右下角坐标 y 在整张图片中的比例

  // 所以 detection.size[2] 是检测到的对象的数量，detection.size[3] 是每次检测的结果的数据（人脸边界框数据和置信度）
  // 类似于：
  // std::array<float, 7> FaceInfo;
  // vector<FaceInfo> face;
  // detection.size[2] = face.size();
  // detection.size[3] = face;
  // Get the result
  cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>()); // 转换成了二维数组，方便运算

  // Draw the rectangles
  for (int i = 0; i < detection_mat.rows; i++) {
    float confidence = detection_mat.at<float>(i, 2);
    // 置信度（confidence score）是指模型对于特定的输入数据的预测结果的确信度，
    // 通常以 0~1 的值表示。一般来说，如果模型的置信度越高，它的预测结果就越可信。
    // 这里一共有 200 个检测到的人脸边界框，我们要选择置信度高的人脸边界框
    if (confidence > 0.7) {
      auto x_top_left = detection_mat.at<float>(i, 3) * static_cast<float>(frame.cols);
      auto y_top_left = detection_mat.at<float>(i, 4) * static_cast<float>(frame.rows);
      auto x_bottom_right = detection_mat.at<float>(i, 5) * static_cast<float>(frame.cols);
      auto y_bottom_right = detection_mat.at<float>(i, 6) * static_cast<float>(frame.rows);

      cv::Rect2f face_rec(cv::Point2f(x_top_left, y_top_left), cv::Point2f(x_bottom_right, y_bottom_right));

      // 把边框画到图片上
      cv::rectangle(frame, face_rec, cv::Scalar(0, 255, 0));
    }
  }
  imshow("detection", frame);
  cv::waitKey(0);
  return 0;
}
