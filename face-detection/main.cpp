#include <string>

#include "opencv2/core/mat.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"

int main() {
  // TODO 配置文件路径要抽取出来
  std::string modelConfig = "model/deploy.prototxt";
  std::string modelBinary =
      "model/res10_300x300_ssd_iter_140000_fp16.caffemodel";

  // Load the model
  // readNet 通过传入的模型后缀来判断使用哪个框架
  // cv::dnn::Net caffe_net = cv::dnn::readNet(modelBinary, modelConfig);
  cv::dnn::Net caffe_net = cv::dnn::readNetFromCaffe(modelConfig, modelBinary);
  if (caffe_net.empty()) {
    std::cerr << "load caffe model failed" << std::endl;
    exit(-1);
  }

  // Read an image
  cv::Mat frame = cv::imread("image/face.jpg");
  if (frame.empty()) {
    std::cerr << "load image failed" << std::endl;
    exit(-1);
  }

  // Pre-process the image
  // Scalar(104, 117, 123) 在这个代码片段中代表的是图像颜色的均值（mean
  // value）。 在图像处理和计算机视觉领域中，通常需要将图像的 RGB
  // 通道的值减去一个固定的均值，以降低图像的复杂度和防止因图像光照变化导致的问题。均值可以认为是图像的背景色。
  // 在此代码中，Scalar(104, 117, 123) 的三个参数分别代表 BGR
  // 通道的均值。当图像的 BGR
  // 通道的值减去这个均值时，会得到一张以该均值为基准的图像。
  // 这样做有助于提高模型的泛化能力，使其能够适用于不同光照条件下的图像
  // 104, 117, 123
  // 这三个值的选择取决于具体的应用场景。它们是预先设定的均值，用于减去图像中
  // RGB 通道的值。
  // 一般来说，不同的应用场景，例如人脸检测、物体识别等，所使用的数据集和模型都不同，因此均值也不同。
  // 一个常见的方法是对所使用的图像数据集计算均值，并使用该均值。这样的话，模型将在与其训练数据集相似的均值环境中进行预测。
  // 有时也可以直接使用一个固定的均值，例如 104, 117,
  // 123。这个均值的选择不一定是最优的，但它已经被广泛应用在各种图像处理和计算机视觉任务中，具有一定的参考价值。
  cv::Mat input_blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300),
                                              cv::Scalar(104, 117, 123));

  caffe_net.setInput(input_blob, "data");  // 输入层

  // Forward pass
  // detection out layer 是 ssd
  // 网络最后一层，用于整合预选框、预选框偏移以及得分三项结果，最终输出满足条件的目标检测框、目标的
  // label 和得分等。
  cv::Mat detection = caffe_net.forward("detection_out");  // compute output
  // std::cout << detection.size[0] << " " << detection.size[1] << " "
  //           << detection.size[2] << " " << detection.size[3] << std::endl;

  // 输出 blob(detection) 为 [1, 1, x, 7]，这是个四维数组
  // 其中 x 是最后保留的框的个数，最后一维存放的数据为
  // [image_id, label, confidence, xmin, ymin, xmax, ymax]
  // 这个 [xmin, ymin] 就是检测到物体的左上角坐标在整张图片中的比例
  // [xmax, ymax] 就是检测到物体的右下角坐标在整张图片中的比例
  // 也就是说 detection.size[2] 是检测到的对象的数量，对应 x
  // detection.size[3] 是每次检测的结果数量（边界框数据和置信度），对应 7
  // 列 0：物体存在的置信度（// TODO 不确定）
  // 列 1：包围盒的置信度（// TODO 不确定）
  // 列 2：检测到的人脸置信度
  // 列 3：左上边界框 x 坐标比例。表示检测到的物体的边界框左上角的
  // x（横）坐标在整张图片中的比例，即 x/300 的值（300 为输入给 caffe
  // 的图片宽的大小）
  // 列 4：左上边界框 y 坐标比例
  // 列 5：右下边界框 x 坐标比例
  // 列 6：右下边界框 y 坐标比例

  // Get the result
  // 构造二维 Mat，detection.size[2] * detection.size[3]
  // CV_32F 是 float 像素是在 0~1.0 之间的任意值
  // 创建行数为 rows，列数为 col，类型为 type
  // 的图像，此构造函数不创建图像数据所需内存，而是直接使用 data
  // 所指内存，图像的行步长由 step 指定。
  // 这个转换有点抽象，就是把多维数组转换成了二维数组
  // TODO 该转换过程画图
  cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F,
                       detection.ptr<float>());

  // Draw the rectangles
  for (int i = 0; i < detection_mat.rows; i++) {
    float confidence = detection_mat.at<float>(i, 2);
    // 置信度（confidence score）是指模型对于特定的输入数据的预测结果的确信度，
    // 通常以0~1的值表示。
    // 一般来说，如果模型的置信度越高，它的预测结果就越可信。
    // 因此，如果检测到的置信度大于0.5，这意味着模型对于该检测结果的置信度较高，即认为该检测结果是有效的。
    // 这个阈值是根据模型训练时设置的阈值进行调整的，如果需要调整置信度阈值，可以通过修改代码进行实现。
    if (confidence > 0.5) {
      // 由于最终的人脸检测结果需要在图片上绘制，因此需要将这个比例值转化为实际的像素坐标，这就是为什么要乘上图片宽度（即
      // frame.cols）的原因。
      int x_left = static_cast<int>(detection_mat.at<float>(i, 3) * frame.cols);
      int y_top = static_cast<int>(detection_mat.at<float>(i, 4) * frame.rows);
      int x_right = static_cast<int>(detection_mat.at<float>(i, 5) * frame.cols);
      int y_bottom =
          static_cast<int>(detection_mat.at<float>(i, 6) * frame.rows);

      cv::Rect face_rec(x_left, y_top, x_right - x_left, y_bottom - y_top);

      // TODO 日志输出优化?
      // std::cout << x_left << ", " << y_top << std::endl;
      // std::cout << x_right << ", " << y_bottom << std::endl;
      // // 返回 rect 的左上顶点的坐标
      // std::cout << face_rec.tl() << std::endl;
      // // 返回 rect 的右下顶点的坐标
      // std::cout << face_rec.br() << std::endl;

      // 把边框画到图片上
      cv::rectangle(frame, face_rec, cv::Scalar(0, 255, 0));
    }
  }
  imshow("detections", frame);
  // cv::imwrite("output.jpg", frame);
  cv::waitKey(0);
  return 0;
}
