#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/opencv.h"
#include "opencv2/opencv.hpp"

int main() {
  // 定义特征点预测器
  dlib::shape_predictor sp;

  // 加载预测模型
  // ">>" 运算符是 dlib 库中的序列化操作符，
  // 它允许你在读取和写入二进制文件时对对象进行序列化和反序列化。
  // 读取 "landmarks.dat" 文件并反序列化到 dlib::shape_predictor 对象 "sp"
  dlib::deserialize("model/shape_predictor_68_face_landmarks.dat") >> sp;

  // 加载图像
  cv::Mat img = cv::imread("image/face.jpg");

  // 将图像转换为 dlib 图像格式
  dlib::cv_image<dlib::bgr_pixel> cimg(img);

  // 定义人脸检测器
  dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

  // 使用人脸检测器检测人脸
  std::vector<dlib::rectangle> faces = detector(cimg);

  // 对每个检测到的人脸检测68个特征点
  for (int64 i = 0; i < faces.size(); ++i) {
    // 定义特征点的位置
    dlib::full_object_detection shape = sp(cimg, faces[i]);

    // 在原图上绘制特征点
    for (int64 j = 0; j < shape.num_parts(); ++j) {
      cv::circle(img, cv::Point(shape.part(j).x(), shape.part(j).y()), 2,
                 cv::Scalar(0, 255, 0), -1);
    }
  }

  // 显示结果
  cv::imshow("68 Point Location", img);
  cv::waitKey(0);

  return 0;
}
