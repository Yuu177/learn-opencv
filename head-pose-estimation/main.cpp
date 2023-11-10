#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  // Read input image
  cv::Mat im = cv::imread("image/face.jpg");

  // 2D image points. If you change the image, you need to change vector
  std::vector<cv::Point2d> image_points;
  image_points.push_back(cv::Point2d(359, 391));  // Nose tip
  image_points.push_back(cv::Point2d(399, 561));  // Chin
  image_points.push_back(cv::Point2d(337, 297));  // Left eye left corner
  image_points.push_back(cv::Point2d(513, 301));  // Right eye right corner
  image_points.push_back(cv::Point2d(345, 465));  // Left Mouth corner
  image_points.push_back(cv::Point2d(453, 469));  // Right mouth corner

  // 3D model points.
  std::vector<cv::Point3d> model_points;
  model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));       // Nose tip
  model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));  // Chin
  model_points.push_back(
      cv::Point3d(-225.0f, 170.0f, -135.0f));  // Left eye left corner
  model_points.push_back(
      cv::Point3d(225.0f, 170.0f, -135.0f));  // Right eye right corner
  model_points.push_back(
      cv::Point3d(-150.0f, -150.0f, -125.0f));  // Left Mouth corner
  model_points.push_back(
      cv::Point3d(150.0f, -150.0f, -125.0f));  // Right mouth corner

  // Camera internals
  double focal_length = im.cols;  // Approximate focal length.
  cv::Point2d center = cv::Point2d(im.cols / 2, im.rows / 2);
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
                           0, focal_length, center.y, 0, 0, 1);
  cv::Mat dist_coeffs = cv::Mat::zeros(
      4, 1, cv::DataType<double>::type);  // Assuming no lens distortion

  std::cout << "Camera Matrix " << std::endl << camera_matrix << std::endl;
  // Output rotation and translation
  cv::Mat rotation_vector;  // Rotation in axis-angle form
  cv::Mat translation_vector;

  // Solve for pose
  cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
               rotation_vector, translation_vector);

  // Project a 3D point (0, 0, 1000.0) onto the image plane.
  // We use this to draw a line sticking out of the nose
  std::vector<cv::Point3d> nose_end_point3D;
  std::vector<cv::Point2d> nose_end_point2D;

  // Nose tip 的 3d 点坐标为 (0, 0, 0)
  // 保持 x，y 轴的坐标不变，修改 z 轴的坐标为 z'。
  // 线段 zz' 就是鼻子的方向（人脸朝向）
  nose_end_point3D.push_back(cv::Point3d(0, 0, 1000.0));

  // 将 3d 点投影到图像上
  // 我们得到 R 和 t 矩阵后，就可以通过一个 3d 点转换为图片上的 2d 点
  cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector,
                    camera_matrix, dist_coeffs, nose_end_point2D);

  // 画出眼睛，鼻子等特征点
  for (int i = 0; i < image_points.size(); i++) {
    cv::circle(im, image_points[i], 3, cv::Scalar(0, 0, 255), -1);
  }

  // 画人脸朝向方向，即上面所说的线段 zz'
  cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

  std::cout << "Rotation Vector " << std::endl << rotation_vector << std::endl;
  std::cout << "Translation Vector" << std::endl
            << translation_vector << std::endl;

  std::cout << nose_end_point2D << std::endl;

  // Display image.
  cv::imshow("Output", im);
  cv::waitKey(0);
}
