#include<opencv2/opencv.hpp>

using namespace std;

int main() {
//读取图片
    cv::Mat rawImage = cv::imread("23.jpg");


    cv::Mat image;
    cv::GaussianBlur(rawImage, image, cv::Size(3, 3), 1);// 高斯模糊，将图片平滑化，去掉干扰的噪声
    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);//图片灰度化
    cv::Sobel(image, image, -1, 1, 0);//Sobel算子（X方向）
    cv::threshold(image, image, 0, 255, cv::THRESH_OTSU); //二值化
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 5));
    cv::morphologyEx(image, image, cv::MORPH_CLOSE, kernel); //闭操作：闭操作可以将目标区域连成一个整体，便于后续轮廓的提取。

    // 膨胀腐蚀(形态学处理)
    cv::Mat kernelX = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 1));
    cv::Mat kernelY = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 19));
    cv::dilate(image, image, kernelX);
    cv::erode(image, image, kernelX);
    cv::erode(image, image, kernelY);
    cv::dilate(image, image, kernelY);

    cv::medianBlur(image, image, 15); //平滑处理，中值滤波
#

    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> vec_4f;
    cv::findContours(image, contours, vec_4f, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);//查找轮廓


//    for (vector<vector<cv::Point>>::iterator it = contours.begin(); it != contours.end(); ++it) { //std里vector标准遍历方法
    for (int i = 0; i < contours.size(); i++) { //常规遍历方法
        cv::Rect rect = cv::boundingRect(contours[i]);
        double x = rect.x, y = rect.y, w = rect.width, h = rect.height;
        if (w > h * 2) {
            cv::Mat carnum = rawImage(cv::Range(y, y + h), cv::Range(x, x + w)); //图像剪裁。这块没教，但大家可以搜索百度直接找到答案
            cv::imshow("car_num", carnum);
        }
    }

    cv::drawContours(rawImage, contours, -1, cv::Scalar(0, 0, 255), 2); //绘制轮廓

    cv::imshow("img_contour", rawImage);
    cv::waitKey(0);

}
