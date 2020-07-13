#ifndef TRACKER_HPP
#define TRACKER_HPP 

  //Haar
  // if(!face_cascade.load("C:\\code\\my\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml"))
  // {
  //   std::cout << "Error loading face cascade\n";
  //   return -1;
  // };
  // if(!eyes_cascade.load("C:\\code\\my\\opencv\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"))
  // {
  //   std::cout << "Error loading eye cascade\n";
  //   return -1;
  // };

// void DetectFacesHaar(cv::Mat& frame)
// {
//     cv::Mat frame_gray;
//     cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
//     equalizeHist(frame_gray, frame_gray);

//     std::vector<cv::Rect> faces;

//     face_cascade.detectMultiScale(frame_gray, faces);

//     for (size_t i = 0; i < faces.size(); i++)
//     {
//       // cv::Mat faceROI = frame_gray(faces[i]);
//       // std::vector<cv::Rect> eyes;
//       // eyes_cascade.detectMultiScale(faceROI, eyes);
//       // if (eyes.size())
//       // {
//       // }
//       cv::rectangle(
//         frame,
//         cv::Point(faces[i].x, faces[i].y),
//         cv::Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
//         cv::Scalar(0, 255, 0), 2, 4);      
//     }
// }

#endif //TRACKER_HPP