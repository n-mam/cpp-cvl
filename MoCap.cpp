#define CAFFE
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <Common.hpp>
#include <Source.hpp>
#include <Tracker.hpp>
#include <Person.hpp>

std::set<Person *> people;
Person* ProcessContour(const std::vector<Point>& contour);

int main(int argc, char *argv[])
{
  _putenv_s("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp");

  Source s("f:/my.mp4");

  if(!s.isOpened())
  {
    std::cout << "Error opening video stream\n";
    return -1;
  }

  cv::namedWindow("Motion Capture");

  cv::Mat ff, frame;

  for (;;)
  {
    if(!s.Read(frame))
	  {
      if(s.HasEnded())
      {
        s.Rewind();
        people.clear();
        continue;
      }
      else
      {
        break;
      }
    }

    auto scale = (float) 600 / frame.cols;

    cv::resize(frame, frame, cv::Size(0, 0), scale, scale);

    if (frame.channels() == 4)
    {
      cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }

    cv::Mat gray, blur, delta, thresh, dilate;

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(gray, blur, cv::Size(21,21), 0);

    if (ff.empty())
    {
      ff = blur;
    }

	  cv::absdiff(ff, blur, delta);

	  cv::threshold(delta, thresh, 25, 255, cv::THRESH_BINARY);

    cv::dilate(thresh, dilate, cv::Mat(), cv::Point(-1, -1), 2);

    std::vector<
      std::vector<cv::Point>
    > contours, filtered;

    cv::findContours(dilate, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) 
    {
      auto bb = cv::boundingRect(contours[i]);

      if (cv::contourArea(contours[i]) < 10000) continue;
      if (bb.width > bb.height) continue;

      bool skip = false;

      for (size_t j = 0; (j < contours.size() && (i != j)); j++)
      {
        if (IsRectInsideRect(bb, cv::boundingRect(contours[j])))
        {
          skip = true;
          break;
        }
      }

      if (!skip)
      {
        ProcessContour(contours[i]);
        cv::rectangle(frame, bb, cv::Scalar(255, 0, 0 ), 1, 1);            
      }
    }

    cv::putText(frame, std::to_string(people.size()), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

	  cv::imshow("Motion Capture", frame);

    cv::imshow("Mask", delta);

    if (!s.HandleUserInput()) break;
  }   
}

    Person* ProcessContour(const std::vector<Point>& contour) 
    {
        time_p now = high_resolution_clock::now();
        Person* person = NULL;
    
        for (std::set<Person*>::iterator it = people.begin(); it != people.end(); ++it) {
            if ((*it)->hasSimilarContour(contour)) {
                person = *it; 
                person->lastSeen = now;
                person->update(contour);
                //countIfPersonIsCrossingTheRefLine(person);
                break;
            }
        }

        if (person == NULL) {
            person = new Person(contour);
            person->firstSeen = now;
            person->lastSeen = now;
            people.insert(person);
        }

        //lastFrameWherePersonWasSeen[person] = frameNumber;

        return person;
    }