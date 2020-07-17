#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std::chrono;
using namespace cv;

typedef high_resolution_clock::time_point time_p;

class Person 
{
  public:

    int id = 0;
    Rect boundingBox;
    std::vector<Point> contour;
    Point contourXY;
    std::vector<Point> trace;

    int x;
    int y;
    
    time_p firstSeen;
    time_p lastSeen;

    Person() {};

    Person(std::vector<Point> contour) 
    {
        this->update(contour);
    }

    // member methods

    bool hasSimilarContour(const std::vector<Point> contour) const 
    {
        Point contourXY = findContourXY(contour);

        bool contourIsClose = abs(x - contourXY.x) <= boundingBox.width && 
                              abs(y - contourXY.y) <= boundingBox.height;

        std::cout << "hasSimilarContour : " << contourIsClose << "\n";

        return contourIsClose;
    }

    void update(const std::vector<Point> contour) 
    {
        this->boundingBox = boundingRect(contour);
        this->contour = contour;
        this->contourXY = findContourXY(contour);
        this->trace.push_back(contourXY);
        this->x = contourXY.x;
        this->y = contourXY.y;
    }

  private:
    
    Point findContourXY(const std::vector<Point> contour) const 
    {
        Moments contourMoments = moments(contour);
        
        return Point(
            (int)(contourMoments.m10 / contourMoments.m00),
            (int)(contourMoments.m01 / contourMoments.m00)
        );
    }
};