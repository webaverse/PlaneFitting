#include <iostream>
#include "GRANSAC.hpp"
#include "PlaneModel.hpp"
#include <omp.h>
#include <opencv2/opencv.hpp>

bool PlaneFitting(const std::vector<Vector3VP> &points_input, double* center, double* normal)
{
	int Num = points_input.size();
	std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
	CandPoints.resize(Num);
#pragma omp parallel for num_threads(6)
	for (int i = 0; i <Num; ++i)
	{
		Vector3VP p=points_input[i];
		std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point3D>(p[0], p[1],p[2]);
		CandPoints[i]=CandPt;
	}
	
	GRANSAC::RANSAC<PlaneModel, 3> Estimator;
    Estimator.Initialize(0.1, 100); // Threshold, iterations

    int64_t start = cv::getTickCount();
	Estimator.Estimate(CandPoints);
    int64_t end = cv::getTickCount();
    // std::cout << "RANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;
	
    // std::cerr << "best inliers size 1: " << Estimator.GetBestInliers().size() << std::endl;

	auto BestPlane = Estimator.GetBestModel();
	if (BestPlane == nullptr)
	{
		return false;
	}
	for (int i = 0; i < 3; i++)
	{
        center[i] = BestPlane->m_PointCenter[i];
	}
    for (int i = 0; i < 4; i++)
    {
        normal[i] = BestPlane->m_PlaneCoefs[i];
    }

	return true;
}
int main()
{
    // read plane as Vector3VP array from raw float values on stdin (binary, not string)
    std::vector<Vector3VP> points;
    float x, y, z;
    while (std::cin.read(reinterpret_cast<char*>(&x), sizeof(x)) && std::cin.read(reinterpret_cast<char*>(&y), sizeof(y)) && std::cin.read(reinterpret_cast<char*>(&z), sizeof(z)))
    {
        Vector3VP point = { x, y, z };
        points.push_back(point);
    }

    double center[3];
    double normal[4];
    for (int i = 0; i < 32; i++) {
        std::cerr << "iteration " << i << ": " << points.size() << std::endl;
        bool ok = PlaneFitting(points, center, normal);
        if (!ok) {
            break;
        }
        // serialize the center and normal in binary to stdout
        float fcenter[3] = { (float)center[0], (float)center[1], (float)center[2] };
        float fnormal[4] = { (float)normal[0], (float)normal[1], (float)normal[2], (float)normal[3] };
        std::cout.write(reinterpret_cast<char*>(&fcenter[0]), sizeof(fcenter[0]));
        std::cout.write(reinterpret_cast<char*>(&fcenter[1]), sizeof(fcenter[1]));
        std::cout.write(reinterpret_cast<char*>(&fcenter[2]), sizeof(fcenter[2]));
        std::cout.write(reinterpret_cast<char*>(&fnormal[0]), sizeof(fnormal[0]));
        std::cout.write(reinterpret_cast<char*>(&fnormal[1]), sizeof(fnormal[1]));
        std::cout.write(reinterpret_cast<char*>(&fnormal[2]), sizeof(fnormal[2]));
        std::cout.write(reinterpret_cast<char*>(&fnormal[3]), sizeof(fnormal[3]));
    }
    // serialize the points in binary to stdout
    // for (int i = 0; i < points.size(); i++) {
    //     std::cout.write(reinterpret_cast<char*>(&points[i][0]), sizeof(points[i][0]));
    //     std::cout.write(reinterpret_cast<char*>(&points[i][1]), sizeof(points[i][1]));
    //     std::cout.write(reinterpret_cast<char*>(&points[i][2]), sizeof(points[i][2]));
    // }
    
    // std::cout << "num points: " << points.size() << std::endl;
    // std::cout << "points 1: " << points[0][0] << " " << points[0][1] << " " << points[0][2] << std::endl;
    // std::cout << "points 2: " << points[1][0] << " " << points[1][1] << " " << points[1][2] << std::endl;
    // std::cout << "points 3: " << points[2][0] << " " << points[2][1] << " " << points[2][2] << std::endl;
    // std::cout << "center: " << center[0] << ", " << center[1] << ", " << center[2] << std::endl;
    // std::cout << "normal: " << normal[0] << ", " << normal[1] << ", " << normal[2] << ", " << normal[3] << std::endl;
    return 0;
}
// to generate binary float points (1.0 .. 9.0) from the command line as test data:
// str=$(for i in {1..32}; do printf '\\x%02x' $i; done); echo -n $str | ./PlaneFittingSample