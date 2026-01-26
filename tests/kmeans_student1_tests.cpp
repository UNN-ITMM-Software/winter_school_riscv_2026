#include <gtest/gtest.h>
#include "all_kmeans.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <chrono>

using namespace cv;
using ClustersCount = int;
using PointsCount = int;
using Attempts = int;
using Perf = bool;
using IsKMeansPlusPlus = bool;

namespace {
static constexpr int PERF_ITERATIONS = 10;

struct KMeansUniversalParams {
    ClustersCount clustersCount;
    PointsCount pointsCount;
    Attempts attempts;
    Perf isPerf;  // also test performance?
    IsKMeansPlusPlus isKMeansPlusPlus;
};

const char* bool2chars(const bool value) {
    return value ? "true" : "false";
}

std::ostream& operator<<(std::ostream& out, const KMeansUniversalParams& params) {
    out << "KMeansUniversalParams{";
    out << "Clusters{" << params.clustersCount;
    out << "}, Points{" << params.pointsCount;
    out << "}, Attempts{" << params.attempts;
    out << "}, Perf{" << bool2chars(params.isPerf);
    out << "}}";
    return out;
}

class SyntheticTestKMeansStudent1 : public testing::TestWithParam<KMeansUniversalParams> {
protected:
    void run(const KMeansUniversalParams& params);
};

void SyntheticTestKMeansStudent1::run(const KMeansUniversalParams& params) {
    const auto perf = params.isPerf;
    const auto attempts = params.attempts;
    const auto pointsCount = params.pointsCount;
    const auto clustersCount = std::min(pointsCount, params.clustersCount);

    const int minValue = -1000;
    const int maxValue = 1000;

    Mat points(pointsCount, 1, CV_32FC4);

    // will generate random (synthetic) data for the test
    const int dimsNum = 4;
    RNG rng(2026);

    cv::Mat initialCenters(clustersCount, dimsNum, CV_32F);
    rng.fill(initialCenters, cv::RNG::UNIFORM, cv::Scalar(minValue), cv::Scalar(maxValue));

    // Distributing the points into clusters
    int pointsPerCluster = pointsCount / clustersCount;
    int remainder = pointsCount % clustersCount;

    int pointIdx = 0;
    for (int clusterIdx = 0; clusterIdx < clustersCount; clusterIdx++) {
        int clusterPoints = pointsPerCluster + (clusterIdx < remainder ? 1 : 0);
        
        for (int i = 0; i < clusterPoints; i++) {
            // For each measurement, we add noise
            cv::Vec4f point;
            for (int d = 0; d < dimsNum; d++) {
                float centerVal = initialCenters.at<float>(clusterIdx, d);
                // Adding Gaussian noise with sigma = 5% of the range
                float noise = rng.gaussian((maxValue - minValue) * 0.05);
                point[d] = centerVal + noise;
            }
            
            // For CV_32FC4, we write it as Vec4f
            points.at<cv::Vec4f>(pointIdx, 0) = point;
            pointIdx++;
        }
    }
    cv::randShuffle(points);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1);

    Mat labelsRef, centersRef, labels, centers;

    int flags = cv::KMEANS_RANDOM_CENTERS;
    if (params.isKMeansPlusPlus) {
        flags = cv::KMEANS_PP_CENTERS;
    }

    double compactnessRef  = kmeans_base(points, clustersCount, labelsRef, 
                                         criteria, attempts, flags, centersRef);
    double compactnessStud = student1_kmeans(points, clustersCount, labels, 
                                            criteria, attempts, flags, centers);
    
    std::cout << "compactnessRef: " << compactnessRef << std::endl;
    std::cout << "compactnessStud: " << compactnessStud << std::endl;
    
    // Check that results are reasonable (within 5% of each other)
    EXPECT_NEAR(compactnessStud, compactnessRef, compactnessRef * 0.05);

    // Performance testing
    if (perf) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < PERF_ITERATIONS; ++i) {
            student1_kmeans(points, clustersCount, labels, 
                           criteria, attempts, flags, centers);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Student1 kmeans took " << duration.count() 
                  << " ms for " << PERF_ITERATIONS << " iterations." << std::endl;
    }
}

TEST_P(SyntheticTestKMeansStudent1, Basic) {
    const KMeansUniversalParams params = GetParam();
    run(params);
}

static std::vector<KMeansUniversalParams> synthetic_data_kmeans_student1 = {
    { ClustersCount{ 4 }, PointsCount{ 1000 }, Attempts{ 3 }, 
      Perf{ false }, IsKMeansPlusPlus{ false } },
};

INSTANTIATE_TEST_SUITE_P(Accuracy, SyntheticTestKMeansStudent1, 
                         testing::ValuesIn(synthetic_data_kmeans_student1));
} // namespace