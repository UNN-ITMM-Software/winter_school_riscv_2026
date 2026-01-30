#include "all_kmeans.h"

#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <chrono>
#include <string>

#include <gtest/gtest.h>

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

class SyntheticTestKMeansStudent2_u8c4 : public testing::TestWithParam<KMeansUniversalParams> {
protected:
    void run(const KMeansUniversalParams& params);
};

template <typename Type>
static inline bool checkDiff(const Type* actual, const Type* ref, const int size, float tolerance, const std::string string) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(actual[i] - ref[i]);
        if (diff > tolerance) {
            std::cout << "[   ERROR  ] reference = " << ref[i] << ", actual = " << actual[i] << ", diff = " << diff << ", idx = " << i << std::endl;
            return false;
        }
    }
    std::cout << "[   INFO   ] All values of " << string << " are within tolerance." << std::endl;

    return true;
}

static inline bool checkDiff_fu(const float* actual, const unsigned char* ref, const int size, float tolerance, const std::string string) {
    for (int i = 0; i < size; i++) {
        float diff = std::abs(actual[i] - static_cast<float>(ref[i]));
        if (diff > tolerance) {
            std::cout << "[   ERROR  ] reference = " << ref[i] << ", actual = " << actual[i] << ", diff = " << diff << ", idx = " << i << std::endl;
            return false;
        }
    }
    std::cout << "[   INFO   ] All values of " << string << " are within tolerance." << std::endl;

    return true;
}

void SyntheticTestKMeansStudent2_u8c4::run(const KMeansUniversalParams& params) {
    std::cout << "[   INFO   ] " << params << std::endl;
    const auto perf = params.isPerf;
    const auto attempts = params.attempts;
    const auto pointsCount = params.pointsCount;
    const auto clustersCount = std::min(pointsCount, params.clustersCount);

    const int minValue = 0;
    const int maxValue = 128;

    Mat points(pointsCount, 1, CV_32FC4);
    Mat pointsU8C4(pointsCount, 1, CV_8UC4);
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
            cv::Vec4b point8U;
            for (int d = 0; d < dimsNum; d++) {
                const float centerVal = initialCenters.at<float>(clusterIdx, d);
                // Adding Gaussian noise with sigma = 5% of the range
                const float noise = rng.gaussian((maxValue - minValue) * 0.05);
                const float roundPoint = std::max(std::round(centerVal + noise), 128.f);
                point[d] = roundPoint;
                point8U[d] = static_cast<unsigned char>(roundPoint);
            }
            
            // For CV_32FC4, we write it as Vec4f
            points.at<cv::Vec4f>(pointIdx, 0) = point;
            pointsU8C4.at<cv::Vec4b>(pointIdx, 0) = point8U;
            pointIdx++;
        }
    }
    setRNGSeed(2026);
    cv::randShuffle(points);
    setRNGSeed(2026);
    cv::randShuffle(pointsU8C4);


    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1);

    Mat labelsRef, centersRef, labels, centers;

    int flags = cv::KMEANS_RANDOM_CENTERS;
    if (params.isKMeansPlusPlus) {
        flags = cv::KMEANS_PP_CENTERS;
    }
    setRNGSeed(2026);
    double compactnessRef  = kmeans_base(points, clustersCount, labelsRef, 
                                         criteria, attempts, flags, centersRef);
    setRNGSeed(2026);
    double compactnessStud = kmeans_uchar(pointsU8C4, clustersCount, labels, 
                                          criteria, attempts, flags, centers);
    
    std::cout << "compactnessRef: " << compactnessRef << std::endl;
    std::cout << "compactnessStud: " << compactnessStud << std::endl;
    ASSERT_TRUE(checkDiff(centers.ptr<float>(0), centersRef.ptr<float>(0), clustersCount * dimsNum, 0.f, "centers"));
    ASSERT_TRUE(checkDiff(labels.ptr<int32_t>(0), labelsRef.ptr<int32_t>(0), pointsCount, 0.f, "lables"));

    // Check that results are reasonable (within 5% of each other)
    // EXPECT_NEAR(compactnessStud, compactnessRef, compactnessRef * 0.05);

    // Performance testing
    if (perf) {

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < PERF_ITERATIONS; ++i) {
            kmeans_uchar(pointsU8C4, clustersCount, labels, 
                           criteria, attempts, flags, centers);
        }

        auto end = std::chrono::high_resolution_clock::now();
        const auto duration_st = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Student1 kmeans took " << duration_st.count() / PERF_ITERATIONS << " ms " << std::endl;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < PERF_ITERATIONS; ++i) {
            kmeans_base(points, clustersCount, labels, 
                        criteria, attempts, flags, centers);
        }

        end = std::chrono::high_resolution_clock::now();
        const auto duration_cv = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "OpenCV kmeans took " << duration_cv.count() / PERF_ITERATIONS << " ms " << std::endl;

        const auto st1_ms =  duration_st.count() / PERF_ITERATIONS;
        const auto cv_ms = duration_cv.count() / PERF_ITERATIONS;
        const double speedup = static_cast<double>(cv_ms) / static_cast<double>(st1_ms);

        std::cout << "Speedup (CV vs Student2) = " << speedup << std::endl;
    }
}

TEST_P(SyntheticTestKMeansStudent2_u8c4, Basic) {
    const KMeansUniversalParams params = GetParam();
    run(params);
}

static std::vector<KMeansUniversalParams> synthetic_data_kmeans_student2_u8c4 = {
    { ClustersCount{ 5 }, PointsCount{ 1000 }, Attempts{ 3 }, 
      Perf{ false }, IsKMeansPlusPlus{ false } },
};

static std::vector<KMeansUniversalParams> synthetic_data_kmeans_student2_u8c4_perf = {
    { ClustersCount{ 5 }, PointsCount{ 20000 }, Attempts{ 5 }, 
      Perf{ true }, IsKMeansPlusPlus{ false } },
};

INSTANTIATE_TEST_SUITE_P(Accuracy, SyntheticTestKMeansStudent2_u8c4, 
                         testing::ValuesIn(synthetic_data_kmeans_student2_u8c4));
INSTANTIATE_TEST_SUITE_P(Performance, SyntheticTestKMeansStudent2_u8c4, 
                         testing::ValuesIn(synthetic_data_kmeans_student2_u8c4_perf));
} // namespace