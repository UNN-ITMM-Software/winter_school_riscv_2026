#include "all_kmeans.h"
#include <opencv2/core.hpp>

double kmeans_base(
    cv::InputArray data,
    int K,
    cv::InputOutputArray bestLabels,
    cv::TermCriteria criteria,
    int attempts,
    int flags,
    cv::OutputArray centers
) {
    return cv::kmeans(
        data,
        K,
        bestLabels,
        criteria,
        attempts,
        flags,
        centers
    );
}