#include "ceres/ceres.h"
#include "glog/logging.h"

#include <random>
#include <ctime>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

// Объект SystemCostFunctor содержит единственную функцию operator() для расчета невязки (residual) для всех уравнений системы.
struct TraceSystem {
    template <typename T> bool operator()(const T* const vars_trace, T* residual) const {

        // (000)
        residual[0] = 6.0 * (vars_trace[0] * vars_trace[2] * vars_trace[4] + vars_trace[1] * vars_trace[3] * vars_trace[5]) +
                      6.0 * (vars_trace[6] * vars_trace[8] * vars_trace[10] + vars_trace[7] * vars_trace[9] * vars_trace[11]) + 3.0;

        // (110)
        residual[1] = 2.0 * (vars_trace[0] + vars_trace[2] + vars_trace[4] + vars_trace[1] + vars_trace[3] + vars_trace[5]) -
                      (vars_trace[7] + vars_trace[9] + vars_trace[11] + vars_trace[6] + vars_trace[8] + vars_trace[10]);

        // (100)
        residual[2] = 2.0 * (vars_trace[2] * vars_trace[4] + vars_trace[0] * vars_trace[4] + vars_trace[0] * vars_trace[2] +
                             vars_trace[3] * vars_trace[5] + vars_trace[1] * vars_trace[5] + vars_trace[1] * vars_trace[3]) -
                      (vars_trace[8] * vars_trace[10] + vars_trace[6] * vars_trace[10] + vars_trace[6] * vars_trace[8] +
                       vars_trace[9] * vars_trace[11] + vars_trace[7] * vars_trace[11] + vars_trace[7] * vars_trace[9]);

        // (002)
        residual[3] = 2.0 * (vars_trace[0] * vars_trace[2] * vars_trace[5] + vars_trace[0] * vars_trace[3] * vars_trace[4] + vars_trace[1] * vars_trace[2] * vars_trace[4] +
                             vars_trace[1] * vars_trace[3] * vars_trace[4] + vars_trace[1] * vars_trace[2] * vars_trace[5] + vars_trace[0] * vars_trace[3] * vars_trace[5]) -
                      (vars_trace[6] * vars_trace[8] * vars_trace[11] + vars_trace[6] * vars_trace[9] * vars_trace[10] + vars_trace[7] * vars_trace[8] * vars_trace[10] +
                       vars_trace[7] * vars_trace[9] * vars_trace[10] + vars_trace[7] * vars_trace[8] * vars_trace[11] + vars_trace[6] * vars_trace[9] * vars_trace[11]);

        // (102)
        residual[4] = 2.0 * (vars_trace[2] * vars_trace[5] + vars_trace[0] * vars_trace[3] + vars_trace[1] * vars_trace[4] +
                             vars_trace[3] * vars_trace[4] + vars_trace[1] * vars_trace[2] + vars_trace[0] * vars_trace[5]) +
                      2.0 * (vars_trace[9] * vars_trace[10] + vars_trace[7] * vars_trace[8] + vars_trace[6] * vars_trace[11] +
                             vars_trace[8] * vars_trace[11] + vars_trace[6] * vars_trace[9] + vars_trace[7] * vars_trace[10]) + 3.0;

        return true;
    }

};

struct ExpandedSystem {
    template <typename T> bool operator()(const T* const vars_trace, const T* const vars_remaining, T* residual) const {

        for (int i = 0; i <= 33; ++i)
            residual[i] = T{0};

        // 1 неследовой элемент
        // (115)
        residual[0] = 2.0 * (vars_remaining[4] + vars_remaining[2] + vars_remaining[0] + vars_remaining[5] + vars_remaining[3] + vars_remaining[1]) -
                      (vars_remaining[11] + vars_remaining[9] + vars_remaining[7] + vars_remaining[10] + vars_remaining[8] + vars_remaining[6]);

        // (105)
        residual[1] = 2.0 * (vars_trace[2] * vars_remaining[4] + vars_trace[0] * vars_remaining[2] + vars_trace[4] * vars_remaining[0] +
                             vars_trace[3] * vars_remaining[5] + vars_trace[1] * vars_remaining[3] + vars_trace[5] * vars_remaining[1]) +
                      2.0 * (vars_trace[9] * vars_remaining[11] + vars_trace[7] * vars_remaining[9] + vars_trace[11] * vars_remaining[7] +
                             vars_trace[8] * vars_remaining[10] + vars_trace[6] * vars_remaining[8] + vars_trace[10] * vars_remaining[6]);

        // (106)
        residual[2] = 2.0 * (vars_trace[2] * vars_remaining[5] + vars_trace[0] * vars_remaining[3] + vars_trace[4] * vars_remaining[1] +
                             vars_trace[3] * vars_remaining[4] + vars_trace[1] * vars_remaining[2] + vars_trace[5] * vars_remaining[0]) -
                      (vars_trace[9] * vars_remaining[10] + vars_trace[7] * vars_remaining[8] + vars_trace[11] * vars_remaining[6] +
                       vars_trace[8] * vars_remaining[11] + vars_trace[6] * vars_remaining[9] + vars_trace[10] * vars_remaining[7]);
        // (005)
        residual[3] = 2.0 * (vars_trace[0] * vars_trace[2] * vars_remaining[4] + vars_trace[0] * vars_trace[4] * vars_remaining[2] + vars_trace[2] * vars_trace[4] * vars_remaining[0] +
                             vars_trace[1] * vars_trace[3] * vars_remaining[5] + vars_trace[1] * vars_trace[5] * vars_remaining[3] + vars_trace[3] * vars_trace[5] * vars_remaining[1]) -
                      (vars_trace[7] * vars_trace[9] * vars_remaining[11] + vars_trace[7] * vars_trace[11] * vars_remaining[9] + vars_trace[9] * vars_trace[11] * vars_remaining[7] +
                       vars_trace[6] * vars_trace[8] * vars_remaining[10] + vars_trace[6] * vars_trace[10] * vars_remaining[8] + vars_trace[8] * vars_trace[10] * vars_remaining[6]);
        // (006)
        residual[4] = 2.0 * (vars_trace[0] * vars_trace[2] * vars_remaining[5] + vars_trace[0] * vars_trace[4] * vars_remaining[3] + vars_trace[2] * vars_trace[4] * vars_remaining[1] +
                             vars_trace[1] * vars_trace[3] * vars_remaining[4] + vars_trace[1] * vars_trace[5] * vars_remaining[2] + vars_trace[3] * vars_trace[5] * vars_remaining[0]) +
                      2.0 * (vars_trace[7] * vars_trace[9] * vars_remaining[10] + vars_trace[7] * vars_trace[11] * vars_remaining[8] + vars_trace[9] * vars_trace[11] * vars_remaining[6] +
                             vars_trace[6] * vars_trace[8] * vars_remaining[11] + vars_trace[6] * vars_trace[10] * vars_remaining[9] + vars_trace[8] * vars_trace[10] * vars_remaining[7]);
        // (015)
        residual[5] = 2.0 * (vars_trace[0] * vars_remaining[4] + vars_trace[4] * vars_remaining[2] + vars_trace[2] * vars_remaining[0] +
                             vars_trace[1] * vars_remaining[5] + vars_trace[5] * vars_remaining[3] + vars_trace[3] * vars_remaining[1]) +
                      2.0 * (vars_trace[7] * vars_remaining[11] + vars_trace[11] * vars_remaining[9] + vars_trace[9] * vars_remaining[7] +
                             vars_trace[6] * vars_remaining[10] + vars_trace[10] * vars_remaining[8] + vars_trace[8] * vars_remaining[6]);
        // (016)
        residual[6] = 2.0 * (vars_trace[0] * vars_remaining[5] + vars_trace[4] * vars_remaining[3] + vars_trace[2] * vars_remaining[1] +
                             vars_trace[1] * vars_remaining[4] + vars_trace[5] * vars_remaining[2] + vars_trace[3] * vars_remaining[0]) -
                      (vars_trace[7] * vars_remaining[10] + vars_trace[11] * vars_remaining[8] + vars_trace[9] * vars_remaining[6] +
                       vars_trace[6] * vars_remaining[11] + vars_trace[10] * vars_remaining[9] + vars_trace[8] * vars_remaining[7]);
        // (025)
        residual[7] = 2.0 * (vars_trace[0] * vars_trace[3] * vars_remaining[4] + vars_trace[1] * vars_trace[4] * vars_remaining[2] + vars_trace[2] * vars_trace[5] * vars_remaining[0] +
                             vars_trace[1] * vars_trace[2] * vars_remaining[5] + vars_trace[0] * vars_trace[5] * vars_remaining[3] + vars_trace[3] * vars_trace[4] * vars_remaining[1]) -
                      (vars_trace[7] * vars_trace[8] * vars_remaining[11] + vars_trace[6] * vars_trace[11] * vars_remaining[9] + vars_trace[9] * vars_trace[10] * vars_remaining[7] +
                       vars_trace[6] * vars_trace[9] * vars_remaining[10] + vars_trace[7] * vars_trace[10] * vars_remaining[8] + vars_trace[8] * vars_trace[11] * vars_remaining[6]);
        // (026)
        residual[8] = 2.0 * (vars_trace[0] * vars_trace[3] * vars_remaining[5] + vars_trace[1] * vars_trace[4] * vars_remaining[3] + vars_trace[2] * vars_trace[5] * vars_remaining[1] +
                             vars_trace[1] * vars_trace[2] * vars_remaining[4] + vars_trace[0] * vars_trace[5] * vars_remaining[2] + vars_trace[3] * vars_trace[4] * vars_remaining[0]) -
                      (vars_trace[7] * vars_trace[8] * vars_remaining[10] + vars_trace[6] * vars_trace[11] * vars_remaining[8] + vars_trace[9] * vars_trace[10] * vars_remaining[6] +
                       vars_trace[6] * vars_trace[9] * vars_remaining[11] + vars_trace[7] * vars_trace[10] * vars_remaining[9] + vars_trace[8] * vars_trace[11] * vars_remaining[7]);
        // (003)
        residual[9] = 2.0 * (vars_trace[0] * vars_trace[2] * vars_remaining[14] + vars_trace[0] * vars_trace[4] * vars_remaining[13] + vars_trace[2] * vars_trace[4] * vars_remaining[12] +
                             vars_trace[1] * vars_trace[3] * vars_remaining[14] + vars_trace[1] * vars_trace[5] * vars_remaining[13] + vars_trace[3] * vars_trace[5] * vars_remaining[12]) -
                      (vars_trace[7] * vars_trace[9] * vars_remaining[17] + vars_trace[7] * vars_trace[11] * vars_remaining[16] + vars_trace[9] * vars_trace[11] * vars_remaining[15] +
                       vars_trace[6] * vars_trace[8] * vars_remaining[17] + vars_trace[6] * vars_trace[10] * vars_remaining[16] + vars_trace[8] * vars_trace[10] * vars_remaining[15]);

        // (023)
        residual[10] = 2.0 * (vars_trace[0] * vars_trace[3] * vars_remaining[14] + vars_trace[1] * vars_trace[4] * vars_remaining[13] + vars_trace[2] * vars_trace[5] * vars_remaining[12] +
                              vars_trace[1] * vars_trace[2] * vars_remaining[14] + vars_trace[0] * vars_trace[5] * vars_remaining[13] + vars_trace[3] * vars_trace[4] * vars_remaining[12]) +
                       2.0 * (vars_trace[7] * vars_trace[8] * vars_remaining[17] + vars_trace[6] * vars_trace[11] * vars_remaining[16] + vars_trace[9] * vars_trace[10] * vars_remaining[15] +
                              vars_trace[6] * vars_trace[9] * vars_remaining[17] + vars_trace[7] * vars_trace[10] * vars_remaining[16] + vars_trace[8] * vars_trace[11] * vars_remaining[15]);

        // (113)
        residual[11] = 4.0 * (vars_remaining[14] + vars_remaining[13] + vars_remaining[12]) +
                       4.0 * (vars_remaining[17] + vars_remaining[16] + vars_remaining[15]);

        // (103)
        residual[12] = 2.0 * (vars_trace[2] * vars_remaining[14] + vars_trace[0] * vars_remaining[13] + vars_trace[4] * vars_remaining[12] +
                              vars_trace[3] * vars_remaining[14] + vars_trace[1] * vars_remaining[13] + vars_trace[5] * vars_remaining[12]) -
                       (vars_trace[9] * vars_remaining[17] + vars_trace[7] * vars_remaining[16] + vars_trace[11] * vars_remaining[15] +
                        vars_trace[8] * vars_remaining[17] + vars_trace[6] * vars_remaining[16] + vars_trace[10] * vars_remaining[15]);

        // (013)
        residual[13] = 2.0 * (vars_trace[0] * vars_remaining[14] + vars_trace[4] * vars_remaining[13] + vars_trace[2] * vars_remaining[12] +
                              vars_trace[1] * vars_remaining[14] + vars_trace[5] * vars_remaining[13] + vars_trace[3] * vars_remaining[12]) -
                       (vars_trace[7] * vars_remaining[17] + vars_trace[11] * vars_remaining[16] + vars_trace[9] * vars_remaining[15] +
                        vars_trace[6] * vars_remaining[17] + vars_trace[10] * vars_remaining[16] + vars_trace[8] * vars_remaining[15]);

        // 2 неследовых элемента
        // (155)
        residual[14] = 2.0 * (vars_remaining[2] * vars_remaining[4] + vars_remaining[0] * vars_remaining[4] + vars_remaining[0] * vars_remaining[2] +
                              vars_remaining[3] * vars_remaining[5] + vars_remaining[1] * vars_remaining[5] + vars_remaining[1] * vars_remaining[3]) -
                       (vars_remaining[9] * vars_remaining[11] + vars_remaining[7] * vars_remaining[11] + vars_remaining[7] * vars_remaining[9] +
                        vars_remaining[8] * vars_remaining[10] + vars_remaining[6] * vars_remaining[10] + vars_remaining[6] * vars_remaining[8]);
        // (156)
        residual[15] = 2.0 * (vars_remaining[2] * vars_remaining[5] + vars_remaining[1] * vars_remaining[4] + vars_remaining[0] * vars_remaining[3] +
                              vars_remaining[3] * vars_remaining[4] + vars_remaining[0] * vars_remaining[5] + vars_remaining[1] * vars_remaining[2]) +
                       2.0 * (vars_remaining[9] * vars_remaining[10] + vars_remaining[6] * vars_remaining[11] + vars_remaining[7] * vars_remaining[8] +
                              vars_remaining[8] * vars_remaining[11] + vars_remaining[7] * vars_remaining[10] + vars_remaining[6] * vars_remaining[9]) + 6.0;
        // (055)
        residual[16] = 2.0 * (vars_trace[0] * vars_remaining[2] * vars_remaining[4] + vars_trace[4] * vars_remaining[0] * vars_remaining[2] + vars_trace[2] * vars_remaining[0] * vars_remaining[4] +
                              vars_trace[1] * vars_remaining[3] * vars_remaining[5] + vars_trace[5] * vars_remaining[1] * vars_remaining[3] + vars_trace[3] * vars_remaining[1] * vars_remaining[5]) -
                       (vars_trace[7] * vars_remaining[9] * vars_remaining[11] + vars_trace[11] * vars_remaining[7] * vars_remaining[9] + vars_trace[9] * vars_remaining[7] * vars_remaining[11] +
                        vars_trace[6] * vars_remaining[8] * vars_remaining[10] + vars_trace[10] * vars_remaining[6] * vars_remaining[8] + vars_trace[8] * vars_remaining[6] * vars_remaining[10]);
        // (056)
        residual[17] = 2.0 * (vars_trace[0] * vars_remaining[2] * vars_remaining[5] + vars_trace[4] * vars_remaining[0] * vars_remaining[3] + vars_trace[2] * vars_remaining[1] * vars_remaining[4] +
                              vars_trace[1] * vars_remaining[3] * vars_remaining[4] + vars_trace[5] * vars_remaining[1] * vars_remaining[2] + vars_trace[3] * vars_remaining[0] * vars_remaining[5]) -
                       (vars_trace[7] * vars_remaining[9] * vars_remaining[10] + vars_trace[11] * vars_remaining[7] * vars_remaining[8] + vars_trace[9] * vars_remaining[6] * vars_remaining[11] +
                        vars_trace[6] * vars_remaining[8] * vars_remaining[11] + vars_trace[10] * vars_remaining[6] * vars_remaining[9] + vars_trace[8] * vars_remaining[7] * vars_remaining[10]);
        // (255)
        residual[18] = 2.0 * (vars_trace[1] * vars_remaining[2] * vars_remaining[4] + vars_trace[5] * vars_remaining[0] * vars_remaining[2] + vars_trace[3] * vars_remaining[0] * vars_remaining[4] +
                              vars_trace[0] * vars_remaining[3] * vars_remaining[5] + vars_trace[4] * vars_remaining[1] * vars_remaining[3] + vars_trace[2] * vars_remaining[1] * vars_remaining[5]) +
                       2.0 * (vars_trace[6] * vars_remaining[9] * vars_remaining[11] + vars_trace[10] * vars_remaining[7] * vars_remaining[9] + vars_trace[8] * vars_remaining[7] * vars_remaining[11] +
                              vars_trace[7] * vars_remaining[8] * vars_remaining[10] + vars_trace[11] * vars_remaining[6] * vars_remaining[8] + vars_trace[9] * vars_remaining[6] * vars_remaining[10]) - 3.0;
        // (256)
        residual[19] = 2.0 * (vars_trace[1] * vars_remaining[2] * vars_remaining[5] + vars_trace[5] * vars_remaining[0] * vars_remaining[3] + vars_trace[3] * vars_remaining[1] * vars_remaining[4] +
                              vars_trace[0] * vars_remaining[3] * vars_remaining[4] + vars_trace[4] * vars_remaining[1] * vars_remaining[2] + vars_trace[2] * vars_remaining[0] * vars_remaining[5]) -
                       (vars_trace[6] * vars_remaining[9] * vars_remaining[10] + vars_trace[10] * vars_remaining[7] * vars_remaining[8] + vars_trace[8] * vars_remaining[6] * vars_remaining[11] +
                        vars_trace[7] * vars_remaining[8] * vars_remaining[11] + vars_trace[11] * vars_remaining[6] * vars_remaining[9] + vars_trace[9] * vars_remaining[7] * vars_remaining[10]);
        // (033)
        residual[20] = 2.0 * (vars_trace[0] * vars_remaining[13] * vars_remaining[14] + vars_trace[4] * vars_remaining[12] * vars_remaining[13] + vars_trace[2] * vars_remaining[12] * vars_remaining[14] +
                              vars_trace[1] * vars_remaining[13] * vars_remaining[14] + vars_trace[5] * vars_remaining[12] * vars_remaining[13] + vars_trace[3] * vars_remaining[12] * vars_remaining[14]) -
                       (vars_trace[7] * vars_remaining[16] * vars_remaining[17] + vars_trace[11] * vars_remaining[15] * vars_remaining[16] + vars_trace[9] * vars_remaining[15] * vars_remaining[17] +
                        vars_trace[6] * vars_remaining[16] * vars_remaining[17] + vars_trace[10] * vars_remaining[15] * vars_remaining[16] + vars_trace[8] * vars_remaining[15] * vars_remaining[17]);

        // (035)
        residual[21] = 2.0 * (vars_trace[0] * vars_remaining[4] * vars_remaining[13] + vars_trace[4] * vars_remaining[2] * vars_remaining[12] + vars_trace[2] * vars_remaining[0] * vars_remaining[14] +
                              vars_trace[1] * vars_remaining[5] * vars_remaining[13] + vars_trace[5] * vars_remaining[3] * vars_remaining[12] + vars_trace[3] * vars_remaining[1] * vars_remaining[14]) +
                       2.0 * (vars_trace[7] * vars_remaining[11] * vars_remaining[16] + vars_trace[11] * vars_remaining[9] * vars_remaining[15] + vars_trace[9] * vars_remaining[7] * vars_remaining[17] +
                              vars_trace[6] * vars_remaining[10] * vars_remaining[16] + vars_trace[10] * vars_remaining[8] * vars_remaining[15] + vars_trace[8] * vars_remaining[6] * vars_remaining[17]) - 3.0;

        // (036)
        residual[22] = 2.0 * (vars_trace[0] * vars_remaining[5] * vars_remaining[13] + vars_trace[4] * vars_remaining[3] * vars_remaining[12] + vars_trace[2] * vars_remaining[1] * vars_remaining[14] +
                              vars_trace[1] * vars_remaining[4] * vars_remaining[13] + vars_trace[5] * vars_remaining[2] * vars_remaining[12] + vars_trace[3] * vars_remaining[0] * vars_remaining[14]) -
                       (vars_trace[7] * vars_remaining[10] * vars_remaining[16] + vars_trace[11] * vars_remaining[8] * vars_remaining[15] + vars_trace[9] * vars_remaining[6] * vars_remaining[17] +
                        vars_trace[6] * vars_remaining[11] * vars_remaining[16] + vars_trace[10] * vars_remaining[9] * vars_remaining[15] + vars_trace[8] * vars_remaining[7] * vars_remaining[17]);

        // (053)
        residual[23] = 2.0 * (vars_trace[0] * vars_remaining[2] * vars_remaining[14] + vars_trace[4] * vars_remaining[0] * vars_remaining[13] + vars_trace[2] * vars_remaining[4] * vars_remaining[12] +
                              vars_trace[1] * vars_remaining[3] * vars_remaining[14] + vars_trace[5] * vars_remaining[1] * vars_remaining[13] + vars_trace[3] * vars_remaining[5] * vars_remaining[12]) +
                       2.0 * (vars_trace[7] * vars_remaining[9] * vars_remaining[17] + vars_trace[11] * vars_remaining[7] * vars_remaining[16] + vars_trace[9] * vars_remaining[11] * vars_remaining[15] +
                              vars_trace[6] * vars_remaining[8] * vars_remaining[17] + vars_trace[10] * vars_remaining[6] * vars_remaining[16] + vars_trace[8] * vars_remaining[10] * vars_remaining[15]) - 3.0;
        // (063)
        residual[24] = 2.0 * (vars_trace[0] * vars_remaining[3] * vars_remaining[14] + vars_trace[4] * vars_remaining[1] * vars_remaining[13] + vars_trace[2] * vars_remaining[5] * vars_remaining[12] +
                              vars_trace[1] * vars_remaining[2] * vars_remaining[14] + vars_trace[5] * vars_remaining[0] * vars_remaining[13] + vars_trace[3] * vars_remaining[4] * vars_remaining[12]) -
                       (vars_trace[7] * vars_remaining[8] * vars_remaining[17] + vars_trace[11] * vars_remaining[6] * vars_remaining[16] + vars_trace[9] * vars_remaining[10] * vars_remaining[15] +
                        vars_trace[6] * vars_remaining[9] * vars_remaining[17] + vars_trace[10] * vars_remaining[7] * vars_remaining[16] + vars_trace[8] * vars_remaining[11] * vars_remaining[15]);
        // (133)
        residual[25] = 4.0 * (vars_remaining[13] * vars_remaining[14] + vars_remaining[12] * vars_remaining[13] + vars_remaining[12] * vars_remaining[14]) +
                       4.0 * (vars_remaining[16] * vars_remaining[17] + vars_remaining[15] * vars_remaining[16] + vars_remaining[15] * vars_remaining[17]) + 6.0;

        // (135)
        residual[26] = 2.0 * (vars_remaining[4] * vars_remaining[13] + vars_remaining[2] * vars_remaining[12] + vars_remaining[0] * vars_remaining[14] +
                              vars_remaining[5] * vars_remaining[13] + vars_remaining[3] * vars_remaining[12] + vars_remaining[1] * vars_remaining[14]) -
                       (vars_remaining[11] * vars_remaining[16] + vars_remaining[9] * vars_remaining[15] + vars_remaining[7] * vars_remaining[17] +
                        vars_remaining[10] * vars_remaining[16] + vars_remaining[8] * vars_remaining[15] + vars_remaining[6] * vars_remaining[17]);

        // (153)
        residual[27] = 2.0 * (vars_remaining[2] * vars_remaining[14] + vars_remaining[0] * vars_remaining[13] + vars_remaining[4] * vars_remaining[12] +
                              vars_remaining[3] * vars_remaining[14] + vars_remaining[1] * vars_remaining[13] + vars_remaining[5] * vars_remaining[12]) -
                       (vars_remaining[9] * vars_remaining[17] + vars_remaining[7] * vars_remaining[16] + vars_remaining[11] * vars_remaining[15] +
                        vars_remaining[8] * vars_remaining[17] + vars_remaining[6] * vars_remaining[16] + vars_remaining[10] * vars_remaining[15]);

        // 3 неследовых элемента

        // (555)
        residual[28] = 6.0 * (vars_remaining[0] * vars_remaining[2] * vars_remaining[4] + vars_remaining[1] * vars_remaining[3] * vars_remaining[5]) +
                       6.0 * (vars_remaining[6] * vars_remaining[8] * vars_remaining[10] + vars_remaining[7] * vars_remaining[9] * vars_remaining[11]) + 6.0;

        // (556)
        residual[29] = 2.0 * (vars_remaining[0] * vars_remaining[2] * vars_remaining[5] + vars_remaining[1] * vars_remaining[2] * vars_remaining[4] + vars_remaining[0] * vars_remaining[3] * vars_remaining[4] +
                              vars_remaining[1] * vars_remaining[3] * vars_remaining[4] + vars_remaining[0] * vars_remaining[3] * vars_remaining[5] + vars_remaining[1] * vars_remaining[2] * vars_remaining[5]) -
                       (vars_remaining[7] * vars_remaining[9] * vars_remaining[10] + vars_remaining[6] * vars_remaining[9] * vars_remaining[11] + vars_remaining[7] * vars_remaining[8] * vars_remaining[11] +
                        vars_remaining[6] * vars_remaining[8] * vars_remaining[11] + vars_remaining[7] * vars_remaining[8] * vars_remaining[10] + vars_remaining[10] * vars_remaining[6] * vars_remaining[9]);
        // (333)
        residual[30] = 12.0 * (vars_remaining[12] * vars_remaining[13] * vars_remaining[14]) +
                       12.0 * (vars_remaining[15] * vars_remaining[16] * vars_remaining[17]) + 6.0;

        // (533)
        residual[31] = 2.0 * (vars_remaining[0] * vars_remaining[13] * vars_remaining[14] + vars_remaining[4] * vars_remaining[12] * vars_remaining[13] + vars_remaining[2] * vars_remaining[12] * vars_remaining[14] +
                              vars_remaining[1] * vars_remaining[13] * vars_remaining[14] + vars_remaining[5] * vars_remaining[12] * vars_remaining[13] + vars_remaining[3] * vars_remaining[12] * vars_remaining[14]) -
                       (vars_remaining[7] * vars_remaining[16] * vars_remaining[17] + vars_remaining[11] * vars_remaining[15] * vars_remaining[16] + vars_remaining[9] * vars_remaining[15] * vars_remaining[17] +
                        vars_remaining[6] * vars_remaining[16] * vars_remaining[17] + vars_remaining[10] * vars_remaining[15] * vars_remaining[16] + vars_remaining[8] * vars_remaining[15] * vars_remaining[17]);

        // (553)
        residual[32] = 2.0 * (vars_remaining[0] * vars_remaining[2] * vars_remaining[14] + vars_remaining[0] * vars_remaining[4] * vars_remaining[13] + vars_remaining[2] * vars_remaining[4] * vars_remaining[12] +
                              vars_remaining[1] * vars_remaining[3] * vars_remaining[14] + vars_remaining[1] * vars_remaining[5] * vars_remaining[13] + vars_remaining[3] * vars_remaining[5] * vars_remaining[12]) -
                       (vars_remaining[7] * vars_remaining[9] * vars_remaining[17] + vars_remaining[7] * vars_remaining[11] * vars_remaining[16] + vars_remaining[9] * vars_remaining[11] * vars_remaining[15] +
                        vars_remaining[6] * vars_remaining[8] * vars_remaining[17] + vars_remaining[6] * vars_remaining[10] * vars_remaining[16] + vars_remaining[8] * vars_remaining[10] * vars_remaining[15]);

        // (563)
        residual[33] = 2.0 * (vars_remaining[0] * vars_remaining[3] * vars_remaining[14] + vars_remaining[1] * vars_remaining[4] * vars_remaining[13] + vars_remaining[2] * vars_remaining[5] * vars_remaining[12] +
                              vars_remaining[1] * vars_remaining[2] * vars_remaining[14] + vars_remaining[0] * vars_remaining[5] * vars_remaining[13] + vars_remaining[3] * vars_remaining[4] * vars_remaining[12]) +
                       2.0 * (vars_remaining[7] * vars_remaining[8] * vars_remaining[17] + vars_remaining[6] * vars_remaining[11] * vars_remaining[16] + vars_remaining[9] * vars_remaining[10] * vars_remaining[15] +
                              vars_remaining[6] * vars_remaining[9] * vars_remaining[17] + vars_remaining[7] * vars_remaining[10] * vars_remaining[16] + vars_remaining[8] * vars_remaining[11] * vars_remaining[15]) - 3.0;
        return true;
    }
};

struct ExpandedSystem2 { // 20 уравнений
    template<typename T>
    bool operator()(const T *const vars_trace, const T *const vars_remaining, const T *vars_remaining_next,
                    T *residual) const
    {
        for (int i = 0; i <= 19; ++i)
            residual[i] = T{0};

        // (004)
        residual[0] =	2.0 *	(vars_trace[0] * vars_trace[2] * vars_remaining_next[2] + vars_trace[0] * vars_trace[4] * vars_remaining_next[1] + vars_trace[2] * vars_trace[4] * vars_remaining_next[0]) -
                         2.0 *	(vars_trace[1] * vars_trace[3] * vars_remaining_next[2] + vars_trace[1] * vars_trace[5] * vars_remaining_next[1] + vars_trace[3] * vars_trace[5] * vars_remaining_next[0]) +
                         (vars_trace[7] * vars_trace[9] * vars_remaining_next[5] + vars_trace[7] * vars_trace[11] * vars_remaining_next[4] + vars_trace[9] * vars_trace[11] * vars_remaining_next[3]) -
                         (vars_trace[6] * vars_trace[8] * vars_remaining_next[5] + vars_trace[6] * vars_trace[10] * vars_remaining_next[4] + vars_trace[8] * vars_trace[10] * vars_remaining_next[3]);

        // (024)
        residual[1] =	2.0 *	(vars_trace[0] * vars_trace[3] * vars_remaining_next[2] + vars_trace[1] * vars_trace[4] * vars_remaining_next[1] + vars_trace[2] * vars_trace[5] * vars_remaining_next[0]) -
                         2.0 *	(vars_trace[1] * vars_trace[2] * vars_remaining_next[2] + vars_trace[0] * vars_trace[5] * vars_remaining_next[1] + vars_trace[3] * vars_trace[4] * vars_remaining_next[0]) -
                         2.0 *	(vars_trace[7] * vars_trace[8] * vars_remaining_next[5] + vars_trace[6] * vars_trace[11] * vars_remaining_next[4] + vars_trace[9] * vars_trace[10] * vars_remaining_next[3]) +
                         2.0 *	(vars_trace[6] * vars_trace[9] * vars_remaining_next[5] + vars_trace[7] * vars_trace[10] * vars_remaining_next[4] + vars_trace[8] * vars_trace[11] * vars_remaining_next[3]);

        // (104)
        residual[2] =	2.0 *	(vars_trace[2] * vars_remaining_next[2] + vars_trace[0] * vars_remaining_next[1] + vars_trace[4] * vars_remaining_next[0]) -
                         2.0 *	(vars_trace[3] * vars_remaining_next[2] + vars_trace[1] * vars_remaining_next[1] + vars_trace[5] * vars_remaining_next[0]) +
                         (vars_trace[9] * vars_remaining_next[5] + vars_trace[7] * vars_remaining_next[4] + vars_trace[11] * vars_remaining_next[3]) -
                         (vars_trace[8] * vars_remaining_next[5] + vars_trace[6] * vars_remaining_next[4] + vars_trace[10] * vars_remaining_next[3]);

        // (014)
        residual[3] =	2.0 *	(vars_trace[0] * vars_remaining_next[2] + vars_trace[4] * vars_remaining_next[1] + vars_trace[2] * vars_remaining_next[0]) -
                         2.0 *	(vars_trace[1] * vars_remaining_next[2] + vars_trace[5] * vars_remaining_next[1] + vars_trace[3] * vars_remaining_next[0]) +
                         (vars_trace[7] * vars_remaining_next[5] + vars_trace[11] * vars_remaining_next[4] + vars_trace[9] * vars_remaining_next[3]) -
                         (vars_trace[6] * vars_remaining_next[5] + vars_trace[10] * vars_remaining_next[4] + vars_trace[8] * vars_remaining_next[3]);
        // (044)
        residual[4] =	2.0 *	(vars_trace[0] * vars_remaining_next[1] * vars_remaining_next[2] + vars_trace[4] * vars_remaining_next[0] * vars_remaining_next[1] + vars_trace[2] * vars_remaining_next[0] * vars_remaining_next[2]) +
                          2.0 *	(vars_trace[1] * vars_remaining_next[1] * vars_remaining_next[2] + vars_trace[5] * vars_remaining_next[0] * vars_remaining_next[1] + vars_trace[3] * vars_remaining_next[0] * vars_remaining_next[2]) -
                          (vars_trace[7] * vars_remaining_next[4] * vars_remaining_next[5] + vars_trace[11] * vars_remaining_next[3] * vars_remaining_next[4] + vars_trace[9] * vars_remaining_next[3] * vars_remaining_next[5]) -
                          (vars_trace[6] * vars_remaining_next[4] * vars_remaining_next[5] + vars_trace[10] * vars_remaining_next[3] * vars_remaining_next[4] + vars_trace[8] * vars_remaining_next[3] * vars_remaining_next[5]);

        // (045)
        residual[5] =	2.0 *	(vars_trace[0] * vars_remaining[4] * vars_remaining_next[1] + vars_trace[4] * vars_remaining[2] * vars_remaining_next[0] + vars_trace[2] * vars_remaining[0] * vars_remaining_next[2]) -
                          2.0 *	(vars_trace[1] * vars_remaining[5] * vars_remaining_next[1] + vars_trace[5] * vars_remaining[3] * vars_remaining_next[0] + vars_trace[3] * vars_remaining[1] * vars_remaining_next[2]) -
                          2.0 *	(vars_trace[7] * vars_remaining[11] * vars_remaining_next[4] + vars_trace[11] * vars_remaining[9] * vars_remaining_next[3] + vars_trace[9] * vars_remaining[7] * vars_remaining_next[5]) +
                          2.0 *	(vars_trace[6] * vars_remaining[10] * vars_remaining_next[4] + vars_trace[10] * vars_remaining[8] * vars_remaining_next[3] + vars_trace[8] * vars_remaining[6] * vars_remaining_next[5]) - 3.0;



        // (046)
        residual[6] =	2.0 *	(vars_trace[0] * vars_remaining[5] * vars_remaining_next[1] + vars_trace[4] * vars_remaining[3] * vars_remaining_next[0] + vars_trace[2] * vars_remaining[1] * vars_remaining_next[2]) -
                          2.0 *	(vars_trace[1] * vars_remaining[4] * vars_remaining_next[1] + vars_trace[5] * vars_remaining[2] * vars_remaining_next[0] + vars_trace[3] * vars_remaining[0] * vars_remaining_next[2]) +
                          (vars_trace[7] * vars_remaining[10] * vars_remaining_next[4] + vars_trace[11] * vars_remaining[8] * vars_remaining_next[3] + vars_trace[9] * vars_remaining[6] * vars_remaining_next[5]) -
                          (vars_trace[6] * vars_remaining[11] * vars_remaining_next[4] + vars_trace[10] * vars_remaining[9] * vars_remaining_next[3] + vars_trace[8] * vars_remaining[7] * vars_remaining_next[5]);



        // (054)
        residual[7] =	2.0 *	(vars_trace[0] * vars_remaining[2] * vars_remaining_next[2] + vars_trace[4] * vars_remaining[0] * vars_remaining_next[1] + vars_trace[2] * vars_remaining[4] * vars_remaining_next[0]) -
                          2.0 *	(vars_trace[1] * vars_remaining[3] * vars_remaining_next[2] + vars_trace[5] * vars_remaining[1] * vars_remaining_next[1] + vars_trace[3] * vars_remaining[5] * vars_remaining_next[0]) -
                          2.0 *	(vars_trace[7] * vars_remaining[9] * vars_remaining_next[5] + vars_trace[11] * vars_remaining[7] * vars_remaining_next[4] + vars_trace[9] * vars_remaining[11] * vars_remaining_next[3]) +
                          2.0 *	(vars_trace[6] * vars_remaining[8] * vars_remaining_next[5] + vars_trace[10] * vars_remaining[6] * vars_remaining_next[4] + vars_trace[8] * vars_remaining[10] * vars_remaining_next[3]) + 3.0;



        // (064)
        residual[8] =	2.0 *	(vars_trace[0] * vars_remaining[3] * vars_remaining_next[2] + vars_trace[4] * vars_remaining[1] * vars_remaining_next[1] + vars_trace[2] * vars_remaining[5] * vars_remaining_next[0]) -
                          2.0 *	(vars_trace[1] * vars_remaining[2] * vars_remaining_next[2] + vars_trace[5] * vars_remaining[0] * vars_remaining_next[1] + vars_trace[3] * vars_remaining[4] * vars_remaining_next[0]) +
                          (vars_trace[7] * vars_remaining[8] * vars_remaining_next[5] + vars_trace[11] * vars_remaining[6] * vars_remaining_next[4] + vars_trace[9] * vars_remaining[10] * vars_remaining_next[3]) -
                          (vars_trace[6] * vars_remaining[9] * vars_remaining_next[5] + vars_trace[10] * vars_remaining[7] * vars_remaining_next[4] + vars_trace[8] * vars_remaining[11] * vars_remaining_next[3]);


        // (144)
        residual[9] =	4.0 *	(vars_remaining_next[1] * vars_remaining_next[2] + vars_remaining_next[0] * vars_remaining_next[1] + vars_remaining_next[0] * vars_remaining_next[2]) +
                          4.0 *	(vars_remaining_next[4] * vars_remaining_next[5] + vars_remaining_next[3] * vars_remaining_next[4] + vars_remaining_next[3] * vars_remaining_next[5]) + 2.0;


        // (145)
        residual[10] =	2.0 *	(vars_remaining[4] * vars_remaining_next[1] + vars_remaining[2] * vars_remaining_next[0] + vars_remaining[0] * vars_remaining_next[2]) -
                          2.0 *	(vars_remaining[5] * vars_remaining_next[1] + vars_remaining[3] * vars_remaining_next[0] + vars_remaining[1] * vars_remaining_next[2]) +
                          (vars_remaining[11] * vars_remaining_next[4] + vars_remaining[9] * vars_remaining_next[3] + vars_remaining[7] * vars_remaining_next[5]) -
                          (vars_remaining[10] * vars_remaining_next[4] + vars_remaining[8] * vars_remaining_next[3] + vars_remaining[6] * vars_remaining_next[5]);


        // (154)
        residual[11] =	2.0 *	(vars_remaining[2] * vars_remaining_next[2] + vars_remaining[0] * vars_remaining_next[1] + vars_remaining[4] * vars_remaining_next[0]) -
                          2.0 *	(vars_remaining[3] * vars_remaining_next[2] + vars_remaining[1] * vars_remaining_next[1] + vars_remaining[5] * vars_remaining_next[0]) +
                          (vars_remaining[9] * vars_remaining_next[5] + vars_remaining[7] * vars_remaining_next[4] + vars_remaining[11] * vars_remaining_next[3]) -
                          (vars_remaining[8] * vars_remaining_next[5] + vars_remaining[6] * vars_remaining_next[4] + vars_remaining[10] * vars_remaining_next[3]);
        // (034)
        residual[12] =	2.0 *	(vars_trace[0] * vars_remaining[13] * vars_remaining_next[2] + vars_trace[4] * vars_remaining[12] * vars_remaining_next[1] + vars_trace[2] * vars_remaining[14] * vars_remaining_next[0]) -
                          2.0 *	(vars_trace[1] * vars_remaining[13] * vars_remaining_next[2] + vars_trace[5] * vars_remaining[12] * vars_remaining_next[1] + vars_trace[3] * vars_remaining[14] * vars_remaining_next[0]) +
                          (vars_trace[7] * vars_remaining[16] * vars_remaining_next[5] + vars_trace[11] * vars_remaining[15] * vars_remaining_next[4] + vars_trace[9] * vars_remaining[17] * vars_remaining_next[3]) -
                          (vars_trace[6] * vars_remaining[16] * vars_remaining_next[5] + vars_trace[10] * vars_remaining[15] * vars_remaining_next[4] + vars_trace[8] * vars_remaining[17] * vars_remaining_next[3]);

        // (043)
        residual[13] =	2.0 *	(vars_trace[0] * vars_remaining[14] * vars_remaining_next[1] + vars_trace[4] * vars_remaining[13] * vars_remaining_next[0] + vars_trace[2] * vars_remaining[12] * vars_remaining_next[2]) -
                          2.0 *	(vars_trace[1] * vars_remaining[14] * vars_remaining_next[1] + vars_trace[5] * vars_remaining[13] * vars_remaining_next[0] + vars_trace[3] * vars_remaining[12] * vars_remaining_next[2]) +
                          (vars_trace[7] * vars_remaining[17] * vars_remaining_next[4] + vars_trace[11] * vars_remaining[16] * vars_remaining_next[3] + vars_trace[9] * vars_remaining[15] * vars_remaining_next[5]) -
                          (vars_trace[6] * vars_remaining[17] * vars_remaining_next[4] + vars_trace[10] * vars_remaining[16] * vars_remaining_next[3] + vars_trace[8] * vars_remaining[15] * vars_remaining_next[5]);

        // (544)
        residual[14] =	2.0 *	(vars_remaining[0] * vars_remaining_next[1] * vars_remaining_next[2] + vars_remaining[4] * vars_remaining_next[0] * vars_remaining_next[1] + vars_remaining[2] * vars_remaining_next[0] * vars_remaining_next[2]) +
                         2.0 *	(vars_remaining[1] * vars_remaining_next[1] * vars_remaining_next[2] + vars_remaining[5] * vars_remaining_next[0] * vars_remaining_next[1] + vars_remaining[3] * vars_remaining_next[0] * vars_remaining_next[2]) -
                         (vars_remaining[7] * vars_remaining_next[4] * vars_remaining_next[5] + vars_remaining[11] * vars_remaining_next[3] * vars_remaining_next[4] + vars_remaining[9] * vars_remaining_next[3] * vars_remaining_next[5]) -
                         (vars_remaining[6] * vars_remaining_next[4] * vars_remaining_next[5] + vars_remaining[10] * vars_remaining_next[3] * vars_remaining_next[4] + vars_remaining[8] * vars_remaining_next[3] * vars_remaining_next[5]);

        // (554)
        residual[15] =	2.0 *	(vars_remaining[0] * vars_remaining[2] * vars_remaining_next[2] + vars_remaining[0] * vars_remaining[4] * vars_remaining_next[1] + vars_remaining[2] * vars_remaining[4] * vars_remaining_next[0]) -
                         2.0 *	(vars_remaining[1] * vars_remaining[3] * vars_remaining_next[2] + vars_remaining[1] * vars_remaining[5] * vars_remaining_next[1] + vars_remaining[3] * vars_remaining[5] * vars_remaining_next[0]) +
                         (vars_remaining[7] * vars_remaining[9] * vars_remaining_next[5] + vars_remaining[7] * vars_remaining[11] * vars_remaining_next[4] + vars_remaining[9] * vars_remaining[11] * vars_remaining_next[3]) -
                         (vars_remaining[6] * vars_remaining[8] * vars_remaining_next[5] + vars_remaining[6] * vars_remaining[10] * vars_remaining_next[4] + vars_remaining[8] * vars_remaining[10] * vars_remaining_next[3]);

        // (564)
        residual[16] =	2.0 *	(vars_remaining[0] * vars_remaining[3] * vars_remaining_next[2] + vars_remaining[1] * vars_remaining[4] * vars_remaining_next[1] + vars_remaining[2] * vars_remaining[5] * vars_remaining_next[0]) -
                         2.0 *	(vars_remaining[1] * vars_remaining[2] * vars_remaining_next[2] + vars_remaining[0] * vars_remaining[5] * vars_remaining_next[1] + vars_remaining[3] * vars_remaining[4] * vars_remaining_next[0]) -
                         2.0 *	(vars_remaining[7] * vars_remaining[8] * vars_remaining_next[5] + vars_remaining[6] * vars_remaining[11] * vars_remaining_next[4] + vars_remaining[9] * vars_remaining[10] * vars_remaining_next[3]) +
                         2.0 *	(vars_remaining[6] * vars_remaining[9] * vars_remaining_next[5] + vars_remaining[7] * vars_remaining[10] * vars_remaining_next[4] + vars_remaining[8] * vars_remaining[11] * vars_remaining_next[3]) - 3.0;

        // (344)
        residual[17] =	4.0 *	(vars_remaining[12] * vars_remaining_next[1] * vars_remaining_next[2] + vars_remaining[14] * vars_remaining_next[0] * vars_remaining_next[1] + vars_remaining[13] * vars_remaining_next[0] * vars_remaining_next[2]) +
                         4.0 *	(vars_remaining[15] * vars_remaining_next[4] * vars_remaining_next[5] + vars_remaining[17] * vars_remaining_next[3] * vars_remaining_next[4] + vars_remaining[16] * vars_remaining_next[3] * vars_remaining_next[5]) - 2.0;

        // (354)
        residual[18] =	2.0 *	(vars_remaining[12] * vars_remaining[2] * vars_remaining_next[2] + vars_remaining[14] * vars_remaining[0] * vars_remaining_next[1] + vars_remaining[13] * vars_remaining[4] * vars_remaining_next[0]) -
                         2.0 *	(vars_remaining[12] * vars_remaining[3] * vars_remaining_next[2] + vars_remaining[14] * vars_remaining[1] * vars_remaining_next[1] + vars_remaining[13] * vars_remaining[5] * vars_remaining_next[0]) +
                         (vars_remaining[15] * vars_remaining[9] * vars_remaining_next[5] + vars_remaining[17] * vars_remaining[7] * vars_remaining_next[4] + vars_remaining[16] * vars_remaining[11] * vars_remaining_next[3]) -
                         (vars_remaining[15] * vars_remaining[8] * vars_remaining_next[5] + vars_remaining[17] * vars_remaining[6] * vars_remaining_next[4] + vars_remaining[16] * vars_remaining[10] * vars_remaining_next[3]);

        // (534)
        residual[19] =	2.0 *	(vars_remaining[13] * vars_remaining[0] * vars_remaining_next[2] + vars_remaining[12] * vars_remaining[4] * vars_remaining_next[1] + vars_remaining[14] * vars_remaining[2] * vars_remaining_next[0]) -
                         2.0 *	(vars_remaining[13] * vars_remaining[1] * vars_remaining_next[2] + vars_remaining[12] * vars_remaining[5] * vars_remaining_next[1] + vars_remaining[14] * vars_remaining[3] * vars_remaining_next[0]) +
                         (vars_remaining[16] * vars_remaining[7] * vars_remaining_next[5] + vars_remaining[15] * vars_remaining[11] * vars_remaining_next[4] + vars_remaining[17] * vars_remaining[9] * vars_remaining_next[3]) -
                         (vars_remaining[16] * vars_remaining[6] * vars_remaining_next[5] + vars_remaining[15] * vars_remaining[10] * vars_remaining_next[4] + vars_remaining[17] * vars_remaining[8] * vars_remaining_next[3]);
        return true;
    }
};

struct ExpandedSystem3 {
    template<typename T>
    bool operator()(const T *const vars_trace, const T *const vars_remaining, const T *vars_remaining_next,
                    T *residual) const
    {
        for (int i = 0; i <= 49; ++i)
            residual[i] = T{0};

        // (007)
        residual[0] =	2.0 *	(vars_trace[0] * vars_trace[2] * vars_remaining_next[10] + vars_trace[0] * vars_trace[4] * vars_remaining_next[8] + vars_trace[2] * vars_trace[4] * vars_remaining_next[6]) -
                         2.0 *	(vars_trace[1] * vars_trace[3] * vars_remaining_next[11] + vars_trace[1] * vars_trace[5] * vars_remaining_next[9] + vars_trace[3] * vars_trace[5] * vars_remaining_next[7]) -
                         2.0 *	(vars_trace[7] * vars_trace[9] * vars_remaining_next[17] + vars_trace[7] * vars_trace[11] * vars_remaining_next[15] + vars_trace[9] * vars_trace[11] * vars_remaining_next[13]) +
                         2.0 *	(vars_trace[6] * vars_trace[8] * vars_remaining_next[16] + vars_trace[6] * vars_trace[10] * vars_remaining_next[14] + vars_trace[8] * vars_trace[10] * vars_remaining_next[12]);
        residual[0] = T{0};
        // (227)
        residual[1] =	2.0 *	(vars_trace[1] * vars_trace[3] * vars_remaining_next[10] + vars_trace[1] * vars_trace[5] * vars_remaining_next[8] + vars_trace[3] * vars_trace[5] * vars_remaining_next[6]) -
                         2.0 *	(vars_trace[0] * vars_trace[2] * vars_remaining_next[11] + vars_trace[0] * vars_trace[4] * vars_remaining_next[9] + vars_trace[2] * vars_trace[4] * vars_remaining_next[7]) +
                         (vars_trace[6] * vars_trace[8] * vars_remaining_next[17] + vars_trace[6] * vars_trace[10] * vars_remaining_next[15] + vars_trace[8] * vars_trace[10] * vars_remaining_next[13]) -
                         (vars_trace[7] * vars_trace[9] * vars_remaining_next[16] + vars_trace[7] * vars_trace[11] * vars_remaining_next[14] + vars_trace[9] * vars_trace[11] * vars_remaining_next[12]);

        // (027)
        residual[2] =	2.0 *	(vars_trace[0] * vars_trace[3] * vars_remaining_next[10] + vars_trace[1] * vars_trace[4] * vars_remaining_next[8] + vars_trace[2] * vars_trace[5] * vars_remaining_next[6]) -
                         2.0 *	(vars_trace[1] * vars_trace[2] * vars_remaining_next[11] + vars_trace[0] * vars_trace[5] * vars_remaining_next[9] + vars_trace[3] * vars_trace[4] * vars_remaining_next[7]) +
                         (vars_trace[7] * vars_trace[8] * vars_remaining_next[17] + vars_trace[6] * vars_trace[11] * vars_remaining_next[15] + vars_trace[9] * vars_trace[10] * vars_remaining_next[13]) -
                         (vars_trace[6] * vars_trace[9] * vars_remaining_next[16] + vars_trace[7] * vars_trace[10] * vars_remaining_next[14] + vars_trace[8] * vars_trace[11] * vars_remaining_next[12]);

        // (207)
        residual[3] =	2.0 *	(vars_trace[1] * vars_trace[2] * vars_remaining_next[10] + vars_trace[0] * vars_trace[5] * vars_remaining_next[8] + vars_trace[3] * vars_trace[4] * vars_remaining_next[6]) -
                         2.0 *	(vars_trace[0] * vars_trace[3] * vars_remaining_next[11] + vars_trace[1] * vars_trace[4] * vars_remaining_next[9] + vars_trace[2] * vars_trace[5] * vars_remaining_next[7]) +
                         (vars_trace[6] * vars_trace[9] * vars_remaining_next[17] + vars_trace[7] * vars_trace[10] * vars_remaining_next[15] + vars_trace[8] * vars_trace[11] * vars_remaining_next[13]) -
                         (vars_trace[7] * vars_trace[8] * vars_remaining_next[16] + vars_trace[6] * vars_trace[11] * vars_remaining_next[14] + vars_trace[9] * vars_trace[10] * vars_remaining_next[12]);

        // (117)
        residual[4] =	2.0 *	(vars_remaining_next[10] + vars_remaining_next[8] + vars_remaining_next[6]) -
                         2.0 *	(vars_remaining_next[11] + vars_remaining_next[9] + vars_remaining_next[7]) +
                         (vars_remaining_next[17] + vars_remaining_next[15] + vars_remaining_next[13]) -
                         (vars_remaining_next[16] + vars_remaining_next[14] + vars_remaining_next[12]);
        // residual[4] = T{0};
        // (107)
        residual[5] =	2.0 *	(vars_trace[2] * vars_remaining_next[10] + vars_trace[0] * vars_remaining_next[8] + vars_trace[4] * vars_remaining_next[6]) -
                         2.0 *	(vars_trace[3] * vars_remaining_next[11] + vars_trace[1] * vars_remaining_next[9] + vars_trace[5] * vars_remaining_next[7]) +
                         (vars_trace[9] * vars_remaining_next[17] + vars_trace[7] * vars_remaining_next[15] + vars_trace[11] * vars_remaining_next[13]) -
                         (vars_trace[8] * vars_remaining_next[16] + vars_trace[6] * vars_remaining_next[14] + vars_trace[10] * vars_remaining_next[12]);

        // (017)
        residual[6] =	2.0 *	(vars_trace[0] * vars_remaining_next[10] + vars_trace[4] * vars_remaining_next[8] + vars_trace[2] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[1] * vars_remaining_next[11] + vars_trace[5] * vars_remaining_next[9] + vars_trace[3] * vars_remaining_next[7]) +
                          (vars_trace[7] * vars_remaining_next[17] + vars_trace[11] * vars_remaining_next[15] + vars_trace[9] * vars_remaining_next[13]) -
                          (vars_trace[6] * vars_remaining_next[16] + vars_trace[10] * vars_remaining_next[14] + vars_trace[8] * vars_remaining_next[12]);

        // (127)
        residual[7] =	2.0 *	(vars_trace[3] * vars_remaining_next[10] + vars_trace[1] * vars_remaining_next[8] + vars_trace[5] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[2] * vars_remaining_next[11] + vars_trace[0] * vars_remaining_next[9] + vars_trace[4] * vars_remaining_next[7]) -
                          2.0 *	(vars_trace[8] * vars_remaining_next[17] + vars_trace[6] * vars_remaining_next[15] + vars_trace[10] * vars_remaining_next[13]) +
                          2.0 *	(vars_trace[9] * vars_remaining_next[16] + vars_trace[7] * vars_remaining_next[14] + vars_trace[11] * vars_remaining_next[12]);

        // (217)
        residual[8] =	2.0 *	(vars_trace[1] * vars_remaining_next[10] + vars_trace[5] * vars_remaining_next[8] + vars_trace[3] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[0] * vars_remaining_next[11] + vars_trace[4] * vars_remaining_next[9] + vars_trace[2] * vars_remaining_next[7]) -
                          2.0 *	(vars_trace[6] * vars_remaining_next[17] + vars_trace[10] * vars_remaining_next[15] + vars_trace[8] * vars_remaining_next[13]) +
                          2.0 *	(vars_trace[7] * vars_remaining_next[16] + vars_trace[11] * vars_remaining_next[14] + vars_trace[9] * vars_remaining_next[12]);
        residual[8] = T{0};
        // (057)
        residual[9] =	2.0 *	(vars_trace[0] * vars_remaining[2] * vars_remaining_next[10] + vars_trace[4] * vars_remaining[0] * vars_remaining_next[8] + vars_trace[2] * vars_remaining[4] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[1] * vars_remaining[3] * vars_remaining_next[11] + vars_trace[5] * vars_remaining[1] * vars_remaining_next[9] + vars_trace[3] * vars_remaining[5] * vars_remaining_next[7]) +
                          (vars_trace[7] * vars_remaining[9] * vars_remaining_next[17] + vars_trace[11] * vars_remaining[7] * vars_remaining_next[15] + vars_trace[9] * vars_remaining[11] * vars_remaining_next[13]) -
                          (vars_trace[6] * vars_remaining[8] * vars_remaining_next[16] + vars_trace[10] * vars_remaining[6] * vars_remaining_next[14] + vars_trace[8] * vars_remaining[10] * vars_remaining_next[12]);
        // (067)
        residual[10] =	2.0 *	(vars_trace[0] * vars_remaining[3] * vars_remaining_next[10] + vars_trace[4] * vars_remaining[1] * vars_remaining_next[8] + vars_trace[2] * vars_remaining[5] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[1] * vars_remaining[2] * vars_remaining_next[11] + vars_trace[5] * vars_remaining[0] * vars_remaining_next[9] + vars_trace[3] * vars_remaining[4] * vars_remaining_next[7]) -
                          2.0 *	(vars_trace[7] * vars_remaining[8] * vars_remaining_next[17] + vars_trace[11] * vars_remaining[6] * vars_remaining_next[15] + vars_trace[9] * vars_remaining[10] * vars_remaining_next[13]) +
                          2.0 *	(vars_trace[6] * vars_remaining[9] * vars_remaining_next[16] + vars_trace[10] * vars_remaining[7] * vars_remaining_next[14] + vars_trace[8] * vars_remaining[11] * vars_remaining_next[12]) + 3.0;
        // (037)
        residual[11] =	2.0 *	(vars_trace[0] * vars_remaining[13] * vars_remaining_next[10] + vars_trace[4] * vars_remaining[12] * vars_remaining_next[8] + vars_trace[2] * vars_remaining[14] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[1] * vars_remaining[13] * vars_remaining_next[11] + vars_trace[5] * vars_remaining[12] * vars_remaining_next[9] + vars_trace[3] * vars_remaining[14] * vars_remaining_next[7]) +
                          (vars_trace[7] * vars_remaining[16] * vars_remaining_next[17] + vars_trace[11] * vars_remaining[15] * vars_remaining_next[15] + vars_trace[9] * vars_remaining[17] * vars_remaining_next[13]) -
                          (vars_trace[6] * vars_remaining[16] * vars_remaining_next[16] + vars_trace[10] * vars_remaining[15] * vars_remaining_next[14] + vars_trace[8] * vars_remaining[17] * vars_remaining_next[12]);

        // (507)
        residual[12] =	2.0 *	(vars_trace[2] * vars_remaining[0] * vars_remaining_next[10] + vars_trace[0] * vars_remaining[4] * vars_remaining_next[8] + vars_trace[4] * vars_remaining[2] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[3] * vars_remaining[1] * vars_remaining_next[11] + vars_trace[1] * vars_remaining[5] * vars_remaining_next[9] + vars_trace[5] * vars_remaining[3] * vars_remaining_next[7]) +
                          (vars_trace[9] * vars_remaining[7] * vars_remaining_next[17] + vars_trace[7] * vars_remaining[11] * vars_remaining_next[15] + vars_trace[11] * vars_remaining[9] * vars_remaining_next[13]) -
                          (vars_trace[8] * vars_remaining[6] * vars_remaining_next[16] + vars_trace[6] * vars_remaining[10] * vars_remaining_next[14] + vars_trace[10] * vars_remaining[8] * vars_remaining_next[12]);
        // (607)
        residual[13] =	2.0 *	(vars_trace[2] * vars_remaining[1] * vars_remaining_next[10] + vars_trace[0] * vars_remaining[5] * vars_remaining_next[8] + vars_trace[4] * vars_remaining[3] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[3] * vars_remaining[0] * vars_remaining_next[11] + vars_trace[1] * vars_remaining[4] * vars_remaining_next[9] + vars_trace[5] * vars_remaining[2] * vars_remaining_next[7]) -
                          2.0 * 	(vars_trace[9] * vars_remaining[6] * vars_remaining_next[17] + vars_trace[7] * vars_remaining[10] * vars_remaining_next[15] + vars_trace[11] * vars_remaining[8] * vars_remaining_next[13]) +
                          2.0 * 	(vars_trace[8] * vars_remaining[7] * vars_remaining_next[16] + vars_trace[6] * vars_remaining[11] * vars_remaining_next[14] + vars_trace[10] * vars_remaining[9] * vars_remaining_next[12]) - 3.0;
        residual[13] = T{0};
        // (307)
        residual[14] =	2.0 *	(vars_trace[2] * vars_remaining[12] * vars_remaining_next[10] + vars_trace[0] * vars_remaining[14] * vars_remaining_next[8] + vars_trace[4] * vars_remaining[13] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[3] * vars_remaining[12] * vars_remaining_next[11] + vars_trace[1] * vars_remaining[14] * vars_remaining_next[9] + vars_trace[5] * vars_remaining[13] * vars_remaining_next[7]) +
                          (vars_trace[9] * vars_remaining[15] * vars_remaining_next[17] + vars_trace[7] * vars_remaining[17] * vars_remaining_next[15] + vars_trace[11] * vars_remaining[16] * vars_remaining_next[13]) -
                          (vars_trace[8] * vars_remaining[15] * vars_remaining_next[16] + vars_trace[6] * vars_remaining[17] * vars_remaining_next[14] + vars_trace[10] * vars_remaining[16] * vars_remaining_next[12]);

        // (257)
        residual[15] =	2.0 *	(vars_trace[1] * vars_remaining[2] * vars_remaining_next[10] + vars_trace[5] * vars_remaining[0] * vars_remaining_next[8] + vars_trace[3] * vars_remaining[4] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[0] * vars_remaining[3] * vars_remaining_next[11] + vars_trace[4] * vars_remaining[1] * vars_remaining_next[9] + vars_trace[2] * vars_remaining[5] * vars_remaining_next[7]) +
                          (vars_trace[6] * vars_remaining[9] * vars_remaining_next[17] + vars_trace[10] * vars_remaining[7] * vars_remaining_next[15] + vars_trace[8] * vars_remaining[11] * vars_remaining_next[13]) -
                          (vars_trace[7] * vars_remaining[8] * vars_remaining_next[16] + vars_trace[11] * vars_remaining[6] * vars_remaining_next[14] + vars_trace[9] * vars_remaining[10] * vars_remaining_next[12]);
        // (267)
        residual[16] =	2.0 *	(vars_trace[1] * vars_remaining[3] * vars_remaining_next[10] + vars_trace[5] * vars_remaining[1] * vars_remaining_next[8] + vars_trace[3] * vars_remaining[5] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[0] * vars_remaining[2] * vars_remaining_next[11] + vars_trace[4] * vars_remaining[0] * vars_remaining_next[9] + vars_trace[2] * vars_remaining[4] * vars_remaining_next[7]) +
                          (vars_trace[6] * vars_remaining[8] * vars_remaining_next[17] + vars_trace[10] * vars_remaining[6] * vars_remaining_next[15] + vars_trace[8] * vars_remaining[10] * vars_remaining_next[13]) -
                          (vars_trace[7] * vars_remaining[9] * vars_remaining_next[16] + vars_trace[11] * vars_remaining[7] * vars_remaining_next[14] + vars_trace[9] * vars_remaining[11] * vars_remaining_next[12]);
        // (237)
        residual[17] =	2.0 *	(vars_trace[1] * vars_remaining[13] * vars_remaining_next[10] + vars_trace[5] * vars_remaining[12] * vars_remaining_next[8] + vars_trace[3] * vars_remaining[14] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[0] * vars_remaining[13] * vars_remaining_next[11] + vars_trace[4] * vars_remaining[12] * vars_remaining_next[9] + vars_trace[2] * vars_remaining[14] * vars_remaining_next[7]) -
                          2.0 *	(vars_trace[6] * vars_remaining[16] * vars_remaining_next[17] + vars_trace[10] * vars_remaining[15] * vars_remaining_next[15] + vars_trace[8] * vars_remaining[17] * vars_remaining_next[13]) +
                          2.0 *	(vars_trace[7] * vars_remaining[16] * vars_remaining_next[16] + vars_trace[11] * vars_remaining[15] * vars_remaining_next[14] + vars_trace[9] * vars_remaining[17] * vars_remaining_next[12]) - 3.0;

        // (527)
        residual[18] =	2.0 *	(vars_trace[3] * vars_remaining[0] * vars_remaining_next[10] + vars_trace[1] * vars_remaining[4] * vars_remaining_next[8] + vars_trace[5] * vars_remaining[2] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[2] * vars_remaining[1] * vars_remaining_next[11] + vars_trace[0] * vars_remaining[5] * vars_remaining_next[9] + vars_trace[4] * vars_remaining[3] * vars_remaining_next[7]) +
                          (vars_trace[8] * vars_remaining[7] * vars_remaining_next[17] + vars_trace[6] * vars_remaining[11] * vars_remaining_next[15] + vars_trace[10] * vars_remaining[9] * vars_remaining_next[13]) -
                          (vars_trace[9] * vars_remaining[6] * vars_remaining_next[16] + vars_trace[7] * vars_remaining[10] * vars_remaining_next[14] + vars_trace[11] * vars_remaining[8] * vars_remaining_next[12]);

        // (627)
        residual[19] =	2.0 *	(vars_trace[3] * vars_remaining[1] * vars_remaining_next[10] + vars_trace[1] * vars_remaining[5] * vars_remaining_next[8] + vars_trace[5] * vars_remaining[3] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[2] * vars_remaining[0] * vars_remaining_next[11] + vars_trace[0] * vars_remaining[4] * vars_remaining_next[9] + vars_trace[4] * vars_remaining[2] * vars_remaining_next[7]) +
                          (vars_trace[8] * vars_remaining[6] * vars_remaining_next[17] + vars_trace[6] * vars_remaining[10] * vars_remaining_next[15] + vars_trace[10] * vars_remaining[8] * vars_remaining_next[13]) -
                          (vars_trace[9] * vars_remaining[7] * vars_remaining_next[16] + vars_trace[7] * vars_remaining[11] * vars_remaining_next[14] + vars_trace[11] * vars_remaining[9] * vars_remaining_next[12]);

        // (327)
        residual[20] =	2.0 *	(vars_trace[3] * vars_remaining[12] * vars_remaining_next[10] + vars_trace[1] * vars_remaining[14] * vars_remaining_next[8] + vars_trace[5] * vars_remaining[13] * vars_remaining_next[6]) -
                          2.0 *	(vars_trace[2] * vars_remaining[12] * vars_remaining_next[11] + vars_trace[0] * vars_remaining[14] * vars_remaining_next[9] + vars_trace[4] * vars_remaining[13] * vars_remaining_next[7]) -
                          2.0 * 	(vars_trace[8] * vars_remaining[15] * vars_remaining_next[17] + vars_trace[6] * vars_remaining[17] * vars_remaining_next[15] + vars_trace[10] * vars_remaining[16] * vars_remaining_next[13]) +
                          2.0 * 	(vars_trace[9] * vars_remaining[15] * vars_remaining_next[16] + vars_trace[7] * vars_remaining[17] * vars_remaining_next[14] + vars_trace[11] * vars_remaining[16] * vars_remaining_next[12]) + 3.0;
        residual[20] = T{0};

        // (157) ----
        residual[21] =	2.0 *	(vars_remaining[2] * vars_remaining_next[10] + vars_remaining[0] * vars_remaining_next[8] + vars_remaining[4] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[3] * vars_remaining_next[11] + vars_remaining[1] * vars_remaining_next[9] + vars_remaining[5] * vars_remaining_next[7]) -
                          2.0 *	(vars_remaining[9] * vars_remaining_next[17] + vars_remaining[7] * vars_remaining_next[15] + vars_remaining[11] * vars_remaining_next[13]) +
                          2.0 * 	(vars_remaining[8] * vars_remaining_next[16] + vars_remaining[6] * vars_remaining_next[14] + vars_remaining[10] * vars_remaining_next[12]);
        // (167)
        residual[22] =	2.0 *	(vars_remaining[3] * vars_remaining_next[10] + vars_remaining[1] * vars_remaining_next[8] + vars_remaining[5] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[2] * vars_remaining_next[11] + vars_remaining[0] * vars_remaining_next[9] + vars_remaining[4] * vars_remaining_next[7]) +
                          (vars_remaining[8] * vars_remaining_next[17] + vars_remaining[6] * vars_remaining_next[15] + vars_remaining[10] * vars_remaining_next[13]) -
                          (vars_remaining[9] * vars_remaining_next[16] + vars_remaining[7] * vars_remaining_next[14] + vars_remaining[11] * vars_remaining_next[12]);
        // (137)
        residual[23] =	2.0 *	(vars_remaining[13] * vars_remaining_next[10] + vars_remaining[12] * vars_remaining_next[8] + vars_remaining[14] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[13] * vars_remaining_next[11] + vars_remaining[12] * vars_remaining_next[9] + vars_remaining[14] * vars_remaining_next[7]) +
                          (vars_remaining[16] * vars_remaining_next[17] + vars_remaining[15] * vars_remaining_next[15] + vars_remaining[17] * vars_remaining_next[13]) -
                          (vars_remaining[16] * vars_remaining_next[16] + vars_remaining[15] * vars_remaining_next[14] + vars_remaining[17] * vars_remaining_next[12]);
        // residual[23] = T{0};
        // (517) ----
        residual[24] =	 2.0 *	(vars_remaining[0] * vars_remaining_next[10] + vars_remaining[4] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[1] * vars_remaining_next[11] + vars_remaining[5] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining_next[7]) -
                          2.0 *	(vars_remaining[7] * vars_remaining_next[17] + vars_remaining[11] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining_next[13]) +
                          2.0 * 	(vars_remaining[6] * vars_remaining_next[16] + vars_remaining[10] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining_next[12]);
        residual[24] = T{0};
        // (617)
        residual[25] =	2.0 *	(vars_remaining[1] * vars_remaining_next[10] + vars_remaining[5] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[0] * vars_remaining_next[11] + vars_remaining[4] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining_next[7]) +
                          (vars_remaining[6] * vars_remaining_next[17] + vars_remaining[10] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining_next[13]) -
                          (vars_remaining[7] * vars_remaining_next[16] + vars_remaining[11] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining_next[12]);

        // (317)
        residual[26] =	2.0 *	(vars_remaining[12] * vars_remaining_next[10] + vars_remaining[14] * vars_remaining_next[8] + vars_remaining[13] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[12] * vars_remaining_next[11] + vars_remaining[14] * vars_remaining_next[9] + vars_remaining[13] * vars_remaining_next[7]) +
                          (vars_remaining[15] * vars_remaining_next[17] + vars_remaining[17] * vars_remaining_next[15] + vars_remaining[16] * vars_remaining_next[13]) -
                          (vars_remaining[15] * vars_remaining_next[16] + vars_remaining[17] * vars_remaining_next[14] + vars_remaining[16] * vars_remaining_next[12]);

        // (077)
        residual[27] =	2.0 *	(vars_trace[0] * vars_remaining_next[8] * vars_remaining_next[10] + vars_trace[4] * vars_remaining_next[6] * vars_remaining_next[8] + vars_trace[2] * vars_remaining_next[6] * vars_remaining_next[10]) +
                          2.0 *	(vars_trace[1] * vars_remaining_next[9] * vars_remaining_next[11] + vars_trace[5] * vars_remaining_next[7] * vars_remaining_next[9] + vars_trace[3] * vars_remaining_next[7] * vars_remaining_next[11]) +
                          2.0 *	(vars_trace[7] * vars_remaining_next[15] * vars_remaining_next[17] + vars_trace[11] * vars_remaining_next[13] * vars_remaining_next[15] + vars_trace[9] * vars_remaining_next[13] * vars_remaining_next[17]) +
                          2.0 * 	(vars_trace[6] * vars_remaining_next[14] * vars_remaining_next[16] + vars_trace[10] * vars_remaining_next[12] * vars_remaining_next[14] + vars_trace[8] * vars_remaining_next[12] * vars_remaining_next[16]) - 1.0;
        //residual[27] = T{0};

  
        // (078)
        residual[28] =	2.0 *	(vars_trace[0] * vars_remaining_next[8] * vars_remaining_next[11] + vars_trace[4] * vars_remaining_next[6] * vars_remaining_next[9] + vars_trace[2] * vars_remaining_next[7] * vars_remaining_next[10]) +
                          2.0 *	(vars_trace[1] * vars_remaining_next[9] * vars_remaining_next[10] + vars_trace[5] * vars_remaining_next[7] * vars_remaining_next[8] + vars_trace[3] * vars_remaining_next[6] * vars_remaining_next[11]) -
                          (vars_trace[7] * vars_remaining_next[15] * vars_remaining_next[16] + vars_trace[11] * vars_remaining_next[13] * vars_remaining_next[14] + vars_trace[9] * vars_remaining_next[12] * vars_remaining_next[17]) -
                          (vars_trace[6] * vars_remaining_next[14] * vars_remaining_next[17] + vars_trace[10] * vars_remaining_next[12] * vars_remaining_next[15] + vars_trace[8] * vars_remaining_next[13] * vars_remaining_next[16]);
        //residual[28] = T{0};

        
        // (277)
        residual[29] =	2.0 *	(vars_trace[1] * vars_remaining_next[8] * vars_remaining_next[10] + vars_trace[5] * vars_remaining_next[6] * vars_remaining_next[8] + vars_trace[3] * vars_remaining_next[6] * vars_remaining_next[10]) +
                          2.0 *	(vars_trace[0] * vars_remaining_next[9] * vars_remaining_next[11] + vars_trace[4] * vars_remaining_next[7] * vars_remaining_next[9] + vars_trace[2] * vars_remaining_next[7] * vars_remaining_next[11]) -
                          (vars_trace[6] * vars_remaining_next[15] * vars_remaining_next[17] + vars_trace[10] * vars_remaining_next[13] * vars_remaining_next[15] + vars_trace[8] * vars_remaining_next[13] * vars_remaining_next[17]) -
                          (vars_trace[7] * vars_remaining_next[14] * vars_remaining_next[16] + vars_trace[11] * vars_remaining_next[12] * vars_remaining_next[14] + vars_trace[9] * vars_remaining_next[12] * vars_remaining_next[16]);
        //residual[29] = T{0};

        // (278)
        residual[30] =	2.0 *	(vars_trace[1] * vars_remaining_next[8] * vars_remaining_next[11] + vars_trace[5] * vars_remaining_next[6] * vars_remaining_next[9] + vars_trace[3] * vars_remaining_next[7] * vars_remaining_next[10]) +
                          2.0 *	(vars_trace[0] * vars_remaining_next[9] * vars_remaining_next[10] + vars_trace[4] * vars_remaining_next[7] * vars_remaining_next[8] + vars_trace[2] * vars_remaining_next[6] * vars_remaining_next[11]) -
                          (vars_trace[6] * vars_remaining_next[15] * vars_remaining_next[16] + vars_trace[10] * vars_remaining_next[13] * vars_remaining_next[14] + vars_trace[8] * vars_remaining_next[12] * vars_remaining_next[17]) -
                          (vars_trace[7] * vars_remaining_next[14] * vars_remaining_next[17] + vars_trace[11] * vars_remaining_next[12] * vars_remaining_next[15] + vars_trace[9] * vars_remaining_next[13] * vars_remaining_next[16]);
        //residual[30] = T{0};
        // (177)
        residual[31] =	2.0 *	(vars_remaining_next[8] * vars_remaining_next[10] + vars_remaining_next[6] * vars_remaining_next[8] + vars_remaining_next[6] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining_next[9] * vars_remaining_next[11] + vars_remaining_next[7] * vars_remaining_next[9] + vars_remaining_next[7] * vars_remaining_next[11]) -
                          (vars_remaining_next[15] * vars_remaining_next[17] + vars_remaining_next[13] * vars_remaining_next[15] + vars_remaining_next[13] * vars_remaining_next[17]) -
                          (vars_remaining_next[14] * vars_remaining_next[16] + vars_remaining_next[12] * vars_remaining_next[14] + vars_remaining_next[12] * vars_remaining_next[16]);

        //residual[31] = T{0};
        // (178)
        residual[32] =	2.0 *	(vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining_next[6] * vars_remaining_next[9] + vars_remaining_next[7] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining_next[7] * vars_remaining_next[8] + vars_remaining_next[6] * vars_remaining_next[11]) +
                          2.0 * 	(vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining_next[13] * vars_remaining_next[14] + vars_remaining_next[12] * vars_remaining_next[17]) +
                          2.0 * 	(vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining_next[12] * vars_remaining_next[15] + vars_remaining_next[13] * vars_remaining_next[16]) + 2.0;
        //residual[32] = T{0};

        // (557)
        residual[33] =	2.0 *	(vars_remaining[0] * vars_remaining[2] * vars_remaining_next[10] + vars_remaining[0] * vars_remaining[4] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining[4] * vars_remaining_next[6]) -
                         2.0 *	(vars_remaining[1] * vars_remaining[3] * vars_remaining_next[11] + vars_remaining[1] * vars_remaining[5] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining[5] * vars_remaining_next[7]) +
                         (vars_remaining[7] * vars_remaining[9] * vars_remaining_next[17] + vars_remaining[7] * vars_remaining[11] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining[11] * vars_remaining_next[13]) -
                         (vars_remaining[6] * vars_remaining[8] * vars_remaining_next[16] + vars_remaining[6] * vars_remaining[10] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining[10] * vars_remaining_next[12]);

        // (567)
        residual[34] =	2.0 *	(vars_remaining[0] * vars_remaining[3] * vars_remaining_next[10] + vars_remaining[1] * vars_remaining[4] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining[5] * vars_remaining_next[6]) -
                         2.0 *	(vars_remaining[1] * vars_remaining[2] * vars_remaining_next[11] + vars_remaining[0] * vars_remaining[5] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining[4] * vars_remaining_next[7]) +
                         (vars_remaining[7] * vars_remaining[8] * vars_remaining_next[17] + vars_remaining[6] * vars_remaining[11] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining[10] * vars_remaining_next[13]) -
                         (vars_remaining[6] * vars_remaining[9] * vars_remaining_next[16] + vars_remaining[7] * vars_remaining[10] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining[11] * vars_remaining_next[12]);

        // (537)
        residual[35] =	2.0 *	(vars_remaining[0] * vars_remaining[13] * vars_remaining_next[10] + vars_remaining[4] * vars_remaining[12] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining[14] * vars_remaining_next[6]) -
                         2.0 *	(vars_remaining[1] * vars_remaining[13] * vars_remaining_next[11] + vars_remaining[5] * vars_remaining[12] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining[14] * vars_remaining_next[7]) -
                         2.0 * 	(vars_remaining[7] * vars_remaining[16] * vars_remaining_next[17] + vars_remaining[11] * vars_remaining[15] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining[17] * vars_remaining_next[13]) +
                         2.0 * 	(vars_remaining[6] * vars_remaining[16] * vars_remaining_next[16] + vars_remaining[10] * vars_remaining[15] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining[17] * vars_remaining_next[12]) + 3.0;

        // (657)
        residual[36] =	2.0 *	(vars_remaining[1] * vars_remaining[2] * vars_remaining_next[10] + vars_remaining[0] * vars_remaining[5] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining[4] * vars_remaining_next[6]) -
                         2.0 *	(vars_remaining[0] * vars_remaining[3] * vars_remaining_next[11] + vars_remaining[1] * vars_remaining[4] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining[5] * vars_remaining_next[7]) +
                         (vars_remaining[6] * vars_remaining[9] * vars_remaining_next[17] + vars_remaining[7] * vars_remaining[10] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining[11] * vars_remaining_next[13]) -
                         (vars_remaining[7] * vars_remaining[8] * vars_remaining_next[16] + vars_remaining[6] * vars_remaining[11] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining[10] * vars_remaining_next[12]);
        // (667)
        residual[37] =	2.0 *	(vars_remaining[1] * vars_remaining[3] * vars_remaining_next[10] + vars_remaining[1] * vars_remaining[5] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining[5] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[0] * vars_remaining[2] * vars_remaining_next[11] + vars_remaining[0] * vars_remaining[4] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining[4] * vars_remaining_next[7]) -
                          2.0 * 	(vars_remaining[6] * vars_remaining[8] * vars_remaining_next[17] + vars_remaining[6] * vars_remaining[10] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining[10] * vars_remaining_next[13]) +
                          2.0 * 	(vars_remaining[7] * vars_remaining[9] * vars_remaining_next[16] + vars_remaining[7] * vars_remaining[11] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining[11] * vars_remaining_next[12]);
        residual[37] = T{0};
        // (637)
        residual[38] =	2.0 *	(vars_remaining[1] * vars_remaining[13] * vars_remaining_next[10] + vars_remaining[5] * vars_remaining[12] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining[14] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[0] * vars_remaining[13] * vars_remaining_next[11] + vars_remaining[4] * vars_remaining[12] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining[14] * vars_remaining_next[7]) +
                          (vars_remaining[6] * vars_remaining[16] * vars_remaining_next[17] + vars_remaining[10] * vars_remaining[15] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining[17] * vars_remaining_next[13]) -
                          (vars_remaining[7] * vars_remaining[16] * vars_remaining_next[16] + vars_remaining[11] * vars_remaining[15] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining[17] * vars_remaining_next[12]);

        // (357)
        residual[39] =	2.0 *	(vars_remaining[2] * vars_remaining[12] * vars_remaining_next[10] + vars_remaining[0] * vars_remaining[14] * vars_remaining_next[8] + vars_remaining[4] * vars_remaining[13] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[3] * vars_remaining[12] * vars_remaining_next[11] + vars_remaining[1] * vars_remaining[14] * vars_remaining_next[9] + vars_remaining[5] * vars_remaining[13] * vars_remaining_next[7]) -
                          2.0 *	(vars_remaining[9] * vars_remaining[15] * vars_remaining_next[17] + vars_remaining[7] * vars_remaining[17] * vars_remaining_next[15] + vars_remaining[11] * vars_remaining[16] * vars_remaining_next[13]) +
                          2.0 * 	(vars_remaining[8] * vars_remaining[15] * vars_remaining_next[16] + vars_remaining[6] * vars_remaining[17] * vars_remaining_next[14] + vars_remaining[10] * vars_remaining[16] * vars_remaining_next[12]) - 3.0;
        residual[39] = T{0};
        // (367)
        residual[40] =	2.0 *	(vars_remaining[3] * vars_remaining[12] * vars_remaining_next[10] + vars_remaining[1] * vars_remaining[14] * vars_remaining_next[8] + vars_remaining[5] * vars_remaining[13] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[2] * vars_remaining[12] * vars_remaining_next[11] + vars_remaining[0] * vars_remaining[14] * vars_remaining_next[9] + vars_remaining[4] * vars_remaining[13] * vars_remaining_next[7]) +
                          (vars_remaining[8] * vars_remaining[15] * vars_remaining_next[17] + vars_remaining[6] * vars_remaining[17] * vars_remaining_next[15] + vars_remaining[10] * vars_remaining[16] * vars_remaining_next[13]) -
                          (vars_remaining[9] * vars_remaining[15] * vars_remaining_next[16] + vars_remaining[7] * vars_remaining[17] * vars_remaining_next[14] + vars_remaining[11] * vars_remaining[16] * vars_remaining_next[12]);

        // (337)
        residual[41] =	2.0 *	(vars_remaining[12] * vars_remaining[13] * vars_remaining_next[10] + vars_remaining[12] * vars_remaining[14] * vars_remaining_next[8] + vars_remaining[13] * vars_remaining[14] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining[12] * vars_remaining[13] * vars_remaining_next[11] + vars_remaining[12] * vars_remaining[14] * vars_remaining_next[9] + vars_remaining[13] * vars_remaining[14] * vars_remaining_next[7]) +
                          (vars_remaining[15] * vars_remaining[16] * vars_remaining_next[17] + vars_remaining[15] * vars_remaining[17] * vars_remaining_next[15] + vars_remaining[16] * vars_remaining[17] * vars_remaining_next[13]) -
                          (vars_remaining[15] * vars_remaining[16] * vars_remaining_next[16] + vars_remaining[15] * vars_remaining[17] * vars_remaining_next[14] + vars_remaining[16] * vars_remaining[17] * vars_remaining_next[12]);

        // (577)
        residual[42] =	2.0 *	(vars_remaining[0] * vars_remaining_next[8] * vars_remaining_next[10] + vars_remaining[4] * vars_remaining_next[6] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining_next[6] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining[1] * vars_remaining_next[9] * vars_remaining_next[11] + vars_remaining[5] * vars_remaining_next[7] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining_next[7] * vars_remaining_next[11]) -
                          (vars_remaining[7] * vars_remaining_next[15] * vars_remaining_next[17] + vars_remaining[11] * vars_remaining_next[13] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining_next[13] * vars_remaining_next[17]) -
                          (vars_remaining[6] * vars_remaining_next[14] * vars_remaining_next[16] + vars_remaining[10] * vars_remaining_next[12] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining_next[12] * vars_remaining_next[16]);

        // residual[42] = T{0};
        // (578)
        residual[43] =	2.0 *	(vars_remaining[0] * vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining[4] * vars_remaining_next[6] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining_next[7] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining[1] * vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining[5] * vars_remaining_next[7] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining_next[6] * vars_remaining_next[11]) -
                          (vars_remaining[7] * vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining[11] * vars_remaining_next[13] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining_next[12] * vars_remaining_next[17]) -
                          (vars_remaining[6] * vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining[10] * vars_remaining_next[12] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining_next[13] * vars_remaining_next[16]);
        //residual[43] = T{0};

        // (677)
        residual[44] =	2.0 *	(vars_remaining[1] * vars_remaining_next[8] * vars_remaining_next[10] + vars_remaining[5] * vars_remaining_next[6] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining_next[6] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining[0] * vars_remaining_next[9] * vars_remaining_next[11] + vars_remaining[4] * vars_remaining_next[7] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining_next[7] * vars_remaining_next[11]) +
                          2.0 * 	(vars_remaining[6] * vars_remaining_next[15] * vars_remaining_next[17] + vars_remaining[10] * vars_remaining_next[13] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining_next[13] * vars_remaining_next[17]) +
                          2.0 * 	(vars_remaining[7] * vars_remaining_next[14] * vars_remaining_next[16] + vars_remaining[11] * vars_remaining_next[12] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining_next[12] * vars_remaining_next[16]) - 2.0;

        // residual[44] = T{0};

        // (678)
        residual[45] =	2.0 *	(vars_remaining[1] * vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining[5] * vars_remaining_next[6] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining_next[7] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining[0] * vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining[4] * vars_remaining_next[7] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining_next[6] * vars_remaining_next[11]) -
                          (vars_remaining[6] * vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining[10] * vars_remaining_next[13] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining_next[12] * vars_remaining_next[17]) -
                          (vars_remaining[7] * vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining[11] * vars_remaining_next[12] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining_next[13] * vars_remaining_next[16]);
        //residual[45] = T{0};

        // (377)
        residual[46] =	2.0 *	(vars_remaining[12] * vars_remaining_next[8] * vars_remaining_next[10] + vars_remaining[14] * vars_remaining_next[6] * vars_remaining_next[8] + vars_remaining[13] * vars_remaining_next[6] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining[12] * vars_remaining_next[9] * vars_remaining_next[11] + vars_remaining[14] * vars_remaining_next[7] * vars_remaining_next[9] + vars_remaining[13] * vars_remaining_next[7] * vars_remaining_next[11]) -
                          (vars_remaining[15] * vars_remaining_next[15] * vars_remaining_next[17] + vars_remaining[17] * vars_remaining_next[13] * vars_remaining_next[15] + vars_remaining[16] * vars_remaining_next[13] * vars_remaining_next[17]) -
                          (vars_remaining[15] * vars_remaining_next[14] * vars_remaining_next[16] + vars_remaining[17] * vars_remaining_next[12] * vars_remaining_next[14] + vars_remaining[16] * vars_remaining_next[12] * vars_remaining_next[16]);
        //residual[46] = T{0};


        // (378)
        residual[47] =	2.0 *	(vars_remaining[12] * vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining[14] * vars_remaining_next[6] * vars_remaining_next[9] + vars_remaining[13] * vars_remaining_next[7] * vars_remaining_next[10]) +
                          2.0 *	(vars_remaining[12] * vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining[14] * vars_remaining_next[7] * vars_remaining_next[8] + vars_remaining[13] * vars_remaining_next[6] * vars_remaining_next[11]) +
                          2.0 *	(vars_remaining[15] * vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining[17] * vars_remaining_next[13] * vars_remaining_next[14] + vars_remaining[16] * vars_remaining_next[12] * vars_remaining_next[17]) +
                          2.0 *	(vars_remaining[15] * vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining[17] * vars_remaining_next[12] * vars_remaining_next[15] + vars_remaining[16] * vars_remaining_next[13] * vars_remaining_next[16]) + 1.0;
        // residual[47] = T{0};


        // (777) ----
        residual[48] =	6.0 *	(vars_remaining_next[6] * vars_remaining_next[8] * vars_remaining_next[10]) -
                          6.0 *	(vars_remaining_next[7] * vars_remaining_next[9] * vars_remaining_next[11]) -
                          6.0 *	(vars_remaining_next[13] * vars_remaining_next[15] * vars_remaining_next[17]) +
                          6.0 *	(vars_remaining_next[12] * vars_remaining_next[14] * vars_remaining_next[16]);
        // residual[48] = T{0};


        // (778)
        residual[49] =	2.0 *	(vars_remaining_next[6] * vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining_next[6] * vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining_next[7] * vars_remaining_next[8] * vars_remaining_next[10]) -
                          2.0 *	(vars_remaining_next[7] * vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining_next[7] * vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining_next[6] * vars_remaining_next[9] * vars_remaining_next[11]) +
                          (vars_remaining_next[13] * vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining_next[13] * vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining_next[12] * vars_remaining_next[15] * vars_remaining_next[17]) -
                          (vars_remaining_next[12] * vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining_next[12] * vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining_next[13] * vars_remaining_next[14] * vars_remaining_next[16]);
        // residual[49] = T{0};


        return true;
    }
};

struct ExpandedSystem4 {
    template <typename T> bool operator()(const T* const vars_trace, const T* const vars_remaining, const T* vars_remaining_next, T* residual) const {

        for (int i = 0; i <= 14; ++i)
            residual[i] = T{0};

        // (547) ----
        residual[0] =	2.0 *	(vars_remaining[0] * vars_remaining_next[1] * vars_remaining_next[10] + vars_remaining[4] * vars_remaining_next[0] * vars_remaining_next[8] + vars_remaining[2] * vars_remaining_next[2] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining[1] * vars_remaining_next[1] * vars_remaining_next[11] + vars_remaining[5] * vars_remaining_next[0] * vars_remaining_next[9] + vars_remaining[3] * vars_remaining_next[2] * vars_remaining_next[7]) +
                          2.0 * 	(vars_remaining[7] * vars_remaining_next[4] * vars_remaining_next[17] + vars_remaining[11] * vars_remaining_next[3] * vars_remaining_next[15] + vars_remaining[9] * vars_remaining_next[5] * vars_remaining_next[13]) +
                          2.0 * 	(vars_remaining[6] * vars_remaining_next[4] * vars_remaining_next[16] + vars_remaining[10] * vars_remaining_next[3] * vars_remaining_next[14] + vars_remaining[8] * vars_remaining_next[5] * vars_remaining_next[12]) + 1.0;
        residual[0] = T{0};
      
        // (647)
        residual[1] =	2.0 *	(vars_remaining[1] * vars_remaining_next[1] * vars_remaining_next[10] + vars_remaining[5] * vars_remaining_next[0] * vars_remaining_next[8] + vars_remaining[3] * vars_remaining_next[2] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining[0] * vars_remaining_next[1] * vars_remaining_next[11] + vars_remaining[4] * vars_remaining_next[0] * vars_remaining_next[9] + vars_remaining[2] * vars_remaining_next[2] * vars_remaining_next[7]) -
                          (vars_remaining[6] * vars_remaining_next[4] * vars_remaining_next[17] + vars_remaining[10] * vars_remaining_next[3] * vars_remaining_next[15] + vars_remaining[8] * vars_remaining_next[5] * vars_remaining_next[13]) -
                          (vars_remaining[7] * vars_remaining_next[4] * vars_remaining_next[16] + vars_remaining[11] * vars_remaining_next[3] * vars_remaining_next[14] + vars_remaining[9] * vars_remaining_next[5] * vars_remaining_next[12]);
        // (347)
        residual[2] =	2.0 *	(vars_remaining[12] * vars_remaining_next[1] * vars_remaining_next[10] + vars_remaining[14] * vars_remaining_next[0] * vars_remaining_next[8] + vars_remaining[13] * vars_remaining_next[2] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining[12] * vars_remaining_next[1] * vars_remaining_next[11] + vars_remaining[14] * vars_remaining_next[0] * vars_remaining_next[9] + vars_remaining[13] * vars_remaining_next[2] * vars_remaining_next[7]) -
                          (vars_remaining[15] * vars_remaining_next[4] * vars_remaining_next[17] + vars_remaining[17] * vars_remaining_next[3] * vars_remaining_next[15] + vars_remaining[16] * vars_remaining_next[5] * vars_remaining_next[13]) -
                          (vars_remaining[15] * vars_remaining_next[4] * vars_remaining_next[16] + vars_remaining[17] * vars_remaining_next[3] * vars_remaining_next[14] + vars_remaining[16] * vars_remaining_next[5] * vars_remaining_next[12]);


        // (457)
        residual[3] =	2.0 *	(vars_remaining[2] * vars_remaining_next[0] * vars_remaining_next[10] + vars_remaining[0] * vars_remaining_next[2] * vars_remaining_next[8] + vars_remaining[4] * vars_remaining_next[1] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining[3] * vars_remaining_next[0] * vars_remaining_next[11] + vars_remaining[1] * vars_remaining_next[2] * vars_remaining_next[9] + vars_remaining[5] * vars_remaining_next[1] * vars_remaining_next[7]) +
                          2.0 * 	(vars_remaining[9] * vars_remaining_next[3] * vars_remaining_next[17] + vars_remaining[7] * vars_remaining_next[5] * vars_remaining_next[15] + vars_remaining[11] * vars_remaining_next[4] * vars_remaining_next[13]) +
                          2.0 * 	(vars_remaining[8] * vars_remaining_next[3] * vars_remaining_next[16] + vars_remaining[6] * vars_remaining_next[5] * vars_remaining_next[14] + vars_remaining[10] * vars_remaining_next[4] * vars_remaining_next[12]) + 1.0;
        residual[3] = T{0};
        // (467)
        residual[4] =	2.0 *	(vars_remaining[3] * vars_remaining_next[0] * vars_remaining_next[10] + vars_remaining[1] * vars_remaining_next[2] * vars_remaining_next[8] + vars_remaining[5] * vars_remaining_next[1] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining[2] * vars_remaining_next[0] * vars_remaining_next[11] + vars_remaining[0] * vars_remaining_next[2] * vars_remaining_next[9] + vars_remaining[4] * vars_remaining_next[1] * vars_remaining_next[7]) -
                          (vars_remaining[8] * vars_remaining_next[3] * vars_remaining_next[17] + vars_remaining[6] * vars_remaining_next[5] * vars_remaining_next[15] + vars_remaining[10] * vars_remaining_next[4] * vars_remaining_next[13]) -
                          (vars_remaining[9] * vars_remaining_next[3] * vars_remaining_next[16] + vars_remaining[7] * vars_remaining_next[5] * vars_remaining_next[14] + vars_remaining[11] * vars_remaining_next[4] * vars_remaining_next[12]);
        // (437)
        residual[5] =	2.0 *	(vars_remaining[13] * vars_remaining_next[0] * vars_remaining_next[10] + vars_remaining[12] * vars_remaining_next[2] * vars_remaining_next[8] + vars_remaining[14] * vars_remaining_next[1] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining[13] * vars_remaining_next[0] * vars_remaining_next[11] + vars_remaining[12] * vars_remaining_next[2] * vars_remaining_next[9] + vars_remaining[14] * vars_remaining_next[1] * vars_remaining_next[7]) -
                          (vars_remaining[16] * vars_remaining_next[3] * vars_remaining_next[17] + vars_remaining[15] * vars_remaining_next[5] * vars_remaining_next[15] + vars_remaining[17] * vars_remaining_next[4] * vars_remaining_next[13]) -
                          (vars_remaining[16] * vars_remaining_next[3] * vars_remaining_next[16] + vars_remaining[15] * vars_remaining_next[5] * vars_remaining_next[14] + vars_remaining[17] * vars_remaining_next[4] * vars_remaining_next[12]);

        // (447)
        residual[6] =	2.0 *	(vars_remaining_next[0] * vars_remaining_next[1] * vars_remaining_next[10] + vars_remaining_next[0] * vars_remaining_next[2] * vars_remaining_next[8] + vars_remaining_next[1] * vars_remaining_next[2] * vars_remaining_next[6]) -
                          2.0 *	(vars_remaining_next[0] * vars_remaining_next[1] * vars_remaining_next[11] + vars_remaining_next[0] * vars_remaining_next[2] * vars_remaining_next[9] + vars_remaining_next[1] * vars_remaining_next[2] * vars_remaining_next[7]) +
                          (vars_remaining_next[3] * vars_remaining_next[4] * vars_remaining_next[17] + vars_remaining_next[3] * vars_remaining_next[5] * vars_remaining_next[15] + vars_remaining_next[4] * vars_remaining_next[5] * vars_remaining_next[13]) -
                          (vars_remaining_next[3] * vars_remaining_next[4] * vars_remaining_next[16] + vars_remaining_next[3] * vars_remaining_next[5] * vars_remaining_next[14] + vars_remaining_next[4] * vars_remaining_next[5] * vars_remaining_next[12]);


        // (477)
        residual[7] =	2.0 *	(vars_remaining_next[0] * vars_remaining_next[8] * vars_remaining_next[10] + vars_remaining_next[2] * vars_remaining_next[6] * vars_remaining_next[8] + vars_remaining_next[1] * vars_remaining_next[6] * vars_remaining_next[10]) -
                          2.0 *	(vars_remaining_next[0] * vars_remaining_next[9] * vars_remaining_next[11] + vars_remaining_next[2] * vars_remaining_next[7] * vars_remaining_next[9] + vars_remaining_next[1] * vars_remaining_next[7] * vars_remaining_next[11]) +
                          (vars_remaining_next[3] * vars_remaining_next[15] * vars_remaining_next[17] + vars_remaining_next[5] * vars_remaining_next[13] * vars_remaining_next[15] + vars_remaining_next[4] * vars_remaining_next[13] * vars_remaining_next[17]) -
                          (vars_remaining_next[3] * vars_remaining_next[14] * vars_remaining_next[16] + vars_remaining_next[5] * vars_remaining_next[12] * vars_remaining_next[14] + vars_remaining_next[4] * vars_remaining_next[12] * vars_remaining_next[16]);
        //residual[7] = T{0};

        // (478)
        residual[8] =	2.0 *	(vars_remaining_next[0] * vars_remaining_next[8] * vars_remaining_next[11] + vars_remaining_next[2] * vars_remaining_next[6] * vars_remaining_next[9] + vars_remaining_next[1] * vars_remaining_next[7] * vars_remaining_next[10]) -
                          2.0 *	(vars_remaining_next[0] * vars_remaining_next[9] * vars_remaining_next[10] + vars_remaining_next[2] * vars_remaining_next[7] * vars_remaining_next[8] + vars_remaining_next[1] * vars_remaining_next[6] * vars_remaining_next[11]) -
                          2.0 *	(vars_remaining_next[3] * vars_remaining_next[15] * vars_remaining_next[16] + vars_remaining_next[5] * vars_remaining_next[13] * vars_remaining_next[14] + vars_remaining_next[4] * vars_remaining_next[12] * vars_remaining_next[17]) +
                          2.0 *	(vars_remaining_next[3] * vars_remaining_next[14] * vars_remaining_next[17] + vars_remaining_next[5] * vars_remaining_next[12] * vars_remaining_next[15] + vars_remaining_next[4] * vars_remaining_next[13] * vars_remaining_next[16]) - 1.0;

        // residual[8] = T{0};


        // (047)
        residual[9] =	2.0 *	(vars_trace[0] * vars_remaining_next[1] * vars_remaining_next[10] + vars_trace[4] * vars_remaining_next[0] * vars_remaining_next[8] + vars_trace[2] * vars_remaining_next[2] * vars_remaining_next[6]) +
                          2.0 *	(vars_trace[1] * vars_remaining_next[1] * vars_remaining_next[11] + vars_trace[5] * vars_remaining_next[0] * vars_remaining_next[9] + vars_trace[3] * vars_remaining_next[2] * vars_remaining_next[7]) -
                          (vars_trace[7] * vars_remaining_next[4] * vars_remaining_next[17] + vars_trace[11] * vars_remaining_next[3] * vars_remaining_next[15] + vars_trace[9] * vars_remaining_next[5] * vars_remaining_next[13]) -
                          (vars_trace[6] * vars_remaining_next[4] * vars_remaining_next[16] + vars_trace[10] * vars_remaining_next[3] * vars_remaining_next[14] + vars_trace[8] * vars_remaining_next[5] * vars_remaining_next[12]);

        // (247) ----
        residual[10] =	2.0 *	(vars_trace[1] * vars_remaining_next[1] * vars_remaining_next[10] + vars_trace[5] * vars_remaining_next[0] * vars_remaining_next[8] + vars_trace[3] * vars_remaining_next[2] * vars_remaining_next[6]) +
                          2.0 *	(vars_trace[0] * vars_remaining_next[1] * vars_remaining_next[11] + vars_trace[4] * vars_remaining_next[0] * vars_remaining_next[9] + vars_trace[2] * vars_remaining_next[2] * vars_remaining_next[7]) +
                          2.   *		(vars_trace[6] * vars_remaining_next[4] * vars_remaining_next[17] + vars_trace[10] * vars_remaining_next[3] * vars_remaining_next[15] + vars_trace[8] * vars_remaining_next[5] * vars_remaining_next[13]) +
                          2.	*	(vars_trace[7] * vars_remaining_next[4] * vars_remaining_next[16] + vars_trace[11] * vars_remaining_next[3] * vars_remaining_next[14] + vars_trace[9] * vars_remaining_next[5] * vars_remaining_next[12]) - 1.0;
        residual[10] = T{0};

        // (147)
        residual[11] =	2.0 *	(vars_remaining_next[1] * vars_remaining_next[10] + vars_remaining_next[0] * vars_remaining_next[8] + vars_remaining_next[2] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining_next[1] * vars_remaining_next[11] + vars_remaining_next[0] * vars_remaining_next[9] + vars_remaining_next[2] * vars_remaining_next[7]) -
                          (vars_remaining_next[4] * vars_remaining_next[17] + vars_remaining_next[3] * vars_remaining_next[15] + vars_remaining_next[5] * vars_remaining_next[13]) -
                          (vars_remaining_next[4] * vars_remaining_next[16] + vars_remaining_next[3] * vars_remaining_next[14] + vars_remaining_next[5] * vars_remaining_next[12]);

        // (407)
        residual[12] =	2.0 *	(vars_trace[2] * vars_remaining_next[0] * vars_remaining_next[10] + vars_trace[0] * vars_remaining_next[2] * vars_remaining_next[8] + vars_trace[4] * vars_remaining_next[1] * vars_remaining_next[6]) +
                          2.0 *	(vars_trace[3] * vars_remaining_next[0] * vars_remaining_next[11] + vars_trace[1] * vars_remaining_next[2] * vars_remaining_next[9] + vars_trace[5] * vars_remaining_next[1] * vars_remaining_next[7]) -
                          (vars_trace[9] * vars_remaining_next[3] * vars_remaining_next[17] + vars_trace[7] * vars_remaining_next[5] * vars_remaining_next[15] + vars_trace[11] * vars_remaining_next[4] * vars_remaining_next[13]) -
                          (vars_trace[8] * vars_remaining_next[3] * vars_remaining_next[16] + vars_trace[6] * vars_remaining_next[5] * vars_remaining_next[14] + vars_trace[10] * vars_remaining_next[4] * vars_remaining_next[12]);

        // (427)
        residual[13] =	2.0 *	(vars_trace[3] * vars_remaining_next[0] * vars_remaining_next[10] + vars_trace[1] * vars_remaining_next[2] * vars_remaining_next[8] + vars_trace[5] * vars_remaining_next[1] * vars_remaining_next[6]) +
                          2.0 *	(vars_trace[2] * vars_remaining_next[0] * vars_remaining_next[11] + vars_trace[0] * vars_remaining_next[2] * vars_remaining_next[9] + vars_trace[4] * vars_remaining_next[1] * vars_remaining_next[7]) +
                          2.0 *		(vars_trace[8] * vars_remaining_next[3] * vars_remaining_next[17] + vars_trace[6] * vars_remaining_next[5] * vars_remaining_next[15] + vars_trace[10] * vars_remaining_next[4] * vars_remaining_next[13]) +
                          2.0 *		(vars_trace[9] * vars_remaining_next[3] * vars_remaining_next[16] + vars_trace[7] * vars_remaining_next[5] * vars_remaining_next[14] + vars_trace[11] * vars_remaining_next[4] * vars_remaining_next[12]) - 1.0;
        residual[13] = T{0};
        // (417)
        residual[14] =	2.0 *	(vars_remaining_next[0] * vars_remaining_next[10] + vars_remaining_next[2] * vars_remaining_next[8] + vars_remaining_next[1] * vars_remaining_next[6]) +
                          2.0 *	(vars_remaining_next[0] * vars_remaining_next[11] + vars_remaining_next[2] * vars_remaining_next[9] + vars_remaining_next[1] * vars_remaining_next[7]) -
                          (vars_remaining_next[3] * vars_remaining_next[17] + vars_remaining_next[5] * vars_remaining_next[15] + vars_remaining_next[4] * vars_remaining_next[13]) -
                          (vars_remaining_next[3] * vars_remaining_next[16] + vars_remaining_next[5] * vars_remaining_next[14] + vars_remaining_next[4] * vars_remaining_next[12]);

        return true;
    }
};



int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    std::mt19937 gen(time(NULL));
    std::uniform_real_distribution<double> rng(-100.0, 100.0);

    int iter_total;

    std::cout << "Enter number of iterations ";
    std::cin >> iter_total;

    // Определение набора переменных, матрицы производных системы и вектора невязок системы.
    double *trace = new double[12];
    double *remaining = new double[18];
    double *remaining_next = new double[18];
    double *trace_best = new double[12];
    double *remaining_best = new double[18];
    double *remaining_next_best = new double[18];

    double **der = new double*[2];
    double *res = new double[22];
    double *trace_res = new double[5];
    double *remaining_res = new double[34];

    double cost, best_cost;
    double full_cost, full_best_cost = 100;
    int cost_regions[7] = { 0, 0, 0, 0, 0, 0, 0 };
    std::time_t clock;

    // Задание системы.
    Problem system, full_system;

    // Объект system_cost_function использует ранее заданный SystemCostFunctor для вычисления невязки и производных.

    CostFunction* trace_cost_function =
            new AutoDiffCostFunction<TraceSystem, 5, 12>(new TraceSystem);

    CostFunction* expanded_cost_function =
            new AutoDiffCostFunction<ExpandedSystem, 34, 12, 18>(new ExpandedSystem);

    CostFunction* expanded_cost_function2 =
            new AutoDiffCostFunction<ExpandedSystem2, 20, 12, 18, 18>(new ExpandedSystem2);

    CostFunction* expanded_cost_function3 =
            new AutoDiffCostFunction<ExpandedSystem3, 50, 12, 18, 18>(new ExpandedSystem3);

    CostFunction* expanded_cost_function4 =
            new AutoDiffCostFunction<ExpandedSystem4, 15, 12, 18, 18>(new ExpandedSystem4);

    // Добавление уравнений в систему.
    for (Problem* system: {&system/*, &full_system*/}) { 
      system->AddResidualBlock(trace_cost_function, NULL, trace);
      system->AddResidualBlock(expanded_cost_function, NULL, trace, remaining);
      system->AddResidualBlock(expanded_cost_function2, NULL, trace, remaining, remaining_next);
      //system->AddResidualBlock(expanded_cost_function3, NULL, trace, remaining, remaining_next);
      //system->AddResidualBlock(expanded_cost_function4, NULL, trace, remaining, remaining_next);
    }
    //full_system.AddResidualBlock(expanded_cost_function3, NULL, trace, remaining, remaining_next);
    //full_system.AddResidualBlock(expanded_cost_function4, NULL, trace, remaining, remaining_next);
    for (int i = 0; i < 12; i++) {
        trace[i] = i;
    }

    for (int i = 0; i < 18; i++) {
        remaining[i] = i;
    }

    for (int i = 0; i < 18; i++) {
        remaining_next[i] = i;
    }

    // Настройки алгоритма решения. Текущие настройки представляют собой
    // явно прописанные настройки по умолчанию.
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.logging_type = ceres::PER_MINIMIZER_ITERATION;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_nonmonotonic_steps = true;
    options.max_consecutive_nonmonotonic_steps = 100;
    options.max_num_iterations = 1000;
    options.function_tolerance = 1e-9;
    options.parameter_tolerance = 1e-11;
    options.num_threads = 4;
    //options.initial_trust_region_radius = 100;
    //options.max_trust_region_radius = 100;
    // options.min_trust_region_radius = 10;

    Solver::Summary summary;
    Solver::Summary full_summary;

    // Одиночный запуск решения системы с фиксированными переменными. Использовался для проверки
    // работы программы. Оставлен для инициализации минимальной невязки.
    Solve(options, &system, &summary);

    std::cout << "Initial run summary:\n";
    std::cout << summary.BriefReport() << "\n";
    system.Evaluate(Problem::EvaluateOptions(), &best_cost, NULL, NULL, NULL);
    trace_cost_function->Evaluate(&trace, trace_res, NULL);
    std::cout << "Evaluated cost is " << best_cost << "(initial run)" << std::endl << std::endl;
    std::cout << "Residual of the first equation is " << trace_res[0] << std::endl;
    std::cout << "Variables:\n";
    for (int i = 0; i < 6; i++)
        std::cout << trace[2 * i] << " " << trace[2 * i + 1] << " | " << remaining[2 * i] << " " << remaining[2 * i + 1] << "  " << remaining[12 + i] << " | "
                  << remaining_next[6 + 2 * i] << " " << remaining_next[6 + 2 * i + 1] << " " << remaining_next[i] << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 12; i++)
        trace_best[i] = trace[i];

    for (int i = 0; i < 18; i++)
        remaining_best[i] = remaining[i];

    for (int i = 0; i < 18; i++)
        remaining_next_best[i] = remaining_next[i];

    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    clock = time(NULL);

    // Основной цикл - выбор начального приближения, запуск решения системы.
    for (int iter_counter = 0; iter_counter < iter_total; iter_counter++) {
        // создание начального приближения
        for (int i = 0; i < 12; i++) {
            trace[i] = rng(gen);
        }
        for (int i = 0; i < 18; i++) {
            remaining[i] = rng(gen);
        }
        for (int i = 0; i < 18; i++) {
            remaining_next[i] = rng(gen);
        }

        // решение системы и получение невязки
        Solve(options, &system, &summary);
        system.Evaluate(Problem::EvaluateOptions(), &cost, NULL, NULL, NULL);

        system.Evaluate()

        // сохранение информации о результате
        if (cost < best_cost) {
            best_cost = cost;
            for (int i = 0; i < 12; i++)
                trace_best[i] = trace[i];
            for (int i = 0; i < 18; i++)
                remaining_best[i] = remaining[i];
            for (int i = 0; i < 18; i++)
                remaining_next_best[i] = remaining_next[i];
            //Solve(options, &full_system, &full_summary);
            //full_system.Evaluate(Problem::EvaluateOptions(), &full_cost, NULL, NULL, NULL);
            //if (full_cost < full_best_cost) {
            //  full_best_cost = full_cost;
            //}
        }
        if (cost > 1.0)
            cost_regions[0]++;
        else {
            int measure = 1;
            while (measure < 7 && cost < pow(10.0, -2.0 * measure))
                measure++;

            if (measure == 7)
                measure--;
            cost_regions[measure]++;
        }

        if (iter_total > 100) {
            if (iter_counter % (iter_total / 100) == 0)
                std::cout << "\rcompleted " << iter_counter / (iter_total / 100) << "% of iterations." << std::flush;
        }
    }
    clock = (time(NULL) - clock) / 60;
    std::cout << "\rcompleted 100% of iterations." << std::endl;
    std::cout << "Time elapsed: " << clock << " minutes" << std::endl;
    std::cout << "Result distribution across cost regions:\n";
    for (int i = 0; i < 7; i++)
        std::cout << cost_regions[i] << " ";
    std::cout << std::endl;
    std::cout << "Best cost achieved is " << best_cost << std::endl;
    std::cout << "Full best cost achieved is " << full_best_cost << std::endl;
    std::cout << "Achieved at variables: " << std::endl;
    trace_cost_function->Evaluate(&trace_best, trace_res, NULL);
    double expanded_res[34];
    double expanded_res2[20];
    double expanded_res3[50];
    double expanded_res4[15];
    double* residual_arr[3] = {trace_best, remaining_best, remaining_next_best};
    expanded_cost_function->Evaluate(residual_arr, expanded_res, NULL);
    expanded_cost_function2->Evaluate(residual_arr, expanded_res2, NULL);
    expanded_cost_function3->Evaluate(residual_arr, expanded_res3, NULL);
    expanded_cost_function4->Evaluate(residual_arr, expanded_res4, NULL);
    for (int i = 0 ; i < 5; ++i) {
      std::cout << i << ' ' << trace_res[i] << std::endl;
      assert(std::abs(trace_res[i]) < 1);
    }
    for (int i = 0 ; i < 34; ++i) {
      std::cout << i << ' ' << expanded_res[i] << std::endl;
      assert(std::abs(expanded_res[i]) < 1);
    }
    for (int i = 0 ; i < 20; ++i) {
      std::cout << i << ' ' << expanded_res2[i] << std::endl;
      assert(std::abs(expanded_res2[i]) < 1);
    }
//    for (int i = 0 ; i < 50; ++i) {
//      std::cout << i << ' ' << expanded_res3[i] << std::endl;
//      assert(std::abs(expanded_res3[i]) < 1);
//    }
//    for (int i = 0 ; i < 15; ++i) {
//      std::cout << i << ' ' << expanded_res4[i] << std::endl;
//      assert(std::abs(expanded_res4[i]) < 1);
//    }
    for (int i = 0; i < 6; i++)
        std::cout << trace_best[2 * i] << " " << trace_best[2 * i + 1] << " | " << remaining_best[2 * i] << " " << remaining_best[2 * i + 1] << "  " << remaining_best[12 + i] << " | "
                  << remaining_next_best[6 + 2 * i] << " " << remaining_next_best[6 + 2 * i + 1] << " " << remaining_next_best[i] << std::endl;
    std::cout << std::endl;

    return 0;
}
