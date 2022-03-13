#pragma once
#include <vector>

template <typename T>
const inline std::vector<T> inverseMatrix3x3(std::vector<T> &A)
{
    double determinant = 0.0f;

    determinant = (A[0] * A[4] * A[8] + A[3] * A[7] * A[2] + A[1] * A[5] * A[6]) -
                  (A[2] * A[4] * A[6] + A[1] * A[3] * A[8] + A[0] * A[5] * A[6]);

    std::vector<T> inv(9);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            inv[i * 3 + j] =
                ((A[((j + 1) % 3) * 3 + ((i + 1) % 3)] * A[((j + 2) % 3) * 3 + ((i + 2) % 3)]) -
                 (A[((j + 1) % 3) * 3 + ((i + 2) % 3)] * A[((j + 2) % 3) * 3 + ((i + 1) % 3)])) /
                determinant;

    return inv;
}

template <typename T>
struct params
{
    std::vector<T> K;
    std::vector<T> R;
    std::vector<T> t;
    std::vector<T> K_inv;
    std::vector<T> R_inv;
    std::vector<T> t_inv;
    params(){};
    params(std::vector<T> _K, std::vector<T> _R, std::vector<T> _t) : K(_K), R(_R), t(_t)
    {
        K_inv = inverseMatrix3x3<double>(K);
        R_inv = inverseMatrix3x3<double>(R);
        t_inv = {-t[0], -t[1], -t[2]};
    };
};

std::vector<params<double>> get_cam_params();