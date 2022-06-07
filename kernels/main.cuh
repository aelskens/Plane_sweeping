#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

#include "../src/cam_params.hpp"
#include "../src/constants.hpp"

#include <vector>
#include <typeinfo>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();
float* frame2frame_matching_naive_baseline(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_naive_float(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_naive_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_partially_shared_float_2D(cam &ref, cam &cam_1, cv::Mat &cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_shared_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_shared_full_float_2D(cam& ref, cam& cam_1, cv::Mat& cost_cube_plane, int zi, int half_window);
float* frame2frame_matching_smart_naive_full_float_2D(cam& ref, cam& cam_1, std::vector<cv::Mat>& cost_cube, int half_window);
float* frame2frame_matching_smart_shared_full_float_2D(cam& ref, cam& cam_1, std::vector<cv::Mat>& cost_cube, int half_window);
float* frame2frame_matching_smart_full_shared_full_float_2D(cam& ref, cam& cam_1, std::vector<cv::Mat>& cost_cube, int half_window);
float* frame2frame_matching_all_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, std::vector<cv::Mat>& cost_cube, int half_window);
float* frame2frame_matching_all_no_fill_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, std::vector<cv::Mat>& cost_cube, int half_window);
float* frame2frame_matching_all_no_fill_better_pad_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, std::vector<cv::Mat>& cost_cube, int half_window);
float* frame2frame_matching_reduced_float_all_no_fill_better_pad_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, int half_window);
uint8_t* frame2frame_matching_reduced_uint8_t_all_no_fill_better_pad_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, int half_window);
uint8_t* frame2frame_matching_reduced_uint8_t_all_no_fill_better_pad_less_global_smart_full_shared_full_float_2D(cam& ref, std::vector<cam>& cam_vector, int half_window);