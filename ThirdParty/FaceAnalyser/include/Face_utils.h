///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __FACE_UTILS_h_
#define __FACE_UTILS_h_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PDM.h"
#include "PAW.h"

namespace FaceAnalysis
{
	//===========================================================================	
	// Defining a set of useful utility functions to be used within FaceAnalyser

	// Aligning a face to a common reference frame
	void AlignFace(cv::Mat& aligned_face, const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, cv::Vec6f params_global, const PDM& pdm, bool rigid = true, float scale = 0.6, int width = 96, int height = 96);
	void AlignFaceMask(cv::Mat& aligned_face, const cv::Mat& frame, const cv::Mat_<float>& detected_landmarks, cv::Vec6f params_global, const PDM& pdm, const cv::Mat_<int>& triangulation, bool rigid = true, float scale = 0.6, int width = 96, int height = 96);

	void Extract_FHOG_descriptor(cv::Mat_<double>& descriptor, const cv::Mat& image, int& num_rows, int& num_cols, int cell_size = 8);

	void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation);

	// The following two methods go hand in hand
	void ExtractSummaryStatistics(const cv::Mat_<double>& descriptors, cv::Mat_<double>& sum_stats, bool mean, bool stdev, bool max_min);
	void AddDescriptor(cv::Mat_<double>& descriptors, cv::Mat_<double> new_descriptor, int curr_frame, int num_frames_to_keep = 120);

	//===========================================================================
	// Point set and landmark manipulation functions
	//===========================================================================
	// Using Kabsch's algorithm for aligning shapes
	//This assumes that align_from and align_to are already mean normalised
	cv::Matx22f AlignShapesKabsch2D(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to);

	//=============================================================================
	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	cv::Matx22f AlignShapesWithScale(cv::Mat_<float>& src, cv::Mat_<float> dst);

	//===========================================================================
	// Visualisation functions
	//===========================================================================
	void Project(cv::Mat_<float>& dest, const cv::Mat_<float>& mesh, float fx, float fy, float cx, float cy);

	//===========================================================================
	// Angle representation conversion helpers
	//===========================================================================
	cv::Matx33f Euler2RotationMatrix(const cv::Vec3f& eulerAngles);
	
	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	cv::Vec3f RotationMatrix2Euler(const cv::Matx33f& rotation_matrix);

	cv::Vec3f Euler2AxisAngle(const cv::Vec3f& euler);

	cv::Vec3f AxisAngle2Euler(const cv::Vec3f& axis_angle);

	cv::Matx33f AxisAngle2RotationMatrix(const cv::Vec3f& axis_angle);

	cv::Vec3f RotationMatrix2AxisAngle(const cv::Matx33f& rotation_matrix);

	//============================================================================
	// Matrix reading functionality
	//============================================================================

	// Reading a matrix written in a binary format
	void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat);

	// Reading in a matrix from a stream
	void ReadMat(std::ifstream& stream, cv::Mat& output_matrix);

	// Skipping comments (lines starting with # symbol)
	void SkipComments(std::ifstream& stream);

}
#endif
