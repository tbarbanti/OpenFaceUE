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

//  Header for all external CLM/CLNF/CLM-Z methods of interest to the user
//
//
#ifndef __LANDMARK_DETECTOR_FUNC_h_
#define __LANDMARK_DETECTOR_FUNC_h_

// OpenCV includes
#include <opencv2/core/core.hpp>

#include <LandmarkDetectorParameters.h>
#include <LandmarkDetectorUtils.h>
#include <LandmarkDetectorModel.h>

using namespace std;

namespace LandmarkDetector
{

	//================================================================================================================
	// Landmark detection in videos, need to provide an image and model parameters (default values work well)
	// Optionally can provide a bounding box from which to start tracking
	//================================================================================================================
	bool DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, FaceModelParameters& params);
	bool DetectLandmarksInVideo(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params);

	//================================================================================================================
	// Landmark detection in image, need to provide an image and optionally CLNF model together with parameters (default values work well)
	// Optionally can provide a bounding box in which detection is performed (this is useful if multiple faces are to be detected in images)
	//================================================================================================================
	bool DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, CLNF& clnf_model, FaceModelParameters& params);
	// Providing a bounding box
	bool DetectLandmarksInImage(const cv::Mat_<uchar> &grayscale_image, const cv::Rect_<double> bounding_box, CLNF& clnf_model, FaceModelParameters& params);

	//================================================================
	// Helper function for getting head pose from CLNF parameters

	// Return the current estimate of the head pose in world coordinates with camera at origin (0,0,0)
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6d GetPose(const CLNF& clnf_model, float fx, float fy, float cx, float cy);

	// Return the current estimate of the head pose in world coordinates with camera at origin (0,0,0), but with rotation representing if the head is looking at the camera
	// The format returned is [Tx, Ty, Tz, Eul_x, Eul_y, Eul_z]
	cv::Vec6d GetPoseWRTCamera(const CLNF& clnf_model, float fx, float fy, float cx, float cy);

	//===========================================================================

}
#endif
