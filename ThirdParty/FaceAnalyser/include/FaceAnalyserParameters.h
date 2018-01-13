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

//  Parameters of the Face analyser
#ifndef __FACE_ANALYSER_PARAM_H
#define __FACE_ANALYSER_PARAM_H

#include <vector>
#include <opencv2/core/core.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace std;

namespace FaceAnalysis
{

struct FaceAnalyserParameters
{
public:
	// Constructors
	FaceAnalyserParameters();
	FaceAnalyserParameters(string root_exe);
	FaceAnalyserParameters(vector<string> &arguments);

	// These are the parameters of training and will not change and are fixed
	const double sim_scale_au = 0.7;
	const int sim_size_au = 112;

	// Should the output aligned faces be grayscale
	bool grayscale;

	// Use getters and setters for these as they might need to reload models and make sure the scale and size ratio makes sense
	void setAlignedOutput(int output_size, double scale=-1);
	// This will also change the model location
	void OptimizeForVideos();
	void OptimizeForImages();

	double getSimScaleOut() const { return sim_scale_out; }
	int getSimSizeOut() const { return sim_size_out; }
	bool getDynamic() const { return dynamic; }
	string getModelLoc() const { return string(model_location); }
	vector<cv::Vec3d> getOrientationBins() const { return vector<cv::Vec3d>(orientation_bins); }

private:

	void init();

	// Aligned face output size
	double sim_scale_out;
	int sim_size_out;

	// Should a video stream be assumed
	bool dynamic;

	// Where to load the models from
	string model_location;
	// The location of the executable
	boost::filesystem::path root;

	vector<cv::Vec3d> orientation_bins;

};

}

#endif // __FACE_ANALYSER_PARAM_H
