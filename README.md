# OpenFaceUE Plugin
This project is a Plugin for [Unreal Engine 4](https://github.com/EpicGames) of [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) library – _OpenFace_ is a state-of-the art tool intended for facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation made by CMU and Cambridge University.



## About

Current developement is only available for Unreal Engine **4.18** branch in **Win64** platforms

Support control objects in level using head pose estimation and gaze inside UE4, camera aquisition is made by OpenCV in running thread and can be used as movie texture inside level to provide Augmented Reality composition. Face data position and analysis can be used too but only data for now. All expose as component to be used in blueprint class.

* Screenshots:

OpenFace:
![Example 1](https://github.com/TadasBaltrusaitis/OpenFace/raw/master/imgs/multi_face_img.png)

![Example 2](https://github.com/TadasBaltrusaitis/OpenFace/raw/master/imgs/au_sample.png)

UE4 Plugin:
![Example Plugin](https://github.com/tbarbanti/OpenFaceUE/raw/master/img/Screenshot_20180113-041047.png)

* Development Road-map:
     - Use landing markers data for realtime facial animation integrated as an animation blueprint of UE skeleton morphtargets;
     - Multithread for aquisition frames;
     - Some filters to improve data aquisition series in order to apply in recorded data; 
  
 * Dependency: (ThirdParty plugin directory)
     - OpenCV 3.0+
     - Boost C++ Library
     - DLib vision computing lib
     - OpenFace
     - OpenBlas
     - Intel TBB
     - NVIDIA CUDA 8+ (not included)


Plugin made by [tbarbanti](https://github.com/tbarbanti). Any question please contact me (_tbarbanti@gmail.com_)

Many thanks to [**TadasBaltrusaitis**](https://github.com/TadasBaltrusaitis) by the great job to bring OpenFace in github a great tool specially when the commercial solutions of facial capture is based em computer vision matlab models or depth sensors and not well documented for researchers in general.


## License

You have to respect OpenFace, Unreal Engine, Boost, TBB, dlib, OpenBLAS, and OpenCV licenses.


## Citation

_OpenFace_ is based in following papers:

Overall system
OpenFace: an open source facial behavior analysis toolkit Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency, in IEEE Winter Conference on Applications of Computer Vision, 2016

Facial landmark detection and tracking
Constrained Local Neural Fields for robust facial landmark detection in the wild Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.

Eye gaze tracking
Rendering of Eyes for Eye-Shape Registration and Gaze Estimation Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling in IEEE International. Conference on Computer Vision (ICCV), 2015

Facial Action Unit detection
Cross-dataset learning and person-specific normalisation for automatic Action Unit detection Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson in Facial Expression Recognition and Analysis Challenge, IEEE International Conference on Automatic Face and Gesture Recognition, 2015


