# OpenFaceUE
Plugin for Unreal Engine 4 (https://github.com/EpicGames) of OpenFace (see description at https://github.com/TadasBaltrusaitis/OpenFace) – a state-of-the art tool intended for facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation made by CMU and Cambridge University.

Current developement is only available for Unreal Engine 4.18 branch in Win64 platforms

Support control objects in level using head pose estimation and gaze inside UE4, camera aquisition is made by OpenCV in running thread and can be used as movie texture inside level to provide Augmented Reality composition. Face data position and analysis can be used too but only data for now. All expose as component to be used in blueprint class.

 - Next dev steps is using landing markers for realtime facial animation as an animation blueprint of UE skeleton morphtargets. I'm work hard on it. 

Dependency (UE ThirdParty plugin directory):
  - OpenCV 3.0+
  - Boost C++ Library
  - DLib vision computing lib
  - OpenFace
  - OpenBlas
  - Intel TBB
  - NVIDIA CUDA 8+ (not included)

You have to respect OpenFace, Unreal Engine, Boost, TBB, dlib, OpenBLAS, and OpenCV licenses.

Plugin made by [tbarbanti](https://github.com/tbarbanti). Any question please contact tbarbanti@gmail.com


