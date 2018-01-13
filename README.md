# OpenFaceUE
 OpenFace Plugin for Unreal Engine 4 (https://github.com/TadasBaltrusaitis/OpenFace) – a state-of-the art tool intended for facial landmark detection, head pose estimation, facial action unit recognition, and eye-gaze estimation made by CMU and Cambridge University.

Current developement is only available for Unreal Engine 4.18 - Win64 platforms

Support control objects using head motion and gaze inside UE4, camera aquisition is made by OpenCV in running thread and can be used as movie texture inside level. Face data position and analysis can be used too. All expose as component to be used in blueprint class. 

Dependency (UE ThirdParty):
  - OpenCV 3.0+
  - Boost C++ Library
  - DLib vision computing lib
  - OpenFace
  - OpenBlas
  - Intel TBB
  - NVIDIA CUDA 8+

You have to respect OpenFace, Unreal Engine, Boost, TBB, dlib, OpenBLAS, and OpenCV licenses.
