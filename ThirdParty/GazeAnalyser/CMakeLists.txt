#TBB library
include_directories(${TBB_ROOT_DIR}/include)
include_directories(${BLAS_ROOT_DIR})

include_directories(${BOOST_INCLUDE_DIR})

#OpenBlas library
include_directories(../../3rdParty/OpenBLAS/include)

#LandmarkDetector library
include_directories(../../local/LandmarkDetector/include)

SET(SOURCE
    src/GazeEstimation.cpp
)

SET(HEADERS
    include/GazeEstimation.h
)

include_directories(./include)
include_directories(${GAZEANALYSER_SOURCE_DIR}/include)

add_library( GazeAnalyser ${SOURCE} ${HEADERS})

install (TARGETS GazeAnalyser DESTINATION lib)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
