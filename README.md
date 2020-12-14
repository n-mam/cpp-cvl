# ocv-ppl-cnt
opencv people counter

download and install cuda toolkit 

download and unzip cudnn

Copy the following files into the CUDA Toolkit directory.
Copy <installpath>\cuda\bin\cudnn*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin.
Copy <installpath>\cuda\include\cudnn*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\include.
Copy <installpath>\cuda\lib\x64\cudnn*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\lib\x64.

clone opencv repo
clone opencv_contrib repo

"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\tbb\bin\tbbvars.bat" intel64
"C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\bin\mklvars.bat" intel64

cmake ^
-GNinja ^
-DWITH_MKL=ON ^
-DMKL_USE_MULTITHREAD=ON ^
-DMKL_WITH_TBB=ON ^
-DWITH_TBB=ON ^
-DWITH_OPENGL=ON ^
-DBUILD_opencv_world=ON ^
-DOPENCV_ENABLE_NONFREE=ON ^
-DWITH_OPENCL=ON ^
-DWITH_NVCUVID=ON ^
-DWITH_CUDA=ON ^
-DWITH_CUDNN=ON ^
-DOPENCV_DNN_CUDA=ON ^
-DENABLE_FAST_MATH=1 ^
-DCUDA_FAST_MATH=1 ^
-DCUDA_ARCH_BIN=6.1 ^
-DWITH_CUBLAS=1 ^
-DOPENCV_EXTRA_MODULES_PATH=C:\opencv_contrib\modules ^
-DCMAKE_INSTALL_PREFIX=C:\opencv\build\install ^
-DBUILD_EXAMPLES=ON ^
-DHAVE_opencv_python3=ON ^
-DBUILD_opencv_rgbd=OFF ^
-DCMAKE_BUILD_TYPE=Release ..

replace ^ with \ for linux

for CUDA_ARCH_BIN, check the make of the gfx card using nvidia-smi or devmgmt.msc (under display adapters) and then head over to: https://developer.nvidia.com/cuda-gpus choose the listed gfx adapter family and then look for the exact adpater name and make. note down the "Compute Capability" version. this becomes the CUDA_ARCH_BIN

cmake --build . --target install --config Release

#tbb and mkl dlls

copy C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\tbb\vc_mt\tbb.dll
copy C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\compiler\libiomp5md.dll
to the run folder or 

set path=%path%;C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\tbb\vc_mt;C:\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\compiler\;C:\opencv\build\install\x64\vc16\bin

#dlib

cmake .. -DUSE_AVX_INSTRUCTIONS=1 -DDLIB_NO_GUI_SUPPORT=0

for building dlib's gui webcam sample

set OPENCV_DIR=C:\opencv\build\install

#opencv with openvino

cmake ^
-GNinja ^
-DWITH_INF_ENGINE=ON ^
-DENABLE_CXX11=ON ^
-DWITH_OPENGL=ON ^
-DBUILD_opencv_world=ON ^
-DOPENCV_ENABLE_NONFREE=ON ^
-DWITH_OPENCL=ON ^
-DENABLE_FAST_MATH=1 ^
-DOPENCV_EXTRA_MODULES_PATH=C:\opencv_contrib\modules ^
-DCMAKE_INSTALL_PREFIX=C:\opencv\build\install ^
-DBUILD_EXAMPLES=ON ^
-DHAVE_opencv_python3=ON ^
-DBUILD_opencv_rgbd=OFF ^
-DCMAKE_BUILD_TYPE=Release ..


smart_classroom_demo.exe ^
 -m_fd F:\cpp-cvl\MODELS\face-detection-adas-0001\FP16\face-detection-adas-0001.xml ^
 -m_lm F:\cpp-cvl\MODELS\landmarks-regression-retail-0009\landmarks-regression-retail-0009.xml ^
 -m_reid F:\cpp-cvl\MODELS\face-reidentification-retail-0095\FP16\face-reidentification-retail-0095.xml ^
 -fg F:\openvino\inference_engine\demos\smart_classroom_demo\faces_gallery.json ^
 -i D:\bush1.mp4

for gesture standing sitting etc add -m_act F:\cpp-cvl\MODELS\person-detection-action-recognition-0005\FP16\person-detection-action-recognition-0005.xml ^

