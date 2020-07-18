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

cmake -D BUILD_opencv_world=ON -D INSTALL_C_EXAMPLES=ON -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=6.1 -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=C:/opencv_contrib/modules -D CMAKE_INSTALL_PREFIX=C:/opencv/build/install -D BUILD_EXAMPLES=ON -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=C:/Python38/python.exe ..

for CUDA_ARCH_BIN, check the make of the gfx card using nvidia-smi or devmgmt.msc (under display adapters) and then head over to: https://developer.nvidia.com/cuda-gpus choose the listed gfx adapter family and then look for the exact adpater name and make. note down the "Compute Capability" version. this becomes the CUDA_ARCH_BIN

cmake --build . --target install --config debug

set path=%path%;C:\opencv\build\bin\Debug