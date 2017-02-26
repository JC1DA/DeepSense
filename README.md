# DeepSense

Download the models below, extract and copy them onto mobile devices and set the link in DeepSense App to load them.

VGG-F Link: https://drive.google.com/file/d/0B_GMfaURPvQDQk9sU3FHdU1sUzA/view?usp=sharing

Yolo Tiny Link: https://drive.google.com/file/d/0B_GMfaURPvQDZVVFMnBXQUU3X2s/view?usp=sharing

## The app is configured to work on Samsung Galaxy S7 with Mali GPU, if you need to run it on Adreno-based devices
- 1) copy the appropriate shared libraries (libllvm-qcom.so and libOpenCL.so) from distribution/opencl/lib/armeabi-v7a/Adreno-Android5 OR distribution/opencl/lib/armeabi-v7a/Adreno-Android6 into distribution/opencl/lib/armeabi-v7a
- 2) comment out Mali-shared library in app/CMakeLists.txt and uncomment Adreno shared library

## To run the app
- 1) Download and extract the model
- 2) Put the whole model's directory onto device's storage
- 3) Change the path in MainActivity.java
- 4) Run :)

Enjoy DeepSense


