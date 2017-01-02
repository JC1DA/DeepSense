# DeepSense

# Download Models at https://github.com/JC1DA/DeepSenseModel

# The app is configured to work on Samsung Galaxy S7 with Mali GPU, if you need to run it on Adreno-based devices
- 1) copy the appropriate shared libraries (libllvm-qcom.so and libOpenCL.so) from distribution/opencl/lib/armeabi-v7a/Adreno-Android5 and distribution/opencl/lib/armeabi-v7a/Adreno-Android6 into distribution/opencl/lib/armeabi-v7a
- 2) comment out Mali-shared library in app/CMakeLists.txt and uncomment Adreno shared library

# To run the app
- 1) Download the model
  2) Put the whole model's directory onto device's storage
  3) Change the path in MainActivity.java
  4) Run :)

Enjoy DeepSense


