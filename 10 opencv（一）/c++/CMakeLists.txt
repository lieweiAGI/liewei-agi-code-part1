# CMakeList.txt: test0 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。

cmake_minimum_required (VERSION 3.8) #cmake源码
    
add_executable (test20210115_c "videoshow.cpp") #设置要编译的源码

set(OpenCV_DIR "E:/lib/opencv/build/x64/vc15/lib") #opencv的编译路径

find_package(OpenCV REQUIRED)  #找到使用opencv需要的别的库

target_link_libraries(test20210115_c ${OpenCV_LIBS})  #链接编译时需要的opencv的动态库到当前的项目

#将动态链接库拷贝到当前程序所在文件夹
if (MSVC)
    file(GLOB OPENCV_DLLS "E:/lib/opencv/build/x64/vc15/bin/*.dll")
    add_custom_command(TARGET test20210115_c  #目标项目
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:test20210115_c>) #项目路径
endif (MSVC)
