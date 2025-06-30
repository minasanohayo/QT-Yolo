QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

win32-g++ {
    message("Using MinGW compiler")
    # hpp files
    INCLUDEPATH += D:\openCV\build-opencv-4.10.0-Desktop_Qt_6_5_3_MinGW_64_bit-Debug\install\include


    # dll.a files
    LIBS += D:\openCV\build-opencv-4.10.0-Desktop_Qt_6_5_3_MinGW_64_bit-Debug\install\x64\mingw\lib\libopencv*
}

msvc {
    message("Using MSVC compiler")
    #带cuda
    #INCLUDEPATH += D:\openCV\install_with_cuda\include
    #LIBS += D:\openCV\install_with_cuda\x64\vc16\lib\opencv_*

    #不带cuda
    INCLUDEPATH += D:\openCV\build-opencv-4.10.0-MSVC2019-Debug\install\include
    LIBS += D:\openCV\build-opencv-4.10.0-MSVC2019-Debug\install\x64\vc17\lib\opencv_*

    INCLUDEPATH += C:/Users/commander/Downloads/Programs/onnxruntime-win-x64-1.20.0/include
    LIBS += -LC:/Users/commander/Downloads/Programs/onnxruntime-win-x64-1.20.0/lib onnxruntime.lib
    #LIBS += -LC:/Users/commander/Downloads/Programs/onnxruntime-win-x64-1.20.0/lib
}




SOURCES += \
    dispwidget.cpp \
    main.cpp \
    onnxruntimethread.cpp \
    yolo.cpp

HEADERS += \
    dispwidget.h \
    onnxruntimethread.h \
    yolo.h

FORMS += \
    dispwidget.ui \
    yolo.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    README.md
