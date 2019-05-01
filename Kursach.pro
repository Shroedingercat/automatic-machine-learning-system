TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    dataframe.cpp \
    decisiontree.cpp \
    mllibrary.cpp \
    randomforestclassifier.cpp \
    differentialevolution.cpp

HEADERS += \
    dataframe.h \
    decisiontree.h \
    mllibrary.h \
    randomforestclassifier.h \
    differentialevolution.h
