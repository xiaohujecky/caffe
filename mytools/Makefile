#INC=../include:../build/src/caffe:/usr/local/include
CAFFE_ROOT=../
INC=-I${CAFFE_ROOT}/build/src -I${CAFFE_ROOT}/include -I/usr/local/cuda/include -I/usr/local/include
LIB=-L${CAFFE_ROOT}/build/lib -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu
C_ARGS=-g -Wno-unused-result -O2 -std=c++11

TARGET=caffemodel
CXX=g++
CC=gcc
OBJS=diveinto_caffe_model.o

%.o: %.cpp
	$(CXX) $(C_ARGS) -c $^  -o $@ $(INC) 

$(TARGET) : $(OBJS) 
	$(CXX) $(C_ARGS) $^ -o $@ $(INC) $(LIB)  -lglog -lgflags -lprotobuf -lcaffe \
		-lboost_system -lm
#$(TARGET) : $(OBJS) 
#	$(CXX) $(C_ARGS) $^ -o $@ $(INC) $(LIB) -lglog -lgflags -lprotobuf \
#		-lboost_system -lm -lopencv_core -lopencv_highgui \
#		-lopencv_imgproc -lcblas -latlas -lcaffe

clean : 
	rm $(OBJS) $(TARGET)
