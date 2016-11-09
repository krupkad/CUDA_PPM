CC=clang
CXX=clang++
CFLAGS=-Wall -g
CXXFLAGS=$(CFLAGS) -std=c++11
NVFLAGS=-arch=sm_50 -std=c++11  -lineinfo -g
NVCC=nvcc
EXE=dcel
OBJS = dcel_cuda.o dcel.o shader.o main.o
LDFLAGS=-lGL -lGLU -lGLEW -lglfw  -lcuda -lcudart -lm -lcublas -llapack

all: $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@

.cpp.o: %.cpp %.hpp %.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.c.o: %.c %.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJS) $(EXE)