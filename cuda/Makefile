TARGET=libtrevisancuda.a

SRC=PolyEvalGF2nBN.cu
OBJ=${SRC:.cu=.o}

TEST_SRC=test.cc
TEST_LIBS=-L/opt/cuda-6.0/lib64 -lcudart -lcudadevrt -l:libtrevisancuda.a
TEST_OBJ=gpuCode.o generated/irreps_cuda.o
TEST_TARGET=test

NVCC=nvcc
CXX=g++
AR=ar
PERL=perl

CCFLAGS=-DLINUXINTEL64 -std=c++11
NVCCFLAGS=-DLINUXINTEL64 -arch=compute_35 -m64 -rdc=true -lineinfo -Xcompiler -rdynamic -lineinfo -g -DCUDA_ERROR_CHECK

all: lib

lib: $(TARGET)
	$(NVCC) -arch=compute_35 $(TARGET) -dlink -lcudadevrt -o gpuCode.o

$(TARGET): $(OBJ)
	$(AR) -rcs $(TARGET) $(OBJ)
#	rm $<

$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

clean:
	rm -f *.cubin
	rm -f *.a
	rm -f *.o
	rm -f test
	rm -f *~
	rm -Rf generated

test: generated	../gen_irreps lib
	../gen_irreps CUDA > generated/irreps_cuda.cc
	$(PERL) -pi -e 's/..\/cuda\/PolyEvalGF2nBN.cuh/..\/PolyEvalGF2nBN.cuh/g' generated/irreps_cuda.cc
	$(CXX) generated/irreps_cuda.cc $(CCFLAGS) -c -o generated/irreps_cuda.o
	$(CXX) $(TEST_OBJ) $(CCFLAGS) $(TEST_LIBS) $(TEST_SRC) -o $(TEST_TARGET) 

generated: 
	mkdir generated

in_one_step:  
	nvcc -lib -DLINUXINTEL64 --cubin -arch=sm_35 -m64 -rdc=true $(SRC)

