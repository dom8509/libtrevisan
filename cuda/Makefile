TARGET=libtrevisancuda.a

OPTIMIZATION=-O0 

SRC=PolyEvalGF2nBN.cu
OBJ=${SRC:.cu=.o}

TEST_SRC=test.cc rsh_test_parameters.cc
TEST_LIBS=-L/opt/cuda-6.0/lib64 -lcudart -lcudadevrt -l:libtrevisancuda.a
TEST_OBJ=gpuCode.o
TEST_TARGET=test

NVCC=nvcc
CXX=g++
AR=ar
PERL=perl

CCFLAGS=-DLINUXINTEL64 -std=c++11 -g -DEXPENSIVE_SANITY_CHECKS $(OPTIMIZATION)
NVCCFLAGS=-DLINUXINTEL64 -arch=compute_35 -m64 -rdc=true -lineinfo -Xcompiler -rdynamic -lineinfo -g -G -DCUDA_ERROR_CHECK -DCUDA_SANITY_CHECKS -DCUDA_DEBUG_INFO3 $(OPTIMIZATION)

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
	rm -f rsh_test_parameters.*
	rm -f rsh_test_results
	rm -Rf generated

test: lib
	rm -f rsh_test_parameters.*
	python createParameterFiles.py -f $(FS) -d $(DG) -ocf rsh_test_parameters.cc -opf rsh_test_parameters.py -v
	$(CXX) $(CCFLAGS) -I. -c -o rsh_test_parameters.o rsh_test_parameters.cc
	$(CXX) $(TEST_OBJ) $(CCFLAGS) -I. $(TEST_LIBS) $(TEST_SRC) -o $(TEST_TARGET) 
	./test
	python analyseResults.py -irf rsh_test_results

generated: 
	mkdir generated

in_one_step:  
	nvcc -lib -DLINUXINTEL64 --cubin -arch=sm_35 -m64 -rdc=true $(SRC)

