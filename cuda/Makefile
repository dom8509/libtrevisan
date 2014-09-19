# destination folders
DESTDIR_release=release
DESTDIR_test=test

TARGET_release=$(DESTDIR_release)/libtrevisancuda.a
TARGET_test=$(DESTDIR_test)/libtrevisancuda_test.a

# define applications
NVCC=nvcc
CXX=g++
AR=ar

# optimization level
OPTIMIZATION_release=-O0
OPTIMIZATION_test=-O3

# define global dependent files
SRC=PolyEvalGF2nBN.cu
OBJ=${SRC:.cu=.o}

# add additional libs
LIBS_test+=-L/opt/cuda-6.0/lib64 -lcudart -lcudadevrt -l:$(TARGET_test)

LIBDIRS+=-L$(HOME)/sw/lib -L/usr/local/lib/R/site-library/RInside/lib -L/usr/local/lib/R/site-library/Rcpp/libs

CCFLAGS_release=-DLINUXINTEL64 -std=c++11 -g $(OPTIMIZATION_release)
CCFLAGS_test=-DLINUXINTEL64 -std=c++11 -g $(OPTIMIZATION_test)

NVCCFLAGS_release=-DLINUXINTEL64 -arch=compute_35 -m64 -rdc=true -lineinfo -Xcompiler -rdynamic -lineinfo $(OPTIMIZATION_release)
NVCCFLAGS_test=-DLINUXINTEL64 -arch=compute_35 -m64 -rdc=true -lineinfo -Xcompiler -rdynamic -lineinfo -g -G -DCUDA_SANITY_CHECKS $(OPTIMIZATION_test)

PYTHONPATH=$(CURDIR)/test:$(CURDIR)/test
export PYTHONPATH

# test stuff
$(DESTDIR_test)/$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS_test) -dc -o $@ $<

$(TARGET_test): $(DESTDIR_test)/$(OBJ)
	$(AR) -rcs $(TARGET_test) $(DESTDIR_test)/$(OBJ)
	rm $<

$(DESTDIR_test)/gpuCode.o: $(TARGET_test)
	$(NVCC) -arch=compute_35 $(TARGET_test) -dlink -lcudadevrt -o $@

../gen_irreps: ../gen_irreps.cc
	$(CXX) $(LIBDIRS) $< -o $@ -lntl -lgf2x -lgmp

$(DESTDIR_test)/irreps_cuda.py: ../gen_irreps
	../gen_irreps CUDA_PY > $(DESTDIR_test)/irreps_cuda.py

.PHONY: create_parameters
create_parameters: $(DESTDIR_test)/irreps_cuda.py
	rm -f $(DESTDIR_test)/rsh_test_parameters.*
	PYTHONPATH=.:$(DESTDIR_test); python createParameterFiles.py -f $(FS) -d $(DG) -ocf $(DESTDIR_test)/rsh_test_parameters.cc -opf $(DESTDIR_test)/rsh_test_parameters.py

test_folder:
	@mkdir -p $(DESTDIR_test)

.PHONY: test
test: test_folder create_parameters $(DESTDIR_test)/gpuCode.o 
	$(CXX) $(CCFLAGS_test) -I. -c -o $(DESTDIR_test)/rsh_test_parameters.o $(DESTDIR_test)/rsh_test_parameters.cc
	$(CXX) $(DESTDIR_test)/gpuCode.o $(DESTDIR_test)/rsh_test_parameters.o $(CCFLAGS_test) -I. $(LIBS_test) test.cc -o $(DESTDIR_test)/test
	cd test; ./test
	python analyseResults.py -irf $(DESTDIR_test)/rsh_test_results

# release stuff
$(DESTDIR_release)/$(OBJ): $(SRC)
	$(NVCC) $(NVCCFLAGS_release) -dc -o $@ $<

$(TARGET_release): $(DESTDIR_release)/$(OBJ)
	$(AR) -rcs $(TARGET_release) $(DESTDIR_release)/$(OBJ)
	rm $<

$(DESTDIR_release)/gpuCode.o: $(TARGET_release)
	$(NVCC) -arch=compute_35 $(TARGET_release) -dlink -lcudadevrt -o $@

release_folder:
	@mkdir -p $(DESTDIR_release)

release: release_folder $(DESTDIR_release)/gpuCode.o

clean:
	rm -f *.cubin
	rm -f *.a
	rm -f *.o
	rm -f *~
	rm -Rf release
	rm -Rf test