TARGET=libtrevisancuda.a

SRC=PolyEvalGF2nBN.cu
HEADER=PolyEvalGF2nBN.cuh \
        CudaTypes.h
OBJ=${SRC:.cu=.o}

CC=nvcc
AR=ar

CCFLAGS=-DLINUXINTEL64 -arch=compute_35 -m64 -rdc=true

all:  
	nvcc -lib -DLINUXINTEL64 --cubin -arch=sm_35 -m64 -rdc=true $(SRC)

lib: $(TARGET)

$(TARGET): $(OBJ)
	$(AR) -rcs $(TARGET) $(OBJ)
	rm $<

$(OBJ): $(SRC)
	$(CC) $(CCFLAGS) -c -o $@ $<