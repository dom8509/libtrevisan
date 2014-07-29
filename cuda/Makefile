TARGET=libtrevisancuda.a

SRC=Montgomery.cu
HEADER=libtrevisancuda.h \
		Montgomery.cuh \
        CudaTypes.h
OBJ=${SRC:.cu=.o}

CC=nvcc
AR=ar

CCFLAGS=-DLINUXINTEL64 -arch=compute_35 -m64 -rdc=true

all:  
	nvcc -lib -DLINUXINTEL64 --cubin -arch=sm_35 -m64 -rdc=true Montgomery.cu

lib: $(TARGET)

$(TARGET): $(OBJ)
	$(AR) -rcs $(TARGET) $(OBJ)
	rm $<

$(OBJ): $(SRC)
	$(CC) $(CCFLAGS) -c -o $@ $<