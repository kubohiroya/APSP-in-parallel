UNAME = $(shell uname)

# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -g -fPIC -O3
CXXFLAGS += $(CXXEXTRA)
LDFLAGS ?= -L./libs

SHAREDFLAGS ?= -shared -fPIC -dynamiclib

CLASSPATH := jna-5.8.0.jar:jna-platform-5.8.0.jar:.
JAVA_OPT := -Djna.library.path=./libs

ISPC ?= ispc
ISPC_FLAGS ?= --arch=x86-64 --emit-obj -g -03

NVCC ?= nvcc
NVCCFLAGS ?= -std=c++11 -O3

OBJ_DIR := objs
LIBS_DIR := libs
SRC_DIR := src

# executable names
SEQ := apsp-seq
OMP := apsp-omp
CUDA := apsp-cuda
SEQ_ISPC = $(SEQ)-ispc
OMP_ISPC = $(OMP)-ispc

CPP_SOURCES := $(shell find $(SRC_DIR) -name '*cpp')
CU_SOURCES := $(shell find $(SRC_DIR) -name '*cu')
ISPC_SOURCES := $(shell find $(SRC_DIR) -name '*ispc')

SEQ_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/seq-%.o)
OMP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/omp-%.o)

CUDA_CPP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/cuda-cpp-%.o)
CUDA_CU_OBJECTS := $(CU_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/cuda-cu-%.o)

ISPC_SEQ_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/ispc-seq-%.o)
ISPC_OMP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/ispc-omp-%.o)
ISPC_OBJECTS := $(ISPC_SOURCES:$(SRC_DIR)/%.ispc=$(OBJ_DIR)/ispc-%.o)

SEQ_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/seq-main.o, $(SEQ_OBJECTS))
OMP_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/omp-main.o, $(OMP_OBJECTS))

CUDA_CPP_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/cuda-cpp-main.o, $(CUDA_CPP_OBJECTS))
CUDA_CU_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/cuda-cu-main.o, $(CUDA_CU_OBJECTS))

ISPC_SEQ_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/ispc-seq-main.o, $(ISPC_SEQ_OBJECTS))
ISPC_OMP_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/ispc-omp-main.o, $(ISPC_OMP_OBJECTS))
ISPC_LIB_OBJECTS := $(filter-out $(OBJ_DIR)/ispc-main.o, $(ISPC_OBJECTS))

JAVA_SRCS := ApspTest.java
JAVA_CLASS := ApspTest.class

ifeq ($(UNAME), Linux)
SEQ_LIB := $(LIBS_DIR)/libapsp-seq.so
OMP_LIB := $(LIBS_DIR)/libapsp-omp.so
CUDA_LIB := $(LIBS_DIR)/libapsp-cuda.so
SEQ_ISPC_LIB := $(LIBS_DIR)/libapsp-seq-ispc.so
OMP_ISPC_LIB := $(LIBS_DIR)/libapsp-omp-ispc.so
$(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): LDFLAGS += -L/usr/lib/llvm-12/lib -lomp
$(CUDA): NVCCFLAGS += -arch=compute_61 -code=sm_61 --compiler-options "-fPIC" 
$(CUDA) $(CUDA_LIB): CXXFLAGS += -DCUDA
$(CUDA) $(CUDA_LIB): LDFLAGS += -DCUDA -L/usr/lib/llvm-12/lib -L/usr/local/cuda/lib64 -lcudart -lomp
LIBS := $(SEQ_LIB) $(OMP_LIB) $(CUDA_LIB) $(SEQ_ISPC_LIB) $(OMP_ISPC_LIB)
all: $(SEQ) $(OMP) $(CUDA) $(SEQ_ISPC) $(OMP_ISPC) $(LIBS) $(JAVA_CLASS)
else
# CXXFLAGS += -I/opt/boost-1.61.0/include -I$(SRC_DIR)/
#$(OMP) $(OMP_ISPC) $(CUDA): CXXFLAGS += -Xpreprocessor -fopenmp
SEQ_LIB := $(LIBS_DIR)/libapsp-seq.dylib
OMP_LIB := $(LIBS_DIR)/libapsp-omp.dylib
SEQ_ISPC_LIB := $(LIBS_DIR)/libapsp-seq-ispc.dylib
OMP_ISPC_LIB := $(LIBS_DIR)/libapsp-omp-ispc.dylib
$(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): LDFLAGS += -lomp
LIBS := $(SEQ_LIB) $(OMP_LIB) $(SEQ_ISPC_LIB) $(OMP_ISPC_LIB)
all: $(SEQ) $(OMP) $(SEQ_ISPC) $(OMP_ISPC) $(LIBS) $(JAVA_CLASS)
endif

$(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): CXXFLAGS += -Xpreprocessor -fopenmp
$(SEQ_ISPC) $(OMP_ISPC): CXXFLAGS += -DISPC
ISPCFLAGS += --target=avx2-i32x8

$(OBJ_DIR):
	mkdir -p $@
$(LIBS_DIR):
	mkdir -p $@

$(SEQ): $(SEQ_OBJECTS) $(SEQ_LIB) $(OBJ_DIR)/seq-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/seq-main.o -o $@ $(LDFLAGS) -lapsp-seq

$(OMP): $(OMP_OBJECTS) $(OMP_LIB) $(OBJ_DIR)/omp-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/omp-main.o -o $@ $(LDFLAGS) -lapsp-omp

$(CUDA): $(CUDA_CPP_OBJECTS) $(CUDA_CU_OBJECTS) $(CUDA_LIB) $(OBJ_DIR)/cuda-cpp-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/cuda-cpp-main.o -o $@ $(LDFLAGS) -lapsp-cuda

$(SEQ_ISPC): $(ISPC_SEQ_OBJECTS) $(ISPC_OBJECTS) $(SEQ_ISPC_LIB) $(OBJ_DIR)/ispc-seq-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/ispc-seq-main.o -o $@ $(LDFLAGS) -lapsp-seq-ispc

$(OMP_ISPC): $(ISPC_OMP_OBJECTS) $(ISPC_OBJECTS) $(OMP_ISPC_LIB) $(OBJ_DIR)/ispc-omp-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/ispc-omp-main.o -o $@ $(LDFLAGS) -lapsp-omp-ispc

$(SEQ_LIB): $(SEQ_LIB_OBJECTS) | $(LIBS_DIR)
	$(CXX) $(SHAREDFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OMP_LIB): $(OMP_LIB_OBJECTS) | $(LIBS_DIR)
	$(CXX) $(SHAREDFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(CUDA_LIB): $(CUDA_CPP_LIB_OBJECTS) $(CUDA_CU_LIB_OBJECTS) | $(LIBS_DIR)
	$(CXX) $(SHAREDFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(SEQ_ISPC_LIB): $(ISPC_SEQ_LIB_OBJECTS) $(ISPC_LIB_OBJECTS) | $(LIBS_DIR)
	$(CXX) $(SHAREDFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OMP_ISPC_LIB): $(ISPC_OMP_LIB_OBJECTS) $(ISPC_LIB_OBJECTS) | $(LIBS_DIR)
	$(CXX) $(SHAREDFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


-include $(SEQ_OBJECTS:%.o=%.d)
-include $(OMP_OBJECTS:%.o=%.d)
-include $(CUDA_CPP_OBJECTS:%.o=%.d)

$(OBJ_DIR)/seq-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/omp-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/cuda-cpp-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -fPIC -MMD -c $< -o $@

$(OBJ_DIR)/cuda-cu-%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/ispc-seq-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/ispc-omp-%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(OBJ_DIR)/ispc-%.o: $(SRC_DIR)/%.ispc | $(OBJ_DIR)
	$(ISPC) $(ISPCFLAGS) $< -o $@
# we do not output a header here on purpose

clean:
	$(RM) -r $(OBJ_DIR) $(LIBS_DIR)
	$(RM) $(SEQ) $(OMP) $(CUDA) $(SEQ_ISPC) $(OMP_ISPC) $(SEQ_LIB) $(OMP_LIB) $(CUDA_LIB) $(SEQ_ISPC_LIB) $(OMP_ISPC_LIB)

$(JAVA_CLASS): $(LIBS) $(JAVA_SRCS)
	javac -cp $(CLASSPATH) $(JAVA_SRCS)

ApspTest: $(JAVA_CLASS)
	java -cp $(CLASSPATH) $(JAVA_OPT) ApspTest

run: ApspTest
