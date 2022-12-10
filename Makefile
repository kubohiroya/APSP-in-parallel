UNAME = $(shell uname)
UNAMEM = $(shell uname -m)

LDFLAGS = -L./libs

ifeq ($(UNAME), Linux)

 PREFIX=/usr/local
 SHARED_LIB_SUFFIX = so
 LLVM=llvm-14
 CXXFLAGS=-fPIC

 ifeq ($(UNAMEM), arm64)
   ISPC_FLAGS ?= --emit-obj --pic --arch=aarch64 --target=avx2-i32x8
   CXXFLAGS += -march=armv8.2-a+fp16+rcpc+dotprod+crypto -O3
 else
   ISPC_FLAGS ?= --emit-obj --pic --arch=x86-64 --target=avx2-i32x8
   CXXFLAGS += -march=native -O3
 endif

 LDFLAGS += -L/usr/lib/$(LLVM)/lib
 $(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): LDFLAGS += -lomp

 ## CUDA
 $(CUDA): CXXFLAGS ?= -DCUDA
 $(CUDA): NVCCFLAGS ?= -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86
 $(CUDA): LIBS ?= $(CUDA_LIB)
 $(CUDA) $(CUDA_LIB): LDFLAGS += -L$(PREFIX)/cuda/lib64 -lcudart -lomp
 BINARIES ?= $(CUDA)
 ## CUDA

else
 ifeq ($(UNAME), Darwin)
   LLVM=llvm-12
   SHARED_LIB_SUFFIX = dylib
   ifeq ($(UNAMEM), arm64)
     PREFIX=/opt/homebrew
     ISPC_FLAGS += --emit-obj --pic --arch=aarch64
     CXXFLAGS = -mcpu=apple-m1 -std=c++11 -fPIC -O3 -I$(PREFIX)/include -I$(PREFIX)/opt/libomp/include
     LDFLAGS += -L$(PREFIX)/opt/libomp/lib -lomp
   else
     PREFIX=/usr/local
     ISPC_FLAGS += --emit-obj --pic --arch=x86-64
     CXXFLAGS = -march=native -std=c++11 -fPIC -O3 -I$(PREFIX)/include
     LDFLAGS += -L$(PREFIX)/lib -lomp
   endif

 endif
endif


CXX ?= g++
CXXFLAGS ?= $(CXXEXTRA)

SHAREDFLAGS ?= -shared -fPIC -dynamiclib
ISPC ?= $(PREFIX)/bin/ispc
NVCC ?= $(PREFIX)/cuda/bin/nvcc
NVCCFLAGS := -std=c++11 -O2 --compiler-options "-fPIC" -DCUDA --expt-relaxeLLVM=llvm-12

SRC_DIR := src
OBJ_DIR := objs
LIBS_DIR := libs

JAR = apsp/target/apsp-1.0.jar
JAVA_OPT := -Djna.library.path=./libs

CPP_SOURCES := $(shell find $(SRC_DIR) -name '*cpp')
CU_SOURCES := $(shell find $(SRC_DIR) -name '*cu')
ISPC_SOURCES := $(shell find $(SRC_DIR) -name '*ispc')

# JAVA_SOURCES := $(shell find $(JAVA_SRC_DIR) -name '*java')
# JAVA_SRC_DIR := apsp/src

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

SEQ_LIB := $(LIBS_DIR)/libapsp-seq.$(SHARED_LIB_SUFFIX)
OMP_LIB := $(LIBS_DIR)/libapsp-omp.$(SHARED_LIB_SUFFIX)
CUDA_LIB := $(LIBS_DIR)/libapsp-cuda.$(SHARED_LIB_SUFFIX)
SEQ_ISPC_LIB := $(LIBS_DIR)/libapsp-seq-ispc.$(SHARED_LIB_SUFFIX)
OMP_ISPC_LIB := $(LIBS_DIR)/libapsp-omp-ispc.$(SHARED_LIB_SUFFIX)

SEQ := apsp-seq
OMP := apsp-omp
CUDA := apsp-cuda
SEQ_ISPC = $(SEQ)-ispc
OMP_ISPC = $(OMP)-ispc

PROFRAW := *.profraw

$(SEQ_ISPC) $(OMP_ISPC): CXXFLAGS += -DISPC
$(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): CXXFLAGS += -Xpreprocessor -fopenmp
# $(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): LDFLAGS += -lomp

LIBS = $(SEQ_LIB) $(OMP_LIB) $(SEQ_ISPC_LIB) $(OMP_ISPC_LIB)
BINARIES = $(SEQ) $(OMP) $(SEQ_ISPC) $(OMP_ISPC)

all: bin $(JAR)

$(OBJ_DIR):
	mkdir -p $@

$(LIBS_DIR):
	mkdir -p $@

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
	$(ISPC) $(ISPC_FLAGS) $< -o $@

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

$(SEQ): $(SEQ_OBJECTS) $(SEQ_LIB) $(OBJ_DIR)/seq-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/seq-main.o -o $@ $(LDFLAGS) -lapsp-seq

$(OMP): $(OMP_OBJECTS) $(OMP_LIB) $(OBJ_DIR)/omp-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/omp-main.o -o $@ $(LDFLAGS) -lapsp-omp

$(CUDA): $(CUDA_CPP_OBJECTS) $(CUDA_LIB) $(OBJ_DIR)/cuda-cpp-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/cuda-cpp-main.o -o $@ $(LDFLAGS) -lapsp-cuda

$(SEQ_ISPC): $(ISPC_SEQ_OBJECTS) $(ISPC_OBJECTS) $(SEQ_ISPC_LIB) $(OBJ_DIR)/ispc-seq-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/ispc-seq-main.o -o $@ $(LDFLAGS) -lapsp-seq-ispc

$(OMP_ISPC): $(ISPC_OMP_OBJECTS) $(ISPC_OBJECTS) $(OMP_ISPC_LIB) $(OBJ_DIR)/ispc-omp-main.o
	$(CXX) $(CXXFLAGS) $(OBJ_DIR)/ispc-omp-main.o -o $@ $(LDFLAGS) -lapsp-omp-ispc

bin: $(BINARIES)

bin-ispc: $(OMP) $(OMP_LIB)

pgo:
	make clean
	make bin-ispc -Bj CXXEXTRA=-fprofile-generate
	make benchmark-j
	make bin-ispc -Bj CXXEXTRA=-fprofile-use

run: ApspMain

benchmark-f:
	export LD_LIBRARY_PATH=./libs; ./benchmark.py -a f -T d -d 256 -b serious2 -c --cuda

benchmark-j:
	export LD_LIBRARY_PATH=./libs; ./benchmark.py -a j -T d -b serious2 -c

benchmark-j-half:
	export LD_LIBRARY_PATH=./libs; ./benchmark.py -a j -T d -b half -c

benchmark: benchmark-f benchmark-j

$(JAR): $(JAVA_SOURCES)
	gradle :apsp:build
	make installLibs

jar:
	make $(JAR)

ApspMain: $(JAR) $(LIBS) installLibs
	java $(JAVA_OPT) -jar $(JAR) omp johnson double time apsp/InputMatrix.csv

javadoc: $(JAR)
	cd apsp; javadoc -cp target/apsp-1.0.jar src/main/java/jp/ac/cuc/hiroya/apsp/*/* -d javadoc

cleanBin:
	$(RM) -r $(OBJ_DIR) $(LIBS_DIR)
	$(RM) $(BINARIES) $(LIBS) $(PROFRAW)

cleanJar:
	(cd apsp; gradle clean)

clean: cleanBin cleanJar

installLibs: $(LIBS)
	mkdir -p apsp/target/classes
	cp $(LIBS) apsp/target/classes

test: $(JAR) installLibs
	(cd apsp; gradle test)
