UNAME = $(shell uname)

# Override using CXX=clang++ make ...
CXX ?= g++
CXXFLAGS ?= -std=c++11 -fPIC -O3

CXXFLAGS += $(CXXEXTRA)
LDFLAGS = -L./libs
LLVM=llvm-12

SHAREDFLAGS ?= -shared -fPIC -dynamiclib

JAR = apsp/target/apsp-1.0.jar
JAVA_OPT := -Djna.library.path=./libs

ISPC ?= /usr/local/bin/ispc
ISPC_FLAGS ?= --emit-obj --pic --arch=x86-64 --target=avx2-i32x8

NVCC ?= /usr/local/cuda/bin/nvcc
NVCCFLAGS := -std=c++11 -O2 --compiler-options "-fPIC" -DCUDA --expt-relaxed-constexpr

OBJ_DIR := objs
LIBS_DIR := libs
SRC_DIR := src
JAVA_SRC_DIR := apsp/src

# executable names
SEQ := apsp-seq
OMP := apsp-omp
CUDA := apsp-cuda
SEQ_ISPC = $(SEQ)-ispc
OMP_ISPC = $(OMP)-ispc

CPP_SOURCES := $(shell find $(SRC_DIR) -name '*cpp')
CU_SOURCES := $(shell find $(SRC_DIR) -name '*cu')
ISPC_SOURCES := $(shell find $(SRC_DIR) -name '*ispc')
JAVA_SOURCES := $(shell find $(JAVA_SRC_DIR) -name '*java')

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

PROFRAW := *.profraw

$(SEQ_ISPC) $(OMP_ISPC): CXXFLAGS += -DISPC
$(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): CXXFLAGS += -Xpreprocessor -fopenmp
$(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): LDFLAGS += -lomp

LIBS := $(SEQ_LIB) $(OMP_LIB) $(SEQ_ISPC_LIB) $(OMP_ISPC_LIB)
BINARIES := $(SEQ) $(OMP) $(SEQ_ISPC) $(OMP_ISPC)

ifeq ($(UNAME), Linux)
 SEQ_LIB := $(LIBS_DIR)/libapsp-seq.so
 OMP_LIB := $(LIBS_DIR)/libapsp-omp.so
 CUDA_LIB := $(LIBS_DIR)/libapsp-cuda.so
 SEQ_ISPC_LIB := $(LIBS_DIR)/libapsp-seq-ispc.so
 OMP_ISPC_LIB := $(LIBS_DIR)/libapsp-omp-ispc.so
 $(OMP) $(OMP_ISPC) $(OMP_LIB) $(OMP_ISPC_LIB): LDFLAGS += -L/usr/lib/$(LLVM)/lib
 $(CUDA): CXXFLAGS += -DCUDA
 $(CUDA): NVCCFLAGS += -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86
 $(CUDA) $(CUDA_LIB): LDFLAGS += -L/usr/lib/$(LLVM)/lib -L/usr/local/cuda/lib64 -lcudart -lomp
 LIBS += $(CUDA_LIB)
 BINARIES += $(CUDA)
else
 SEQ_LIB := $(LIBS_DIR)/libapsp-seq.dylib
 OMP_LIB := $(LIBS_DIR)/libapsp-omp.dylib
 SEQ_ISPC_LIB := $(LIBS_DIR)/libapsp-seq-ispc.dylib
 OMP_ISPC_LIB := $(LIBS_DIR)/libapsp-omp-ispc.dylib
endif


all: bin $(JAR)

$(OBJ_DIR):
	mkdir -p $@

$(LIBS_DIR):
	mkdir -p $@

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
	$(ISPC) $(ISPC_FLAGS) $< -o $@
# we do not output a header here on purpose


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
	(cd apsp; mvn clean compile assembly:single)
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
	(cd apsp; mvn clean)

clean: cleanBin cleanJar

installLibs: $(LIBS)
	mkdir -p apsp/target/classes
	cp $(LIBS) apsp/target/classes

test: $(JAR) installLibs
	(cd apsp; mvn test)
