# APSP-in-parallel

Implementation of Floyd-Warshall's Algorithm and Johnson's Algorithm using different methods of expressing parallelism. We sought to take strong performing sequential versions of both algorithms and compare them to the their parallel counterparts.

Our full project description and results can be found here: https://moorejs.github.io/APSP-in-parallel/

Floyd-Warshall has its sequential version, OpenMP, CUDA and ISPC while Johnson's Algorithm has its sequential version, OpenMP and CUDA.

For Johnson's Algorithm we require the Boost Graph Library as we use it to create our baseline sequential implementation.

The sequential versions of the Floyd-Warshall and Johnson can be further optimized using PGO in profile.py. Follow the instructions on our site to achieve guided optimization speedup for the sequential version.

Once you compile the code, you can run different benchmarks using benchmark.py. You can use ./benchmark.py -h to see the different parameters


# Additional features by forked project

* The original project implementation only supports "int" weight edges. The folked project implementation also supports "float" and "double" weight edges.
* The original project implementation only exports the distance matrix. The folked project implementation also exports the successor matrix.
* The original project's Makefile is designed for Linux. The folked project's Makefile has additional support of macOS(CUDA version is disabled, though).
* The original project builds executable files apsp-seq, apsp-omp, apsp-omp-ispc and apsp-cuda. The folked project also builds its shared libraries such as libapsp-seq.so, libapsp-omp.so, libapsp-omp-ispc.so and libapsp-cuda.so for Linux, libapsp-seq.dylib, libapsp-omp.dylib and libapsp-omp-ispc.dylib for macOS.
* The original project was designed for benchmarkings of parallelism. The folked project is re-designed to build the reusable libraries for C++ and Java project.
* Unit tests (using JUnit).
* Python 3 support.
