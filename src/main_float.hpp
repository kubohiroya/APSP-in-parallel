void bench_floyd_warshall_float(int iterations, unsigned long seed, int block_size, bool check_correctness);
void bench_johnson_float(int iterations, unsigned long seed, bool check_correctness);
int do_main_float(
    unsigned long seed,
    int n,
    double p,
    bool use_floyd_warshall,
    bool benchmark,
    bool check_correctness,
    int block_size,
    int thread_count
);
