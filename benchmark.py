#!/usr/bin/python3

import subprocess
import os
import re # regex
import argparse


all_benchmarks = {
    'normal': [
        [
            { 'n': 2560, 't': 8, 'p': 0.5 },
            { 'n': 5120, 't': 8, 'p': 0.5 },
            { 'n': 10240, 't': 8, 'p': 0.5 },
            { 'n': 10240+5120, 't': 8, 'p': 0.5 },
        ],
        [
            { 'n': 2560, 't': 8, 'p': 0.5 },
            { 'n': 5120, 't': 8, 'p': 0.5 },
            { 'n': 10240, 't': 8, 'p': 0.5 },
            { 'n': 10240+5120, 't': 8, 'p': 0.5 },
        ],
        [
            { 'n': 2560, 't': 8, 'p': 0.5 },
            { 'n': 5120, 't': 8, 'p': 0.5 },
            { 'n': 10240, 't': 8, 'p': 0.5 },
            { 'n': 10240+5120, 't': 8, 'p': 0.5 },
        ],
        [
            { 'n': 2560, 't': 8, 'p': 0.5 },
            { 'n': 5120, 't': 8, 'p': 0.5 },
            { 'n': 10240, 't': 8, 'p': 0.5 },
            { 'n': 10240+5120, 't': 8, 'p': 0.5 },
        ],
        [
            { 'n': 2560, 't': 8, 'p': 1.0 },
            { 'n': 5120, 't': 8, 'p': 1.0 },
            { 'n': 10240, 't': 8, 'p': 1.0 },
            { 'n': 10240+5120, 't': 8, 'p': 1.0 },
        ]
    ],
    'profile': [
        [
            { 'n': 11129, 't': 16, 'p': 0.00016 },
        ]
    ],
    'thread_scale': [
        [
            { 'n': 1024, 't': 1, 'p': 0.0001 },
            { 'n': 1024, 't': 2, 'p': 0.0001 },
            { 'n': 1024, 't': 3, 'p': 0.0001 },
            { 'n': 1024, 't': 4, 'p': 0.0001 },
            { 'n': 1024, 't': 5, 'p': 0.0001 },
            { 'n': 1024, 't': 6, 'p': 0.0001 },
            { 'n': 1024, 't': 7, 'p': 0.0001 },
            { 'n': 1024, 't': 8, 'p': 0.0001 },
            { 'n': 1024, 't': 9, 'p': 0.0001 },
            { 'n': 1024, 't': 10, 'p': 0.0001 },
            { 'n': 1024, 't': 11, 'p': 0.0001 },
            { 'n': 1024, 't': 12, 'p': 0.0001 },
        ]
    ],
    'serious': [
        [
            { 'n': 512, 't': 16, 'p': 0.00016 },
            { 'n': 1024, 't': 16, 'p': 0.00016 },
            { 'n': 2048, 't': 16, 'p': 0.00016 },
            { 'n': 4096, 't': 16, 'p': 0.00016 },
        ]
    ],
    'serious2': [
        [
            { 'n': 512, 't': 16, 'p': 0.00016 },
            { 'n': 1024, 't': 16, 'p': 0.00016 },
            { 'n': 2048, 't': 16, 'p': 0.00016 },
            { 'n': 4096, 't': 16, 'p': 0.00016 },
            { 'n': 11129, 't': 16, 'p': 0.00016 },
        ]
    ],
    'half': [
        [
            { 'n': 512, 't': 4, 'p': 0.00016 },
            { 'n': 1024, 't': 4, 'p': 0.00016 },
            { 'n': 2048, 't': 4, 'p': 0.00016 },
            { 'n': 4096, 't': 4, 'p': 0.00016 },
        ]
    ]
}

DEFAULT_BENCH = 'normal'
DEFAULT_BLOCK_SIZE = 8

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm', choices=['f', 'j'], required=True,
                    help='Algorithm to benchmark')
parser.add_argument('-s', '--seed', default=42,
                    help='Seed for graph generation')
parser.add_argument('-d', '--block_size', default=DEFAULT_BLOCK_SIZE,
                    help='The block size of the graph for Floyd-Warshall')
parser.add_argument('-b', '--benchmark', choices=all_benchmarks.keys(), default=DEFAULT_BENCH,
                    help='The name of the benchmark to run')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print commands as they run')
parser.add_argument('-g', '--cuda', action='store_true', help='Run CUDA version')
parser.add_argument('-c', '--compare', action='store_true', help='Compare different parallel schemes. Recommended to be used with "-b serious"')
parser.add_argument('-T', '--wtype', type=str, choices=['i', 'f', 'd'], default='i', help='weight type')

args = parser.parse_args()

def create_cmd(params):
    cmd = []
    for attr, value in params.items():
        cmd += ['-' + attr, str(value)]
    return cmd

def run_cmd(command, verbose):
    if verbose:
        print('Running command ' + ' '.join(command))

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

def extract_time(stdout):
    if len(stdout):
        return float(re.search(r'(\d*\.?\d*)ms', stdout.decode('utf-8')).group(1))
    return float(0.0)

def run_bench(bench_list, algorithm, wtype, seed, block_size, verbose, cuda, caching_seq=True, seq_cache={}):
    
    print('')
    print(' {0:-^55} '.format(''))
    print('|{0:^55}|'.format('  Benchmark for {0}\'s Algorithm '
                             .format('Floyd-Warshall' if algorithm == 'f' else 'Johnson')))
    print('|{0:^55}|'.format('seed = {0}{1}'.format(seed, ', block size = {0}'.format(block_size) if algorithm == 'f' else '')))
    print(' {0:-^55} '.format(''))
    print('| {0:<7} | {1:<5} | {2:<2} | {3:<12} | {4:<8} | {5:<8} |'.format('p', 'n', 't', 'seq (ms)',
                                                                     'par (ms)', 'speedup'))

    for bench in bench_list:
        print(' {0:-^55} '.format(''))

        for param_obj in bench:
            param_obj['a'] = algorithm
            param_obj['s'] = seed
            param_obj['d'] = block_size
            param_obj['T'] = wtype
            params = create_cmd(param_obj)

            cache_key = str(param_obj['p']) + str(param_obj['n'])
            if not caching_seq or cache_key not in seq_cache:
                stdout, stderr = run_cmd(['./apsp-seq'] + params, verbose)

                if len(stderr):
                    print('Sequential Error: ', stderr)
                    return

                seq_cache[cache_key] = extract_time(stdout)

            seq_time = seq_cache[cache_key]

            if cuda: stdout, stderr = run_cmd(['./apsp-cuda'] + params,verbose)
            else: stdout, stderr = run_cmd(['./apsp-omp'] + params, verbose)

            if len(stderr):
                print( 'Parallel Error: ' + stderr)
                return

            par_time = extract_time(stdout)

            print( '| {p:>1.5f} | {n:>5} | {t:>2} | {0:>11.1f} | {1:>8.1f} | {2:>7.1f}x |'.format(seq_time, par_time,
                                                                                             seq_time / par_time,
                                                                                             **param_obj))

    print(' {0:-^55} '.format(''))
    print('')

def run_par_bench(bench_list, algorithm, wtype, seed, block_size, verbose, cuda=False, caching_seq=True, seq_cache={}):
    
    print('')
    print(' {0:-^72} '.format(''))
    print('|{0:^72}|'.format('  Benchmark for {0}\'s Algorithm  of type = {1} weights'
                             .format('Floyd-Warshall' if algorithm == 'f' else 'Johnson', wtype)))
    print('|{0:^72}|'.format('seed = {0}{1}'.format(seed, ', block size = {0}'.format(block_size) if algorithm == 'f' else '')))
    print(' {0:-^72} '.format(''))
    print('| {0:<7} | {1:<5} | {2:<2} | {3:<13} | {4:<8} | {5:<8} | {6:<8} |'.format('p', 'n', 't', ' SEQ  (ms)', 
                                                                     'OMP (ms)', 'ISPC(ms)', 'CUDA (ms)'))

    for bench in bench_list:
        print(' {0:-^72} '.format(''))

        for param_obj in bench:
            param_obj['a'] = algorithm
            param_obj['s'] = seed
            param_obj['d'] = block_size
            param_obj['T'] = wtype
            params = create_cmd(param_obj)

            if os.path.exists('./apsp-seq'):
              stdout, stderr = run_cmd(['./apsp-seq'] + params, verbose)
              if len(stderr):
                  print('SEQ Error: ', stderr)
                  return
              seq_time = extract_time(stdout)
            else:
              seq_time = 0

            if os.path.exists('./apsp-omp'):
              stdout, stderr = run_cmd(['./apsp-omp'] + params, verbose)
              if len(stderr):
                  print('OMP Error: ', stderr)
                  return
              omp_time = extract_time(stdout)
            else:
              omp_time = 0

            if os.path.exists('./apsp-omp-ispc'):
              stdout, stderr = run_cmd(['./apsp-omp-ispc'] + params, verbose)
              if len(stderr):
                  print('OMP ISPC Error: ', stderr)
                  return
              omp_ispc_time = extract_time(stdout)
            else:
              omp_ispc_time = 0

            if os.path.exists('./apsp-cuda'):
              stdout, stderr = run_cmd(['./apsp-cuda'] + params,verbose)
              if len(stderr):
                  print('CUDA Error: ', stderr)
                  return
              if len(stdout):
                  cuda_time = extract_time(stdout)
              else:
                cuda_time = 0
            else:
              cuda_time = 0

            print('| {p:>1.5f} | {n:>5} | {t:>2} | {0:>13.1f} | {1:>8.1f} | {2:>8.1f} | {3:>9.1f} |'.format(seq_time,
                                                                                               omp_time, omp_ispc_time, 
                                                                                               cuda_time,
                                                                                               **param_obj))

    print(' {0:-^72} '.format(''))
    print('')


def choose_benchmark():
    if (args.compare):
        run_par_bench(all_benchmarks[args.benchmark], args.algorithm, args.wtype, args.seed, args.block_size, args.verbose, args.cuda)
    else:
        run_bench(all_benchmarks[args.benchmark], args.algorithm, args.wtype, args.seed, args.block_size, args.verbose, args.cuda)

choose_benchmark()
