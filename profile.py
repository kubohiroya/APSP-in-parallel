#!/usr/bin/python3

import subprocess
import re # regex
import argparse

profile_benchmarks = {
    'both': [
        { 'a': 'f', 'n': 1024 },
    ],
    'floyd_warshall': [
        { 'a': 'f', 'n': 4096, 's': 32, 'd': 96 },
        { 'a': 'f', 'n': 4096, 's': 5, 'd': 96 },
        { 'a': 'f', 'n': 4096, 'd': 96 },
    ],
    'johnson': [
        { 'a': 'j', 'n': 11129, 'p': 0.00016 }, 
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument('executable', type=str, help='The Make target name (e.g. apsp-seq)')
parser.add_argument('-b', '--benchmark', choices=profile_benchmarks.keys(),
                    required=True, help='The algorithm you would like to profile.')
parser.add_argument('-t', '--thread_count', required=True, help='Thread count for CPU based implementations.')

args = parser.parse_args()

def create_cmd(params):
    cmd = []
    for attr, value in params.items():
        cmd += ['-' + attr, str(value)]

    return cmd

def run_cmd(command, verbose):
    if verbose:
        print(' '.join(command))

    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

def extract_time(stdout):
    return float(re.search(r'(\d*\.?\d*)ms', stdout.decode('utf-8')).group(1))

run_cmd(['make', 'clean', '-j'], True)

print('Compiling plain executale to benchmark')
stdout, stderr = run_cmd(['make', args.executable, '-Bj', 'CXXEXTRA=-w'], True)
print(stdout.decode('utf-8'))
if stderr.decode('utf-8'):
    print(stderr.decode('utf-8'))
    exit(1)

print('Benchmarking to get basic time...')
for param_obj in profile_benchmarks[args.benchmark]:
    param_obj['t'] = args.thread_count
    params = create_cmd(param_obj)
    stdout, stderr = run_cmd(['./' + args.executable] + params, True)
    if stderr.decode('utf-8'):
        print(stderr.decode('utf-8'))
        exit(1)
    time = extract_time(stdout)
    print(stdout.decode('utf-8'))

run_cmd(['make', 'clean', '-j'], True)

print('Compiling to generate profile files...')
stdout, stderr = run_cmd(['make', args.executable, '-Bj', 'CXXEXTRA=-fprofile-generate -w'], True)
print(stdout.decode('utf-8'))
if stderr.decode('utf-8'):
    print(stderr.decode('utf-8'))
    exit(1)

print('Benchmarking to inform profiler...')
for param_obj in profile_benchmarks[args.benchmark]:
    param_obj['t'] = args.thread_count
    params = create_cmd(param_obj)
    stdout, stderr = run_cmd(['./' + args.executable] + params, True)
    if stderr.decode('utf-8'):
        print(stderr.decode('utf-8'))
        exit(1)

print('\nCompiling with use of profiler...')
stdout, stderr = run_cmd(['make', args.executable, '-Bj', 'CXXEXTRA=-fprofile-use -fprofile-correction'], True)
if stderr.decode('utf-8'):
    print(stderr.decode('utf-8'))
    exit(1)
print(stdout.decode('utf-8'))

stdout, stderr = run_cmd(['./' + args.executable] + create_cmd(profile_benchmarks[args.benchmark][-1]), True)
if stderr.decode('utf-8'):
    print(stderr.decode('utf-8'))
    exit(1)
new_time = extract_time(stdout)
print(stdout.decode('utf-8'))

print('Approximate speedup achieved: {0:4.2f}x'.format(time / new_time))
