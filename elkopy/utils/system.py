import os
import time
import psutil

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def print_performance_report(start_time, mem_start):
    end_time = time.time()
    mem_final = get_memory_usage()
    print("\n" + "="*30 + " PERFORMANCE REPORT " + "="*30)
    print(f"Elapsed time:    {end_time - start_time:.2f} seconds")
    print(f"Initial Memory:  {mem_start:.2f} MB")
    print(f"Peak Memory:     {mem_final:.2f} MB")
    print(f"Memory Overhead: {mem_final - mem_start:.2f} MB")
    print("="*80)