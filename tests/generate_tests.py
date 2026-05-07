import os
import random

def generate_test_case(directory, size):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path = os.path.join(directory, 'fft.in')
    with open(path, 'w') as f:
        f.write(str(size) + '\n')
        f.write(''.join([str(random.randint(0, 9)) for _ in range(size)]) + '\n')
        f.write(''.join([str(random.randint(0, 9)) for _ in range(size)]) + '\n')
    print(f"Generated {path} with number size {size}")

def main():
    # Small test cases
    small_dir = 'tests/small_tests'
    for i in range(4, 10): # 16, 32, 64, 128, 256, 512
        generate_test_case(os.path.join(small_dir, str(2**i)), 2**i)

    # Medium test cases
    medium_dir = 'tests/medium_tests'
    for i in range(10, 14): # 1024, 2048, 4096, 8192
        generate_test_case(os.path.join(medium_dir, str(2**i)), 2**i)

    # Large test cases
    large_dir = 'tests/large_tests'
    for i in range(14, 17): # 16384, 32768, 65536
        generate_test_case(os.path.join(large_dir, str(2**i)), 2**i)

if __name__ == "__main__":
    main()
