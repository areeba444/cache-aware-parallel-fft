CXX = g++
CXXFLAGS = -O2 -std=c++17

OMPFLAGS = -fopenmp
AVXFLAGS = -mavx2 -mfma -fopenmp

BIN_DIR = bin
SRC_DIR = src

SERIAL_SRC = $(SRC_DIR)/core/serial_fft.cpp
OMP_SRC = $(SRC_DIR)/parallel/fft_omp.cpp
OMP_TILED_SRC = $(SRC_DIR)/parallel/fft_omp_tiled.cpp
AVX_SRC = $(SRC_DIR)/simd/fft_avx.cpp

SERIAL_BIN = $(BIN_DIR)/serial_fft
OMP_BIN = $(BIN_DIR)/fft_omp
OMP_TILED_BIN = $(BIN_DIR)/fft_omp_tiled
AVX_BIN = $(BIN_DIR)/fft_avx

all: directories serial omp omp_tiled avx

directories:
	mkdir -p $(BIN_DIR)
	mkdir -p output

serial:
	$(CXX) $(CXXFLAGS) $(SERIAL_SRC) -o $(SERIAL_BIN)

omp:
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OMP_SRC) -o $(OMP_BIN)

omp_tiled:
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OMP_TILED_SRC) -o $(OMP_TILED_BIN)

avx:
	$(CXX) $(CXXFLAGS) $(AVXFLAGS) $(AVX_SRC) -o $(AVX_BIN)

run_serial:
	./$(SERIAL_BIN)

run_omp:
	./$(OMP_BIN)

run_omp_tiled:
	./$(OMP_TILED_BIN)

run_avx:
	./$(AVX_BIN)

clean:
	rm -rf $(BIN_DIR) output/*.out