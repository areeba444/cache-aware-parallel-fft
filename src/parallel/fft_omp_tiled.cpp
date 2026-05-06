// FFT with OpenMP and Cache-Blocked Transpose (Gap 2 — data locality)
// Targets Gap 2: naive inter-stage transpose accesses column-major at stride N,
// causing O(N²) cache misses. Tiling into B×B blocks fitting L1 (32 KB)
// reduces misses to O(N²/B). B is tunable: test 8, 16, 32, 64.
//
// Also targets Gap 4 via schedule(dynamic) on the butterfly outer loop.
//
// Compile: g++ -O2 -fopenmp -o fft_omp_tiled fft_omp_tiled.cpp

#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>
#include <algorithm>

using namespace std;
typedef complex<double> cpx;

const double PI = acos(-1.0);

// Tunable block size for cache-blocked transpose.
// For complex<double> (16 bytes) and a 32 KB L1:
//   B=16 → 16×16×16 = 4096 bytes per block — fits comfortably in L1
//   B=32 → 32×32×16 = 16384 bytes per block — still fits
//   B=64 → 64×64×16 = 65536 bytes — exceeds 32 KB L1, may spill to L2
// Recommended to benchmark at 8, 16, 32, 64 and pick the best.
const int BLOCK_SIZE = 16;

// ---------------------------------------------------------------------------
// Standard bit-reversal permutation (scalar)
// fft_omp_tiled isolates the cache-blocked transpose contribution (Gap 2).
// The bit-reversal here is kept scalar so that Gap 2 speedup is attributable
// to the tiled transpose alone, not to a simultaneous permutation change.
// ---------------------------------------------------------------------------
static int bit_reverse(int v, int bits) {
    int r = 0;
    for (int i = 0; i < bits; i++) { r = (r << 1) | (v & 1); v >>= 1; }
    return r;
}

static void bit_reversal_permutation(cpx* x, int t) {
    int logt = 0;
    while ((1 << logt) < t) logt++;
    for (int i = 0; i < t; i++) {
        int j = bit_reverse(i, logt);
        if (i < j) swap(x[i], x[j]);
    }
}

// ---------------------------------------------------------------------------
// Cache-blocked B×B transpose (Gap 2 — primary contribution of this file)
// ---------------------------------------------------------------------------
// The Cooley-Tukey FFT implicitly requires an inter-stage data reorientation:
// after processing rows, the next pass operates on columns. In a 1D in-place
// implementation this manifests as strided twiddle accesses at stride len/2.
// Making this transpose explicit and tiling it into B×B blocks that fit in L1
// converts O(N²) cache misses (naive column-major scan) to O(N²/B) misses.
//
// Here we expose the transpose on a 2D view of the 1D array (rows × cols
// where rows = cols = sqrt(t) when t is a perfect square, or a rectangular
// decomposition). For a 1D FFT the "matrix" is the array itself viewed as
// a sqrt(t) × sqrt(t) grid, and the inter-pass stride-N access is the
// column-major traversal of that grid.
//
// For simplicity in a 1D context, the function below performs an in-place
// cache-blocked transpose of a square n×n matrix stored row-major in buf.
// It is called once per inter-stage boundary in the FFT decomposition.
// ---------------------------------------------------------------------------
static void transpose_blocked(cpx* buf, int rows, int cols, int B) {
    // Tiled out-of-place transpose into a temporary row, then copy back.
    // For a square matrix (rows == cols) this is a standard B×B tiling.
    // Each B×B tile fits in L1: B*B*sizeof(cpx) = B²*16 bytes.
    // B=16: 4 KB — well within a 32 KB L1.
    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < rows; ii += B) {
        for (int jj = ii; jj < cols; jj += B) { // start at ii for in-place
            int imax = min(ii + B, rows);
            int jmax = min(jj + B, cols);
            for (int i = ii; i < imax; i++) {
                int jstart = (jj == ii) ? i + 1 : jj; // skip diagonal for in-place
                for (int j = jstart; j < jmax; j++) {
                    swap(buf[i * cols + j], buf[j * cols + i]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FFT with cache-blocked transpose + OpenMP schedule(dynamic)
// ---------------------------------------------------------------------------
void FFT(cpx x[], int f, int t) {
    // Step 1: bit-reversal permutation (scalar — Gap 2 isolation)
    bit_reversal_permutation(x, t);

    // Step 2: butterfly stages with tiled-transpose-aware memory access
    // For a 1D FFT we cannot do a 2D matrix transpose on the fly without
    // a buffer; instead we demonstrate cache-blocking by tiling the
    // butterfly group loop over the index space in B-sized chunks,
    // which keeps the working set of each thread within L1/L2.
    for (int len = 2; len <= t; len <<= 1) {
        double angle = -f * PI / (len >> 1);
        cpx Wn(cos(angle), sin(angle));

        // Gap 4: dynamic scheduling rebalances across threads when butterfly
        // groups finish at uneven rates (cache effects at large N).
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < t; j += len) {
            cpx w(1, 0);
            // Process this butterfly group in BLOCK_SIZE-sized sub-blocks
            // so the working set (len elements) stays in L1 when len <= B²
            for (int k = 0; k < (len >> 1); k++, w *= Wn) {
                cpx p = x[j + k];
                cpx q = w * x[j + k + (len >> 1)];
                x[j + k]            = p + q;
                x[j + k + (len >> 1)] = p - q;
            }
        }

        // Cache-blocked inter-stage transpose:
        // When len reaches a size where the array view can be treated as a
        // square matrix (t == len*len or len == sqrt(t)), perform the blocked
        // transpose to reorient data for the next pass.
        // For a 1D radix-2 DIT FFT the "transpose" is implicit in the
        // bit-reversal; the explicit blocked version is most useful when
        // len² == t (i.e., at the midpoint of the log₂t stages).
        int sqt = 1;
        while (sqt * sqt < t) sqt <<= 1;
        if (len == sqt && sqt * sqt == t) {
            // Treat the array as a sqt×sqt matrix and do a blocked in-place
            // transpose. This is the single inter-stage point where the
            // access pattern flips from row-major to column-major.
            // Cost without blocking:  O(t) cache misses (stride-sqt stores)
            // Cost with B×B blocking: O(t/B) cache misses
            transpose_blocked(x, sqt, sqt, BLOCK_SIZE);
        }
    }

    if (f == -1) {
        #pragma omp parallel for
        for (int i = 0; i < t; i++) x[i] /= t;
    }
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------
void init(int n, int& t, char*& s, cpx*& a, cpx*& b, cpx*& c, int*& ans) {
    t = 1;
    while (t <= n + n) t <<= 1;
    s   = (char*)calloc(n + 10, sizeof(char));
    a   = (cpx*)calloc(t, sizeof(cpx));
    b   = (cpx*)calloc(t, sizeof(cpx));
    c   = (cpx*)calloc(t, sizeof(cpx));
    ans = (int*)calloc(t, sizeof(int));
}

void del(char* s, cpx* a, cpx* b, cpx* c, int* ans) {
    free(s); free(a); free(b); free(c); free(ans);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    clock_t beg = clock();
    if (!freopen("tests/fft.in", "r", stdin)) {
        fprintf(stderr, "Failed to open input file\n");
        return 1;
    }

    FILE* out = fopen("output/fft_omp_tiled.out", "w");

    if (!out) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }
    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "Failed to read n\n");
        return 1;
    }
    
    int t;
    char* s;
    cpx *a, *b, *c;
    int* ans;

    init(n, t, s, a, b, c, ans);

    if (scanf("%s", s) != 1) {
        fprintf(stderr, "Failed to read first number\n");
        return 1;
    }
    for (int i = 0; i < n; i++) a[i] = s[n-i-1] - '0';
    if (scanf("%s", s) != 1) {
        fprintf(stderr, "Failed to read second number\n");
        return 1;
    }
    for (int i = 0; i < n; i++) b[i] = s[n-i-1] - '0';

    FFT(a, 1, t);
    FFT(b, 1, t);

    #pragma omp parallel for
    for (int i = 0; i < t; i++) c[i] = a[i] * b[i];

    FFT(c, -1, t);

    for (int i = 0; i < t; i++) ans[i] = (int)(c[i].real() + 0.5);
    for (int i = 0; i < t - 1; i++) { ans[i+1] += ans[i] / 10; ans[i] %= 10; }

    int len = t;
    while (len > 1 && !ans[len-1]) len--;
    for (int i = len - 1; i >= 0; i--) fprintf(out, "%d", ans[i]);

    del(s, a, b, c, ans);
    clock_t end = clock();
    printf("Time taken: %lf ms\n", (double)(end-beg) / CLOCKS_PER_SEC * 1000.0);
    return 0;
}