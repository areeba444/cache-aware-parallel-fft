// ============================================================================
// Cache-Aware Parallel FFT Project
// AVX2 FFT implementation with OpenMP, caching, and SIMD butterfly updates.
// ============================================================================
// FFT with OpenMP, Tiled Transpose, and AVX2 SIMD Optimizations
// ---------------------------------------------------------------
// BUGS FIXED (all three caused wrong output):
//
// BUG 1 — Wrong twiddle vector: w_vec loaded the SAME w for both lanes.
//   The 256-bit register processes two complex numbers simultaneously:
//   lane0 = x[i+j], lane1 = x[i+j+1].
//   lane0 needs twiddle w, but lane1 needs twiddle w*w_len.
//   Old code:  _mm256_set_pd(w.imag(), w.real(), w.imag(), w.real())
//              ↑ identical twiddle in both 128-bit lanes — lane1 was wrong.
//   Fix: compute w1 = w * w_len separately and pack [w1.im, w1.re, w.im, w.re].
//
// BUG 2 — Wrong blend mask reconstructing interleaved result:
//   Memory layout of complex<double>: [re0, im0, re1, im1] (doubles).
//   After computing q_mul_w_real = [re0, re0, re1, re1]
//                   q_mul_w_imag = [im0, im0, im1, im1]
//   we need result  q_mul_w      = [re0, im0, re1, im1].
//   Old mask 0b1010 picked lanes 1,3 from imag → [re0, im0, re0, im0]  WRONG.
//   Fix: after the subtract/add we have:
//     q_mul_w_real = [r0-i0, r0-i0, r1-i1, r1-i1]  (duplicated reals)
//     q_mul_w_imag = [r0+i0, r0+i0, r1+i1, r1+i1]  (duplicated imags)
//   We want even indices (0,2) from _real and odd indices (1,3) from _imag.
//   blend mask bit=0 → take from first src, bit=1 → take from second src.
//   Lanes: 3  2  1  0  (AVX bit ordering)
//   Want:  im re im re → mask = 0b1010 picks lanes 1,3 from second src.
//   Wait — but the shuffle is duplicating, so lane layout is:
//     q_mul_w_real doubles: [lane0=real0, lane1=real0, lane2=real1, lane3=real1]
//     q_mul_w_imag doubles: [lane0=imag0, lane1=imag0, lane2=imag1, lane3=imag1]
//   We want output:         [real0,       imag0,       real1,       imag1      ]
//                  lanes:    0             1            2            3
//   So: take lane0 from real, lane1 from imag, lane2 from real, lane3 from imag
//   blend mask (bit i=1 means take from src2=imag): bits 3210 = 1010 = 0xA
//   That IS 0b1010 — so the mask was actually correct for this layout!
//   The real problem with blend was that q_mul_w_real after _mm256_sub_pd is
//   NOT [r,r,r,r] — it is [r0-x, r0-x, r1-x, r1-x] ONLY if we shuffled to
//   broadcast reals. Let's trace carefully:
//     q_vec_interleaved = [re(q0), im(q0), re(q1), im(q1)]
//     shuffle mask 0b0000: for each 128-bit lane, picks element 0 of each pair
//       → [re(q0), re(q0), re(q1), re(q1)]   ✓ that's q_real
//     shuffle mask 0b1111: picks element 1 of each pair
//       → [im(q0), im(q0), im(q1), im(q1)]   ✓ that's q_imag
//   Similarly w_real = [wr0, wr0, wr1, wr1], w_imag = [wi0, wi0, wi1, wi1]
//   term1 = q_real * w_real = [re(q0)*wr0, re(q0)*wr0, re(q1)*wr1, re(q1)*wr1]
//   term2 = q_imag * w_imag = [im(q0)*wi0, im(q0)*wi0, im(q1)*wi1, im(q1)*wi1]
//   q_mul_w_real = term1-term2 = [A, A, B, B]  where A=re(q0*w0), B=re(q1*w1)
//   q_mul_w_imag = term3+term4 = [C, C, D, D]  where C=im(q0*w0), D=im(q1*w1)
//   blend 0b1010: lane0←real(A), lane1←imag(C), lane2←real(B), lane3←imag(D)
//   result = [A, C, B, D] = [re(q0*w0), im(q0*w0), re(q1*w1), im(q1*w1)] ✓
//   So blend mask WAS correct. The actual corruption came from Bugs 1 and 3.
//
// BUG 3 — Twiddle advance at end of loop body advanced by only 1 step:
//   The inner loop does j += 2, processing two elements per iteration.
//   After handling x[i+j] (needs w) and x[i+j+1] (needs w*w_len),
//   the next iteration needs w' = w * w_len^2.
//   Old code: w *= w_len  → only one step, so next iteration uses wrong twiddle.
//   Fix: advance by two steps: w *= w_len * w_len  (precomputed as w_len2).
//
// BUG 4 (edge case) — When half_len is odd (can't happen in radix-2, but
//   guarding the tail: if half_len is not a multiple of 2, the last element
//   must be handled with scalar fallback. Added tail loop for safety.

#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cmath>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

using namespace std;
typedef complex<double> cpx;

const int BLOCK_SIZE = 16;
static int g_fft_min_groups = 0;
static vector<unsigned int> g_bitrev;
static int g_bitrev_t = 0;

static int read_env_int(const char* name, int default_value) {
    const char* v = getenv(name);
    if (!v || !*v) return default_value;
    char* end = nullptr;
    long x = strtol(v, &end, 10);
    if (end == v || x <= 0) return default_value;
    return (int)x;
}

// ---------------------------------------------------------------------------
// Bit-reversal permutation (parallelised, same as fft_avx original)
// ---------------------------------------------------------------------------
static unsigned int reverse_bits(unsigned int n, unsigned int bits) {
    unsigned int r = 0;
    for (unsigned int i = 0; i < bits; i++) {
        if ((n >> i) & 1) r |= 1u << (bits - 1 - i);
    }
    return r;
}

static void build_bitrev_cache(int t) {
    if (g_bitrev_t == t && !g_bitrev.empty()) return;
    int logt = 0;
    while ((1 << logt) < t) logt++;
    g_bitrev.assign((size_t)t, 0u);
    for (int i = 0; i < t; i++) g_bitrev[(size_t)i] = reverse_bits((unsigned int)i, (unsigned int)logt);
    g_bitrev_t = t;
}

static void bit_reversal_permutation(cpx* x, int t) {
    build_bitrev_cache(t);

    // Note: parallel swap is safe for bit-reversal because each pair (i,j=rev(i))
    // is disjoint — no two pairs share an index.
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < t; i++) {
        unsigned int j = g_bitrev[(size_t)i];
        if ((int)j > i) swap(x[i], x[j]);
    }
}

// ---------------------------------------------------------------------------
// FFT_AVX — corrected
// ---------------------------------------------------------------------------
void FFT_AVX(cpx x[], int f, int t) {
    // Step 1: bit-reversal permutation
    bit_reversal_permutation(x, t);

    // Step 2: butterfly stages
    for (int len = 2; len <= t; len <<= 1) {
        int half_len = len >> 1;
        const int groups = t / len;

        // Twiddle root for this stage: e^{i * f * pi / half_len}
        double angle = f * M_PI / half_len;
        cpx w_len(cos(angle), sin(angle));

        // Stage twiddle cache to reduce loop-carried dependency on w*=w_len
        vector<cpx> tw(half_len);
        tw[0] = cpx(1.0, 0.0);
        for (int k = 1; k < half_len; k++) tw[k] = tw[k - 1] * w_len;

        if (groups >= g_fft_min_groups) {
            #pragma omp parallel for schedule(runtime)
            for (int i = 0; i < t; i += len) {
                int j = 0;

                // --- AVX2 vectorised path: process 2 complex numbers per iteration ---
                for (; j + 1 < half_len; j += 2) {
                const cpx& w = tw[j];
                const cpx& w1 = tw[j + 1];
                // FIX BUG 1: pack TWO different twiddle factors into the 256-bit reg.
                // Memory order expected by storeu/loadu: [re0, im0, re1, im1]
                // _mm256_set_pd fills in REVERSE lane order: arg3,arg2,arg1,arg0
                // We want: lane0=w.re, lane1=w.im, lane2=w1.re, lane3=w1.im
                // So:      set_pd(w1.im, w1.re, w.im, w.re)
                __m256d w_vec = _mm256_set_pd(w1.imag(), w1.real(),
                                               w.imag(),  w.real());

                // Load p = [re(p0), im(p0), re(p1), im(p1)]
                __m256d p_vec = _mm256_loadu_pd((double*)&x[i + j]);
                // Load q = [re(q0), im(q0), re(q1), im(q1)]
                __m256d q_vec = _mm256_loadu_pd((double*)&x[i + j + half_len]);

                // Complex multiplication q * w (two pairs simultaneously):
                //   result_re = q.re * w.re - q.im * w.im
                //   result_im = q.re * w.im + q.im * w.re
                //
                // Broadcast reals and imags within each 128-bit lane:
                //   shuffle mask 0b0000: for each 64-bit pair in each 128-bit lane,
                //     pick index 0 → duplicates the lower double of each pair
                //   shuffle mask 0b1111: picks index 1 → duplicates upper double
                __m256d q_re = _mm256_shuffle_pd(q_vec, q_vec, 0b0000); // [qr0,qr0,qr1,qr1]
                __m256d q_im = _mm256_shuffle_pd(q_vec, q_vec, 0b1111); // [qi0,qi0,qi1,qi1]
                __m256d w_re = _mm256_shuffle_pd(w_vec, w_vec, 0b0000); // [wr0,wr0,wr1,wr1]
                __m256d w_im = _mm256_shuffle_pd(w_vec, w_vec, 0b1111); // [wi0,wi0,wi1,wi1]

                // Compute real and imag parts of q*w (each result duplicated):
                __m256d res_re = _mm256_sub_pd(_mm256_mul_pd(q_re, w_re),
                                               _mm256_mul_pd(q_im, w_im));
                // res_re = [A, A, B, B]  A=re(q0*w0), B=re(q1*w1)

                __m256d res_im = _mm256_add_pd(_mm256_mul_pd(q_re, w_im),
                                               _mm256_mul_pd(q_im, w_re));
                // res_im = [C, C, D, D]  C=im(q0*w0), D=im(q1*w1)

                // Interleave: blend mask 0b1010 →
                //   bit0=0: lane0 from res_re → A
                //   bit1=1: lane1 from res_im → C
                //   bit2=0: lane2 from res_re → B
                //   bit3=1: lane3 from res_im → D
                //   result = [A, C, B, D] = [re(q0w0), im(q0w0), re(q1w1), im(q1w1)] ✓
                __m256d q_mul_w = _mm256_blend_pd(res_re, res_im, 0b1010);

                // Butterfly: p±q*w
                _mm256_storeu_pd((double*)&x[i + j],            _mm256_add_pd(p_vec, q_mul_w));
                _mm256_storeu_pd((double*)&x[i + j + half_len], _mm256_sub_pd(p_vec, q_mul_w));

                }

                // --- Scalar tail: handles last element when half_len is odd ---
                // (In radix-2 FFT half_len is always a power of 2, so this only
                //  fires when half_len == 1, i.e. len == 2, where j starts at 0
                //  and the AVX loop requires j+1 < 1 = false → scalar handles it.)
                for (; j < half_len; j++) {
                    const cpx& w = tw[j];
                    cpx p = x[i + j];
                    cpx q = w * x[i + j + half_len];
                    x[i + j]            = p + q;
                    x[i + j + half_len] = p - q;
                }
            }
        } else {
            for (int i = 0; i < t; i += len) {
                int j = 0;
                for (; j + 1 < half_len; j += 2) {
                    const cpx& w = tw[j];
                    const cpx& w1 = tw[j + 1];
                    __m256d w_vec = _mm256_set_pd(w1.imag(), w1.real(),
                                                   w.imag(),  w.real());
                    __m256d p_vec = _mm256_loadu_pd((double*)&x[i + j]);
                    __m256d q_vec = _mm256_loadu_pd((double*)&x[i + j + half_len]);
                    __m256d q_re = _mm256_shuffle_pd(q_vec, q_vec, 0b0000);
                    __m256d q_im = _mm256_shuffle_pd(q_vec, q_vec, 0b1111);
                    __m256d w_re = _mm256_shuffle_pd(w_vec, w_vec, 0b0000);
                    __m256d w_im = _mm256_shuffle_pd(w_vec, w_vec, 0b1111);
                    __m256d res_re = _mm256_sub_pd(_mm256_mul_pd(q_re, w_re),
                                                   _mm256_mul_pd(q_im, w_im));
                    __m256d res_im = _mm256_add_pd(_mm256_mul_pd(q_re, w_im),
                                                   _mm256_mul_pd(q_im, w_re));
                    __m256d q_mul_w = _mm256_blend_pd(res_re, res_im, 0b1010);
                    _mm256_storeu_pd((double*)&x[i + j],            _mm256_add_pd(p_vec, q_mul_w));
                    _mm256_storeu_pd((double*)&x[i + j + half_len], _mm256_sub_pd(p_vec, q_mul_w));
                }
                for (; j < half_len; j++) {
                    const cpx& w = tw[j];
                    cpx p = x[i + j];
                    cpx q = w * x[i + j + half_len];
                    x[i + j]            = p + q;
                    x[i + j + half_len] = p - q;
                }
            }
        }
    }

    if (f == -1) {
        if (t >= g_fft_min_groups * 2) {
            #pragma omp parallel for schedule(runtime)
            for (int i = 0; i < t; i++) x[i] /= (double)t;
        } else {
            for (int i = 0; i < t; i++) x[i] /= (double)t;
        }
    }
}

// ---------------------------------------------------------------------------
// Memory management
// ---------------------------------------------------------------------------
void init(int n, int& t, char*& s, cpx*& a, cpx*& b, cpx*& c, int*& ans) {
    t = 1;
    while (t <= n + n) t <<= 1;
    s   = (char*)aligned_alloc(32, (size_t)(n + 10) * sizeof(char));
    a   = (cpx* )aligned_alloc(32, (size_t)t * sizeof(cpx));
    b   = (cpx* )aligned_alloc(32, (size_t)t * sizeof(cpx));
    c   = (cpx* )aligned_alloc(32, (size_t)t * sizeof(cpx));
    ans = (int* )aligned_alloc(32, (size_t)t * sizeof(int));
    memset(a, 0, t * sizeof(cpx));
    memset(b, 0, t * sizeof(cpx));
    memset(c, 0, t * sizeof(cpx));
    memset(ans, 0, t * sizeof(int));
}

void del(char* s, cpx* a, cpx* b, cpx* c, int* ans) {
    free(s); free(a); free(b); free(c); free(ans);
}

static void print_metrics(const char* config_name, int transform_size,
                          double elapsed_seconds) {
    const double log_terms = log2((double)transform_size);
    const double flops = 15.0 * transform_size * log_terms + 6.0 * transform_size;
    const double bytes = 6.0 * transform_size * sizeof(cpx);
    const double gflops = flops / elapsed_seconds / 1e9;
    const double ai = flops / bytes;
    printf("%s,%d,%.9f,%.6f,%.6f\n",
           config_name, transform_size, elapsed_seconds, gflops, ai);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    int fft_chunk = read_env_int("FFT_OMP_CHUNK", 16);
    const char* sched_env = getenv("FFT_OMP_SCHEDULE");
    omp_sched_t sched_kind = omp_sched_dynamic;
    if (sched_env && strcmp(sched_env, "static") == 0) sched_kind = omp_sched_static;
    omp_set_schedule(sched_kind, fft_chunk);
    g_fft_min_groups = read_env_int("FFT_MIN_GROUPS", 8 * omp_get_max_threads());

    if (!freopen("tests/fft.in", "r", stdin)) {
        fprintf(stderr, "Failed to open input file\n");
        return 1;
    }

    FILE* out = fopen("output/fft_avx.out", "w");

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

    const int transform_size = t;
    struct timespec start_time;
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    FFT_AVX(a,  1, t);
    FFT_AVX(b,  1, t);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < t; i++) c[i] = a[i] * b[i];

    FFT_AVX(c, -1, t);

    for (int i = 0; i < t; i++)     ans[i] = (int)(c[i].real() + 0.5);
    for (int i = 0; i < t - 1; i++) { ans[i+1] += ans[i] / 10; ans[i] %= 10; }

    int len = t;
    while (len > 1 && !ans[len-1]) len--;

    clock_gettime(CLOCK_MONOTONIC, &end_time);

    for (int i = len - 1; i >= 0; i--) fprintf(out, "%d", ans[i]);

    del(s, a, b, c, ans);

    const double elapsed_seconds = (end_time.tv_sec - start_time.tv_sec)
        + (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
    print_metrics("avx", transform_size, elapsed_seconds);
    return 0;
}