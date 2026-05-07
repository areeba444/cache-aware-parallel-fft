// FFT with OpenMP parallelism
// Based on serial_fft.cpp
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <time.h>
#include <omp.h> // Include OpenMP header

using namespace std;
typedef complex<double> cpx;
int const N=10000100,M=34000000;
double const PI=acos(-1);
int n,t; char *s;
cpx *a,*b,*c;
int *ans,*pos;

void FFT(cpx x[],int f)
{
    for(int i=0;i<t;i++) if(i<pos[i]) swap(x[i],x[pos[i]]);
    for(int i=1;i<t;i<<=1)
    {
        cpx Wn=cpx(cos(PI/i),f*sin(PI/i));
        #pragma omp parallel for schedule(dynamic)
        for(int j=0;j<t;j+=i<<1)
        {
            cpx w=1;
            for(int k=0;k<i;k++,w*=Wn)
            {
                cpx p=x[j+k],q=w*x[j+k+i];
                x[j+k]=p+q,x[j+k+i]=p-q;
            }
        }
    }
    if(f==-1) for(int i=0;i<t;i++) x[i]/=t;
}

// void FFT(cpx x[], int f)
// {
//     for(int i = 0; i < t; i++)
//         if(i < pos[i]) swap(x[i], x[pos[i]]);

//     for(int i = 1; i < t; i <<= 1)
//     {
//         cpx Wn = cpx(cos(PI/i), f*sin(PI/i));
//         int half = i, full = i << 1;

//         // Only parallelize when there's enough work to amortize thread overhead
//         if(full <= PARALLEL_THRESHOLD)
//         {
//             // Small stages: serial is faster, avoid thread overhead
//             for(int j = 0; j < t; j += full)
//             {
//                 cpx w = 1;
//                 for(int k = 0; k < half; k++, w *= Wn)
//                 {
//                     cpx p = x[j+k], q = w * x[j+k+half];
//                     x[j+k] = p+q;  x[j+k+half] = p-q;
//                 }
//             }
//         }
//         else
//         {
//             // Large stages: parallelize the k loop across all butterfly groups
//             // Flatten j+k into a single index to maximize parallel granularity
//             #pragma omp parallel for schedule(static)
//             for(int k = 0; k < half; k++)
//             {
//                 cpx w = cpx(cos(PI*k/i), f*sin(PI*k/i));  // precompute per-thread
//                 for(int j = 0; j < t; j += full)
//                 {
//                     cpx p = x[j+k], q = w * x[j+k+half];
//                     x[j+k] = p+q;  x[j+k+half] = p-q;
//                 }
//             }
//         }
//     }

//     if(f == -1)
//     {
//         #pragma omp parallel for schedule(static)
//         for(int i = 0; i < t; i++) x[i] /= t;
//     }
// }

void init(){
    s = (char*)calloc(n+10, sizeof(char));
    a = (cpx*)calloc(t,sizeof(cpx));
    b = (cpx*)calloc(t,sizeof(cpx));
    c = (cpx*)calloc(t,sizeof(cpx));
    ans = (int*)calloc(t,sizeof(int));
    pos = (int*)calloc(t,sizeof(int));
}

void del(){
    free(s);free(a);free(b);
    free(c);free(ans);free(pos);
}

static void print_metrics(const char* config_name, int transform_size,
                          double elapsed_seconds) {
    const double log_terms = log2((double)transform_size);
    const double flops = 15.0 * transform_size * log_terms + 6.0 * transform_size;
    const double bytes = 6.0 * transform_size * sizeof(cpx);
    const double gflops = flops / elapsed_seconds / 1e9;
    const double ai = flops / bytes;
    printf("%s,%d,%.6f,%.6f,%.6f\n",
           config_name, transform_size, elapsed_seconds, gflops, ai);
}

int main()
{
    if (!freopen("tests/fft.in", "r", stdin)) {
        fprintf(stderr, "Failed to open input file\n");
        return 1;
    }

    FILE *out = fopen("output/fft_omp.out", "w");

    if (!out) {
        fprintf(stderr, "Failed to open output file\n");
        return 1;
    }

    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "Failed to read n\n");
        return 1;
    }
    t=1; int n0=0; while(t<=n+n) t<<=1,n0++;
    init();

    if (scanf("%s", s) != 1) {
        fprintf(stderr, "Failed to read first number\n");
        return 1;
    }
    for(int i=0;i<n;i++) a[i]=s[n-i-1]-'0';
    if (scanf("%s", s) != 1) {
        fprintf(stderr, "Failed to read second number\n");
        return 1;
    }
    for(int i=0;i<n;i++) b[i]=s[n-i-1]-'0';
    for(int i=0;i<t;i++) pos[i]=(pos[i>>1]>>1)|((i&1)<<(n0-1));
    
    const int transform_size = t;
    struct timespec start_time;
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    FFT(a,1),FFT(b,1);
    
    for(int i=0;i<t;i++) c[i]=a[i]*b[i];
    
    FFT(c,-1);
    
    for(int i=0;i<t;i++) ans[i]=(int)(c[i].real()+0.5);
    for(int i=0;i<t;i++) ans[i+1]+=ans[i]/10,ans[i]%=10;
    
    int len = t;
    while(len > 1 && !ans[len-1]) len--;
    
    for(int i=len-1;i>=0;i--) fprintf(out,"%d",ans[i]);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    del();

    const double elapsed_seconds = (end_time.tv_sec - start_time.tv_sec)
        + (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
    print_metrics("omp", transform_size, elapsed_seconds);
    
    return 0;
}
