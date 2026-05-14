/* bench_policy.c — Time the C inference, also dump output for parity check. */
#include "policy_inference.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    float obs[POLICY_OBS_DIM];
    float action[POLICY_ACT_DIM];

    /* Read 20 obs values from stdin (one per line) for parity testing */
    if (argc > 1 && argv[1][0] == 'p') {
        for (int i = 0; i < POLICY_OBS_DIM; ++i) {
            if (scanf("%f", &obs[i]) != 1) { fprintf(stderr, "bad input\n"); return 1; }
        }
        policy_forward(obs, action);
        for (int i = 0; i < POLICY_ACT_DIM; ++i) printf("%.8f\n", action[i]);
        return 0;
    }

    /* Timing mode: random obs, 100k forward passes */
    srand(42);
    for (int i = 0; i < POLICY_OBS_DIM; ++i) obs[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    /* Warmup */
    for (int k = 0; k < 1000; ++k) policy_forward(obs, action);

    /* Time 100k runs */
    const int N = 100000;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k = 0; k < N; ++k) {
        obs[0] += 0.0001f;  /* prevent compiler from CSE'ing */
        policy_forward(obs, action);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    printf("Mean over %d calls: %.3f us\n", N, ns / N / 1000.0);
    printf("Action: %+.4f %+.4f %+.4f %+.4f\n", action[0], action[1], action[2], action[3]);
    return 0;
}
