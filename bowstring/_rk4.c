/*
 * _rk4.c
 * RK4 integration of one elastica segment and its variational equations
 * (the C version of segment.integrate; same equations, same state order).
 *
 * state: x, y, t, M, E, then (dx, dy, dt, dM) for each of nx, ny, M0   (17 doubles)
 *   x' = cos t, y' = sin t, t' = M, M' = nx sin t - ny cos t, E' = M^2 / 2
 */
#include <math.h>

static void rhs(const double *s, double nx, double ny, double *o)
{
    double t = s[2], M = s[3];
    double c = cos(t), sn = sin(t);
    double k = nx * c + ny * sn;
    o[0] = c;
    o[1] = sn;
    o[2] = M;
    o[3] = nx * sn - ny * c;
    o[4] = 0.5 * M * M;
    o[5] = -sn * s[7];  o[6] = c * s[7];  o[7] = s[8];  o[8] = k * s[7] + sn;
    o[9] = -sn * s[11]; o[10] = c * s[11]; o[11] = s[12]; o[12] = k * s[11] - c;
    o[13] = -sn * s[15]; o[14] = c * s[15]; o[15] = s[16]; o[16] = k * s[15];
}

/* Integrates over [0, 1] in N steps. out: the 17 end values. thetas: N + 1 angles, or 0. */
void integrate(double nx, double ny, double M0, int N, double *out, double *thetas)
{
    double s[17] = {0, 0, 0, M0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    double k1[17], k2[17], k3[17], k4[17], tmp[17];
    double h = 1.0 / N;
    int i, j;
    if (thetas) thetas[0] = 0.0;
    for (i = 0; i < N; i++) {
        rhs(s, nx, ny, k1);
        for (j = 0; j < 17; j++) tmp[j] = s[j] + 0.5 * h * k1[j];
        rhs(tmp, nx, ny, k2);
        for (j = 0; j < 17; j++) tmp[j] = s[j] + 0.5 * h * k2[j];
        rhs(tmp, nx, ny, k3);
        for (j = 0; j < 17; j++) tmp[j] = s[j] + h * k3[j];
        rhs(tmp, nx, ny, k4);
        for (j = 0; j < 17; j++)
            s[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        if (thetas) thetas[i + 1] = s[2];
    }
    for (j = 0; j < 17; j++) out[j] = s[j];
}
