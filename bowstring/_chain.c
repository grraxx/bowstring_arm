/*
 * _chain.c
 * ChainArm.evaluate (hessian=False) in C: the residual and Jacobian of the arm's disk
 * equilibria in the segment base loads. Same equations and output layout as chain.py;
 * compiled because, as in Till et al. (2019, sec. 2.7), the per-evaluation cost is what
 * bounds real-time simulation, and in Python it was ~95% interpreter overhead.
 *
 * Inputs: n segments, loads L (3n), tensions T (2n, row k = (T+, T-)), hole offsets (n),
 *         bending stiffnesses B (n).
 * Outputs (base disk dropped where chain.py drops it):
 *   P     (n+1)*3     disk poses, P_0 = 0
 *   G     3n          dU/dP_1..P_n
 *   dGdL  3n x 3n     row-major
 *   dPdL  3n x 3n     row-major
 *   lg    n*2*3n      tendon length gradients in P_1..P_n
 *   lam   2n          tendon lengths
 *   U     1           energy
 * Returns 0, or 1 if dr/dP is singular.
 */
#include <math.h>
#include <string.h>

#include "_rk4.c"

#define NMAX 6
#define NV (3 * (NMAX + 1))

static int steps_for(double nx, double ny)
{
    double m = fabs(nx) > fabs(ny) ? fabs(nx) : fabs(ny);
    if (m < 1.0) m = 1.0;
    double s = 25.0 * sqrt(m);
    if (s < 200.0) s = 200.0;
    if (s > 4000.0) s = 4000.0;
    return (int)s;
}

/* Solves A X = Bm in place (A m x m, Bm m x m, row-major, leading dim ld). 0 or 1 (singular). */
static int lu_solve(double *A, double *Bm, int m, int ld)
{
    int i, j, k, p;
    for (k = 0; k < m; k++) {
        p = k;
        for (i = k + 1; i < m; i++)
            if (fabs(A[i * ld + k]) > fabs(A[p * ld + k])) p = i;
        if (A[p * ld + k] == 0.0) return 1;
        if (p != k)
            for (j = 0; j < m; j++) {
                double t = A[k * ld + j]; A[k * ld + j] = A[p * ld + j]; A[p * ld + j] = t;
                t = Bm[k * ld + j]; Bm[k * ld + j] = Bm[p * ld + j]; Bm[p * ld + j] = t;
            }
        for (i = k + 1; i < m; i++) {
            double f = A[i * ld + k] / A[k * ld + k];
            if (f == 0.0) continue;
            for (j = k; j < m; j++) A[i * ld + j] -= f * A[k * ld + j];
            for (j = 0; j < m; j++) Bm[i * ld + j] -= f * Bm[k * ld + j];
        }
    }
    for (k = m - 1; k >= 0; k--)
        for (j = 0; j < m; j++) {
            double s = Bm[k * ld + j];
            for (i = k + 1; i < m; i++) s -= A[k * ld + i] * Bm[i * ld + j];
            Bm[k * ld + j] = s / A[k * ld + k];
        }
    return 0;
}

int chain_evaluate(int n, const double *L, const double *T, const double *off, const double *B,
                   double *P, double *Gout, double *dGdL, double *dPdL, double *lgout,
                   double *lam, double *U)
{
    int nv = 3 * (n + 1), m = 3 * n;
    double G[NV] = {0}, Hgeo[NV][NV], A[3 * NMAX][NV], Bdir[NV][3 * NMAX];
    double g[NMAX][3], dg[NMAX][3][3], dpose[NMAX][3][3];
    double lg[NMAX][2][NV];
    double out[17];
    int i, a, b, c, k, s, j;

    memset(Hgeo, 0, sizeof Hgeo);
    memset(A, 0, sizeof A);
    memset(Bdir, 0, sizeof Bdir);
    memset(lg, 0, sizeof lg);
    *U = 0.0;

    /* segments, integrated forward from their loads; disk poses by composition */
    P[0] = P[1] = P[2] = 0.0;
    for (i = 0; i < n; i++) {
        double nx = L[3 * i] / B[i], ny = L[3 * i + 1] / B[i], M0 = L[3 * i + 2] / B[i];
        integrate(nx, ny, M0, steps_for(nx, ny), out, 0);
        /* out: x y t M E, then (dx dy dt dM) for nx, ny, M0 */
        g[i][0] = L[3 * i]; g[i][1] = L[3 * i + 1]; g[i][2] = B[i] * out[3];
        *U += B[i] * out[4];
        for (b = 0; b < 3; b++) {
            for (a = 0; a < 3; a++) dpose[i][a][b] = out[5 + 4 * b + a] / B[i];
            dg[i][0][b] = (b == 0); dg[i][1][b] = (b == 1);
            dg[i][2][b] = out[5 + 4 * b + 3];
        }
        double ca = cos(P[3 * i + 2]), sa = sin(P[3 * i + 2]);
        P[3 * i + 3] = P[3 * i] + ca * out[0] - sa * out[1];
        P[3 * i + 4] = P[3 * i + 1] + sa * out[0] + ca * out[1];
        P[3 * i + 5] = P[3 * i + 2] + out[2];
    }

    /* relative poses: G, geometric Hessian, dr/dP and dG/dL at fixed P */
    for (i = 0; i < n; i++) {
        const double *Pa = P + 3 * i, *Pb = P + 3 * i + 3;
        double co = cos(Pa[2]), si = sin(Pa[2]);
        double dx = Pb[0] - Pa[0], dy = Pb[1] - Pa[1];
        double rx = co * dx + si * dy, ry = -si * dx + co * dy;
        double J[3][6] = {{-co, -si, ry, co, si, 0.0},
                          {si, -co, -rx, -si, co, 0.0},
                          {0.0, 0.0, -1.0, 0.0, 0.0, 1.0}};
        double *gi = g[i];
        double row[6] = {gi[0] * si + gi[1] * co, -gi[0] * co + gi[1] * si, 0.0,
                         -gi[0] * si - gi[1] * co, gi[0] * co - gi[1] * si, 0.0};
        int o = 3 * i;
        for (a = 0; a < 6; a++) {
            for (c = 0; c < 3; c++) G[o + a] += J[c][a] * gi[c];
            Hgeo[o + 2][o + a] += row[a];
            Hgeo[o + a][o + 2] += row[a];
        }
        Hgeo[o + 2][o + 2] += -(gi[0] * rx + gi[1] * ry);
        for (c = 0; c < 3; c++)
            for (a = 0; a < 6; a++) A[o + c][o + a] = J[c][a];
        for (a = 0; a < 6; a++)
            for (b = 0; b < 3; b++) {
                double v = 0.0;
                for (c = 0; c < 3; c++) v += J[c][a] * dg[i][c][b];
                Bdir[o + a][o + b] = v;
            }
    }

    /* tendons: straight chords hole to hole, pair k passes disks 0..k, anchored at k + 1 */
    for (k = 0; k < n; k++) lam[2 * k] = lam[2 * k + 1] = 0.0;
    for (k = 0; k < n; k++)
        for (s = 0; s < 2; s++) {
            double sg = s == 0 ? 1.0 : -1.0, d = off[k] * sg, t = T[2 * k + s];
            for (j = 0; j <= k; j++) {
                const double *pa = P + 3 * j, *pb = P + 3 * j + 3;
                double ca = cos(pa[2]), sa = sin(pa[2]), cb = cos(pb[2]), sb = sin(pb[2]);
                double v0 = (pb[0] - d * sb) - (pa[0] - d * sa);
                double v1 = (pb[1] + d * cb) - (pa[1] + d * ca);
                double Lc = hypot(v0, v1), u0 = v0 / Lc, u1 = v1 / Lc;
                double dpa0 = -d * ca, dpa1 = -d * sa, dpb0 = -d * cb, dpb1 = -d * sb;
                double gl[6] = {-u0, -u1, -(u0 * dpa0 + u1 * dpa1), u0, u1, u0 * dpb0 + u1 * dpb1};
                double D[2][6] = {{-1.0, 0.0, -dpa0, 1.0, 0.0, dpb0},
                                  {0.0, -1.0, -dpa1, 0.0, 1.0, dpb1}};
                double Pm[2][2] = {{(1.0 - u0 * u0) / Lc, -u0 * u1 / Lc},
                                   {-u0 * u1 / Lc, (1.0 - u1 * u1) / Lc}};
                int o = 3 * j;
                lam[2 * k + s] += Lc;
                *U += t * Lc;
                for (a = 0; a < 6; a++) {
                    lg[k][s][o + a] += gl[a];
                    G[o + a] += t * gl[a];
                }
                double PD[2][6];
                for (a = 0; a < 2; a++)
                    for (b = 0; b < 6; b++) PD[a][b] = Pm[a][0] * D[0][b] + Pm[a][1] * D[1][b];
                for (a = 0; a < 6; a++)
                    for (b = 0; b < 6; b++)
                        Hgeo[o + a][o + b] += t * (D[0][a] * PD[0][b] + D[1][a] * PD[1][b]);
                Hgeo[o + 5][o + 5] += t * (u0 * d * sb - u1 * d * cb);
                Hgeo[o + 2][o + 2] -= t * (u0 * d * sa - u1 * d * ca);
            }
        }

    /* drop the base disk; dP/dL = (dr/dP)^-1 blockdiag(dpose/dL) */
    for (a = 0; a < m; a++) {
        Gout[a] = G[3 + a];
        for (b = 0; b < m; b++) dPdL[a * m + b] = 0.0;
    }
    for (i = 0; i < n; i++)
        for (a = 0; a < 3; a++)
            for (b = 0; b < 3; b++) dPdL[(3 * i + a) * m + 3 * i + b] = dpose[i][a][b];
    double Ar[3 * NMAX * 3 * NMAX];
    for (a = 0; a < m; a++)
        for (b = 0; b < m; b++) Ar[a * m + b] = A[a][3 + b];
    if (lu_solve(Ar, dPdL, m, m)) return 1;
    for (a = 0; a < m; a++)
        for (b = 0; b < m; b++) {
            double v = Bdir[3 + a][b];
            for (c = 0; c < m; c++) v += Hgeo[3 + a][3 + c] * dPdL[c * m + b];
            dGdL[a * m + b] = v;
        }
    for (k = 0; k < n; k++)
        for (s = 0; s < 2; s++)
            for (a = 0; a < m; a++) lgout[(2 * k + s) * m + a] = lg[k][s][3 + a];
    return 0;
}
