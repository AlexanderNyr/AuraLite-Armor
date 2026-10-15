// ============================================================================
//  sph_core2.cpp — Armor-penetration SPH solver (2D + 3D) with hybrid TL/UL,
//  Johnson–Cook D1–D5 damage, a Mie–Grüneisen EOS, leapfrog integration,
//  adaptive time steps, symmetric force accumulation, and Morton Z-order locality.
//
//  Build:
//    g++ -O3 -march=native -fopenmp -std=c++17 sph_core2.cpp -o sph_core2
//    g++ -O3 -march=native -fopenmp -std=c++17 -DDIM=3 sph_core2.cpp -o sph_core3
//
//  Model summary:
//   * Hybrid: Total-Lagrangian SPH (F, PK1 = sigma*Cof(F), renormalization B0)
//     for elastic/solid response + Updated-Lagrangian SPH (hydrodynamic response)
//     in regions of large deformation. TL->UL switching is one-way and triggered by
//     (strain rate > sw) | (J outside [0.55,1.8]) | (D > 0.75).
//   * Johnson–Cook plasticity (A,B,n,C,m) + D1–D5 damage with stress triaxiality,
//     damage softening Y*(1-D); Taylor–Quinney factor 0.9 converts plastic work to heat.
//   * Mie–Grüneisen EOS: P_H = rho0 c0^2 mu/(1-(s-1)mu)^2 in compression, linear
//     tension, and a bounded thermal pressure term Gamma*rho0*cp*(T-T0).
//   * Monaghan artificial stress (pairwise pressure-like term) + XSPH.
//   * Leapfrog (velocity-Verlet) with adaptive dt (CFL + sqrt(h/|a|)).
//
//  Performance notes:
//   * one CSR neighbor-list pass per step (allocated capacity is reused;
//     full rebuilds are infrequent);
//   * symmetric UL pairs: each pair is computed once and contributes to both
//     particles through thread-private arrays (bitwise comparison to the naive v2 path);
//   * Z-order (Morton) array reordering every z steps improves
//     cache locality for large particle counts; TL reference lists store
//     stable particle IDs, preserving the invariant that projectile particles come first.
//
//  Output: <out>/meta.txt (VER=2, DIM, NFIELDS=8|9), frames.bin,
//         <out>/vtk/*.vtk (vtk=1), ./summary.csv. The stdout protocol is compatible with v1.
//
//  Parameters: see parseArgs. 3D adds w_plate (plate width along z, mm).
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Spatial dimension: 2 (plane strain) or 3 (full 3D)
#ifndef DIM
#define DIM 2
#endif

// pass-level profiling (prof=1)
static int PROF = 0;
static double T_CSR=0, T_RATE=0, T_UPD=0, T_FORCE=0, T_SORT=0, T_KICK=0;
static long N_COLD=0;
static double NOWSEC() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ----------------------------- materials -----------------------------------
struct Material {
    const char* name;
    double rho0, c0, s; // shock Hugoniot: Us = c0 + s*up
    double Gamma;       // Grüneisen parameter
    double G;           // shear modulus, Pa
    double A, B, n;     // Johnson–Cook yield parameters
    double C, m;        // rate / thermal exponents
    double Tm, cp;
    double D1, D2, D3, D4, D5; // Johnson–Cook damage parameters
};

static Material MATERIALS[] = {
//  name       rho0   c0    s    Gamma  G      A        B        n     C      m    Tm   cp    D1     D2    D3     D4    D5
    {"steel",    7850, 4570, 1.49, 2.0,  80e9, 1.20e9, 0.50e9, 0.26, 0.014, 1.03, 1793, 477, 0.05,  3.44, -2.12, 0.002, 0.61},
    {"tungsten",17600, 4030, 1.24, 1.6, 160e9, 1.51e9, 0.18e9, 0.12, 0.016, 1.00, 1723, 134, 0.09,  0.33, -1.50, 0.012, 1.00},
    {"rha",      7850, 4570, 1.49, 2.0,  80e9, 0.78e9, 0.51e9, 0.26, 0.014, 1.03, 1793, 477, 0.10,  2.00, -1.80, 0.002, 0.61},
    {"hha",      7850, 4570, 1.49, 2.2,  80e9, 1.50e9, 0.35e9, 0.20, 0.014, 1.03, 1793, 477, 0.02,  0.60, -1.00, 0.000, 0.30},
    {"aluminum", 2660, 5240, 1.40, 2.0,26.9e9, 0.17e9, 0.43e9, 0.68, 0.010, 1.70,  893, 910, 0.0261,0.49, -1.618,0.013, 1.64},
};
static int matIndex(const std::string& s) {
    for (size_t k = 0; k < sizeof(MATERIALS)/sizeof(Material); ++k)
        if (s == MATERIALS[k].name) return (int)k;
    fprintf(stderr, "Unknown material: %s\n", s.c_str());
    exit(1);
}

// ----------------------------- parameters ------------------------------------
struct Params {
    double v_impact  = 800.0;   // m/s
    double angle_deg = 60.0;    // plate obliquity from vertical, degrees
    double t_plate   = 15.0;    // mm
    double h_plate   = 120.0;   // mm (plate extent in the impact plane)
    double w_plate   = 60.0;    // mm (width along z, 3D only)
    double proj_l    = 20.0;    // mm
    double proj_d    = 8.0;     // mm
    double dx        = 0.5;     // mm
    double t_end     = 60.0;    // μs
    int    n_frames  = 70;
    std::string proj_mat = "steel";
    std::string arm_mat  = "rha";
    std::string nose     = "flat";   // flat | ogive | pointed
    std::string out      = "run";
    // --- v2 ---
    int    use_tl    = 1;       // 1 = hybrid TL/UL; 0 = hydrodynamics only
    double cfl       = 0.3;     // Courant number
    double xs_eps    = 0.3;     // XSPH (0 = disabled)
    double as_chi    = 0.3;     // artificial stress (0 = disabled)
    double sw_edot   = 5e4;     // TL->UL threshold, 1/s
    int    vtk       = 0;
    // --- v3 ---
    int    z_every   = 100;     // Z-order sort every N steps (0 = disabled)
};

static Params parseArgs(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto eq = a.find('=');
        if (eq == std::string::npos) continue;
        std::string k = a.substr(0, eq), v = a.substr(eq + 1);
        if      (k == "v")        p.v_impact  = atof(v.c_str());
        else if (k == "angle")    p.angle_deg = atof(v.c_str());
        else if (k == "t_plate")  p.t_plate   = atof(v.c_str());
        else if (k == "h_plate")  p.h_plate   = atof(v.c_str());
        else if (k == "w_plate")  p.w_plate   = atof(v.c_str());
        else if (k == "proj_l")   p.proj_l    = atof(v.c_str());
        else if (k == "proj_d")   p.proj_d    = atof(v.c_str());
        else if (k == "dx")       p.dx        = atof(v.c_str());
        else if (k == "t_end")    p.t_end     = atof(v.c_str());
        else if (k == "frames")   p.n_frames  = atoi(v.c_str());
        else if (k == "proj_mat") p.proj_mat  = v;
        else if (k == "arm_mat")  p.arm_mat   = v;
        else if (k == "nose")     p.nose      = v;
        else if (k == "out")      p.out       = v;
        else if (k == "tl")       p.use_tl    = atoi(v.c_str());
        else if (k == "cfl")      p.cfl       = atof(v.c_str());
        else if (k == "xs")       p.xs_eps    = atof(v.c_str());
        else if (k == "as")       p.as_chi    = atof(v.c_str());
        else if (k == "sw")       p.sw_edot   = atof(v.c_str());
        else if (k == "vtk")      p.vtk       = atoi(v.c_str());
        else if (k == "z")        p.z_every   = atoi(v.c_str());
        else if (k == "prof")     PROF        = atoi(v.c_str());
    }
    return p;
}

// bit interleaving for Morton keys
static inline uint64_t part1by2(uint32_t x) {
    uint64_t v = x;
    v = (v | (v << 16)) & 0x0000FFFF0000FFFFULL;
    v = (v | (v << 8))  & 0x00FF00FF00FF00FFULL;
    v = (v | (v << 4))  & 0x0F0F0F0F0F0F0F0FULL;
    v = (v | (v << 2))  & 0x3333333333333333ULL;
    v = (v | (v << 1))  & 0x5555555555555555ULL;
    return v;
}
static inline uint64_t morton3(uint32_t x, uint32_t y, uint32_t z) {
    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2);
}

// ============================ main solver ==================================
int main(int argc, char** argv) {
    Params par = parseArgs(argc, argv);
    if (!(par.dx > 0.0) || !(par.t_end > 0.0) || par.n_frames < 1 ||
        !(par.t_plate > 0.0) || !(par.h_plate > 0.0) ||
        !(par.proj_l > 0.0) || !(par.proj_d > 0.0) ||
        (DIM == 3 && !(par.w_plate > 0.0))) {
        fprintf(stderr, "invalid geometry/time parameters\n");
        return 2;
    }
    const double DX = par.dx * 1e-3;
    const double H  = 1.3 * DX;
    const double H2 = 2.0 * H;
    // cubic spline normalization: 2D — 10/(7 pi h^2), 3D — 1/(pi h^3)
    const double SIGK = (DIM == 2) ? 10.0/(7.0*M_PI*H*H) : 1.0/(M_PI*H*H*H);
    const double T0K = 293.0;
    const double V0  = (DIM == 2) ? DX*DX : DX*DX*DX;
    const long   MAX_NB = (DIM == 2) ? 256 : 512;
    const long   NB_PAD = (DIM == 2) ? 8 : 20;

    const Material& MP = MATERIALS[matIndex(par.proj_mat)];
    const Material& MA = MATERIALS[matIndex(par.arm_mat)];

    // ---------------- particle generation ----------------
    std::vector<double> X, Y, Z; std::vector<int> matid; std::vector<char> fixv;
    const double th = par.angle_deg * M_PI / 180.0;
    const double nx_ = cos(th), ny_ = sin(th);
    const double tx_ = -sin(th), ty_ = cos(th);

    const double T_ = par.t_plate * 1e-3, HH = par.h_plate * 1e-3;
    const double WW = par.w_plate * 1e-3;
    for (double xi = DX/2; xi < T_; xi += DX)
        for (double s = -HH/2 + DX/2; s < HH/2; s += DX) {
            char fx = (fabs(s) > HH/2 - 3*DX) ? 1 : 0;
            if (DIM == 2) {
                X.push_back(xi*nx_ + s*tx_);
                Y.push_back(xi*ny_ + s*ty_);
                Z.push_back(0.0);
                matid.push_back(1); fixv.push_back(fx);
            } else {
                for (double u = -WW/2 + DX/2; u < WW/2; u += DX) {
                    X.push_back(xi*nx_ + s*tx_);
                    Y.push_back(xi*ny_ + s*ty_);
                    Z.push_back(u);
                    matid.push_back(1); fixv.push_back(fx);
                }
            }
        }
    const size_t NA = X.size();
    if (NA == 0) {
        fprintf(stderr, "empty armor geometry: increase h_plate/t_plate or decrease dx\n");
        return 2;
    }

    const double L = par.proj_l*1e-3, D = par.proj_d*1e-3, R = D/2;
    std::vector<double> px, py, pz;
    for (double x = -L + DX/2; x < 0; x += DX) {
        if (DIM == 2) {
            for (double y = -R + DX/2; y < R; y += DX) {
                bool keep = true;
                if (par.nose == "ogive") {
                    double rc = R;
                    if (x > -rc) keep = ((x+rc)*(x+rc) + y*y <= rc*rc);
                } else if (par.nose == "pointed") {
                    double nl = 1.2 * D;
                    if (x > -nl) keep = (fabs(y) <= R * (-x) / nl);
                }
                if (keep) { px.push_back(x); py.push_back(y); pz.push_back(0.0); }
            }
        } else {
            for (double y = -R + DX/2; y < R; y += DX)
                for (double z = -R + DX/2; z < R; z += DX) {
                    if (y*y + z*z > R*R) continue;      // circular cross-section
                    bool keep = true;
                    if (par.nose == "ogive") {
                        double rc = R;
                        if (x > -rc) keep = ((x+rc)*(x+rc) + y*y + z*z <= rc*rc);
                    } else if (par.nose == "pointed") {
                        double nl = 1.2 * D;
                        if (x > -nl) keep = (sqrt(y*y+z*z) <= R * (-x) / nl);
                    }
                    if (keep) { px.push_back(x); py.push_back(y); pz.push_back(z); }
                }
        }
    }
    double xmin_arm = 1e9;
    for (size_t k = 0; k < NA; ++k)
        if (fabs(Y[k]) < D && (DIM == 2 || fabs(Z[k]) < D))
            xmin_arm = std::min(xmin_arm, X[k]);
    const double gap = 1.5 * DX;
    const size_t NP = px.size();
    if (NP == 0) {
        fprintf(stderr, "empty projectile geometry: increase proj_l/proj_d or decrease dx\n");
        return 2;
    }
    for (size_t k = 0; k < NP; ++k) {
        X.insert(X.begin() + k, px[k] + xmin_arm - gap);
        Y.insert(Y.begin() + k, py[k]);
        Z.insert(Z.begin() + k, pz[k]);
        matid.insert(matid.begin() + k, 0);
        fixv.insert(fixv.begin() + k, 0);
    }
    const size_t N = X.size();
    printf("particles: proj=%zu armor=%zu total=%zu dim=%d (TL=%s)\n",
           NP, NA, N, DIM, par.use_tl ? "hybrid" : "off");

    // ---------------- state arrays ----------------
    const std::vector<double> X0 = X, Y0 = Y, Z0 = Z;   // reference configuration
    std::vector<double> vx(N,0), vy(N,0), vz(N,0), rho(N), mass(N);
    std::vector<double> Sxx(N,0), Syy(N,0), Szz(N,0);   // deviatoric stress (6 components)
    std::vector<double> Sxy(N,0), Sxz(N,0), Syz(N,0);
    std::vector<double> epsp(N,0), dmg(N,0), Tk(N,T0K);
    std::vector<double> Pfld(N,0), cson(N);
    std::vector<double> ax(N,0), ay(N,0), az(N,0);      // accelerations (leapfrog)
    std::vector<double> xsx(N,0), xsy(N,0), xsz(N,0);   // XSPH correction
    std::vector<double> F11(N,1), F12(N,0), F13(N,0),
                        F21(N,0), F22(N,1), F23(N,0),
                        F31(N,0), F32(N,0), F33(N,1);
    std::vector<double> B11(N,1), B12(N,0), B13(N,0),
                        B21(N,0), B22(N,1), B23(N,0),
                        B31(N,0), B33(N,1), B32(N,0);
    std::vector<char> isTL(N, par.use_tl ? 1 : 0), isFix(N);
    std::vector<int> pid(N), id2idx(N);
    for (size_t k = 0; k < N; ++k) { pid[k] = (int)k; id2idx[k] = (int)k; }
    // workspace: velocity gradient (L), drho, total stresses, inverse density
    std::vector<double> L11(N),L12(N),L13(N),L21(N),L22(N),L23(N),
                        L31(N),L32(N),L33(N), drho(N);
    std::vector<double> sxx(N), syy(N), szzf(N), sxy(N), sxz(N), syz(N);
    std::vector<double> invRho(N), invRho2(N);

    for (size_t k = 0; k < N; ++k) {
        const Material& M = matid[k] ? MA : MP;
        rho[k] = M.rho0; mass[k] = M.rho0 * V0; cson[k] = M.c0;
        isFix[k] = fixv[k];
        if (matid[k] == 0) vx[k] = par.v_impact;
    }
    double c_long0 = 0;
    for (auto& M : {MP, MA})
        c_long0 = std::max(c_long0, sqrt((M.rho0*M.c0*M.c0 + 4*M.G/3)/M.rho0));

    // ---------------- neighbor grid ----------------
    std::vector<int> cellOf(N), order(N);
    int ncx = 0, ncy = 0, ncz = 1; double gx0 = 0, gy0 = 0, gz0 = 0;
    std::vector<int> cellStart, cellCount;
    auto buildGrid = [&]() {
        double xmin=1e9, xmax=-1e9, ymin=1e9, ymax=-1e9, zmin=1e9, zmax=-1e9;
        for (size_t k=0;k<N;++k){
            xmin=std::min(xmin,X[k]); xmax=std::max(xmax,X[k]);
            ymin=std::min(ymin,Y[k]); ymax=std::max(ymax,Y[k]);
            if (DIM==3){ zmin=std::min(zmin,Z[k]); zmax=std::max(zmax,Z[k]); }
        }
        gx0 = xmin - H2; gy0 = ymin - H2; gz0 = zmin - H2;
        ncx = (int)((xmax - gx0)/H2) + 2;
        ncy = (int)((ymax - gy0)/H2) + 2;
        ncz = (DIM==3) ? (int)((zmax - gz0)/H2) + 2 : 1;
        size_t ncells = (size_t)ncx*ncy*ncz;
        cellStart.assign(ncells + 1, 0);
        cellCount.assign(ncells, 0);
        for (size_t k=0;k<N;++k){
            int cx = (int)((X[k]-gx0)/H2), cy = (int)((Y[k]-gy0)/H2);
            int cz = (DIM==3) ? (int)((Z[k]-gz0)/H2) : 0;
            cellOf[k] = (cz*ncy + cy)*ncx + cx;
            cellCount[cellOf[k]]++;
        }
        for (size_t c=0;c<ncells;++c) cellStart[c+1]=cellStart[c]+cellCount[c];
        std::vector<int> fill(cellStart.begin(), cellStart.end()-1);
        for (size_t k=0;k<N;++k) order[fill[cellOf[k]]++] = (int)k;
    };

    // ---------------- kernel (cubic spline) ----------------
    auto kernelW = [&](double r) -> double {
        double q = r / H;
        if (q <= 1.0)      return SIGK * (1.0 - 1.5*q*q + 0.75*q*q*q);
        else if (q < 2.0) { double u = 2.0-q; return SIGK * 0.25*u*u*u; }
        return 0.0;
    };
    auto kernelDW = [&](double r) -> double {
        double q = r / H;
        if (q <= 1.0)      return  SIGK/H * (-3.0*q + 2.25*q*q);
        else if (q < 2.0)  return -SIGK/H * 0.75*(2.0-q)*(2.0-q);
        return 0.0;
    };

    // ---------------- CSR neighbor lists (reused capacity allocation) ----------------
    std::vector<long>  nbStart(N+1, 0);
    std::vector<long>  nbCnt(N, 0);
    std::vector<int>   nbJ;                        // neighbor particle ID
    std::vector<float> nbW, nbGx, nbGy, nbGz;
    auto scanFill = [&]() -> bool {
        char overflow = 0;
        #pragma omp parallel for schedule(dynamic, 512)
        for (long i = 0; i < (long)N; ++i) {
            long cur = nbStart[i];
            const long cap = nbStart[i+1] - nbStart[i];
            long c = 0;
            int cx=(int)((X[i]-gx0)/H2), cy=(int)((Y[i]-gy0)/H2);
            int cz=(DIM==3)?(int)((Z[i]-gz0)/H2):0;
            for (int oz=-1;oz<=1 && c<=cap;++oz)
            for (int oy=-1;oy<=1;++oy) for (int ox=-1;ox<=1;++ox) {
                if (DIM==2 && oz!=0) continue;
                int cc=((cz+oz)*ncy+(cy+oy))*ncx+(cx+ox);
                if (cc<0||cc>=(long)ncx*ncy*ncz) continue;
                for (int q=cellStart[cc]; q<cellStart[cc+1]; ++q) {
                    int j = order[q]; if ((long)j == i) continue;
                    double dxp=X[i]-X[j], dyp=Y[i]-Y[j];
                    double dzp=(DIM==3)?(Z[i]-Z[j]):0.0;
                    double r2=dxp*dxp+dyp*dyp+dzp*dzp;
                    if (r2 >= H2*H2) continue;
                    if (c >= cap) { overflow = 1; c = cap + 1; break; }
                    double r=sqrt(r2);
                    double dwdr=kernelDW(r), inv=(r>1e-12)?1.0/r:0.0;
                    nbJ[cur]  = j;                 // current list uses array indices
                    nbW[cur]  = (float)kernelW(r);
                    nbGx[cur] = (float)(dwdr*dxp*inv);
                    nbGy[cur] = (float)(dwdr*dyp*inv);
                    if (DIM==3) nbGz[cur] = (float)(dwdr*dzp*inv);
                    ++cur; ++c;
                }
            }
            nbCnt[i] = std::min(c, cap);
            if (c > cap) overflow = 1;
        }
        return overflow != 0;
    };
    auto buildCSR = [&]() {
        buildGrid();
        if (nbStart[N] > 0 && !scanFill()) return;
        N_COLD++;
        std::vector<long> cnt(N, 0);
        #pragma omp parallel for schedule(dynamic, 512)
        for (long i = 0; i < (long)N; ++i) {
            long c = 0;
            int cx=(int)((X[i]-gx0)/H2), cy=(int)((Y[i]-gy0)/H2);
            int cz=(DIM==3)?(int)((Z[i]-gz0)/H2):0;
            for (int oz=-1;oz<=1;oz++)
            for (int oy=-1;oy<=1;++oy) for (int ox=-1;ox<=1;++ox) {
                if (DIM==2 && oz!=0) continue;
                int cc=((cz+oz)*ncy+(cy+oy))*ncx+(cx+ox);
                if (cc<0||cc>=(long)ncx*ncy*ncz) continue;
                for (int q=cellStart[cc]; q<cellStart[cc+1]; ++q) {
                    int j = order[q]; if ((long)j == i) continue;
                    double dxp=X[i]-X[j], dyp=Y[i]-Y[j];
                    double dzp=(DIM==3)?(Z[i]-Z[j]):0.0;
                    if (dxp*dxp+dyp*dyp+dzp*dzp < H2*H2) ++c;
                }
            }
            if (c + NB_PAD > MAX_NB) { fprintf(stderr,"too many neighbors\n"); exit(1); }
            cnt[i] = c;
        }
        nbStart[0] = 0;
        for (size_t i = 0; i < N; ++i) nbStart[i+1] = nbStart[i] + cnt[i] + NB_PAD;
        long total = nbStart[N];
        nbJ.assign(total, 0); nbW.assign(total, 0);
        nbGx.assign(total, 0); nbGy.assign(total, 0);
        if (DIM==3) nbGz.assign(total, 0);
        scanFill();
    };

    // -------- TL reference list (by particle ID, compact and sorted) --------
    std::vector<long>  nb0Start;
    std::vector<int>   nb0J;
    std::vector<float> nb0Gx, nb0Gy, nb0Gz;
    bool refCaptured = false;
    auto captureReference = [&]() {
        nb0Start.assign(N+1, 0);
        for (size_t i = 0; i < N; ++i) nb0Start[i+1] = nb0Start[i] + nbCnt[i];
        long total = nb0Start[N];
        nb0J.resize(total); nb0Gx.resize(total); nb0Gy.resize(total);
        if (DIM==3) nb0Gz.resize(total);
        for (size_t i = 0; i < N; ++i) {
            long b0 = nb0Start[i];
            for (long q = nbStart[i]; q < nbStart[i]+nbCnt[i]; ++q, ++b0) {
                nb0J[b0]=nbJ[q]; nb0Gx[b0]=nbGx[q]; nb0Gy[b0]=nbGy[q];
                if (DIM==3) nb0Gz[b0]=nbGz[q];
            }
        }
        for (size_t i = 0; i < N; ++i) {
            long b = nb0Start[i], e = nb0Start[i+1];
            std::vector<long> idx(e-b);
            for (long k = 0; k < e-b; ++k) idx[k] = b + k;
            std::sort(idx.begin(), idx.end(),
                      [&](long a, long c){ return nb0J[a] < nb0J[c]; });
            std::vector<int> Jc(e-b);
            std::vector<float> Xc(e-b), Yc(e-b), Zc(DIM==3?e-b:0);
            for (long k = 0; k < e-b; ++k) {
                Jc[k]=nb0J[idx[k]]; Xc[k]=nb0Gx[idx[k]]; Yc[k]=nb0Gy[idx[k]];
                if (DIM==3) Zc[k]=nb0Gz[idx[k]];
            }
            for (long k = 0; k < e-b; ++k) {
                nb0J[b+k]=Jc[k]; nb0Gx[b+k]=Xc[k]; nb0Gy[b+k]=Yc[k];
                if (DIM==3) nb0Gz[b+k]=Zc[k];
            }
        }
        // B0 = -M^{-1}; M = sum_j V0 (X0j-X0i) (x) grad0W_ij
        for (size_t i = 0; i < N; ++i) {
            if (DIM == 2) {
                double m11=0,m12=0,m21=0,m22=0;
                for (long q = nb0Start[i]; q < nb0Start[i+1]; ++q) {
                    int j = id2idx[nb0J[q]];
                    double dx0 = X0[j]-X0[i], dy0 = Y0[j]-Y0[i];
                    m11 += V0*dx0*nb0Gx[q]; m12 += V0*dx0*nb0Gy[q];
                    m21 += V0*dy0*nb0Gx[q]; m22 += V0*dy0*nb0Gy[q];
                }
                double detm = m11*m22 - m12*m21;
                if (fabs(detm) > 0.05) {
                    double iv = -1.0/detm;
                    double b11= iv*m22,  b12=-iv*m12;
                    double b21=-iv*m21,  b22= iv*m11;
                    auto cl=[](double v){return std::min(std::max(v,-10.0),10.0);};
                    B11[i]=cl(b11); B12[i]=cl(b12); B21[i]=cl(b21); B22[i]=cl(b22);
                }
            } else {
                double m11=0,m12=0,m13=0,m21=0,m22=0,m23=0,m31=0,m32=0,m33=0;
                for (long q = nb0Start[i]; q < nb0Start[i+1]; ++q) {
                    int j = id2idx[nb0J[q]];
                    double dx0 = X0[j]-X0[i], dy0 = Y0[j]-Y0[i], dz0 = Z0[j]-Z0[i];
                    double g1=nb0Gx[q], g2=nb0Gy[q], g3=nb0Gz[q];
                    m11 += V0*dx0*g1; m12 += V0*dx0*g2; m13 += V0*dx0*g3;
                    m21 += V0*dy0*g1; m22 += V0*dy0*g2; m23 += V0*dy0*g3;
                    m31 += V0*dz0*g1; m32 += V0*dz0*g2; m33 += V0*dz0*g3;
                }
                double detm = m11*(m22*m33-m23*m32) - m12*(m21*m33-m23*m31)
                            + m13*(m21*m32-m22*m31);
                if (fabs(detm) > 0.02) {
                    double iv = -1.0/detm;
                    // B = -M^{-1} = iv * adj(M)
                    auto cl=[](double v){return std::min(std::max(v,-10.0),10.0);};
                    B11[i]=cl(iv*(m22*m33-m23*m32)); B12[i]=cl(iv*(m13*m32-m12*m33));
                    B13[i]=cl(iv*(m12*m23-m13*m22));
                    B21[i]=cl(iv*(m23*m31-m21*m33)); B22[i]=cl(iv*(m11*m33-m13*m31));
                    B23[i]=cl(iv*(m13*m21-m11*m23));
                    B31[i]=cl(iv*(m21*m32-m22*m31)); B32[i]=cl(iv*(m12*m31-m11*m32));
                    B33[i]=cl(iv*(m11*m22-m12*m21));
                }
            }
        }
        refCaptured = true;
    };
    // whether particle ID j is a reference neighbor of particle i (binary search)
    auto isRefNbr = [&](long i, int jpid) -> bool {
        long b = nb0Start[i], e = nb0Start[i+1];
        while (b < e) { long m=(b+e)/2; int v=nb0J[m];
                        if (v<jpid) b=m+1; else if (v>jpid) e=m; else return true; }
        return false;
    };
    // corrected reference gradient g' = B0 * g0
    auto corrGrad = [&](long i, float gx, float gy, float gz,
                        double& ox, double& oy, double& oz) {
        ox = B11[i]*gx + B12[i]*gy + B13[i]*gz;
        oy = B21[i]*gx + B22[i]*gy + B23[i]*gz;
        oz = B31[i]*gx + B32[i]*gy + B33[i]*gz;
    };

    // ---------------- output ----------------
    const int NF_OUT = (DIM == 3) ? 9 : 8;
    std::error_code ec;
    std::filesystem::create_directories(par.out, ec);
    if (par.vtk) std::filesystem::create_directories(par.out + "/vtk", ec);
    FILE* ff = fopen((par.out + "/frames.bin").c_str(), "wb");
    if (!ff) { fprintf(stderr, "cannot open output in %s\n", par.out.c_str()); return 1; }

    int frameNo = 0;
    auto writeVtk = [&](double t) {
        char fn[512]; snprintf(fn, sizeof fn, "%s/vtk/frame_%04d.vtk",
                               par.out.c_str(), frameNo);
        FILE* fv = fopen(fn, "w");
        if (!fv) return;
        fprintf(fv, "# vtk DataFile Version 3.0\nsph_core%d t=%g\nASCII\n", DIM, t);
        fprintf(fv, "DATASET POLYDATA\nPOINTS %zu float\n", N);
        for (size_t k=0;k<N;++k) fprintf(fv, "%g %g %g\n", X[k], Y[k], Z[k]);
        fprintf(fv, "POINT_DATA %zu\n", N);
        fprintf(fv, "VECTORS velocity float\n");
        for (size_t k=0;k<N;++k) fprintf(fv, "%g %g %g\n", vx[k], vy[k], vz[k]);
        fprintf(fv, "SCALARS damage float 1\nLOOKUP_TABLE default\n");
        for (size_t k=0;k<N;++k) fprintf(fv, "%g\n", dmg[k]);
        fprintf(fv, "SCALARS pressure float 1\nLOOKUP_TABLE default\n");
        for (size_t k=0;k<N;++k) fprintf(fv, "%g\n", Pfld[k]);
        fprintf(fv, "SCALARS temperature float 1\nLOOKUP_TABLE default\n");
        for (size_t k=0;k<N;++k) fprintf(fv, "%g\n", Tk[k]);
        fprintf(fv, "SCALARS epsp float 1\nLOOKUP_TABLE default\n");
        for (size_t k=0;k<N;++k) fprintf(fv, "%g\n", epsp[k]);
        fprintf(fv, "SCALARS matid float 1\nLOOKUP_TABLE default\n");
        for (size_t k=0;k<N;++k) fprintf(fv, "%d\n", matid[k]);
        fclose(fv);
    };
    auto writeFrame = [&](double t) {
        float tf = (float)t; fwrite(&tf, 4, 1, ff);
        std::vector<float> buf(N);
        // frame layout: [X][Y]([Z])[spd][dmg][P][T][epsp][svm]
        for (size_t k=0;k<N;++k) buf[k]=(float)X[k];
        fwrite(buf.data(),4,N,ff);
        for (size_t k=0;k<N;++k) buf[k]=(float)Y[k];
        fwrite(buf.data(),4,N,ff);
        if (DIM==3) {
            for (size_t k=0;k<N;++k) buf[k]=(float)Z[k];
            fwrite(buf.data(),4,N,ff);
        }
        for (size_t k=0;k<N;++k)
            buf[k]=(float)sqrt(vx[k]*vx[k]+vy[k]*vy[k]+vz[k]*vz[k]);
        fwrite(buf.data(),4,N,ff);
        for (size_t k=0;k<N;++k) buf[k]=(float)dmg[k];
        fwrite(buf.data(),4,N,ff);
        for (size_t k=0;k<N;++k) buf[k]=(float)Pfld[k];
        fwrite(buf.data(),4,N,ff);
        for (size_t k=0;k<N;++k) buf[k]=(float)Tk[k];
        fwrite(buf.data(),4,N,ff);
        for (size_t k=0;k<N;++k) buf[k]=(float)epsp[k];
        fwrite(buf.data(),4,N,ff);
        for (size_t k=0;k<N;++k)
            buf[k]=(float)sqrt(1.5*(Sxx[k]*Sxx[k]+Syy[k]*Syy[k]+Szz[k]*Szz[k]
                    +2*(Sxy[k]*Sxy[k]+Sxz[k]*Sxz[k]+Syz[k]*Syz[k])));
        fwrite(buf.data(),4,N,ff);
        if (par.vtk) writeVtk(t);
        ++frameNo;
    };

    const double ALPHA=1.0, BETA=1.5;

    // ========================== solver passes ==========================

    // ---- pass 1: velocity gradients (+ XSPH is integrated here) ----
    auto ratesPass = [&](const std::vector<double>& vhx,
                         const std::vector<double>& vhy,
                         const std::vector<double>& vhz,
                         bool doXSPH) {
        #pragma omp parallel for schedule(dynamic, 256)
        for (long i = 0; i < (long)N; ++i) {
            double dr=0;
            double l11=0,l12=0,l13=0,l21=0,l22=0,l23=0,l31=0,l32=0,l33=0;
            double xsv=0, ysv=0, zsv=0;
            if (isTL[i]) {
                // L0 = sum_j V0 (v_j - v_i) (x) B0*grad0W_ij
                for (long q = nb0Start[i]; q < nb0Start[i+1]; ++q) {
                    int j = id2idx[nb0J[q]];
                    double g0x, g0y, g0z;
                    float gz = (DIM==3)?nb0Gz[q]:0.0f;
                    corrGrad(i, nb0Gx[q], nb0Gy[q], gz, g0x, g0y, g0z);
                    double dux = vhx[j]-vhx[i], duy = vhy[j]-vhy[i];
                    double duz = vhz[j]-vhz[i];
                    l11 += V0*dux*g0x; l12 += V0*dux*g0y; l13 += V0*dux*g0z;
                    l21 += V0*duy*g0x; l22 += V0*duy*g0y; l23 += V0*duy*g0z;
                    l31 += V0*duz*g0x; l32 += V0*duz*g0y; l33 += V0*duz*g0z;
                }
            } else {
                for (long q = nbStart[i]; q < nbStart[i]+nbCnt[i]; ++q) {
                    int j = nbJ[q];
                    double gxv=nbGx[q], gyv=nbGy[q];
                    double gzv=(DIM==3)?nbGz[q]:0.0;
                    double dux=vhx[i]-vhx[j], duy=vhy[i]-vhy[j], duz=vhz[i]-vhz[j];
                    dr += mass[j]*(dux*gxv+duy*gyv+duz*gzv);
                    double Vj = mass[j]*invRho[j];
                    l11 += Vj*(-dux)*gxv; l12 += Vj*(-dux)*gyv; l13 += Vj*(-dux)*gzv;
                    l21 += Vj*(-duy)*gxv; l22 += Vj*(-duy)*gyv; l23 += Vj*(-duy)*gzv;
                    l31 += Vj*(-duz)*gxv; l32 += Vj*(-duz)*gyv; l33 += Vj*(-duz)*gzv;
                    if (doXSPH) {
                        double irh=0.5*(invRho[i]+invRho[j]);
                        xsv -= par.xs_eps*mass[j]*irh*dux*nbW[q];
                        ysv -= par.xs_eps*mass[j]*irh*duy*nbW[q];
                        zsv -= par.xs_eps*mass[j]*irh*duz*nbW[q];
                    }
                }
            }
            // XSPH correction: UL uses the current neighbor list; TL uses an elastic
            // drift without it (contact regions quickly transition to UL);
            // when xs=0, xsv=0 -> unmodified drift
            if (!isFix[i]) { xsx[i]=xsv; xsy[i]=ysv; xsz[i]=zsv; }
            else { xsx[i]=xsy[i]=xsz[i]=0; }
            drho[i]=dr;
            L11[i]=l11;L12[i]=l12;L13[i]=l13;L21[i]=l21;L22[i]=l22;L23[i]=l23;
            L31[i]=l31;L32[i]=l32;L33[i]=l33;
        }
    };

    // ---- pass 2: F, constitutive model, EOS, and damage ----
    auto updatePass = [&](double dtl, double& cmaxOut, long& tlCntOut) {
        double cmax = 0; long tlCnt = 0;
        #pragma omp parallel
        {
            double cmaxL = 0; long tlL = 0;
            #pragma omp for schedule(static) nowait
            for (long i = 0; i < (long)N; ++i) {
                const Material& M = matid[i]?MA:MP;
                double e11,e22,e33,e12,e13,e23, wxy,wxz,wyz;
                bool tlActive = isTL[i] != 0;
                if (tlActive) {
                    double a=L11[i],b=L12[i],c=L13[i],
                           d=L21[i],e=L22[i],f=L23[i],
                           g=L31[i],h=L32[i],k=L33[i];
                    if (dtl > 0) {
                        // F' = L0 * F
                        double n11=a*F11[i]+b*F21[i]+c*F31[i];
                        double n12=a*F12[i]+b*F22[i]+c*F32[i];
                        double n13=a*F13[i]+b*F23[i]+c*F33[i];
                        double n21=d*F11[i]+e*F21[i]+f*F31[i];
                        double n22=d*F12[i]+e*F22[i]+f*F32[i];
                        double n23=d*F13[i]+e*F23[i]+f*F33[i];
                        double n31=g*F11[i]+h*F21[i]+k*F31[i];
                        double n32=g*F12[i]+h*F22[i]+k*F32[i];
                        double n33=g*F13[i]+h*F23[i]+k*F33[i];
                        F11[i]+=dtl*n11; F12[i]+=dtl*n12; F13[i]+=dtl*n13;
                        F21[i]+=dtl*n21; F22[i]+=dtl*n22; F23[i]+=dtl*n23;
                        F31[i]+=dtl*n31; F32[i]+=dtl*n32; F33[i]+=dtl*n33;
                    }
                    double f11=F11[i],f12=F12[i],f13=F13[i],
                           f21=F21[i],f22=F22[i],f23=F23[i],
                           f31=F31[i],f32=F32[i],f33=F33[i];
                    double J;
                    if (DIM==2) J = f11*f22 - f12*f21;
                    else J = f11*(f22*f33-f23*f32) - f12*(f21*f33-f23*f31)
                           + f13*(f21*f32-f22*f31);
                    double edot_chk;
                    if (DIM==2)
                        edot_chk = sqrt(2.0*(a*a+e*e+0.5*(b+d)*(b+d))/3.0);
                    else
                        edot_chk = sqrt(2.0*(a*a+e*e+k*k
                            +0.5*((b+d)*(b+d)+(c+g)*(c+g)+(f+h)*(f+h)))/3.0);
                    if (J < 0.55 || J > 1.8 || edot_chk > par.sw_edot
                        || dmg[i] > 0.75) {
                        isTL[i] = 0; tlActive = false;
                        rho[i] = std::min(std::max(M.rho0/J, 0.3*M.rho0), 3.0*M.rho0);
                        F11[i]=1; F12[i]=0; F13[i]=0;
                        F21[i]=0; F22[i]=1; F23[i]=0;
                        F31[i]=0; F32[i]=0; F33[i]=1;
                    } else {
                        rho[i] = M.rho0/J;
                        // L = L0 * F^{-1}; F^{-1} = Cof^T / J
                        double c11= (f22*f33-f23*f32), c12=-(f21*f33-f23*f31),
                               c13= (f21*f32-f22*f31);
                        double c21=-(f12*f33-f13*f32), c22= (f11*f33-f13*f31),
                               c23=-(f11*f32-f12*f31);
                        double c31= (f12*f23-f13*f22), c32=-(f11*f23-f13*f21),
                               c33= (f11*f22-f12*f21);
                        double iv=1.0/J;
                        double lx=(a*c11+b*c12+c*c13)*iv;
                        double ly=(a*c21+b*c22+c*c23)*iv;
                        double lz=(a*c31+b*c32+c*c33)*iv;
                        double mx=(d*c11+e*c12+f*c13)*iv;
                        double my=(d*c21+e*c22+f*c23)*iv;
                        double mz=(d*c31+e*c32+f*c33)*iv;
                        double nx=(g*c11+h*c12+k*c13)*iv;
                        double ny=(g*c21+h*c22+k*c23)*iv;
                        double nz=(g*c31+h*c32+k*c33)*iv;
                        e11=lx; e22=my; e33=nz;
                        e12=0.5*(ly+mx); e13=0.5*(lz+nx); e23=0.5*(mz+ny);
                        wxy=0.5*(ly-mx); wxz=0.5*(lz-nx); wyz=0.5*(mz-ny);
                    }
                }
                if (!tlActive) {
                    if (dtl > 0) {
                        rho[i] += drho[i]*dtl;
                        rho[i] = std::min(std::max(rho[i], 0.3*M.rho0), 3.0*M.rho0);
                    }
                    double a=L11[i],b=L12[i],c=L13[i],
                           d=L21[i],e=L22[i],f=L23[i],
                           g=L31[i],h=L32[i],k=L33[i];
                    e11=a; e22=e; e33=k;
                    e12=0.5*(b+d); e13=0.5*(c+g); e23=0.5*(f+h);
                    wxy=0.5*(b-d); wxz=0.5*(c-g); wyz=0.5*(f-h);
                }
                if (tlActive) ++tlL;
                // elastic trial stress + Jaumann rate (S' = 2G dev e + [W,S])
                double trE = (e11+e22+e33)/3.0;
                if (dtl > 0) {
                    double r_xx=2*(wxy*Sxy[i]+wxz*Sxz[i]);
                    double r_yy=2*(-wxy*Sxy[i]+wyz*Syz[i]);
                    double r_zz=2*(-wxz*Sxz[i]-wyz*Syz[i]);
                    double r_xy=wxy*(Syy[i]-Sxx[i])+wxz*Syz[i]+wyz*Sxz[i];
                    double r_xz=wxz*(Szz[i]-Sxx[i])+wxy*Syz[i]-wyz*Sxy[i];
                    double r_yz=wyz*(Szz[i]-Syy[i])-wxy*Sxz[i]-wxz*Sxy[i];
                    Sxx[i]+=dtl*(2*M.G*(e11-trE)+r_xx);
                    Syy[i]+=dtl*(2*M.G*(e22-trE)+r_yy);
                    Szz[i]+=dtl*(2*M.G*(e33-trE)+r_zz);
                    Sxy[i]+=dtl*(2*M.G*e12+r_xy);
                    Sxz[i]+=dtl*(2*M.G*e13+r_xz);
                    Syz[i]+=dtl*(2*M.G*e23+r_yz);
                }
                // Mie–Grüneisen EOS + thermal pressure
                double mu = rho[i]/M.rho0 - 1.0;
                double P;
                if (mu > 0) {
                    double den = 1.0 - (M.s-1.0)*mu;
                    den = std::max(den, 0.15);
                    P = M.rho0*M.c0*M.c0*mu/(den*den);
                } else {
                    P = M.rho0*M.c0*M.c0*mu;
                }
                double Pth = M.Gamma*M.rho0*M.cp*(Tk[i]-T0K);
                Pth = std::min(Pth, 0.5*M.rho0*M.c0*M.c0);
                P += Pth;
                // Johnson–Cook + damage D (triaxiality, strain rate, temperature)
                double edot = sqrt(2.0*(e11*e11+e22*e22+e33*e33
                                   +2*(e12*e12+e13*e13+e23*e23))/3.0);
                double rateF = 1.0 + M.C*log(std::max(edot,1.0));
                double Tstar = std::min(std::max((Tk[i]-T0K)/(M.Tm-T0K),0.0),0.999);
                double thermF = 1.0 - pow(Tstar, M.m);
                double soft = std::max(1.0 - dmg[i], 0.0);
                double Yld = (M.A + M.B*pow(std::max(epsp[i],1e-8),M.n))
                             * rateF * thermF * soft;
                double seq = sqrt(1.5*(Sxx[i]*Sxx[i]+Syy[i]*Syy[i]+Szz[i]*Szz[i]
                             +2*(Sxy[i]*Sxy[i]+Sxz[i]*Sxz[i]+Syz[i]*Syz[i])))+1e-30;
                if (dtl > 0 && seq > Yld && dmg[i] < 1.0) {
                    double sc = Yld/seq;
                    double dep = (seq-Yld)/(3*M.G);
                    epsp[i]+=dep;
                    Tk[i]+=0.9*seq*dep/(rho[i]*M.cp);
                    double triax = std::min(std::max(-P/seq, -1.5), 1.5);
                    double epf = (M.D1 + M.D2*exp(M.D3*triax))
                               * (1.0 + M.D4*log(std::max(edot,1.0)))
                               * (1.0 + M.D5*Tstar);
                    epf = std::min(std::max(epf, 0.03), 3.0);
                    dmg[i] = std::min(dmg[i] + dep/epf, 1.2);
                    Sxx[i]*=sc; Syy[i]*=sc; Szz[i]*=sc;
                    Sxy[i]*=sc; Sxz[i]*=sc; Syz[i]*=sc;
                } else if (dmg[i] >= 1.0) {
                    Sxx[i]=Syy[i]=Szz[i]=Sxy[i]=Sxz[i]=Syz[i]=0.0;
                }
                double pmin = (dmg[i] >= 1.0) ? 0.0 : -3.0e9 - Pth;
                Pfld[i] = std::max(P, pmin);
                Tk[i] = std::min(Tk[i], 1.3*M.Tm);
                sxx[i]=-Pfld[i]+Sxx[i]; syy[i]=-Pfld[i]+Syy[i];
                szzf[i]=-Pfld[i]+Szz[i];
                sxy[i]=Sxy[i]; sxz[i]=Sxz[i]; syz[i]=Syz[i];
                double ir = 1.0/rho[i];
                invRho[i] = ir; invRho2[i] = ir*ir;
                double cs2 = M.c0*M.c0*(1.0 + 2.0*M.s*std::max(mu,0.0))
                           + 4.0*M.G/(3.0*rho[i]);
                cson[i] = sqrt(std::max(cs2, 1.0));
                cmaxL = std::max(cmaxL, cson[i]+sqrt(vx[i]*vx[i]+vy[i]*vy[i]
                                                     +vz[i]*vz[i]));
            }
            #pragma omp critical
            { cmax = std::max(cmax, cmaxL); tlCnt += tlL; }
        }
        cmaxOut = cmax; tlCntOut = tlCnt;
    };

    // ---- pass 3: forces — symmetric UL pairs + TL-TL pairs from the reference list ----
    std::vector<double> TAx, TAy, TAz;   // thread-private accumulators
    int NT = 1;
    #ifdef _OPENMP
    NT = omp_get_max_threads();
    #endif
    TAx.assign((size_t)NT*N, 0.0);
    TAy.assign((size_t)NT*N, 0.0);
    TAz.assign((size_t)NT*N, 0.0);

    auto forcesPass = [&](const std::vector<double>& vhx,
                          const std::vector<double>& vhy,
                          const std::vector<double>& vhz) -> double {
        double amax = 0;
        #pragma omp parallel
        {
            int tid = 0;
            #ifdef _OPENMP
            tid = omp_get_thread_num();
            #endif
            double* __restrict__ A_x = TAx.data() + (size_t)tid*N;
            double* __restrict__ A_y = TAy.data() + (size_t)tid*N;
            double* __restrict__ A_z = TAz.data() + (size_t)tid*N;
            const double wDx = kernelW(DX);
            // zero the entire thread-owned stripe (each thread clears its own)
            for (long i = 0; i < (long)N; ++i) {
                A_x[i]=0; A_y[i]=0; A_z[i]=0;
            }
            // zero direct accumulators (for the TL part)
            #pragma omp for schedule(static)
            for (long i = 0; i < (long)N; ++i) { ax[i]=0; ay[i]=0; az[i]=0; }
            // ------- symmetric UL pairs (each pair once, j>i) -------
            #pragma omp for schedule(dynamic, 256) nowait
            for (long i = 0; i < (long)N; ++i) {
                bool fi = isFix[i];
                double mi = mass[i], ri2 = invRho2[i];
                for (long q = nbStart[i]; q < nbStart[i]+nbCnt[i]; ++q) {
                    int j = nbJ[q];
                    if (j <= i || (fi && isFix[j])) continue;
                    bool tlPair = par.use_tl && isTL[i] && isTL[j]
                                  && refCaptured && isRefNbr(i, pid[j]);
                    if (tlPair) continue;
                    double dxp=X[i]-X[j], dyp=Y[i]-Y[j];
                    double dzp=(DIM==3)?(Z[i]-Z[j]):0.0;
                    double r2=dxp*dxp+dyp*dyp+dzp*dzp;
                    double gxv=nbGx[q], gyv=nbGy[q];
                    double gzv=(DIM==3)?nbGz[q]:0.0;
                    // artificial viscosity
                    double dux=vhx[i]-vhx[j], duy=vhy[i]-vhy[j];
                    double duz=vhz[i]-vhz[j];
                    double vr=dux*dxp+duy*dyp+duz*dzp, Pi=0;
                    if (vr<0) {
                        double muv=H*vr/(r2+0.01*H*H);
                        double cij=0.5*(cson[i]+cson[j]);
                        Pi=(-ALPHA*cij*muv+BETA*muv*muv)/(0.5*(rho[i]+rho[j]));
                    }
                    // artificial stress added to the isotropic component
                    double tas = 0;
                    if (par.as_chi > 0 && r2 > 1e-16) {
                        double sni=(sxx[i]*dxp*dxp+syy[i]*dyp*dyp+szzf[i]*dzp*dzp
                              +2*(sxy[i]*dxp*dyp+sxz[i]*dxp*dzp+syz[i]*dyp*dzp))/r2;
                        double snj=(sxx[j]*dxp*dxp+syy[j]*dyp*dyp+szzf[j]*dzp*dzp
                              +2*(sxy[j]*dxp*dyp+sxz[j]*dxp*dzp+syz[j]*dyp*dzp))/r2;
                        double sn=0.5*(sni+snj);
                        if (sn>0) {
                            double phi=nbW[q]/wDx;
                            tas=par.as_chi*sn*phi*phi*phi*invRho2[i];
                        }
                    }
                    double rj2=invRho2[j];
                    double txx=sxx[i]*ri2+sxx[j]*rj2-Pi-tas;
                    double tyy=syy[i]*ri2+syy[j]*rj2-Pi-tas;
                    double tzz=(DIM==3)?(szzf[i]*ri2+szzf[j]*rj2-Pi-tas):0.0;
                    double txy=sxy[i]*ri2+sxy[j]*rj2;
                    double txz=(DIM==3)?(sxz[i]*ri2+sxz[j]*rj2):0.0;
                    double tyz=(DIM==3)?(syz[i]*ri2+syz[j]*rj2):0.0;
                    double ux=txx*gxv+txy*gyv+txz*gzv;
                    double uy=txy*gxv+tyy*gyv+tyz*gzv;
                    double uz=txz*gxv+tyz*gyv+tzz*gzv;
                    double mj=mass[j];
                    if (!fi) { A_x[i]+=mj*ux; A_y[i]+=mj*uy; A_z[i]+=mj*uz; }
                    if (!isFix[j]) { A_x[j]-=mi*ux; A_y[j]-=mi*uy; A_z[j]-=mi*uz; }
                }
            }
            // ------- TL-TL pairs (per particle, using the reference list) -------
            if (par.use_tl && refCaptured) {
                #pragma omp for schedule(dynamic, 256) nowait
                for (long i = 0; i < (long)N; ++i) {
                    if (isFix[i] || !isTL[i]) continue;
                    const Material& Mi = matid[i]?MA:MP;
                    // PK1 = sigma * Cof(F) = J sigma F^{-T}
                    double c11= F22[i]*F33[i]-F23[i]*F32[i];
                    double c12=-(F21[i]*F33[i]-F23[i]*F31[i]);
                    double c13= F21[i]*F32[i]-F22[i]*F31[i];
                    double c21=-(F12[i]*F33[i]-F13[i]*F32[i]);
                    double c22= F11[i]*F33[i]-F13[i]*F31[i];
                    double c23=-(F11[i]*F32[i]-F12[i]*F31[i]);
                    double c31= F12[i]*F23[i]-F13[i]*F22[i];
                    double c32=-(F11[i]*F23[i]-F13[i]*F21[i]);
                    double c33= F11[i]*F22[i]-F12[i]*F21[i];
                    double p11=sxx[i]*c11+sxy[i]*c21+sxz[i]*c31;
                    double p12=sxx[i]*c12+sxy[i]*c22+sxz[i]*c32;
                    double p13=sxx[i]*c13+sxy[i]*c23+sxz[i]*c33;
                    double p21=sxy[i]*c11+syy[i]*c21+syz[i]*c31;
                    double p22=sxy[i]*c12+syy[i]*c22+syz[i]*c32;
                    double p23=sxy[i]*c13+syy[i]*c23+syz[i]*c33;
                    double p31=sxz[i]*c11+syz[i]*c21+szzf[i]*c31;
                    double p32=sxz[i]*c12+syz[i]*c22+szzf[i]*c32;
                    double p33=sxz[i]*c13+syz[i]*c23+szzf[i]*c33;
                    double axx=0, ayy=0, azz=0;
                    for (long q = nb0Start[i]; q < nb0Start[i+1]; ++q) {
                        int j = id2idx[nb0J[q]];
                        if (!isTL[j]) continue;
                        double g0x, g0y, g0z;
                        float gz = (DIM==3)?nb0Gz[q]:0.0f;
                        corrGrad(i, nb0Gx[q], nb0Gy[q], gz, g0x, g0y, g0z);
                        double jc11= F22[j]*F33[j]-F23[j]*F32[j];
                        double jc12=-(F21[j]*F33[j]-F23[j]*F31[j]);
                        double jc13= F21[j]*F32[j]-F22[j]*F31[j];
                        double jc21=-(F12[j]*F33[j]-F13[j]*F32[j]);
                        double jc22= F11[j]*F33[j]-F13[j]*F31[j];
                        double jc23=-(F11[j]*F32[j]-F12[j]*F31[j]);
                        double jc31= F12[j]*F23[j]-F13[j]*F22[j];
                        double jc32=-(F11[j]*F23[j]-F13[j]*F21[j]);
                        double jc33= F11[j]*F22[j]-F12[j]*F21[j];
                        double q11=sxx[j]*jc11+sxy[j]*jc21+sxz[j]*jc31;
                        double q12=sxx[j]*jc12+sxy[j]*jc22+sxz[j]*jc32;
                        double q13=sxx[j]*jc13+sxy[j]*jc23+sxz[j]*jc33;
                        double q21=sxy[j]*jc11+syy[j]*jc21+syz[j]*jc31;
                        double q22=sxy[j]*jc12+syy[j]*jc22+syz[j]*jc32;
                        double q23=sxy[j]*jc13+syy[j]*jc23+syz[j]*jc33;
                        double q31=sxz[j]*jc11+syz[j]*jc21+szzf[j]*jc31;
                        double q32=sxz[j]*jc12+syz[j]*jc22+szzf[j]*jc32;
                        double q33=sxz[j]*jc13+syz[j]*jc23+szzf[j]*jc33;
                        axx += V0*((p11-q11)*g0x+(p12-q12)*g0y+(p13-q13)*g0z)/Mi.rho0;
                        ayy += V0*((p21-q21)*g0x+(p22-q22)*g0y+(p23-q23)*g0z)/Mi.rho0;
                        azz += V0*((p31-q31)*g0x+(p32-q32)*g0y+(p33-q33)*g0z)/Mi.rho0;
                        // viscosity based on the current geometry
                        double dxp=X[i]-X[j], dyp=Y[i]-Y[j];
                        double dzp=(DIM==3)?(Z[i]-Z[j]):0.0;
                        double r2=dxp*dxp+dyp*dyp+dzp*dzp;
                        if (r2 < H2*H2 && r2 > 1e-16) {
                            double dux=vhx[i]-vhx[j], duy=vhy[i]-vhy[j];
                            double duz=vhz[i]-vhz[j];
                            double vr=dux*dxp+duy*dyp+duz*dzp;
                            if (vr<0) {
                                double r=sqrt(r2);
                                double muv=H*vr/(r2+0.01*H*H);
                                double cij=0.5*(cson[i]+cson[j]);
                                double Pi=(-ALPHA*cij*muv+BETA*muv*muv)
                                          /(0.5*(rho[i]+rho[j]));
                                double dwdr=kernelDW(r);
                                axx -= mass[j]*Pi*dwdr*dxp/r;
                                ayy -= mass[j]*Pi*dwdr*dyp/r;
                                azz -= mass[j]*Pi*dwdr*dzp/r;
                            }
                        }
                    }
                    ax[i]=axx; ay[i]=ayy; az[i]=azz;
                }
            }
            // ------- reduction of thread-private accumulators -------
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (long i = 0; i < (long)N; ++i) {
                double sx=ax[i], sy=ay[i], sz=az[i];
                for (int t=0;t<NT;++t) {
                    sx+=TAx[(size_t)t*N+i]; sy+=TAy[(size_t)t*N+i];
                    sz+=TAz[(size_t)t*N+i];
                }
                ax[i]=sx; ay[i]=sy; az[i]=sz;
            }
            // amax
            double amaxL = 0;
            #pragma omp for schedule(static) nowait
            for (long i = 0; i < (long)N; ++i) {
                amaxL = std::max(amaxL, ax[i]*ax[i]+ay[i]*ay[i]+az[i]*az[i]);
            }
            #pragma omp critical
            amax = std::max(amax, amaxL);
        }
        return sqrt(amax);
    };

    // ---------------- Z-order reordering ----------------
    auto zorderSort = [&]() {
        double xmin=1e9, ymin=1e9, zmin=1e9;
        for (size_t k=0;k<N;++k){ xmin=std::min(xmin,X[k]);
                                  ymin=std::min(ymin,Y[k]);
                                  if (DIM==3) zmin=std::min(zmin,Z[k]); }
        std::vector<uint64_t> keys(N);
        for (size_t k=0;k<N;++k) {
            uint32_t cx=(uint32_t)std::min((X[k]-xmin)/H2+1, 2e6);
            uint32_t cy=(uint32_t)std::min((Y[k]-ymin)/H2+1, 2e6);
            uint32_t cz=(DIM==3)?(uint32_t)std::min((Z[k]-zmin)/H2+1, 2e6):1u;
            keys[k] = (uint64_t)(matid[k]?1:0)<<63 | morton3(cx,cy,cz);
        }
        std::vector<int> ord(N);
        for (size_t k=0;k<N;++k) ord[k]=(int)k;
        std::sort(ord.begin(), ord.end(),
                  [&](int a, int b){ return keys[a]<keys[b]; });
        auto permD=[&](std::vector<double>& v){ auto t=v;
            for(size_t k=0;k<N;++k) v[k]=t[ord[k]]; };
        auto permI=[&](std::vector<int>& v){ auto t=v;
            for(size_t k=0;k<N;++k) v[k]=t[ord[k]]; };
        auto permC=[&](std::vector<char>& v){ auto t=v;
            for(size_t k=0;k<N;++k) v[k]=t[ord[k]]; };
        permD(X);permD(Y);permD(Z);
        permD(vx);permD(vy);permD(vz);permD(ax);permD(ay);permD(az);
        permD(xsx);permD(xsy);permD(xsz);permD(rho);permD(mass);
        permD(Sxx);permD(Syy);permD(Szz);permD(Sxy);permD(Sxz);permD(Syz);
        permD(epsp);permD(dmg);permD(Tk);permD(Pfld);permD(cson);
        permD(invRho);permD(invRho2);
        permD(F11);permD(F12);permD(F13);permD(F21);permD(F22);permD(F23);
        permD(F31);permD(F32);permD(F33);
        permD(B11);permD(B12);permD(B13);permD(B21);permD(B22);permD(B23);
        permD(B31);permD(B32);permD(B33);
        permD(L11);permD(L12);permD(L13);permD(L21);permD(L22);permD(L23);
        permD(L31);permD(L32);permD(L33);permD(drho);
        permD(sxx);permD(syy);permD(szzf);permD(sxy);permD(sxz);permD(syz);
        permI(matid); permI(pid);
        permC(isTL); permC(isFix);
        // the segmented TL reference-list layout is keyed by array indices ->
        // rebuild it together with the particles (particle-ID records are unchanged)
        if (refCaptured) {
            std::vector<long> oldStart = nb0Start;
            std::vector<int>   oldJ = nb0J;
            std::vector<float> oldGx = nb0Gx, oldGy = nb0Gy, oldGz = nb0Gz;
            std::vector<long> cnts(N);
            for (size_t k=0;k<N;++k) cnts[k] = oldStart[ord[k]+1] - oldStart[ord[k]];
            nb0Start.assign(N+1, 0);
            for (size_t k=0;k<N;++k) nb0Start[k+1] = nb0Start[k] + cnts[k];
            long total = nb0Start[N];
            nb0J.assign(total,0); nb0Gx.assign(total,0); nb0Gy.assign(total,0);
            if (DIM==3) nb0Gz.assign(total,0);
            for (size_t k=0;k<N;++k) {
                long src = oldStart[ord[k]], dst = nb0Start[k];
                for (long m=0;m<cnts[k];++m) {
                    nb0J[dst+m]  = oldJ[src+m];
                    nb0Gx[dst+m] = oldGx[src+m];
                    nb0Gy[dst+m] = oldGy[src+m];
                    if (DIM==3) nb0Gz[dst+m] = oldGz[src+m];
                }
            }
        }
        // X0/Y0/Z0 remain unchanged: reference configuration and B0 are computed
        // once at step 0 (before sorting), when pid == index.
        for (size_t k=0;k<N;++k) id2idx[pid[k]]=(int)k;
        // the CSR layout is keyed by array indices -> invalidate it (rebuild on demand)
        nbStart.assign(N+1, 0);
        nbCnt.assign(N, 0);
    };

    // ================== initialization (step 0, dt=0) ==================
    buildCSR();
    if (par.use_tl) captureReference();
    double cmax = c_long0 + par.v_impact; long tlCnt = N;
    ratesPass(vx, vy, vz, par.xs_eps > 0);
    updatePass(0.0, cmax, tlCnt);
    double amax = forcesPass(vx, vy, vz);
    double dt = std::min(par.cfl*H/cmax, 0.25*sqrt(H/std::max(amax,1.0)));
    dt = std::min(std::max(dt, 1e-11), 1e-7);

    // ================== main loop (leapfrog / velocity-Verlet) ==================
    const double t_end = par.t_end * 1e-6;
    const long max_steps = 5000000;
    int n_frames_plan = std::max(par.n_frames, 2);
    double t = 0;
    long step = 0;
    int prog_last = -1;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> vh_x(N), vh_y(N), vh_z(N, 0.0);

    writeFrame(0.0);
    int frame_idx = 1;
    double next_frame_t = t_end / (n_frames_plan - 1);
    bool failed = false;

    for (; step < max_steps && t < t_end; ++step) {
        double _t0s = NOWSEC();
        // periodic Z-order reorder (state is consistent at the step boundary)
        if (par.z_every > 0 && step > 0 && step % par.z_every == 0) {
            double _zs=NOWSEC();
            zorderSort();
            if (PROF) T_SORT += NOWSEC()-_zs;
        }
        // kick #1 + drift (using XSPH from the previous pass)
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)N; ++i) {
            if (isFix[i]) { vh_x[i]=vh_y[i]=vh_z[i]=0; continue; }
            vh_x[i] = vx[i] + 0.5*dt*ax[i];
            vh_y[i] = vy[i] + 0.5*dt*ay[i];
            vh_z[i] = vz[i] + 0.5*dt*az[i];
            X[i] += (vh_x[i] + xsx[i])*dt;
            Y[i] += (vh_y[i] + xsy[i])*dt;
            if (DIM==3) Z[i] += (vh_z[i] + xsz[i])*dt;
        }
        // neighbors + all physics
        double _t1=NOWSEC();
        buildCSR();
        double _t2=NOWSEC();
        ratesPass(vh_x, vh_y, vh_z, par.xs_eps > 0);
        double _t3=NOWSEC();
        updatePass(dt, cmax, tlCnt);
        double _t4=NOWSEC();
        amax = forcesPass(vh_x, vh_y, vh_z);
        double _t5=NOWSEC();
        if (PROF) { T_KICK+=_t1-_t0s; T_CSR+=_t2-_t1; T_RATE+=_t3-_t2;
                    T_UPD+=_t4-_t3; T_FORCE+=_t5-_t4; }
        // kick#2
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < (long)N; ++i) {
            if (isFix[i]) { vx[i]=vy[i]=vz[i]=0; continue; }
            vx[i] = vh_x[i] + 0.5*dt*ax[i];
            vy[i] = vh_y[i] + 0.5*dt*ay[i];
            vz[i] = vh_z[i] + 0.5*dt*az[i];
        }
        t += dt;
        double dt_new = std::min(par.cfl*H/std::max(cmax,1.0),
                                 0.25*sqrt(H/std::max(amax,1.0)));
        dt = std::min(dt_new, 1.3*dt);
        dt = std::min(std::max(dt, 1e-11), 1e-7);
        if (t >= next_frame_t && frame_idx < n_frames_plan - 1) {
            writeFrame(t);
            ++frame_idx;
            next_frame_t = t_end * frame_idx / (n_frames_plan - 1);
        }
        int prog = (int)(1000.0 * t / t_end);
        if (prog >= prog_last + 10 || prog >= 1000) {
            prog_last = prog;
            double mvx=0,mvy=0;
            for (size_t k=0;k<NP;++k){mvx+=vx[k];mvy+=vy[k];}
            mvx/=NP; mvy/=NP;
            auto el = std::chrono::duration<double>(
                std::chrono::steady_clock::now()-t0).count();
            printf("PROGRESS %d 1000 t=%.1f v=(%.0f,%.0f) dt=%.1fns tl=%ld%% elapsed=%.1f\n",
                   std::min(prog,1000), t*1e6, mvx, mvy, dt*1e9,
                   100L*tlCnt/(long)N, el);
            fflush(stdout);
        }
        if ((step % 200) == 0) {
            bool finite = true;
            #pragma omp parallel for reduction(&:finite)
            for (long i = 0; i < (long)N; ++i) {
                finite = finite && std::isfinite(X[i]) && std::isfinite(Y[i]) &&
                         std::isfinite(Z[i]) && std::isfinite(vx[i]) &&
                         std::isfinite(vy[i]) && std::isfinite(vz[i]) &&
                         std::isfinite(rho[i]) && std::isfinite(Pfld[i]) &&
                         std::isfinite(Tk[i]) && std::isfinite(dmg[i]);
            }
            if (!finite) {
                fprintf(stderr, "NUMERICAL FAILURE at step %ld (t=%g)\n", step, t);
                failed = true; break;
            }
        }
    }
    writeFrame(t);
    fclose(ff);
    if (failed) return 1;
    if (PROF)
        fprintf(stderr, "PROF: kick=%.2f sort=%.2f csr=%.2f rate=%.2f upd=%.2f "
                "force=%.2f s (steps=%ld, cold=%ld)\n",
                T_KICK, T_SORT, T_CSR, T_RATE, T_UPD, T_FORCE, step, N_COLD);

    // ---------------- verdict ----------------
    double mvx=0,mvy=0,mxi=0,intact=0,mvz=0;
    for (size_t k=0;k<NP;++k){
        mvx+=vx[k]; mvy+=vy[k]; mvz+=vz[k];
        mxi += X[k]*nx_ + Y[k]*ny_;
        if (dmg[k] < 1.0) intact+=1;
    }
    mvx/=NP; mvy/=NP; mvz/=NP; mxi/=NP; intact/=NP;
    double vres = sqrt(mvx*mvx+mvy*mvy+mvz*mvz);
    double vn = mvx*nx_+mvy*ny_;
    double angOut = atan2(mvy,mvx)*180/M_PI;
    const char* verdict;
    if (mxi > T_ && vn > 50)        verdict = "PENETRATION";
    else if (vn < -20)              verdict = "RICOCHET";
    else                            verdict = "EMBEDDED";
    printf("RESULT verdict=%s vres=%.0f angle_out=%.1f intact=%.2f tl=%.2f\n",
           verdict, vres, angOut, intact, (double)tlCnt/N);

    // ---------------- meta ----------------
    FILE* fm = fopen((par.out+"/meta.txt").c_str(),"w");
    if (!fm) {
        fprintf(stderr, "cannot write metadata in %s\n", par.out.c_str());
        return 1;
    }
    fprintf(fm,"VER=2\nDIM=%d\nNFIELDS=%d\nN=%zu\nNP=%zu\n"
               "verdict=%s\nvres=%.1f\nangle_out=%.1f\nintact=%.3f\ntl=%.3f\n"
               "v=%g\nangle=%g\nt_plate=%g\nh_plate=%g\nw_plate=%g\nproj_l=%g\n"
               "proj_d=%g\ndx=%g\nt_end=%g\nproj_mat=%s\narm_mat=%s\nnose=%s\n",
            DIM, NF_OUT, N, NP, verdict, vres, angOut, intact, (double)tlCnt/N,
            par.v_impact, par.angle_deg, par.t_plate, par.h_plate, par.w_plate,
            par.proj_l, par.proj_d, par.dx, par.t_end,
            par.proj_mat.c_str(), par.arm_mat.c_str(), par.nose.c_str());
    fclose(fm);

    // ---------------- shared CSV log ----------------
    FILE* fc = fopen("summary.csv","a");
    if (fc) {
        fseek(fc, 0, SEEK_END);
        if (ftell(fc) == 0)
            fprintf(fc, "out,dim,v,angle,t_plate,w_plate,proj_l,proj_d,dx,"
                        "proj_mat,arm_mat,nose,tlmode,verdict,vres,angle_out,"
                        "intact,tl_left,elapsed_s\n");
        auto el = std::chrono::duration<double>(
            std::chrono::steady_clock::now()-t0).count();
        fprintf(fc, "%s,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.2f,%s,%s,%s,%d,%s,"
                    "%.0f,%.1f,%.3f,%.3f,%.1f\n",
                par.out.c_str(), DIM, par.v_impact, par.angle_deg, par.t_plate,
                par.w_plate, par.proj_l, par.proj_d, par.dx,
                par.proj_mat.c_str(), par.arm_mat.c_str(), par.nose.c_str(),
                par.use_tl, verdict, vres, angOut, intact,
                (double)tlCnt/N, el);
        fclose(fc);
    }
    auto el = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-t0).count();
    printf("DONE %.1f s\n", el);
    return 0;
}
