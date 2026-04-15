/**
 * @file kernel_benchmark.cu
 * @brief Standalone kernel performance benchmark
 * Purpose: Analyze key kernel GPU metrics under Nsight Compute (ncu)
 */

#include "solver.cuh"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#ifdef _WIN32
#include <windows.h>
#endif

static constexpr int DEFAULT_NX = 1024;
static constexpr int DEFAULT_NY = 512;
static constexpr int WARMUP_ITERATIONS = 3;    // Warmup iterations (not timed, pre-heat GPU cache)
static constexpr int BENCHMARK_ITERATIONS = 5; // Benchmark iterations (ncu samples each call)

#define CUDA_BENCH_CHECK(call)                                 \
    do                                                         \
    {                                                          \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess)                                \
        {                                                      \
            std::cerr << "[CUDA Error] " << __FILE__ << ":"    \
                      << __LINE__ << ": "                      \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main(int argc, char *argv[])
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    // ---- Parse command-line arguments ----
    int nx = (argc > 1) ? atoi(argv[1]) : DEFAULT_NX;
    int ny = (argc > 2) ? atoi(argv[2]) : DEFAULT_NY;
    if (nx <= 0 || ny <= 0)
    {
        std::cerr << "Usage: " << argv[0] << " [nx=" << DEFAULT_NX << "] [ny=" << DEFAULT_NY << "]" << std::endl;
        return 1;
    }

    std::cout << "================================================================" << std::endl;
    std::cout << "  FVM Kernel Performance Benchmark (for Nsight Compute)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Grid size:  " << nx << " x " << ny << " = " << nx * ny << " cells" << std::endl;
    std::cout << "Warmup:     " << WARMUP_ITERATIONS << std::endl;
    std::cout << "Benchmark:  " << BENCHMARK_ITERATIONS << std::endl;
    std::cout << "================================================================" << std::endl
              << std::endl;

    // ---- GPU info ----
    int device;
    cudaDeviceProp prop;
    CUDA_BENCH_CHECK(cudaGetDevice(&device));
    CUDA_BENCH_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU:        " << prop.name << std::endl;
    std::cout << "SM count:   " << prop.multiProcessorCount << std::endl;
    std::cout << "VRAM:       " << std::fixed << std::setprecision(0) << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Compute:    " << prop.major << "." << prop.minor << std::endl
              << std::endl;

    // ---- Initialize solver ----
    std::cout << "[1/4] Initializing solver..." << std::endl;

    SimParams params;
    params.nx = nx;
    params.ny = ny;
    params.mach = 3.0f;
    params.T_inf = 300.0f;
    params.p_inf = 101325.0f;
    params.enable_viscosity = true;
    params.cfl = 0.5f;
    params.cfl_visc = 0.4f;
    params.obstacle_shape = ObstacleShape::CIRCLE;
    params.obstacle_r = 0.5f;
    params.domain_width = 10.0f;
    params.domain_height = 5.0f;
    params.computeDerived();

    CFDSolver solver;
    solver.initialize(params);
    solver.reset(params);

    // Compute initial dt (based on CFL condition)
    float dt = solver.computeStableTimeStep(params);
    params.dt = dt;
    std::cout << "            Grid spacing: dx=" << std::fixed << std::setprecision(4) << params.dx << ", dy=" << params.dy << std::endl;
    std::cout << "            Time step:    dt=" << std::scientific << std::setprecision(2) << params.dt << " s  (CFL=" << std::fixed << std::setprecision(2) << params.cfl << ")" << std::endl;

    // ---- Advance flow field to non-uniform state ----
    std::cout << "[2/4] Advancing flow field to non-uniform state (10 steps)..." << std::endl;
    for (int i = 0; i < 10; i++)
    {
        solver.step(params);
    }
    CUDA_BENCH_CHECK(cudaDeviceSynchronize());
    std::cout << "            Flow advanced " << params.step << " steps, t=" << std::scientific << std::setprecision(4) << params.t_current << " s" << std::endl
              << std::endl;

    // ---- Warmup phase ----
    std::cout << "[3/4] GPU warmup (" << WARMUP_ITERATIONS << " steps, not timed)..." << std::endl;
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        solver.step(params);
    }
    CUDA_BENCH_CHECK(cudaDeviceSynchronize());

    // ---- Benchmark phase ----
    std::cout << "[4/4] Running benchmark (" << BENCHMARK_ITERATIONS << " steps)..." << std::endl
              << std::endl;
    std::cout << "  Each step contains a full Strang-splitting cycle:" << std::endl;
    std::cout << "    ViscousTerms -> DiffusionStep (dt/2)" << std::endl;
    std::cout << "    Primitives -> Fluxes -> Update" << std::endl;
    std::cout << "    Primitives -> ViscousTerms -> DiffusionStep" << std::endl;
    std::cout << "    BoundaryConditions -> Primitives" << std::endl;
    std::cout << "  ncu will automatically sample each kernel call" << std::endl
              << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        solver.step(params);
    }
    CUDA_BENCH_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Benchmark completed:" << std::endl;
    std::cout << "  Total time:     " << std::fixed << std::setprecision(2) << elapsed << " ms" << std::endl;
    std::cout << "  Avg per step:   " << std::fixed << std::setprecision(2) << (elapsed / BENCHMARK_ITERATIONS) << " ms" << std::endl;
    std::cout << "  Total steps:    " << params.step << std::endl;
    std::cout << "  Final sim time: " << std::scientific << std::setprecision(4) << params.t_current << " s" << std::endl
              << std::endl;

    return 0;
}
