#pragma once

#include "common.h"
#include <iostream>

#pragma region 宏定义
// 宏定义：检查 CUDA 函数调用的返回值
#define CUDA_CHECK(call)                                               \
    do                                                                 \
    {                                                                  \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess)                                        \
        {                                                              \
            std::cerr << "CUDA 相关函数调用返回错误结果 '" << __FILE__ \
                      << "' in line " << __LINE__ << " : "             \
                      << cudaGetErrorString(err) << "." << std::endl;  \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)
#pragma endregion

class CFDSolver
{
public:
    CFDSolver();
    ~CFDSolver();

    // 初始化求解器，分配内存等
    void initialize(const SimParams &params);
    void resize(int nx, int ny);

    // 进行单步仿真计算
    void step(SimParams &params);

    // 按照当前参数重置求解器状态
    void reset(const SimParams &params);

    // 动态更新障碍物几何（不重置仿真，仅更新SDF/网格类型并修正新暴露区域）
    // 适用于：障碍物位置、大小、旋转、形状、襟翼角度的任意变化
    void updateObstacleGeometry(const SimParams &params);

    // 主机端数据传输（网格类型更新用）
    void getCellTypes(uint8_t *host_types);

    // GPU端网格类型转换（uint8→float，直接写入设备指针，用于CUDA-GL互操作零拷贝更新）
    void convertCellTypesToDevice(float *devOutput);

    // GPU零拷贝路径（CUDA-OpenGL互操作）
    void computeTemperatureToDevice(float *dev_dst);
    void computePressureToDevice(float *dev_dst);
    void computeDensityToDevice(float *dev_dst);
    void computeVelocityMagToDevice(float *dev_dst);
    void computeMachToDevice(float *dev_dst);

    // 生成速度矢量箭头，直接写入OpenGL VBO
    // 返回实际生成的顶点数，失败返回-1
    int generateVectorArrows(float *dev_vertexData, int maxVertices,
                             int step, float u_inf,
                             float maxArrowLength, float arrowHeadAngle, float arrowHeadLength);

    // 获取网格尺寸
    int getNx() const { return _nx; }
    int getNy() const { return _ny; }

    // 计算稳定时间步长(CFL条件)
    float computeStableTimeStep(const SimParams &params);

    // 异步时间步长计算流水线（消除GPU空闲气泡）
    // queueTimeStepComputation: 将归约核函数+异步回传排入GPU流，不阻塞CPU
    // readTimeStepResult: 从锁页内存读取已完成的结果（调用前需确保GPU已同步）
    void queueTimeStepComputation(const SimParams &params);
    float readTimeStepResult(const SimParams &params) const;

    // 获取统计数据
    float getMaxTemperature();
    float getMaxMach();

    static void getGPUMemoryInfo(size_t &totalMem, size_t &freeMem);

    size_t getSimulationMemoryUsage();

private:
    // 显存分配器
    void allocateMemory();
    void freeMemory();

    // 网格维度
    int _nx = 1024;
    int _ny = 512;

    // 保守量的显存位置指针
    float *d_rho_ = nullptr;   // 网格空气密度
    float *d_rho_u_ = nullptr; // 网格空气水平速度
    float *d_rho_v_ = nullptr; // 网格空气垂直速度
    float *d_E_ = nullptr;     // 网格内空气总能量（内能+动能）
    float *d_rho_e_ = nullptr; // 网格内空气内能密度（用于双能量法）

    // 为实现双缓冲需要再保存下一个状态的保守量
    float *d_rho_new_ = nullptr;
    float *d_rho_u_new_ = nullptr;
    float *d_rho_v_new_ = nullptr;
    float *d_E_new_ = nullptr;
    float *d_rho_e_new_ = nullptr;

    // SSP-RK2 第二阶段的中间缓冲区
    float *d_rho_rk_ = nullptr;
    float *d_rho_u_rk_ = nullptr;
    float *d_rho_v_rk_ = nullptr;
    float *d_E_rk_ = nullptr;
    float *d_rho_e_rk_ = nullptr;

    // 为实现可视化和通量计算保存的原始变量
    float *d_u_ = nullptr; // x轴空气速度
    float *d_v_ = nullptr; // y轴空气速度
    float *d_p_ = nullptr; // 空气压强
    float *d_T_ = nullptr; // 空气温度

    // 用于 IBM 的辅助数据场
    uint8_t *d_cell_type_ = nullptr; // 网格类型标记
    float *d_sdf_ = nullptr;         // 符号距离场

    // 通量，中间存储
    float *d_flux_rho_x_ = nullptr;   // x轴-质量通量场
    float *d_flux_rho_u_x_ = nullptr; // x轴-x方向动量-通量场
    float *d_flux_rho_v_x_ = nullptr; // x轴-y方向动量-通量场
    float *d_flux_E_x_ = nullptr;     // x轴-能量-通量场
    float *d_flux_rho_e_x_ = nullptr; // x轴-内能-通量场

    float *d_flux_rho_y_ = nullptr;   // y轴-质量通量场
    float *d_flux_rho_u_y_ = nullptr; // y轴-x方向动量-通量场
    float *d_flux_rho_v_y_ = nullptr; // y轴-y方向动量-通量场
    float *d_flux_E_y_ = nullptr;     // y轴-能量-通量场
    float *d_flux_rho_e_y_ = nullptr; // y轴-内能-通量场

    // 归约缓冲区及其大小
    float *d_reduction_buffer_ = nullptr; // CUB临时存储空间
    float *d_reduction_output_ = nullptr; // 归约结果输出（单个float）
    size_t reduction_buffer_size_ = 0;    // 动态计算的缓冲区字节数

    // 预分配的临时计算缓冲区（用于归约前的中间计算，避免每次调用cudaMalloc/cudaFree）
    float *d_scratch_ = nullptr;

    // 锁页主机内存（用于computeStableTimeStep异步归约流水线，消除CPU-GPU同步停顿）
    // [0]=最大波速, [1]=最大粘性数
    float *h_pinnedReduction_ = nullptr;

    // 预分配的原子计数器（用于矢量箭头生成，避免每帧cudaMalloc/cudaFree）
    int *d_atomic_counter_ = nullptr;

    // 粘性相关中间量 (Navier-Stokes方程)
    float *d_mu_ = nullptr; // 动力粘性系数场（供CFL条件使用，由 fusedViscousDiffusionKernel 输出）

    // 功能:启动初始化核函数，将全场设为来流条件
    // 输入:守恒变量数组指针，仿真参数，网格尺寸
    // 输出:初始化后的守恒变量场
    void launchInitializeKernel(float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
                                const SimParams &params, int nx, int ny);

    // 功能:启动原始变量计算核函数，从守恒变量推导原始变量
    // 输入:守恒变量场(rho, rho_u, rho_v, E, rho_e)，网格尺寸
    // 输出:原始变量场(u, v, p, T)，使用双能量法保证精度
    void launchComputePrimitivesKernel(const float *rho, const float *rho_u,
                                       const float *rho_v, const float *E,
                                       const float *rho_e,
                                       float *u, float *v, float *p, float *T,
                                       int nx, int ny);

    // 功能:启动通量计算核函数，使用MUSCL重构和HLLC Riemann求解器
    // 输入:密度(rho)和原始变量(u,v,p)，网格类型，仿真参数
    // 输出:X和Y方向的数值通量(flux_*_x/y)，包括质量/动量/能量/内能通量
    // 注意:仅需rho和原始变量，守恒量(rho_u,rho_v,E,rho_e)和T在内核中未使用
    void launchComputeFluxesKernel(const float *rho,
                                   const float *u, const float *v,
                                   const float *p,
                                   const uint8_t *cell_type,
                                   float *flux_rho_x, float *flux_rho_u_x,
                                   float *flux_rho_v_x, float *flux_E_x,
                                   float *flux_rho_e_x,
                                   float *flux_rho_y, float *flux_rho_u_y,
                                   float *flux_rho_v_y, float *flux_E_y,
                                   float *flux_rho_e_y,
                                   const SimParams &params, int nx, int ny);

    // 功能:启动更新核函数，使用有限体积法和双能量法更新守恒变量
    // 输入:当前守恒变量，已计算的原始变量(u,v,p)，X/Y方向通量，网格类型
    // 输出:下一时间步的守恒变量(rho_new, rho_u_new等)，通过双缓冲实现
    // 注意:直接复用原始变量避免重复的除法(rho_u/rho)和压强反算
    void launchUpdateKernel(const float *rho, const float *rho_u,
                            const float *rho_v, const float *E, const float *rho_e,
                            const float *u, const float *v, const float *p,
                            const float *flux_rho_x, const float *flux_rho_u_x,
                            const float *flux_rho_v_x, const float *flux_E_x,
                            const float *flux_rho_e_x,
                            const float *flux_rho_y, const float *flux_rho_u_y,
                            const float *flux_rho_v_y, const float *flux_E_y,
                            const float *flux_rho_e_y,
                            const uint8_t *cell_type,
                            float *rho_new, float *rho_u_new,
                            float *rho_v_new, float *E_new, float *rho_e_new,
                            const SimParams &params, int nx, int ny);

    // 功能:启动边界条件应用核函数，处理所有边界类型
    // 输入:守恒变量，网格类型，SDF场，仿真参数
    // 输出:应用边界条件后的守恒变量(流入/流出/固体壁面/Ghost Cell)
    void launchApplyBoundaryConditionsKernel(float *rho, float *rho_u, float *rho_v, float *E,
                                             float *rho_e,
                                             const uint8_t *cell_type, const float *sdf,
                                             const SimParams &params, int nx, int ny);

    // 功能:启动SSP-RK2凸组合混合核函数，将 U^n 与 U** 按 0.5:0.5 加权得到 U^{n+1}
    // 输入:dst守恒变量(U^n)，src守恒变量(U**)，网格尺寸
    // 输出:覆写dst为 U^{n+1} = 0.5*U^n + 0.5*U**
    void launchSSPRK2BlendKernel(float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
                                 const float *rho_rk, const float *rho_u_rk, const float *rho_v_rk,
                                 const float *E_rk, const float *rho_e_rk, int nx, int ny);

    // 功能:启动SDF计算核函数，计算带符号距离场并初始化网格类型
    // 输入:障碍物几何参数(位置/尺寸/旋转/形状)，网格尺寸
    // 输出:SDF场和网格类型标记(流体/固体/虚拟/流入/流出)
    void launchComputeSDFKernel(float *sdf, uint8_t *cell_type,
                                const SimParams &params, int nx, int ny);

    // 功能:使用CUB库归约计算全场最大温度
    // 输入:温度场数组，网格尺寸
    // 输出:最大温度值(用于监控激波强度和数值稳定性)
    // 实现:使用CUB库直接对温度数组归约，最高效的方式
    float launchComputeMaxTemperature(const float *T, int nx, int ny);

    // 功能:归约计算全场最大马赫数
    // 输入:速度场(u,v)，压强场，密度场，网格尺寸
    // 输出:最大马赫数 Ma = |v| / c_local (用于判断流动类型：亚音速/超音速)
    // 实现:使用CUB库高效归约，单阶段GPU完成（已替换旧的两阶段手写实现）
    float launchComputeMaxMach(const float *u, const float *v, const float *p,
                               const float *rho, int nx, int ny);

    // 功能:归约计算全场最大波速(速度+声速)
    // 输入:速度场(u,v)，压强场，密度场，网格尺寸
    // 输出:最大波速 = |v| + c (用于CFL条件的时间步长限制)
    // 实现:使用CUB库高效归约，单阶段GPU完成
    float launchComputeMaxWaveSpeed(const float *u, const float *v, const float *p,
                                    const float *rho, int nx, int ny);

    // 功能:融合粘性-扩散核函数(替代 ViscousTerms + DiffusionStep 双核函数调用)
    // 原理:每线程按需重算中心+4邻居的粘性量(寄存器级)，消除 tau/q 中间数组全局内存往返
    // 输出:mu(供CFL), 直接更新 rho_u/rho_v/E/rho_e
    void launchFusedViscousDiffusionKernel(const float *u, const float *v, const float *T,
                                           float *mu,
                                           float *rho_u, float *rho_v, float *E, float *rho_e,
                                           const uint8_t *cell_type,
                                           float dt, float dx, float dy, int nx, int ny);

    // 功能:归约计算全场最大运动粘性系数 nu = mu / rho
    // 输入:动力粘性系数场，密度场，网格尺寸
    // 输出:最大运动粘性系数(用于粘性CFL条件: dt <= dx^2 / nu)
    // 实现:使用CUB库高效归约，单阶段GPU完成
    float launchComputeMaxViscousNumber(const float *mu, const float *rho, int nx, int ny);
};
