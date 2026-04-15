// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "common.h"
#include "solver.cuh"
#include "renderer.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef ENABLE_STREAMING
#include "streaming/stream_server.h"
#endif
// clang-format on

#pragma region 常量和全局变量

// 窗口尺寸
int windowWidth = 1600;
int windowHeight = 900;
const float DEFAULT_FONT_SIZE = 18.0f;
const std::string DEFAULT_FONT_PATH = "assets/fonts/msyh.ttc";

// 性能监控
float fps = 0.0f;
float simTimePerStep = 0.0f; // 每步仿真耗时，单位秒
int stepsPerFrame = 1;       // 进行多少步仿真计算后再渲染一帧

// 仿真参数和求解器
SimParams params;
CFDSolver solver;

// 主机存储缓冲区（仅保留网格类型所需的缓冲区）
std::vector<uint8_t> h_cellTypes;

// 可视化
Renderer renderer;
FieldType currentField = FieldType::TEMPERATURE;
const char *colormapNames[] = {"Jet", "Hot", "Plasma", "Inferno", "Viridis"};
int currentColormap = 0;

// 多线程：求解器与渲染器帧级同步（生产者-消费者模型）
std::mutex g_solverMutex;
std::condition_variable g_solverCV;
std::atomic<bool> g_solverShouldStop{false};
std::thread g_solverThread;
enum class SolverState { IDLE, RUNNING, DONE };
SolverState g_solverState = SolverState::IDLE; // 受 g_solverMutex 保护

// 延迟操作标志（键盘回调中设置，主循环安全区域执行）
std::atomic<bool> g_resetRequested{false};

// 障碍物几何更新标志（UI线程设置，求解器线程消费）
std::atomic<bool> g_obstacleGeometryDirty{false};
// 形状改变标志（需要完整reset，UI线程设置，求解器线程消费）
std::atomic<bool> g_shapeResetRequested{false};

// dt重算标志（重置/几何变化后设置，强制求解器线程同步重算dt）
std::atomic<bool> g_dtNeedsRecompute{true};

// 缓存的仿真统计数据（在渲染线程安全区域更新）
float cachedMaxTemp = 0.0f;
float cachedMaxMach = 0.0f;

// 垂直同步控制
bool vsyncEnabled = true;

#ifdef ENABLE_STREAMING
// 串流服务器实例
StreamServer g_streamServer;
bool g_streamingEnabled = false;
#endif

// 求解器帧结果（求解器线程DONE前写入，渲染线程DONE后读取，通过条件变量同步保护）
struct SolverFrameResults {
    float maxTemp = 0.0f;
    float maxMach = 0.0f;
    float simTimePerStep = 0.0f;
    int vectorVertexCount = 0;  // 本帧生成的矢量箭头顶点数
    bool cellTypesDirty = false; // 求解器线程更新了网格类型，渲染线程应提交到GL纹理
};
SolverFrameResults g_solverResults;

// 颜色映射范围控制变量（绝对值）
float temperature_min = 200.0f; // 温度下限 (K)
float temperature_max = 400.0f; // 温度上限 (K)
float pressure_min = 50000.0f;  // 压强下限 (Pa)
float pressure_max = 200000.0f; // 压强上限 (Pa)
float density_min = 0.5f;       // 密度下限 (kg/m³)
float density_max = 2.0f;       // 密度上限 (kg/m³)
float velocity_max = 500.0f;    // 速度上限 (m/s)
float mach_max = 2.0f;          // 马赫数上限

// UI滑块范围限制常量
constexpr float TEMPERATURE_MIN_LIMIT = 100.0f;
constexpr float TEMPERATURE_MAX_LIMIT = 5000.0f;
constexpr float PRESSURE_MIN_LIMIT = 1000.0f;
constexpr float PRESSURE_MAX_LIMIT = 500000.0f;
constexpr float DENSITY_MIN_LIMIT = 0.01f;
constexpr float DENSITY_MAX_LIMIT = 10.0f;
constexpr float VELOCITY_MAX_LIMIT = 2000.0f;
constexpr float MACH_MAX_LIMIT = 10.0f;

// 矢量箭头渲染常量
constexpr float ARROW_HEAD_ANGLE = 0.5f;               // 箭头头部张角 [rad]
constexpr float ARROW_HEAD_LENGTH_RATIO = 0.3f;        // 箭头头部占箭身比例
constexpr float ARROW_LENGTH_SCALE = 0.8f;             // 箭头长度缩放因子（相对于网格间距）
constexpr int VERTICES_PER_ARROW = 8;                  // 每箭头顶点数（箭身2+头部6）

// 性能监控节流常量
constexpr float STATS_THROTTLE_INTERVAL = 0.25f;       // 统计归约节流间隔 [秒]
constexpr float FPS_UPDATE_INTERVAL = 0.5f;            // FPS显示更新间隔 [秒]

// CUDA环境检测常量
constexpr int MIN_COMPUTE_CAPABILITY_MAJOR = 7;        // 最低计算能力主版本号
constexpr int MIN_COMPUTE_CAPABILITY_MINOR = 5;        // 最低计算能力次版本号

#pragma endregion

#pragma region 窗口回调函数
void framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;
    renderer.resize(width, height);
#ifdef ENABLE_STREAMING
    if (g_streamingEnabled)
        g_streamServer.requestResize(width, height);
#endif
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        params.paused = !params.paused;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
    {
        g_resetRequested = true;
    }
}
#pragma endregion

#pragma region 工具函数
void setupImGuiFont(ImGuiIO &io, const std::string &fontPath, float fontSize)
{
    ImFontConfig fontConfig;
    fontConfig.OversampleH = 3;   // 水平过采样
    fontConfig.OversampleV = 1;   // 垂直过采样
    fontConfig.PixelSnapH = true; // 像素对齐

    if (std::filesystem::exists(fontPath))
    {
        io.Fonts->AddFontFromFileTTF(fontPath.c_str(), fontSize, &fontConfig,
                                     io.Fonts->GetGlyphRangesChineseFull());
    }
    else
    {
        std::cerr << "[警告] 字体文件未找到，请检查assets/fonts，使用默认字体。" << std::endl;
        std::cerr << "程序正在试图查找的字体路径：" << fontPath << std::endl;
        std::cerr << "当前工作目录：" << std::filesystem::current_path() << std::endl;
        io.Fonts->AddFontDefault(&fontConfig);
    }
}
#pragma endregion

#pragma region 求解器设定
void resizeBuffers()
{
    size_t size = params.nx * params.ny;
    h_cellTypes.resize(size);
}

bool initializeSimulation()
{
    params.computeDerived();
    resizeBuffers();
    solver.initialize(params);

    // Get initial cell types
    solver.getCellTypes(h_cellTypes.data());
    renderer.updateCellTypes(h_cellTypes.data(), params.nx, params.ny);

    // 初始化GPU零拷贝互操作
    if (!renderer.initCudaInterop(params.nx, params.ny))
    {
        std::cerr << "[错误] CUDA-OpenGL互操作初始化失败" << std::endl;
        return false;
    }

    return true;
}
#pragma endregion

#pragma region 求解器线程
// 求解器线程函数：独立于渲染线程运行仿真计算
// 采用"生产者-消费者"帧同步模型：
//   每批计算 stepsPerFrame 步 + 可视化写入PBO/VBO + 统计归约
//   全部GPU工作完成后通知渲染线程，实现最大化并行
void solverThreadFunc()
{
    cudaSetDevice(0);

    while (true)
    {
        // 等待渲染线程发出"开始计算"信号
        {
            std::unique_lock<std::mutex> lock(g_solverMutex);
            g_solverCV.wait(lock, [] {
                return g_solverState == SolverState::RUNNING || g_solverShouldStop.load();
            });
            if (g_solverShouldStop) return;
        }

        // ==================== 0. 障碍物几何更新（延迟到求解器线程执行，消除主线程CUDA同步开销） ====================
        if (g_shapeResetRequested.exchange(false))
        {
            // 形状/大小变化：完整重置流场
            solver.reset(params);
            g_dtNeedsRecompute = true;
            // 将cell type写入预映射的PBO
            float *ctPtr = renderer.getMappedCellTypePtr();
            if (ctPtr)
            {
                solver.convertCellTypesToDevice(ctPtr);
                g_solverResults.cellTypesDirty = true;
            }
        }
        else if (g_obstacleGeometryDirty.exchange(false))
        {
            // 位置/旋转/襟翼等变化：热更新SDF（不重置流场）
            solver.updateObstacleGeometry(params);
            // 将cell type写入预映射的PBO
            float *ctPtr = renderer.getMappedCellTypePtr();
            if (ctPtr)
            {
                solver.convertCellTypesToDevice(ctPtr);
                g_solverResults.cellTypesDirty = true;
            }
        }

        // ==================== 1. 仿真计算 ====================
        auto simStart = std::chrono::high_resolution_clock::now();

        // 延迟dt模式：首帧或重置后同步计算，否则使用上帧异步计算的结果
        // 这消除了computeStableTimeStep中的GPU空闲气泡
        static float cachedDt = 0.0f;
        if (g_dtNeedsRecompute.exchange(false))
        {
            cachedDt = solver.computeStableTimeStep(params); // 同步版本，仅用于首帧/重置
        }
        params.dt = cachedDt;
        int n = stepsPerFrame; // 读一次当前值
        for (int i = 0; i < n; i++)
        {
            solver.step(params);
        }

        // ==================== 2. 可视化：直接写入预映射的PBO（零拷贝，无需GL上下文） ====================
        // PBO[writeIndex]已由GL线程预映射，此处获取的设备指针可直接写入
        float *devPtr = renderer.getMappedFieldPtr();
        if (devPtr)
        {
            switch (currentField)
            {
            case FieldType::TEMPERATURE:
                solver.computeTemperatureToDevice(devPtr);
                break;
            case FieldType::PRESSURE:
                solver.computePressureToDevice(devPtr);
                break;
            case FieldType::DENSITY:
                solver.computeDensityToDevice(devPtr);
                break;
            case FieldType::VELOCITY_MAG:
                solver.computeVelocityMagToDevice(devPtr);
                break;
            case FieldType::MACH:
                solver.computeMachToDevice(devPtr);
                break;
            }
        }

        // ==================== 3. 矢量箭头：直接写入预映射的VBO（零拷贝） ====================
        g_solverResults.vectorVertexCount = 0;
        if (renderer.getShowVectors())
        {
            int step = renderer.getVectorDensity();
            float cellWidth = 2.0f / params.nx;
            float cellHeight = 2.0f / params.ny;
            float maxArrowLength = std::min(cellWidth, cellHeight) * (step * ARROW_LENGTH_SCALE);

            int vboCapacity = renderer.getMappedVectorCapacity();
            float *devVertexData = renderer.getMappedVectorPtr();
            if (devVertexData && vboCapacity > 0)
            {
                int numVertices = solver.generateVectorArrows(
                    devVertexData, vboCapacity,
                    step, params.u_inf,
                    maxArrowLength, ARROW_HEAD_ANGLE, ARROW_HEAD_LENGTH_RATIO);
                g_solverResults.vectorVertexCount = numVertices;
            }
        }

        // ==================== 4. 统计归约（节流至0.25秒一次） ====================
        {
            static auto lastStatsTime = std::chrono::high_resolution_clock::now();
            auto statsNow = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<float>(statsNow - lastStatsTime).count() >= STATS_THROTTLE_INTERVAL)
            {
                g_solverResults.maxTemp = solver.getMaxTemperature();
                g_solverResults.maxMach = solver.getMaxMach();
                lastStatsTime = statsNow;
            }
        }

        // ==================== 5. 异步排入下帧dt计算（GPU流水线，无CPU停顿） ====================
        // 归约核函数紧跟step+viz内核后执行，GPU零空闲
        solver.queueTimeStepComputation(params);

        // ==================== 6. 同步所有GPU工作（step+viz+dt归约） ====================
        cudaDeviceSynchronize();

        // 读取异步dt结果（锁页内存已就绪，零延迟）
        cachedDt = solver.readTimeStepResult(params);

        auto simEnd = std::chrono::high_resolution_clock::now();
        g_solverResults.simTimePerStep = std::chrono::duration<float>(simEnd - simStart).count() / n;

        // ==================== 6. 通知渲染线程 ====================
        {
            std::lock_guard<std::mutex> lock(g_solverMutex);
            g_solverState = SolverState::DONE;
        }
        g_solverCV.notify_one();
    }
}
#pragma endregion

#pragma region 控制面板渲染
void renderUI()
{
    ImGui::Begin(u8"有限体积法空气动力学模拟控制面板");
    static float inputFontSize = DEFAULT_FONT_SIZE;
    if (ImGui::SliderFloat(u8"字体大小", &inputFontSize, 20.0f, 32.0f))
    {
        ImGuiIO &io = ImGui::GetIO();
        io.FontGlobalScale = inputFontSize / DEFAULT_FONT_SIZE;
    };

    if (ImGui::CollapsingHeader("性能监控"))
    {
        ImGui::Text(u8"帧率: %.1f FPS", fps);
        ImGui::Text(u8"单步耗时: %.3f 毫秒", simTimePerStep * 1000.0f);
        ImGui::Text(u8"仿真时间: %.6f 秒", params.t_current);
        ImGui::Text(u8"迭代步数: %d", params.step);
        ImGui::SliderInt(u8"每帧迭代数", &stepsPerFrame, 1, 100);

        // 垂直同步热切换
        if (ImGui::Checkbox(u8"垂直同步", &vsyncEnabled))
        {
            glfwSwapInterval(vsyncEnabled ? 1 : 0);
        }

        ImGui::Separator();

        // GPU 显存信息（节流查询：每1秒更新一次，避免驱动调用造成性能开销）
        static float cachedUsedMem = 0.0f;
        static float cachedTotalMemMB = 0.0f;
        static float cachedMemUsagePercent = 0.0f;
        static auto lastMemQueryTime = std::chrono::high_resolution_clock::now();
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<float>(now - lastMemQueryTime).count() >= 1.0f)
        {
            lastMemQueryTime = now;
            size_t freeMem, totalMem;
            CFDSolver::getGPUMemoryInfo(freeMem, totalMem);
            cachedUsedMem = (totalMem - freeMem) / (1024.0f * 1024.0f);
            cachedTotalMemMB = totalMem / (1024.0f * 1024.0f);
            cachedMemUsagePercent = (totalMem - freeMem) * 100.0f / totalMem;
        }

        ImGui::Text(u8"GPU 显存使用");
        ImGui::SameLine();

        // 用一个进度条显示
        // RGBA 绿色->黄色->红色
        ImVec4 barColor = (cachedMemUsagePercent < 70.0f) ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : (cachedMemUsagePercent < 90.0f) ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f)
                                                                                                                 : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
        char memInfoLabel[64];
        std::snprintf(memInfoLabel, sizeof(memInfoLabel), u8"GPU 显存使用: %.1f MB / %.1f MB (%.1f%%)", cachedUsedMem, cachedTotalMemMB, cachedMemUsagePercent);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
        ImGui::ProgressBar(cachedMemUsagePercent / 100.0f, ImVec2(200, 0), memInfoLabel);
        ImGui::PopStyleColor(2);

        size_t simMemory = solver.getSimulationMemoryUsage();
        ImGui::Text(u8"仿真数据占用显存: %.1f MB", simMemory / (1024.0f * 1024.0f));

        ImGui::Separator();

        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), u8"GPU零拷贝加速已启用");
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"GPU直接写入显示纹理，跳过所有中间传输");
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"仿真控制"))
    {
        if (ImGui::Button(params.paused ? u8"开始（space）" : u8"暂停（space）"))
        {
            params.paused = !params.paused;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"重置（R）"))
        {
            g_resetRequested = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"单步（N）") && params.paused)
        {
            if (params.paused)
            {
                solver.step(params);
                params.t_current += params.dt;
                params.step += 1;
                g_dtNeedsRecompute = true;
            }
        }
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"网格设置"))
    {
        static int nx_ui = 1024;
        static int ny_ui = 512;

        ImGui::SliderInt(u8"X轴网格分辨率", &nx_ui, 64, 4096);
        ImGui::SliderInt(u8"Y轴网格分辨率", &ny_ui, 32, 4096);

        // 如果调整了，先显示再决定要不要应用修改
        ImGui::Text(u8"当前网格分辨率：%d x %d", params.nx, params.ny);
        if (nx_ui != params.nx || ny_ui != params.ny)
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), u8" -> %d x %d", nx_ui, ny_ui);
        }

        if (ImGui::Button(u8"应用网格尺寸"))
        {
            params.nx = nx_ui;
            params.ny = ny_ui;
            params.computeDerived();
            // 重新初始化仿真（重新分配显存和缓冲区）
            initializeSimulation();
            g_dtNeedsRecompute = true;
            // 重置时间记录
            params.t_current = 0.0f;
            params.step = 0;
        }

        ImGui::Text(u8"dx = %.4f m, dy = %.4f m", params.dx, params.dy);
        ImGui::Text(u8"计算域: %.1f x %.1f m", params.domain_width, params.domain_height);
        ImGui::Text(u8"总网格数: %d", params.nx * params.ny);
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"来流条件", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool changed = false;

        changed |= ImGui::SliderFloat(u8"马赫数", &params.mach, 0.01f, 10.0f);
        changed |= ImGui::SliderFloat(u8"来流温度 (K)", &params.T_inf, 200.0f, 400.0f);
        changed |= ImGui::SliderFloat(u8"来流压强 (Pa)", &params.p_inf, 10000.0f, 101325.0f);

        if (changed)
        {
            params.computeDerived();
        }

        ImGui::Text(u8"来流密度 = %.4f kg/m^3", params.rho_inf);
        ImGui::Text(u8"来流速度 = %.1f m/s", params.u_inf);
        ImGui::Text(u8"声速 = %.1f m/s", params.c_inf);

        ImGui::SliderFloat(u8"CFL数", &params.cfl, 0.1f, 0.9f);
        ImGui::Text(u8"时间步长 = %.2e s", params.dt);
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"粘性设置 (Navier-Stokes)"))
    {
        bool viscosityChanged = false;

        if (ImGui::Checkbox(u8"启用粘性模拟", &params.enable_viscosity))
        {
            viscosityChanged = true;
        }

        if (params.enable_viscosity)
        {
            ImGui::SliderFloat(u8"扩散CFL数", &params.cfl_visc, 0.1f, 0.5f);

            ImGui::Separator();

            ImGui::Text(u8"壁面边界条件:");

            if (ImGui::Checkbox(u8"绝热壁面", &params.adiabatic_wall))
            {
                viscosityChanged = true;
            }

            if (!params.adiabatic_wall)
            {
                if (ImGui::SliderFloat(u8"壁面温度 (K)", &params.T_wall, 200.0f, 1000.0f))
                {
                    viscosityChanged = true;
                }
            }

            ImGui::Separator();

            // 计算来流粘性，用Sutherland公式，只代表刚飞进来的气体
            float mu_inf = MU_REF * powf(params.T_inf / T_REF, 1.5f) *
                           (T_REF + S_SUTHERLAND) / (params.T_inf + S_SUTHERLAND);
            // 雷诺值，惯性力与粘性力之比
            float Re = params.rho_inf * params.u_inf * (2.0f * params.obstacle_r) / mu_inf;
            ImGui::Text(u8"雷诺数 Re = %.0f", Re);
            ImGui::Text(u8"来流粘性 mu = %.2e Pa·s", mu_inf);

            ImGui::Separator();
            ImGui::TextWrapped(u8"注意：启用粘性后，计算量增加约50%%。粘性CFL通常比对流CFL更严格。");
        }
        else
        {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), u8"注意：当前为无粘性欧拉方程求解");
        }

        if (viscosityChanged)
        {
            g_resetRequested = true;
        }
    }

    ImGui::Separator();

    // 障碍物设置
    if (ImGui::CollapsingHeader(u8"障碍物设置", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool shapeChanged = false;
        bool obstacleChanged = false;

        // Quick shape buttons
        ImGui::Text(u8"障碍物形状:");
        ImGui::SameLine();
        if (ImGui::Button(u8"圆形"))
        {
            params.obstacle_shape = ObstacleShape::CIRCLE;
            shapeChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"五角星"))
        {
            params.obstacle_shape = ObstacleShape::STAR;
            shapeChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"菱形"))
        {
            params.obstacle_shape = ObstacleShape::DIAMOND;
            shapeChanged = true;
        }
        if (ImGui::Button(u8"胶囊形"))
        {
            params.obstacle_shape = ObstacleShape::CAPSULE;
            shapeChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"三角形"))
        {
            params.obstacle_shape = ObstacleShape::TRIANGLE;
            shapeChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"星舰"))
        {
            params.obstacle_shape = ObstacleShape::STARSHIP;
            shapeChanged = true;
        }

        ImGui::Separator();

        obstacleChanged |= ImGui::SliderFloat(u8"中心 X 坐标", &params.obstacle_x, 0.5f, params.domain_width * 0.5f);
        shapeChanged |= ImGui::SliderFloat(u8"大小 (半径)", &params.obstacle_r, 0.1f, 1.5f);

        // 障碍物旋转角度
        float rotationDeg = params.obstacle_rotation * 180.0f / PI;
        if (ImGui::SliderFloat(u8"旋转角度 (度)", &rotationDeg, -180.0f, 180.0f))
        {
            params.obstacle_rotation = rotationDeg * PI / 180.0f;
            obstacleChanged = true;
        }
        if (ImGui::InputFloat(u8"精确旋转角度 (度)", &rotationDeg))
        {
            params.obstacle_rotation = rotationDeg * PI / 180.0f;
            obstacleChanged = true;
        }

        if (params.obstacle_shape == ObstacleShape::STARSHIP)
        {
            float wingRotationDeg = params.wing_rotation * 180.0f / PI;
            if (ImGui::SliderFloat(u8"襟翼旋转角度 (度)", &wingRotationDeg, 0.0f, 90.0f))
            {
                params.wing_rotation = wingRotationDeg * PI / 180.0f;
                obstacleChanged = true;
            }
            if (ImGui::InputFloat(u8"襟翼精确旋转角度 (度)", &wingRotationDeg))
            {
                wingRotationDeg = std::clamp(wingRotationDeg, 0.0f, 90.0f);
                params.wing_rotation = wingRotationDeg * PI / 180.0f;
                obstacleChanged = true;
            }
        }

        if (shapeChanged)
        {
            params.obstacle_y = params.domain_height / 2.0f;
            // 延迟到求解器线程执行（避免安全区域的CUDA同步开销）
            g_shapeResetRequested = true;
        }
        else if (obstacleChanged)
        {
            // 延迟到求解器线程执行（避免安全区域的CUDA同步开销）
            g_obstacleGeometryDirty = true;
        }
    }
    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"可视化设置", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const char *fieldNames[] = {u8"温度", u8"压强", u8"密度", u8"速度大小", u8"马赫数"};
        int fieldIdx = static_cast<int>(currentField);
        if (ImGui::Combo(u8"显示物理量", &fieldIdx, fieldNames, 5))
        {
            currentField = static_cast<FieldType>(fieldIdx);
        }

        if (ImGui::Combo(u8"色图", &currentColormap, colormapNames, 5))
        {
            renderer.setColormap(static_cast<ColormapType>(currentColormap));
        }

        // 根据当前显示的物理量动态调整范围控制
        ImGui::Separator();
        ImGui::Text(u8"颜色映射范围调整:");

        switch (currentField)
        {
        case FieldType::TEMPERATURE:
            ImGui::SliderFloat(u8"温度下限 (K)", &temperature_min, TEMPERATURE_MIN_LIMIT, TEMPERATURE_MAX_LIMIT);
            ImGui::SliderFloat(u8"温度上限 (K)", &temperature_max, TEMPERATURE_MIN_LIMIT, TEMPERATURE_MAX_LIMIT);
            if (temperature_min > temperature_max)
                temperature_min = temperature_max;
            break;
        case FieldType::PRESSURE:
            ImGui::SliderFloat(u8"压强下限 (Pa)", &pressure_min, PRESSURE_MIN_LIMIT, PRESSURE_MAX_LIMIT);
            ImGui::SliderFloat(u8"压强上限 (Pa)", &pressure_max, PRESSURE_MIN_LIMIT, PRESSURE_MAX_LIMIT);
            if (pressure_min > pressure_max)
                pressure_min = pressure_max;
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"参考: 来流压强 = %.0f Pa", params.p_inf);
            break;
        case FieldType::DENSITY:
            ImGui::SliderFloat(u8"密度下限 (kg/m³)", &density_min, DENSITY_MIN_LIMIT, DENSITY_MAX_LIMIT);
            ImGui::SliderFloat(u8"密度上限 (kg/m³)", &density_max, DENSITY_MIN_LIMIT, DENSITY_MAX_LIMIT);
            if (density_min > density_max)
                density_min = density_max;
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"参考: 来流密度 = %.3f kg/m³", params.rho_inf);
            break;
        case FieldType::VELOCITY_MAG:
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"速度下限固定为 0 m/s");
            ImGui::SliderFloat(u8"速度上限 (m/s)", &velocity_max, 0.0f, VELOCITY_MAX_LIMIT);
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"参考: 来流速度 = %.1f m/s", params.u_inf);
            break;
        case FieldType::MACH:
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"马赫数下限固定为 0");
            ImGui::SliderFloat(u8"马赫数上限", &mach_max, 0.0f, MACH_MAX_LIMIT);
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"参考: 来流马赫数 = %.2f", params.mach);
            break;
        }
        ImGui::Separator();

        // 速度矢量显示（仅在速度大小可视化模式下可用）
        if (currentField == FieldType::VELOCITY_MAG)
        {
            bool showVectors = renderer.getShowVectors();
            if (ImGui::Checkbox(u8"显示速度矢量", &showVectors))
            {
                renderer.setShowVectors(showVectors);
            }

            // 如果显示矢量开启，显示密度控制滑块
            if (showVectors)
            {
                int vectorDensity = renderer.getVectorDensity();
                if (ImGui::SliderInt(u8"矢量箭头间隔", &vectorDensity, 5, 100, u8"%d 格"))
                {
                    renderer.setVectorDensity(vectorDensity);
                }
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), u8"（值越小箭头越密集）");
            }
        }
        else
        {
            // 非速度可视化模式时，自动关闭矢量显示
            renderer.setShowVectors(false);
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), u8"（速度矢量仅在[速度大小]模式下可用）");
        }

        // Statistics（使用缓存值，实际查询在主循环安全区域以0.25秒间隔执行）
        ImGui::Text(u8"最高温度: %.1f K", cachedMaxTemp);
        ImGui::Text(u8"最大马赫数: %.2f", cachedMaxMach);
    }

    ImGui::Separator();

    // Colorbar
    if (ImGui::CollapsingHeader(u8"色标", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImDrawList *drawList = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float barWidth = 200.0f;
        float barHeight = 20.0f;

        // Draw colorbar
        for (int i = 0; i < (int)barWidth; i++)
        {
            float t = (float)i / barWidth;
            float r, g, b;

            // 使用解耦的颜色映射函数
            getColormapColor(static_cast<ColormapType>(currentColormap), t, r, g, b);

            ImU32 color = IM_COL32(r * 255, g * 255, b * 255, 255);
            drawList->AddRectFilled(
                ImVec2(pos.x + i, pos.y),
                ImVec2(pos.x + i + 1, pos.y + barHeight),
                color);
        }

        ImGui::Dummy(ImVec2(barWidth, barHeight + 5));

        // Labels
        const char *unit = "";
        float minVal = 0.0f;
        float maxVal = 1.0f;

        switch (currentField)
        {
        case FieldType::TEMPERATURE:
            unit = "K";
            minVal = temperature_min;
            maxVal = temperature_max;
            break;
        case FieldType::PRESSURE:
            unit = "Pa";
            minVal = pressure_min;
            maxVal = pressure_max;
            break;
        case FieldType::DENSITY:
            unit = "kg/m^3";
            minVal = density_min;
            maxVal = density_max;
            break;
        case FieldType::VELOCITY_MAG:
            unit = "m/s";
            minVal = 0;
            maxVal = velocity_max;
            break;
        case FieldType::MACH:
            unit = "";
            minVal = 0;
            maxVal = mach_max;
            break;
        }

        ImGui::Text("%.1f %s", minVal, unit);
        ImGui::SameLine(barWidth - 50);
        ImGui::Text("%.1f %s", maxVal, unit);
    }

    ImGui::Separator();
    ImGui::End();
}
#pragma endregion

#pragma region 可视化函数
// GPU零拷贝可视化更新
// 数据流：保守变量(GPU) -> 直接计算到PBO(GPU) -> OpenGL纹理(GPU)
void updateVisualization()
{
    float minVal, maxVal;

    // 计算值域范围
    switch (currentField)
    {
    case FieldType::TEMPERATURE:
        minVal = temperature_min;
        maxVal = temperature_max;
        break;
    case FieldType::PRESSURE:
        minVal = pressure_min;
        maxVal = pressure_max;
        break;
    case FieldType::DENSITY:
        minVal = density_min;
        maxVal = density_max;
        break;
    case FieldType::VELOCITY_MAG:
        minVal = 0.0f;
        maxVal = velocity_max;
        break;
    case FieldType::MACH:
        minVal = 0.0f;
        maxVal = mach_max;
        break;
    }

    renderer.setFieldRange(minVal, maxVal, currentField);

    // PBO[writeIndex]已预映射，直接获取设备指针写入
    float *devPtr = renderer.getMappedFieldPtr();
    if (!devPtr)
    {
        std::cerr << "[错误] PBO未映射" << std::endl;
        return;
    }

    // 直接从保守变量计算到PBO，完全避免GPU-GPU拷贝
    switch (currentField)
    {
    case FieldType::TEMPERATURE:
        solver.computeTemperatureToDevice(devPtr);
        break;
    case FieldType::PRESSURE:
        solver.computePressureToDevice(devPtr);
        break;
    case FieldType::DENSITY:
        solver.computeDensityToDevice(devPtr);
        break;
    case FieldType::VELOCITY_MAG:
        solver.computeVelocityMagToDevice(devPtr);
        break;
    case FieldType::MACH:
        solver.computeMachToDevice(devPtr);
        break;
    }

    // 提交PBO：unmap → 上传纹理 → swap → remap
    renderer.submitField();

    // 生成矢量箭头（如果启用）
    if (renderer.getShowVectors())
    {
        int step = renderer.getVectorDensity();

        // 计算箭头参数

        // 计算单个格子在NDC中的尺寸
        float cellWidth = 2.0f / params.nx;
        float cellHeight = 2.0f / params.ny;
        float maxArrowLength = std::min(cellWidth, cellHeight) * (step * ARROW_LENGTH_SCALE);

        // 计算最大可能的箭头数量（每个箭头8个顶点）
        int numArrowsX = (params.nx + step - 1) / step;
        int numArrowsY = (params.ny + step - 1) / step;
        int maxVertices = numArrowsX * numArrowsY * 8;

        // 确保VBO有足够容量（同时确保VBO已映射）
        renderer.ensureVectorVBOCapacity(maxVertices);

        // VBO[writeIndex]已预映射，直接获取设备指针写入
        float *devVertexData = renderer.getMappedVectorPtr();
        int vboCapacity = renderer.getMappedVectorCapacity();
        if (devVertexData && vboCapacity > 0)
        {
            int numVertices = solver.generateVectorArrows(
                devVertexData, vboCapacity,
                step, params.u_inf,
                maxArrowLength, ARROW_HEAD_ANGLE, ARROW_HEAD_LENGTH_RATIO);

            // 提交VBO：unmap → swap → remap
            renderer.submitVectors(numVertices);
            renderer.prepareVectorRender();
        }
    }
}
#pragma endregion

#pragma region CUDA环境检测
// 检查CUDA/GPU可用性，返回false表示不可用
bool checkCudaAvailability()
{
    std::cout << "[信息] 正在检测 CUDA 环境..." << std::endl;

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess)
    {
        std::cerr << "========================================" << std::endl;
        std::cerr << "[错误] CUDA 初始化失败！" << std::endl;
        std::cerr << "错误代码: " << static_cast<int>(error) << std::endl;

        const char *errorStr = cudaGetErrorString(error);
        const char *errorName = cudaGetErrorName(error);

        if (errorStr && strlen(errorStr) > 0)
        {
            std::cerr << "错误信息: " << errorStr << std::endl;
        }
        if (errorName && strlen(errorName) > 0)
        {
            std::cerr << "错误名称: " << errorName << std::endl;
        }

        std::cerr << std::endl;

        // 根据错误代码给出具体建议
        if (error == cudaErrorNoDevice)
        {
            std::cerr << "[诊断] 未检测到 NVIDIA GPU" << std::endl;
            std::cerr << std::endl;
            std::cerr << "本程序需要 NVIDIA 显卡才能运行。" << std::endl;
            std::cerr << "如果您有 NVIDIA 显卡，请尝试：" << std::endl;
            std::cerr << "  1. 更新 NVIDIA 显卡驱动" << std::endl;
            std::cerr << "  2. 在 BIOS 中启用独立显卡" << std::endl;
        }
        else if (error == cudaErrorInsufficientDriver)
        {
            std::cerr << "[诊断] 显卡驱动版本过旧" << std::endl;
            std::cerr << std::endl;
            std::cerr << "请更新 NVIDIA 显卡驱动到最新版本：" << std::endl;
            std::cerr << "  https://www.nvidia.cn/Download/index.aspx" << std::endl;
        }
        else if (error == cudaErrorInitializationError)
        {
            std::cerr << "[诊断] CUDA 驱动初始化失败" << std::endl;
            std::cerr << std::endl;
            std::cerr << "可能的原因：" << std::endl;
            std::cerr << "  1. 显卡驱动损坏或不兼容" << std::endl;
            std::cerr << "  2. 其他程序占用了 GPU" << std::endl;
            std::cerr << "  3. 系统刚从休眠恢复" << std::endl;
            std::cerr << std::endl;
            std::cerr << "建议：重启电脑后再试" << std::endl;
        }
        else
        {
            std::cerr << "可能的原因：" << std::endl;
            std::cerr << "  1. 您的电脑没有 NVIDIA 显卡" << std::endl;
            std::cerr << "  2. NVIDIA 显卡驱动未安装或版本过旧" << std::endl;
            std::cerr << "  3. CUDA 运行时库 (cudart64_*.dll) 缺失" << std::endl;
            std::cerr << std::endl;
            std::cerr << "解决方案：" << std::endl;
            std::cerr << "  - 本程序需要 NVIDIA 显卡才能运行" << std::endl;
            std::cerr << "  - 请更新显卡驱动: https://www.nvidia.cn/Download/index.aspx" << std::endl;
        }

        std::cerr << "========================================" << std::endl;
        return false;
    }

    if (deviceCount == 0)
    {
        std::cerr << "========================================" << std::endl;
        std::cerr << "[错误] 未检测到支持 CUDA 的 GPU！" << std::endl;
        std::cerr << std::endl;
        std::cerr << "本程序需要 NVIDIA 显卡进行 GPU 加速计算。" << std::endl;
        std::cerr << "请确保您的电脑配备了 NVIDIA 独立显卡。" << std::endl;
        std::cerr << "========================================" << std::endl;
        return false;
    }

    // 获取并显示GPU信息
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess)
    {
        std::cerr << "[警告] 无法获取 GPU 详细信息" << std::endl;
    }
    else
    {
        std::cout << "[信息] 检测到 GPU: " << prop.name << std::endl;
        std::cout << "[信息] 计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "[信息] 显存大小: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;

        // 检查计算能力是否足够（至少需要 7.5，对应 RTX 20 系列）
        int computeCapability = prop.major * 10 + prop.minor;
        if (computeCapability < MIN_COMPUTE_CAPABILITY_MAJOR * 10 + MIN_COMPUTE_CAPABILITY_MINOR)
        {
            std::cerr << "========================================" << std::endl;
            std::cerr << "[警告] GPU 计算能力较低 (" << prop.major << "." << prop.minor << ")" << std::endl;
            std::cerr << "本程序针对 RTX 20 系列及更新显卡优化。" << std::endl;
            std::cerr << "旧显卡可能无法运行或性能较差。" << std::endl;
            std::cerr << "========================================" << std::endl;
            // 不返回false，尝试继续运行
        }
    }

    std::cout << "[信息] CUDA 环境检测通过" << std::endl;
    return true;
}

#pragma endregion

int main(int argc, char *argv[])
{
// 设置控制台编码为 UTF-8，支持中文输出
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    //  设置工作目录为程序所在目录，方便加载资源文件
    std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    std::filesystem::current_path(exePath);

    // 【关键】首先检查CUDA/GPU可用性
    if (!checkCudaAvailability())
    {
        std::cerr << std::endl;
        std::cerr << "按任意键退出..." << std::endl;
        system("pause");
        return -1;
    }

    // 初始化GLFW
    if (!glfwInit())
    {
        std::cerr << "[错误] 程序在初始化GLFW阶段失败并退出" << std::endl;
        system("pause");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(windowWidth, windowHeight, "FVM空气动力学模拟器", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "[错误] 程序在创建窗口阶段失败并退出";
        system("pause");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // 使用真实帧缓冲像素尺寸初始化渲染/串流，避免高DPI下分辨率偏差。
    glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
    
    // 设置回调函数
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    // 初始化GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "[错误] 程序在初始化GLAD阶段失败并退出";
        system("pause");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 初始化ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;                                             // 用于避免未使用警告
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // 启用键盘控制
    ImGui::StyleColorsDark();                             // 设置深色主题
    ImGui_ImplGlfw_InitForOpenGL(window, true);           // ImGui接管GLFW输入
    ImGui_ImplOpenGL3_Init("#version 430 core");          // 指定GLSL版本，编写Shader时会用到
    // 设置中文字体
    setupImGuiFont(io, DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE);

    // 初始化渲染器
    if (!renderer.initialize(windowWidth, windowHeight))
    {
        std::cerr << "[错误] 程序在初始化Renderer阶段失败并退出" << std::endl;
        system("pause");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 初始化仿真器
    if (!initializeSimulation())
    {
        std::cerr << "[错误] 程序在初始化Solver阶段失败并退出" << std::endl;
        system("pause");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 启动求解器线程
    g_solverShouldStop = false;
    g_solverThread = std::thread(solverThreadFunc);

#ifdef ENABLE_STREAMING
    // 初始化串流服务器（可选功能，失败不影响主程序）
    {
        StreamServer::Config streamCfg;
        streamCfg.httpPort = 8080;
        streamCfg.wsPort = 8081;
        streamCfg.bitrateMbps = 15;
        streamCfg.fps = 60;
        // web 目录路径：相对于可执行文件
        streamCfg.webRoot = (std::filesystem::current_path() / "web").string();
        if (g_streamServer.initialize(streamCfg, windowWidth, windowHeight)) {
            g_streamingEnabled = true;
            std::cout << "[串流] 浏览器串流已启动，访问 http://<本机IP>:8080\n";
        } else {
            std::cerr << "[串流] 串流初始化失败（主程序正常运行，串流功能不可用）\n";
        }
    }
#endif

    // 初始化时间和帧率
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    // 主渲染循环
    // 架构：渲染线程与求解器线程使用条件变量进行帧级同步
    // 每帧流程：
    //   1. 等待求解器完成上一批计算
    //   2. [安全区域] 处理UI、可视化、统计（求解器不运行）
    //   3. 信号求解器开始下一批计算
    //   4. [并行区域] OpenGL渲染当前帧（求解器同时计算下一帧数据）
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

#ifdef ENABLE_STREAMING
        if (g_streamingEnabled) {
            g_streamServer.inputHandler().processEvents(window, windowWidth, windowHeight,
                [window](int targetFbW, int targetFbH) {
                    targetFbW = std::max(64, targetFbW & ~1);
                    targetFbH = std::max(64, targetFbH & ~1);

                    int currentFbW = 0, currentFbH = 0;
                    glfwGetFramebufferSize(window, &currentFbW, &currentFbH);
                    if (std::abs(currentFbW - targetFbW) <= 1 &&
                        std::abs(currentFbH - targetFbH) <= 1)
                    {
                        return;
                    }

                    float scaleX = 1.0f, scaleY = 1.0f;
                    glfwGetWindowContentScale(window, &scaleX, &scaleY);
                    if (scaleX <= 0.0f) scaleX = 1.0f;
                    if (scaleY <= 0.0f) scaleY = 1.0f;

                    int winW = std::max(1, (int)std::lround(targetFbW / scaleX));
                    int winH = std::max(1, (int)std::lround(targetFbH / scaleY));
                    glfwSetWindowSize(window, winW, winH);
                });
        }
#endif

        // ==================== 等待求解器完成 ====================
        bool hasNewResults = false;
        {
            std::unique_lock<std::mutex> lock(g_solverMutex);
            if (g_solverState == SolverState::RUNNING)
            {
                g_solverCV.wait(lock, [] {
                    return g_solverState != SolverState::RUNNING;
                });
            }
            // 消费DONE状态：标记有新结果可用
            hasNewResults = (g_solverState == SolverState::DONE);
            if (hasNewResults) g_solverState = SolverState::IDLE;
        }

        // ========= 安全区域：求解器不运行，可自由访问GPU数据 =========

        // 处理延迟的键盘操作（回调中设置标志，此处安全执行）
        if (g_resetRequested.exchange(false))
        {
            params.t_current = 0.0f;
            params.step = 0;
            solver.reset(params);
            g_dtNeedsRecompute = true;
            // 更新网格类型到PBO
            float *ctPtr = renderer.getMappedCellTypePtr();
            if (ctPtr)
            {
                solver.convertCellTypesToDevice(ctPtr);
                renderer.submitCellTypes();
            }
        }

        // 消费求解器帧结果：提交PBO/VBO（unmap → upload → swap → remap）+ 读取统计
        if (hasNewResults)
        {
            renderer.submitField();                                  // GL: unmap→纹理→swap→remap
            if (g_solverResults.vectorVertexCount > 0)
            {
                renderer.submitVectors(g_solverResults.vectorVertexCount); // GL: unmap→swap→remap
            }
            renderer.prepareVectorRender();                          // 快照VBO读取索引

            // 如果求解器线程更新了网格类型，提交到GL纹理
            if (g_solverResults.cellTypesDirty)
            {
                renderer.submitCellTypes();  // GL: unmap PBO → glTexSubImage2D → remap
                g_solverResults.cellTypesDirty = false;
            }

            cachedMaxTemp = g_solverResults.maxTemp;
            cachedMaxMach = g_solverResults.maxMach;
            simTimePerStep = g_solverResults.simTimePerStep;
        }

        // ImGui帧开始 + UI逻辑（所有solver交互在此安全发生）
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        renderUI();

        // 设置场值范围（用于渲染着色器）
        {
            float minVal, maxVal;
            switch (currentField)
            {
            case FieldType::TEMPERATURE: minVal = temperature_min; maxVal = temperature_max; break;
            case FieldType::PRESSURE:    minVal = pressure_min;    maxVal = pressure_max;    break;
            case FieldType::DENSITY:     minVal = density_min;     maxVal = density_max;     break;
            case FieldType::VELOCITY_MAG: minVal = 0.0f;           maxVal = velocity_max;    break;
            case FieldType::MACH:        minVal = 0.0f;            maxVal = mach_max;        break;
            }
            renderer.setFieldRange(minVal, maxVal, currentField);
        }

        // 确保矢量VBO容量（GL操作，必须在GL线程；在solver使用前完成）
        if (renderer.getShowVectors())
        {
            int step = renderer.getVectorDensity();
            int numArrowsX = (params.nx + step - 1) / step;
            int numArrowsY = (params.ny + step - 1) / step;
            int maxVertices = numArrowsX * numArrowsY * VERTICES_PER_ARROW;
            renderer.ensureVectorVBOCapacity(maxVertices);
        }

        // ==================== 信号求解器或处理暂停 ====================
        if (!params.paused)
        {
            // 非暂停：启动求解器线程（计算、可视化、统计全部在solver线程完成）
            std::lock_guard<std::mutex> lock(g_solverMutex);
            g_solverState = SolverState::RUNNING;
            g_solverCV.notify_one();
        }
        else
        {
            // 暂停时：求解器线程处于阻塞状态，安全区域内可直接执行CUDA操作

            // 处理暂停期间的障碍物几何变更（否则用户看不到即时反馈）
            bool geometryUpdated = false;
            if (g_shapeResetRequested.exchange(false))
            {
                // 形状/大小变化 → 完整重置流场
                solver.reset(params);
                g_dtNeedsRecompute = true;
                geometryUpdated = true;
            }
            else if (g_obstacleGeometryDirty.exchange(false))
            {
                // 位置/旋转/襟翼变化 → 热更新SDF（不重置流场）
                solver.updateObstacleGeometry(params);
                geometryUpdated = true;
            }

            if (geometryUpdated)
            {
                // 将更新后的网格类型写入PBO并提交到GL纹理
                float *ctPtr = renderer.getMappedCellTypePtr();
                if (ctPtr)
                {
                    solver.convertCellTypesToDevice(ctPtr);
                    renderer.submitCellTypes();
                }
            }

            // 更新可视化（用户可能切换显示物理量，或上方刚变更了几何）
            updateVisualization();
        }

        // ====== 并行区域：OpenGL渲染 ======
        // 此时求解器线程正在GPU上计算下一帧：
        //   - solver CUDA直接写入PBO[writeIndex]（预映射），GL从纹理（来自另一个PBO）渲染
        //   - solver CUDA直接写入VBO[writeIndex]（预映射），GL从另一个VBO绘制矢量箭头
        //   完全零拷贝，无staging中转
        renderer.render(params);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

#ifdef ENABLE_STREAMING
        if (g_streamingEnabled)
            g_streamServer.onFrameReady();
#endif

        glfwSwapBuffers(window);

        // 计算帧率
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(currentTime - lastTime).count();

        if (elapsed >= FPS_UPDATE_INTERVAL)
        {
            fps = frameCount / elapsed;
            frameCount = 0;
            lastTime = currentTime;
        }
    }

    // 停止求解器线程
    {
        std::lock_guard<std::mutex> lock(g_solverMutex);
        g_solverShouldStop = true;
    }
    g_solverCV.notify_one();
    if (g_solverThread.joinable())
        g_solverThread.join();

#ifdef ENABLE_STREAMING
    if (g_streamingEnabled)
        g_streamServer.shutdown();
#endif

    // 例行清理代码
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}