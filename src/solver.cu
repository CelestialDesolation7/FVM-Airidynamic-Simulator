#include "solver.cuh"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cub/cub.cuh>

#pragma region 常量定义
// === GPU设备常量（用于CUDA核函数内部的物理约束） ===
__device__ __constant__ float MIN_DENSITY = 1e-4f;
__device__ __constant__ float MIN_PRESSURE = 1e-2f;
__device__ __constant__ float MAX_TEMPERATURE = 5000.0f;
__device__ __constant__ float MIN_TEMPERATURE = 50.0f;
__device__ __constant__ float MAX_VELOCITY = 10000.0f;

// === 主机端编译期常量 ===
constexpr int BLOCK_SIZE = 16; // CUDA线程块尺寸（2D）

// 数值方法超参数
constexpr float ENTROPY_FIX_FACTOR = 0.1f;        // 熵修正系数：delta = factor * max(cL, cR)
constexpr float DUAL_ENERGY_ETA_THRESHOLD = 0.1f; // 双能量法切换阈值：eta < threshold 时使用内能法
constexpr float LOW_DENSITY_THRESHOLD = 0.01f;    // 低密度阈值：低于此值时降级为一阶格式
constexpr float MAX_VELOCITY_LIMIT = 5000.0f;     // 最大速度限制 [m/s]（防止非物理超高速）
constexpr float MAX_GHOST_VELOCITY = 2500.0f;     // 虚拟网格最大速度 [m/s]

// SDF几何常量
constexpr float SDF_BAND_WIDTH_FACTOR = 1.5f;       // Ghost Cell层宽度因子（相对于网格间距）
constexpr float STAR_INNER_OUTER_RATIO = 0.38f;     // 五角星内外圆半径比（标准比例）
constexpr float DIAMOND_HALF_SQRT2 = 0.7071f;       // 菱形缩放因子 ≈ 1/√2
constexpr float CAPSULE_HALF_HEIGHT = 1.5f;         // 胶囊体半高（相对于半径）
constexpr float CAPSULE_HALF_WIDTH = 0.5f;          // 胶囊体半宽（相对于半径）
constexpr float TRIANGLE_HALF_SQRT3 = 0.866025404f; // 三角形高度因子 = √3/2

// 壁面边界条件超参数
constexpr float SLIP_REFERENCE_VELOCITY = 100.0f; // 滑移因子参考速度 [m/s]
constexpr float SLIP_FACTOR_MAX = 0.8f;           // 最大滑移因子
#pragma endregion

#pragma region 数学工具函数
// 功能:计算点到线段的最短距离(用于多边形SDF)
// 输入:点坐标(px, py), 线段端点A(ax, ay)和B(bx, by)
// 输出:点到线段的欧几里得距离
__device__ __forceinline__ float distToSegment(float px, float py, float ax, float ay, float bx, float by)
{
    // 线段向量 AB = B - A
    float abx = bx - ax, aby = by - ay;
    // 点向量 AP = P - A
    float apx = px - ax, apy = py - ay;

    // 投影参数 t = (AP · AB) / |AB|^2
    // t 表示点P在线段AB上的投影位置(0表示A点，1表示B点)
    float t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-10f);

    // 将 t 限制在 [0,1] 范围内(对应线段上的点)
    t = fmaxf(0.0f, fminf(1.0f, t));

    // 线段上最近点的坐标 = A + t * AB
    float closestX = ax + t * abx;
    float closestY = ay + t * aby;

    // 计算点到最近点的距离
    float dx = px - closestX;
    float dy = py - closestY;
    return sqrtf(dx * dx + dy * dy);
}

// 功能:计算二维向量的叉积(用于判断点的相对位置和环绕数)
// 输入:两个二维向量 A(ax, ay) 和 B(bx, by)
// 输出:叉积标量值 = ax*by - ay*bx (正值表示B在A的左侧)
__device__ __forceinline__ float cross2d(float ax, float ay, float bx, float by)
{
    return ax * by - ay * bx;
}

// 功能:Minmod 限制器函数，用于MUSCL重构
// 输入:两个相邻网格的斜率 a 和 b
// 输出:限制后的斜率，确保单调性，避免数值振荡
__device__ __forceinline__ float minmod(float a, float b)
{
    // 使用无分支(Branchless)形式，依靠符号提取与内置最值函数消除Warp分化
    // 异号时 copysignf 项相加为0；同号时提出符号，乘以最值
    return 0.5f * (copysignf(1.0f, a) + copysignf(1.0f, b)) * fminf(fabsf(a), fabsf(b));
}

// 功能:Harten 熵修正，避免跨音速区域的数值不稳定(膨胀激波)
// 输入:波速 lambda 和修正阈值 delta
// 输出:修正后的波速绝对值
__device__ __forceinline__ float entropyFix(float lambda, float delta)
{
    // 当波速接近零时(亚音速区域)，使用二次插值避免除零
    if (fabsf(lambda) < delta)
    {
        // 二次熵修正公式: |lambda*| = (lambda^2 + delta^2) / (2*delta)
        // 确保波速不会完全为零，避免Riemann求解器退化
        return (lambda * lambda + delta * delta) / (2.0f * delta);
    }
    return fabsf(lambda);
}

// 功能:MUSCL重构辅助函数，计算限制后的斜率
// 输入:三个相邻网格的物理量值 qm1(i-1), q0(i), qp1(i+1)
// 输出:限制后的斜率，用于二阶精度的变量重构
__device__ __forceinline__ float musclSlope(float qm1, float q0, float qp1)
{
    // 计算左侧斜率: q(i) - q(i-1)
    float dL = q0 - qm1;
    // 计算右侧斜率: q(i+1) - q(i)
    float dR = qp1 - q0;
    // 使用minmod限制器选择更保守的斜率，避免在极值点产生振荡
    return minmod(dL, dR);
}
#pragma endregion

#pragma region 多边形SDF场计算算法
// 功能:计算点到圆形的带符号距离
// 输入:点坐标(px, py), 圆心(cx, cy), 半径 r
// 输出:带符号距离(负值表示在圆内，正值表示在圆外)
__device__ __forceinline__ float sdfCircle(float px, float py, float cx, float cy, float r)
{
    // SDF(圆) = |点到圆心距离| - 半径
    return sqrtf((px - cx) * (px - cx) + (py - cy) * (py - cy)) - r;
}

// 功能:计算点到五角星的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 外接圆半径 r, 旋转角度 rotation
// 输出:带符号距离
__device__ __forceinline__ float sdfStar(float px, float py, float cx, float cy, float r, float rotation)
{
    const int N = 5;                                 // 5个尖角
    const float outerR = r;                          // 外圆半径(尖角处)
    const float innerR = r * STAR_INNER_OUTER_RATIO; // 内圆半径(凹陷处，0.38是五角星的标准比例)

    // 转换到局部坐标系(以中心为原点)
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转(旋转矩阵的逆变换)
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float qx = lx * cosR - ly * sinR;
    float qy = lx * sinR + ly * cosR;

    // 生成五角星的10个顶点(5个外顶点 + 5个内顶点，交替排列)
    float verts[20]; // 10个顶点 * 2坐标 = 20个浮点数
    for (int i = 0; i < N; i++)
    {
        // 外顶点角度(从-90度开始，保证尖角朝右)
        float outerAngle = 2.0f * PI * i / N - PI / 2.0f;
        // 内顶点角度(在两个外顶点中间)
        float innerAngle = outerAngle + PI / N;

        // 存储外顶点坐标
        verts[i * 4 + 0] = outerR * cosf(outerAngle);
        verts[i * 4 + 1] = outerR * sinf(outerAngle);
        // 存储内顶点坐标
        verts[i * 4 + 2] = innerR * cosf(innerAngle);
        verts[i * 4 + 3] = innerR * sinf(innerAngle);
    }

    // 计算点到所有边的最短距离
    float minDist = 1e10f;
    float windingSum = 0.0f; // 环绕数累加器

    for (int i = 0; i < 2 * N; i++)
    {
        // 当前边的两个端点(循环连接)
        int j = (i + 1) % (2 * N);
        float ax = verts[i * 2], ay = verts[i * 2 + 1];
        float bx = verts[j * 2], by = verts[j * 2 + 1];

        // 更新最短距离
        float d = distToSegment(qx, qy, ax, ay, bx, by);
        minDist = fminf(minDist, d);

        // 计算环绕数(winding number)贡献
        // 环绕数通过累加每条边对应的角度变化来判断点是否在多边形内
        float eax = ax - qx, eay = ay - qy; // 边起点到查询点的向量
        float ebx = bx - qx, eby = by - qy; // 边终点到查询点的向量
        // atan2(叉积, 点积) 给出两个向量之间的夹角
        windingSum += atan2f(cross2d(eax, eay, ebx, eby), eax * ebx + eay * eby);
    }

    // 根据环绕数判断内外
    // 如果环绕数的绝对值接近2*PI或更大，说明点在多边形内
    float sign = (fabsf(windingSum) > PI) ? -1.0f : 1.0f;

    return sign * minDist;
}

// 功能:计算点到菱形(旋转正方形)的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 半边长 r, 旋转角度 rotation
// 输出:带符号距离
__device__ __forceinline__ float sdfDiamond(float px, float py, float cx, float cy, float r, float rotation)
{
    // 转换到局部坐标
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float rx = lx * cosR - ly * sinR;
    float ry = lx * sinR + ly * cosR;

    // 菱形的SDF使用L1范数(曼哈顿距离)
    // 将坐标归一化到半径
    float ndx = fabsf(rx) / r;
    float ndy = fabsf(ry) / r;

    // SDF = (|x| + |y| - 1) * r / sqrt(2)
    // 0.7071 = 1/sqrt(2)，用于将L1距离转换为欧几里得等效距离
    return (ndx + ndy - 1.0f) * r * DIAMOND_HALF_SQRT2;
}

// 功能:计算点到胶囊形(圆角矩形)的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 长度 r, 旋转角度 rotation
// 输出:带符号距离
__device__ __forceinline__ float sdfCapsule(float px, float py, float cx, float cy, float r, float rotation)
{
    // 转换到局部坐标
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float rx = lx * cosR - ly * sinR;
    float ry = lx * sinR + ly * cosR;

    // 核心思想是，胶囊形可以看作是一个半径为 capRadius 的圆，沿着一条线段“扫过”形成的区域
    // 胶囊参数: 水平方向的伸展
    float halfWidth = r * CAPSULE_HALF_HEIGHT; // 胶囊的半长(中轴长度的一半)
    float capRadius = r * CAPSULE_HALF_WIDTH;  // 两端圆弧的半径

    // 将x坐标限制到中轴线段上
    // 中轴线段范围: [-halfWidth + capRadius, halfWidth - capRadius]
    float clampedX = fmaxf(-halfWidth + capRadius, fminf(halfWidth - capRadius, rx));

    // 计算点到中轴线段最近点的距离
    float dx = rx - clampedX;
    float dy = ry;

    // SDF = 点到中轴距离 - 半径
    return sqrtf(dx * dx + dy * dy) - capRadius;
}

// 功能:计算点到三角形(等边，尖端向右)的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 外接圆半径 r, 旋转角度 rotation
// 输出:带符号距离
__device__ __forceinline__ float sdfTriangle(float px, float py, float cx, float cy, float r, float rotation)
{
    // 转换到局部坐标
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float qx = lx * cosR - ly * sinR;
    float qy = lx * sinR + ly * cosR;

    // 定义等边三角形的三个顶点
    // 顶点0(右尖): (r, 0)
    // 顶点1(左上): (-r/2, r*sqrt(3)/2)
    // 顶点2(左下): (-r/2, -r*sqrt(3)/2)
    const float sqrt3_2 = TRIANGLE_HALF_SQRT3; // sqrt(3)/2
    float v0x = r, v0y = 0.0f;
    float v1x = -r * 0.5f, v1y = r * sqrt3_2;
    float v2x = -r * 0.5f, v2y = -r * sqrt3_2;

    // 计算点到三条边的最短距离
    float d0 = distToSegment(qx, qy, v0x, v0y, v1x, v1y); // 边 v0-v1
    float d1 = distToSegment(qx, qy, v1x, v1y, v2x, v2y); // 边 v1-v2
    float d2 = distToSegment(qx, qy, v2x, v2y, v0x, v0y); // 边 v2-v0

    float minDist = fminf(d0, fminf(d1, d2));

    // 使用叉积判断点是否在三角形内
    // 对于逆时针排列的顶点，如果所有叉积同号，则点在内部
    float c0 = cross2d(v1x - v0x, v1y - v0y, qx - v0x, qy - v0y); // 边0的法向判断
    float c1 = cross2d(v2x - v1x, v2y - v1y, qx - v1x, qy - v1y); // 边1的法向判断
    float c2 = cross2d(v0x - v2x, v0y - v2y, qx - v2x, qy - v2y); // 边2的法向判断

    // 如果所有叉积都非负或都非正，说明点在三角形内
    bool inside = (c0 >= 0 && c1 >= 0 && c2 >= 0) || (c0 <= 0 && c1 <= 0 && c2 <= 0);

    // 内部返回负距离，外部返回正距离
    return inside ? -minDist : minDist;
}

// 功能:计算点到凸四边形的带符号距离
// 输入:点坐标(px, py), 四个顶点坐标（要求逆时针绕序）
// 输出:带符号距离（负值表示在四边形内，正值表示在外）
__device__ __forceinline__ float sdfQuad(float px, float py,
                                         float v0x, float v0y,
                                         float v1x, float v1y,
                                         float v2x, float v2y,
                                         float v3x, float v3y)
{
    // 计算点到四条边的最短距离
    float d0 = distToSegment(px, py, v0x, v0y, v1x, v1y);
    float d1 = distToSegment(px, py, v1x, v1y, v2x, v2y);
    float d2 = distToSegment(px, py, v2x, v2y, v3x, v3y);
    float d3 = distToSegment(px, py, v3x, v3y, v0x, v0y);

    float minDist = fminf(fminf(d0, d1), fminf(d2, d3));

    // 使用叉积判断点是否在凸四边形内部
    // 对于逆时针排列的顶点，如果所有叉积同号，则点在内部
    float c0 = cross2d(v1x - v0x, v1y - v0y, px - v0x, py - v0y);
    float c1 = cross2d(v2x - v1x, v2y - v1y, px - v1x, py - v1y);
    float c2 = cross2d(v3x - v2x, v3y - v2y, px - v2x, py - v2y);
    float c3 = cross2d(v0x - v3x, v0y - v3y, px - v3x, py - v3y);

    bool inside = (c0 >= 0 && c1 >= 0 && c2 >= 0 && c3 >= 0) ||
                  (c0 <= 0 && c1 <= 0 && c2 <= 0 && c3 <= 0);

    return inside ? -minDist : minDist;
}

// 功能:计算点到星舰模型的带符号距离
// 星舰由三部分组成：圆形主体 + 上下对称的固定襟翼根部 + 上下对称的可旋转襟翼
// 输入:点坐标(px, py), 中心(cx, cy), 主体半径 r, 全局旋转角度 rotation, 襟翼旋转角度 wingRotation
// 输出:带符号距离（联合体的SDF = 各部件SDF的最小值）
__device__ __forceinline__ float sdfStarship(float px, float py, float cx, float cy, float r, float rotation, float wingRotation)
{
    // 转换到局部坐标（以星舰中心为原点）
    float lx = px - cx;
    float ly = py - cy;

    // 应用全局旋转（将查询点旋转到星舰的参考坐标系）
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float qx = lx * cosR - ly * sinR;
    float qy = lx * sinR + ly * cosR;

    // ===== 第一部分：圆形主体的SDF =====
    float distToCircle = sqrtf(qx * qx + qy * qy) - r;

    // ===== 第二部分：襟翼SDF =====
    // 利用上下对称性：取|qy|将查询点镜像到上半平面，只需计算上襟翼
    // 上襟翼顺时针旋转 ↔ 镜像后的下襟翼向外偏转，对称性天然成立
    float scale = r / wingRootRadius; // 参考坐标系（wingRootRadius=90）到实际坐标系的缩放因子

    // 将查询点从实际坐标转换到参考坐标系
    float refX = qx / scale;
    float refY = fabsf(qy) / scale;

    // --- 固定襟翼根部（四边形，两个底点固定在圆周上）---
    // 顶点逆时针绕序：LB -> RB -> RT -> LT
    float distToRoot = sdfQuad(refX, refY,
                               wingRootLB.x, wingRootLB.y,
                               wingRootRB.x, wingRootRB.y,
                               wingRootRT.x, wingRootRT.y,
                               wingRootLT.x, wingRootLT.y) *
                       scale;

    // --- 可动襟翼（等腰梯形，绕wingLB顺时针旋转wingRotation弧度）---
    // 旋转中心 = wingLB = wingRootLT = (-6, 112)
    float pivotX = wingLB.x;
    float pivotY = wingLB.y;
    float cosW = cosf(-wingRotation); // 顺时针旋转 = 数学上的负角度
    float sinW = sinf(-wingRotation);

    // wingLB 是旋转中心，坐标不变
    float wv0x = pivotX;
    float wv0y = pivotY;

    // wingRB 绕旋转中心旋转
    float dx1 = wingRB.x - pivotX; // = 12
    float dy1 = wingRB.y - pivotY; // = 0
    float wv1x = pivotX + dx1 * cosW - dy1 * sinW;
    float wv1y = pivotY + dx1 * sinW + dy1 * cosW;

    // wingRT 绕旋转中心旋转
    float dx2 = wingRT.x - pivotX; // = 9
    float dy2 = wingRT.y - pivotY; // = 74
    float wv2x = pivotX + dx2 * cosW - dy2 * sinW;
    float wv2y = pivotY + dx2 * sinW + dy2 * cosW;

    // wingLT 绕旋转中心旋转
    float dx3 = wingLT.x - pivotX; // = 3
    float dy3 = wingLT.y - pivotY; // = 74
    float wv3x = pivotX + dx3 * cosW - dy3 * sinW;
    float wv3y = pivotY + dx3 * sinW + dy3 * cosW;

    // 顶点逆时针绕序：LB -> RB -> RT -> LT
    float distToWing = sdfQuad(refX, refY,
                               wv0x, wv0y,
                               wv1x, wv1y,
                               wv2x, wv2y,
                               wv3x, wv3y) *
                       scale;

    // ===== 取联合体SDF：各部件SDF的最小值 =====
    return fminf(distToCircle, fminf(distToRoot, distToWing));
}

// 功能:统一的SDF计算接口，根据形状类型调用相应的SDF函数
// 输入:点坐标(px, py), 形状中心(cx, cy), 尺寸参数 r, 旋转角度 rotation, 形状类型 shapeType
// 输出:带符号距离
__device__ __forceinline__ float computeShapeSDF(float px, float py, float cx, float cy, float r,
                                                 float rotation, ObstacleShape shapeType, float wingRotation)
{
    // 根据形状类型分发到具体的SDF函数
    switch (shapeType)
    {
    case ObstacleShape::CIRCLE: // 圆形
        return sdfCircle(px, py, cx, cy, r);
    case ObstacleShape::STAR: // 五角星
        return sdfStar(px, py, cx, cy, r, rotation);
    case ObstacleShape::DIAMOND: // 菱形
        return sdfDiamond(px, py, cx, cy, r, rotation);
    case ObstacleShape::CAPSULE: // 胶囊形
        return sdfCapsule(px, py, cx, cy, r, rotation);
    case ObstacleShape::TRIANGLE: // 三角形
        return sdfTriangle(px, py, cx, cy, r, rotation);
    case ObstacleShape::STARSHIP: // 星舰
        return sdfStarship(px, py, cx, cy, r, rotation, wingRotation);
    default: // 默认返回圆形
        return sdfCircle(px, py, cx, cy, r);
    }
}

// 功能:并行计算全场的SDF并初始化网格类型
// 输入:障碍物参数(位置、大小、旋转、形状), 网格间距 dx,dy, 网格尺寸 nx,ny
// 输出:SDF场 sdf[], 网格类型标记 cell_type[]
__global__ void computeSDFKernel(float *sdf, uint8_t *cell_type,
                                 float obs_x, float obs_y, float obs_r,
                                 float rotation, ObstacleShape shapeType,
                                 float dx, float dy, int nx, int ny, float wingRotation)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // 计算网格单元中心的物理坐标
    float x = (i + 0.5f) * dx;
    float y = (j + 0.5f) * dy;

    // 根据形状类型计算带符号距离
    float dist = computeShapeSDF(x, y, obs_x, obs_y, obs_r, rotation, shapeType, wingRotation);

    sdf[idx] = dist;

    // 根据距离分类网格单元
    // band 是边界层的宽度，通常取1-2个网格间距
    float band = SDF_BAND_WIDTH_FACTOR * fmaxf(dx, dy);

    if (dist < -band)
    {
        // 深入固体内部 -> 固体单元
        cell_type[idx] = CELL_SOLID;
    }
    else if (dist < band && dist >= -band)
    {
        // 距离边界很近 -> 虚拟单元(Ghost Cell，用于边界条件处理)
        cell_type[idx] = CELL_GHOST;
    }
    else
    {
        // 远离边界 -> 流体单元
        cell_type[idx] = CELL_FLUID;
    }

    // 计算域边界条件覆盖
    if (i == 0)
    {
        // 左边界 -> 流入边界
        cell_type[idx] = CELL_INFLOW;
    }
    else if (i == nx - 1)
    {
        // 右边界 -> 流出边界
        cell_type[idx] = CELL_OUTFLOW;
    }
}

// 功能:重新计算SDF和网格类型（用于动态襟翼旋转），并对类型变化的网格做物理修正
// 与 computeSDFKernel 不同，此核函数读取旧的 cell_type，对比新类型：
// - 固体/Ghost → 流体：用来流条件初始化守恒变量（新暴露的流体区域）
// - 流体 → 固体/Ghost：保持守恒变量不变（边界条件内核会在之后正确处理）
// 输入:障碍物参数, 来流参数(rho_inf, u_inf, v_inf, p_inf), 守恒变量指针
// 输出:更新后的SDF、cell_type，以及新暴露区域的守恒变量
__global__ void updateSDFWithFixupKernel(
    float *sdf, uint8_t *cell_type,
    float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
    float obs_x, float obs_y, float obs_r,
    float rotation, ObstacleShape shapeType,
    float dx, float dy, int nx, int ny, float wingRotation,
    float rho_inf, float u_inf, float v_inf, float p_inf)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // 保存旧的网格类型
    uint8_t oldType = cell_type[idx];

    // 计算新的SDF
    float x = (i + 0.5f) * dx;
    float y = (j + 0.5f) * dy;
    float dist = computeShapeSDF(x, y, obs_x, obs_y, obs_r, rotation, shapeType, wingRotation);
    sdf[idx] = dist;

    // 计算新的网格类型
    float band = SDF_BAND_WIDTH_FACTOR * fmaxf(dx, dy);
    uint8_t newType;

    if (dist < -band)
        newType = CELL_SOLID;
    else if (dist < band && dist >= -band)
        newType = CELL_GHOST;
    else
        newType = CELL_FLUID;

    // 计算域边界条件覆盖
    if (i == 0)
        newType = CELL_INFLOW;
    else if (i == nx - 1)
        newType = CELL_OUTFLOW;

    cell_type[idx] = newType;

    // 物理修正：如果网格从固体/Ghost变为流体，用来流条件填充
    // 这样做的物理意义是：襟翼收起后，暴露的区域被新鲜来流填充
    bool wasObstacle = (oldType == CELL_SOLID || oldType == CELL_GHOST);
    bool nowFluid = (newType == CELL_FLUID);

    if (wasObstacle && nowFluid)
    {
        rho[idx] = 0;
        rho_u[idx] = 0;
        rho_v[idx] = 0;
        E[idx] = 0;
        rho_e[idx] = 0;
    }
}
#pragma endregion

#pragma region 物理逻辑操作核函数
#pragma region 内联函数
// 功能: 计算x方向的Riemann不变量（用于左右边界）
// 输入: 原始变量(rho, u, v, p)
// 输出: 四个Riemann不变量(R1, R2, R3, R4)
__device__ __forceinline__ void computeRiemannInvarFromPrimitiveX(float rho, float u, float v, float p,
                                                                  float &R1, float &R2, float &R3, float &R4)
{
    // 声速
    float c = sqrtf(GAMMA * p / rho);
    // Riemann不变量（X方向）
    // 优化：使用 exp(-GAMMA * log(rho)) 代替 pow(rho, -GAMMA)
    float log_rho = logf(rho);
    R1 = expf(-GAMMA * log_rho) * p; // 熵波（特征速度 = u）
    R2 = v;                          // 剪切波（特征速度 = u）

    // 预计算常数以减少除法
    constexpr float two_over_gamma_minus_1 = 2.0f / (GAMMA - 1.0f); // 对于 GAMMA=1.4，这是 5.0
    R3 = u + c * two_over_gamma_minus_1;                            // 向右声波（特征速度 = u + c）
    R4 = u - c * two_over_gamma_minus_1;                            // 向左声波（特征速度 = u - c）
}

// 功能: 计算y方向的Riemann不变量（用于上下边界）
// 输入: 原始变量(rho, u, v, p)
// 输出: 四个Riemann不变量(R1, R2, R3, R4)
__device__ __forceinline__ void computeRiemannInvarFromPrimitiveY(float rho, float u, float v, float p,
                                                                  float &R1, float &R2, float &R3, float &R4)
{
    // 声速
    float c = sqrtf(GAMMA * p / rho);
    // Riemann不变量
    // 优化：使用 exp(-GAMMA * log(rho)) 代替 pow(rho, -GAMMA)，速度提升约20%
    float log_rho = logf(rho);
    R1 = expf(-GAMMA * log_rho) * p; // 熵波
    R2 = u;                          // 剪切波

    // 预计算常数以减少除法
    constexpr float two_over_gamma_minus_1 = 2.0f / (GAMMA - 1.0f); // 对于 GAMMA=1.4，这是 5.0
    R3 = v + c * two_over_gamma_minus_1;                            // 向上声波
    R4 = v - c * two_over_gamma_minus_1;                            // 向下声波
}

// 功能: 从X方向Riemann不变量重构原始变量
// 输入: Riemann不变量(R1, R2, R3, R4)
// 输出: 原始变量(rho, u, v, p)
__device__ __forceinline__ void computePrimitiveFromRiemannInvarX(float R1, float R2, float R3, float R4,
                                                                  float &rho, float &u, float &v, float &p)
{
    v = R2;               // 剪切波不变量直接给出垂直速度
    u = (R3 + R4) * 0.5f; // 水平速度（优化：乘法比除法快）

    // 预计算常数
    constexpr float gamma_minus_1_over_4 = (GAMMA - 1.0f) * 0.25f; // 0.1
    constexpr float inv_gamma_minus_1 = 1.0f / (GAMMA - 1.0f);     // 2.5 (对于 GAMMA=1.4)
    constexpr float inv_gamma = 1.0f / GAMMA;                      // 0.714286

    float c = (R3 - R4) * gamma_minus_1_over_4; // 本地声速
    float c_sq = c * c;                         // 声速平方

    // 优化：使用 exp 和 log 代替 pow
    // rho = ((c^2) / (GAMMA * R1))^(1/(GAMMA-1))
    float log_arg = c_sq * inv_gamma / R1;
    rho = expf(inv_gamma_minus_1 * logf(log_arg));

    p = rho * c_sq * inv_gamma; // 压强
}

// 功能: 从Y方向Riemann不变量重构原始变量
// 输入: Riemann不变量(R1, R2, R3, R4)
// 输出: 原始变量(rho, u, v, p)
__device__ __forceinline__ void computePrimitiveFromRiemannInvarY(float R1, float R2, float R3, float R4,
                                                                  float &rho, float &u, float &v, float &p)
{
    u = R2;               // 剪切波不变量直接给出横向速度
    v = (R3 + R4) * 0.5f; // 速度（优化：乘法比除法快）

    // 预计算常数
    constexpr float gamma_minus_1_over_4 = (GAMMA - 1.0f) * 0.25f; // 0.1
    constexpr float inv_gamma_minus_1 = 1.0f / (GAMMA - 1.0f);     // 2.5 (对于 GAMMA=1.4)
    constexpr float inv_gamma = 1.0f / GAMMA;                      // 0.714286

    float c = (R3 - R4) * gamma_minus_1_over_4; // 本地声速
    float c_sq = c * c;                         // 声速平方

    // 优化：使用 exp 和 log 代替 pow
    // rho = ((c^2) / (GAMMA * R1))^(1/(GAMMA-1))
    float log_arg = c_sq * inv_gamma / R1;
    rho = expf(inv_gamma_minus_1 * logf(log_arg));

    p = rho * c_sq * inv_gamma; // 压强
}

// 功能: 计算远场条件对应的Riemann不变量（X方向边界条件使用）
// 输入:来流参数(rho_inf, u_inf, v_inf, p_inf)
// 输出:来流对应的Riemann不变量(R1_inf, R2_inf, R3_inf, R4_inf)
__device__ __forceinline__ void computeFarfieldRiemannInvarX(float rho_inf, float u_inf, float v_inf, float p_inf,
                                                             float &R1_inf, float &R2_inf, float &R3_inf, float &R4_inf)
{
    // 计算来流的声速
    float c_inf = sqrtf(GAMMA * p_inf / rho_inf);

    // 优化：使用 exp(-GAMMA * log(rho)) 代替 pow(rho, -GAMMA)
    float log_rho_inf = logf(rho_inf);
    R1_inf = expf(-GAMMA * log_rho_inf) * p_inf; // 熵波不变量

    R2_inf = v_inf; // 剪切波不变量（X方向是v）

    // 预计算常数
    constexpr float two_over_gamma_minus_1 = 2.0f / (GAMMA - 1.0f);
    R3_inf = u_inf + c_inf * two_over_gamma_minus_1; // 向右声波不变量
    R4_inf = u_inf - c_inf * two_over_gamma_minus_1; // 向左声波不变量
}

// 功能: 计算远场条件对应的Riemann不变量（Y方向边界条件使用）
// 输入:来流参数(rho_inf, u_inf, v_inf, p_inf)
// 输出:来流对应的Riemann不变量(R1_inf, R2_inf, R3_inf, R4_inf)
__device__ __forceinline__ void computeFarfieldRiemannInvarY(float rho_inf, float u_inf, float v_inf, float p_inf,
                                                             float &R1_inf, float &R2_inf, float &R3_inf, float &R4_inf)
{
    // 计算来流的声速
    float c_inf = sqrtf(GAMMA * p_inf / rho_inf);

    // 优化：使用 exp(-GAMMA * log(rho)) 代替 pow(rho, -GAMMA)
    float log_rho_inf = logf(rho_inf);
    R1_inf = expf(-GAMMA * log_rho_inf) * p_inf; // 熵波不变量

    R2_inf = u_inf; // 剪切波不变量

    // 预计算常数
    constexpr float two_over_gamma_minus_1 = 2.0f / (GAMMA - 1.0f);
    R3_inf = v_inf + c_inf * two_over_gamma_minus_1; // 向上声波不变量
    R4_inf = v_inf - c_inf * two_over_gamma_minus_1; // 向下声波不变量
}

// 功能: 从守恒变量计算原始变量，并应用双能量法切换以提高数值稳定性
// 输入:守恒变量(rho, rho_u, rho_v, E, rho_e)，其中 rho_e 是从双能量方程追踪的内能密度
// 输出:原始变量(u, v, p, T)，其中 T 是温度
__device__ __forceinline__ void computePrimitivesKernelInline(const float rho, const float rho_u,
                                                              const float rho_v, const float E,
                                                              const float rho_e, // Internal energy from dual-energy equation
                                                              float &u, float &v, float &p, float &T)
{
    // 读取守恒变量并应用物理下限
    float rho_val = fmaxf(rho, MIN_DENSITY);
    // 从动量密度计算速度: u = (rho*u) / rho
    float u_val = rho_u / rho_val;
    float v_val = rho_v / rho_val;
    // 计算动能密度: KE = 0.5 * rho * (u^2 + v^2)
    float ke = 0.5f * rho_val * (u_val * u_val + v_val * v_val);

    // ========== 双能量法 ==========
    // 方法1(E法): 从总能量计算内能
    // e_internal = E_total - KE
    float e_from_E = E - ke;
    // 由内能计算压强: p = (gamma - 1) * e_internal
    float p_from_E = (GAMMA - 1.0f) * e_from_E;

    // 方法2(e法): 直接使用追踪的内能密度
    float e_from_e = rho_e;
    float p_from_e = (GAMMA - 1.0f) * e_from_e;

    // 切换准则: eta = e_internal / E_total
    // eta 是内能占总能量的比例
    // 当 eta 很小时，动能占主导，E-KE的减法不可靠
    float eta = e_from_e / (E + 1e-20f);

    // 切换阈值(通常取 0.001 到 0.1)
    // 低于此值时，说明动能远大于内能，应使用e法
    float eta_switch = 0.1f;

    float p_val;
    if (eta > eta_switch && e_from_E > 0.0f)
    {
        // 情况1: 内能占比足够大(激波区、高温区)
        // 使用 E 法保证守恒性，准确捕捉激波
        p_val = p_from_E;
    }
    else
    {
        // 情况2: 动能占主导(膨胀区、低密度尾流)
        // 使用 e 法避免精度损失
        p_val = p_from_e;
    }

    // 最终安全限制
    p_val = fmaxf(p_val, MIN_PRESSURE);

    // 由理想气体状态方程计算温度: T = p / (rho * R)
    float T_val = p_val / (rho_val * R_GAS);
    T_val = fminf(T_val, MAX_TEMPERATURE);
    T_val = fmaxf(T_val, MIN_TEMPERATURE);

    // 写入原始变量
    u = u_val;
    v = v_val;
    p = p_val;
    T = T_val;
}
#pragma endregion

#pragma region 无条件初始化
// 功能:初始化流场为来流(自由流)条件
// 输入:来流密度 rho_inf, 速度 u_inf, v_inf, 压强 p_inf, 网格尺寸 nx, ny
// 输出:初始化的守恒变量数组 rho, rho_u, rho_v, E, rho_e
__global__ void initializeKernel(float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
                                 float rho_inf, float u_inf, float v_inf, float p_inf,
                                 int nx, int ny)
{
    // 计算当前线程对应的网格索引
    int i = blockIdx.x * blockDim.x + threadIdx.x; // X方向索引
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Y方向索引

    // 边界检查，防止越界访问
    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i; // 展平的一维索引

    // 设置来流密度
    rho[idx] = rho_inf;
    // 设置来流X动量 = 密度 * X速度
    rho_u[idx] = rho_inf * u_inf;
    // 设置来流Y动量 = 密度 * Y速度
    rho_v[idx] = rho_inf * v_inf;

    // 计算动能密度: KE = 0.5 * rho * (u^2 + v^2)
    float ke = 0.5f * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    // 计算内能密度(理想气体): e_internal = p / (gamma - 1)
    float e_internal = p_inf / (GAMMA - 1.0f);
    // 总能量密度 = 内能密度 + 动能密度
    E[idx] = e_internal + ke;
    // 单独存储内能密度用于双能量法
    rho_e[idx] = e_internal;
}
#pragma endregion

#pragma region 无粘性逻辑

// 功能:从守恒变量计算原始变量，使用双能量法避免数值精度问题
// 输入:守恒变量 rho, rho_u, rho_v, E(总能量), rho_e(内能密度), 网格尺寸 nx, ny
// 输出:原始变量 u, v, p, T
// 说明:双能量法在动能占主导时使用单独追踪的内能，避免 E-KE 的大数减小数精度损失
__global__ void computePrimitivesKernel(const float *rho, const float *rho_u,
                                        const float *rho_v, const float *E,
                                        const float *rho_e, // Internal energy from dual-energy equation
                                        float *u, float *v, float *p, float *T,
                                        int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;
    int idx = j * nx + i;

    computePrimitivesKernelInline(rho[idx], rho_u[idx], rho_v[idx], E[idx], rho_e[idx], u[idx], v[idx], p[idx], T[idx]);
}

// 功能:从原始变量计算 X 方向(水平)的通量向量 F
// 输入:密度 rho, 速度分量 u,v, 压强 p, 总能量密度 E
// 输出:X方向通量的四个分量(质量、X动量、Y动量、能量通量)
__device__ __forceinline__ void computeFluxX(float rho, float u, float v, float p, float E,
                                             float &f_rho, float &f_rho_u, float &f_rho_v, float &f_E)
{
    // 欧拉方程的X方向通量:
    // F = [rho*u, rho*u^2 + p, rho*u*v, (E+p)*u]^T
    f_rho = rho * u;           // 质量通量 = 密度 * X速度
    f_rho_u = rho * u * u + p; // X动量通量 = 动量密度*u + 压力
    f_rho_v = rho * u * v;     // Y动量通量 = Y动量密度 * X速度
    f_E = (E + p) * u;         // 能量通量 = (总能量+压强功) * X速度
}

// 功能:从原始变量计算 Y 方向(垂直)的通量向量 G
// 输入:密度 rho, 速度分量 u,v, 压强 p, 总能量密度 E
// 输出:Y方向通量的四个分量(质量、X动量、Y动量、能量通量)
__device__ __forceinline__ void computeFluxY(float rho, float u, float v, float p, float E,
                                             float &g_rho, float &g_rho_u, float &g_rho_v, float &g_E)
{
    // 欧拉方程的Y方向通量:
    // G = [rho*v, rho*u*v, rho*v^2 + p, (E+p)*v]^T
    g_rho = rho * v;           // 质量通量 = 密度 * Y速度
    g_rho_u = rho * u * v;     // X动量通量 = X动量密度 * Y速度
    g_rho_v = rho * v * v + p; // Y动量通量 = 动量密度*v + 压力
    g_E = (E + p) * v;         // 能量通量 = (总能量+压强功) * Y速度
}

// 功能:HLL (Harten-Lax-van Leer) Riemann 求解器(X方向)，比HLLC更稳健但稍微扩散
// 输入:界面左侧状态(rhoL, uL, vL, pL, EL)和右侧状态(rhoR, uR, vR, pR, ER)
// 输出:界面处的数值通量(f_rho, f_rho_u, f_rho_v, f_E)
__device__ __forceinline__ void hllFluxX(
    float rhoL, float uL, float vL, float pL, float EL,
    float rhoR, float uR, float vR, float pR, float ER,
    float &f_rho, float &f_rho_u, float &f_rho_v, float &f_E)
{
    // 计算左右状态的声速: c = sqrt(gamma * p / rho)
    float cL = sqrtf(GAMMA * pL / (rhoL + 1e-10f));
    float cR = sqrtf(GAMMA * pR / (rhoR + 1e-10f));

    // 估算最快和最慢的波速(Davis估计)
    // SL: 向左传播的最快波(负速度方向)
    // SR: 向右传播的最快波(正速度方向)
    float SL = fminf(uL - cL, uR - cR); // 取左右两侧的最小值
    float SR = fmaxf(uL + cL, uR + cR); // 取左右两侧的最大值

    // 熵修正：避免跨音速区域的膨胀激波问题
    float delta = ENTROPY_FIX_FACTOR * fmaxf(cL, cR); // 修正阈值取决于声速
    SL = -entropyFix(-SL, delta);                     // 对负波速应用修正
    SR = entropyFix(SR, delta);                       // 对正波速应用修正

    // 计算左右两侧的物理通量
    float fL_rho, fL_rho_u, fL_rho_v, fL_E;
    float fR_rho, fR_rho_u, fR_rho_v, fR_E;
    computeFluxX(rhoL, uL, vL, pL, EL, fL_rho, fL_rho_u, fL_rho_v, fL_E);
    computeFluxX(rhoR, uR, vR, pR, ER, fR_rho, fR_rho_u, fR_rho_v, fR_E);

    // HLL 通量公式(三种情况):
    if (SL >= 0.0f)
    {
        // 情况1: 全部波向右传播(超音速右行) -> 使用左状态通量
        f_rho = fL_rho;
        f_rho_u = fL_rho_u;
        f_rho_v = fL_rho_v;
        f_E = fL_E;
    }
    else if (SR <= 0.0f)
    {
        // 情况2: 全部波向左传播(超音速左行) -> 使用右状态通量
        f_rho = fR_rho;
        f_rho_u = fR_rho_u;
        f_rho_v = fR_rho_v;
        f_E = fR_E;
    }
    else
    {
        // 情况3: 波在两侧传播(跨音速或亚音速) -> 使用加权平均
        // HLL 公式: F* = (SR*FL - SL*FR + SL*SR*(UR - UL)) / (SR - SL)
        float denom = SR - SL + 1e-10f; // 避免除零
        f_rho = (SR * fL_rho - SL * fR_rho + SL * SR * (rhoR - rhoL)) / denom;
        f_rho_u = (SR * fL_rho_u - SL * fR_rho_u + SL * SR * (rhoR * uR - rhoL * uL)) / denom;
        f_rho_v = (SR * fL_rho_v - SL * fR_rho_v + SL * SR * (rhoR * vR - rhoL * vL)) / denom;
        f_E = (SR * fL_E - SL * fR_E + SL * SR * (ER - EL)) / denom;
    }
}

// 功能:HLLC (HLL-Contact) Riemann 求解器(X方向)，考虑接触间断，比HLL更精确
// 输入:界面左侧状态(rhoL, uL, vL, pL, EL)和右侧状态(rhoR, uR, vR, pR, ER)
// 输出:界面处的数值通量(f_rho, f_rho_u, f_rho_v, f_E)
__device__ __forceinline__ void hllcFluxX(
    float rhoL, float uL, float vL, float pL, float EL,
    float rhoR, float uR, float vR, float pR, float ER,
    float &f_rho, float &f_rho_u, float &f_rho_v, float &f_E)
{
    // 强制密度和压强满足物理约束，防止负值
    rhoL = fmaxf(rhoL, MIN_DENSITY);
    rhoR = fmaxf(rhoR, MIN_DENSITY);
    pL = fmaxf(pL, MIN_PRESSURE);
    pR = fmaxf(pR, MIN_PRESSURE);

    // 计算左右状态的声速
    float cL = sqrtf(GAMMA * pL / rhoL);
    float cR = sqrtf(GAMMA * pR / rhoR);

    // Roe 平均(用于更准确的波速估计)
    // Roe平均密度权重
    float sqrtRhoL = sqrtf(rhoL);
    float sqrtRhoR = sqrtf(rhoR);
    float denom = sqrtRhoL + sqrtRhoR + 1e-10f;

    // Roe平均速度(两个分量都需要)
    float u_roe = (sqrtRhoL * uL + sqrtRhoR * uR) / denom;
    float v_roe = (sqrtRhoL * vL + sqrtRhoR * vR) / denom;

    // Roe平均比焓: H = (E + p) / rho
    float H_L = (EL + pL) / rhoL;
    float H_R = (ER + pR) / rhoR;
    float H_roe = (sqrtRhoL * H_L + sqrtRhoR * H_R) / denom;

    // 从Roe平均量计算声速: c^2 = (gamma-1) * (H - 0.5*|v|^2)
    // 注意: 必须减去完整的动能(u^2+v^2)，否则声速被高估导致多余数值耗散
    float c_roe_sq = (GAMMA - 1.0f) * (H_roe - 0.5f * (u_roe * u_roe + v_roe * v_roe));
    float c_roe = (c_roe_sq > 0.0f) ? sqrtf(c_roe_sq) : 0.5f * (cL + cR);

    // Davis 波速估计(结合左右和Roe平均)
    float SL = fminf(uL - cL, u_roe - c_roe);
    float SR = fmaxf(uR + cR, u_roe + c_roe);

    // 熵修正，防止膨胀激波
    float delta = ENTROPY_FIX_FACTOR * fmaxf(cL, cR);
    if (fabsf(SL) < delta)
        SL = -delta;
    if (fabsf(SR) < delta)
        SR = delta;

    // 计算接触波速度 S*
    // S* = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR)) / (rhoL*(SL-uL) - rhoR*(SR-uR))
    float denom_star = rhoL * (SL - uL) - rhoR * (SR - uR);
    // 强制规避除零问题，避免分支退化执行HLL
    float safe_denom_star = copysignf(fmaxf(fabsf(denom_star), 1e-10f), denom_star);
    float S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / safe_denom_star;

    // HLLC 通量公式(四种情况，比HLL多考虑接触间断):
    if (SL >= 0.0f)
    {
        // 情况1: 所有波向右 -> 使用左状态通量
        computeFluxX(rhoL, uL, vL, pL, EL, f_rho, f_rho_u, f_rho_v, f_E);
    }
    else if (SR <= 0.0f)
    {
        // 情况2: 所有波向左 -> 使用右状态通量
        computeFluxX(rhoR, uR, vR, pR, ER, f_rho, f_rho_u, f_rho_v, f_E);
    }
    else if (S_star >= 0.0f)
    {
        // 情况3: 接触波在界面右侧 -> 使用左星区状态
        // 计算左星区(L*)状态
        float coeff_denom = SL - S_star;
        if (fabsf(coeff_denom) < 1e-10f)
            coeff_denom = (coeff_denom >= 0) ? 1e-10f : -1e-10f;

        float rho_star = rhoL * (SL - uL) / coeff_denom;                        // 星区密度: rho* = rho * (S - u) / (S - S*)
        rho_star = fmaxf(rho_star, MIN_DENSITY);                                // 确保密度不为负，避免数值不稳定
        float rhoU_star = rho_star * S_star;                                    // 星区X动量
        float rhoV_star = rho_star * vL;                                        // 星区Y动量
        float pLR = pL + rhoL * (SL - uL) * (S_star - uL);                      // 星区压强: p* = pL + rhoL*(SL-uL)*(S*-uL)
        pLR = fmaxf(pLR, MIN_PRESSURE);                                         // 确保压强不为负，避免数值不稳定
        float E_star = (EL * (SL - uL) + pLR * S_star - pL * uL) / coeff_denom; // 星区总能量密度

        // 计算左通量
        float fL_rho, fL_rho_u, fL_rho_v, fL_E;
        computeFluxX(rhoL, uL, vL, pL, EL, fL_rho, fL_rho_u, fL_rho_v, fL_E);

        // HLLC修正: F* = FL + SL*(U* - UL)
        f_rho = fL_rho + SL * (rho_star - rhoL);
        f_rho_u = fL_rho_u + SL * (rhoU_star - rhoL * uL);
        f_rho_v = fL_rho_v + SL * (rhoV_star - rhoL * vL);
        f_E = fL_E + SL * (E_star - EL);
    }
    else
    {
        // 情况4: 接触波在界面左侧 -> 使用右星区状态
        // 计算右星区(R*)状态(类似左星区但使用右侧数据)
        float coeff_denom = SR - S_star;
        if (fabsf(coeff_denom) < 1e-10f)
            coeff_denom = (coeff_denom >= 0) ? 1e-10f : -1e-10f;
        float coeff = rhoR * (SR - uR) / coeff_denom;
        coeff = fmaxf(coeff, MIN_DENSITY);

        float rho_star = coeff;
        float rhoU_star = coeff * S_star;
        float rhoV_star = coeff * vR;

        float pLR = pR + rhoR * (SR - uR) * (S_star - uR);
        pLR = fmaxf(pLR, MIN_PRESSURE);
        float E_star = (ER * (SR - uR) + pLR * S_star - pR * uR) / coeff_denom;

        float fR_rho, fR_rho_u, fR_rho_v, fR_E;
        computeFluxX(rhoR, uR, vR, pR, ER, fR_rho, fR_rho_u, fR_rho_v, fR_E);

        // HLLC修正: F* = FR + SR*(U* - UR)
        f_rho = fR_rho + SR * (rho_star - rhoR);
        f_rho_u = fR_rho_u + SR * (rhoU_star - rhoR * uR);
        f_rho_v = fR_rho_v + SR * (rhoV_star - rhoR * vR);
        f_E = fR_E + SR * (E_star - ER);
    }
}

// 功能:HLL Riemann 求解器(Y方向)，适用于垂直方向的通量计算
// 输入:界面下方状态(rhoB, uB, vB, pB, EB)和上方状态(rhoT, uT, vT, pT, ET)
// 输出:界面处的Y方向数值通量(g_rho, g_rho_u, g_rho_v, g_E)
__device__ __forceinline__ void hllFluxY(
    float rhoB, float uB, float vB, float pB, float EB,
    float rhoT, float uT, float vT, float pT, float ET,
    float &g_rho, float &g_rho_u, float &g_rho_v, float &g_E)
{
    // 强制物理约束
    rhoB = fmaxf(rhoB, MIN_DENSITY);
    rhoT = fmaxf(rhoT, MIN_DENSITY);
    pB = fmaxf(pB, MIN_PRESSURE);
    pT = fmaxf(pT, MIN_PRESSURE);

    // 计算声速
    float cB = sqrtf(GAMMA * pB / rhoB);
    float cT = sqrtf(GAMMA * pT / rhoT);

    // Roe 平均(Y方向，主方向用v，但两个分量都需要)
    float sqrtRhoB = sqrtf(rhoB);
    float sqrtRhoT = sqrtf(rhoT);
    float denom = sqrtRhoB + sqrtRhoT + 1e-10f;

    float u_roe = (sqrtRhoB * uB + sqrtRhoT * uT) / denom;
    float v_roe = (sqrtRhoB * vB + sqrtRhoT * vT) / denom;
    float H_B = (EB + pB) / rhoB;
    float H_T = (ET + pT) / rhoT;
    float H_roe = (sqrtRhoB * H_B + sqrtRhoT * H_T) / denom;

    // 注意: 必须减去完整的动能(u^2+v^2)，否则声速被高估导致多余数值耗散
    float c_roe_sq = (GAMMA - 1.0f) * (H_roe - 0.5f * (u_roe * u_roe + v_roe * v_roe));
    float c_roe = (c_roe_sq > 0.0f) ? sqrtf(c_roe_sq) : 0.5f * (cB + cT);

    // 波速估计
    float SB = fminf(vB - cB, v_roe - c_roe);
    float ST = fmaxf(vT + cT, v_roe + c_roe);

    // 熵修正
    float delta = ENTROPY_FIX_FACTOR * fmaxf(cB, cT);
    SB = -entropyFix(-SB, delta);
    ST = entropyFix(ST, delta);

    // 计算下方和上方的Y方向物理通量
    float gB_rho, gB_rho_u, gB_rho_v, gB_E;
    float gT_rho, gT_rho_u, gT_rho_v, gT_E;
    computeFluxY(rhoB, uB, vB, pB, EB, gB_rho, gB_rho_u, gB_rho_v, gB_E);
    computeFluxY(rhoT, uT, vT, pT, ET, gT_rho, gT_rho_u, gT_rho_v, gT_E);

    // HLL 通量公式(三种情况)
    if (SB >= 0.0f)
    {
        // 所有波向上 -> 使用下方通量
        g_rho = gB_rho;
        g_rho_u = gB_rho_u;
        g_rho_v = gB_rho_v;
        g_E = gB_E;
    }
    else if (ST <= 0.0f)
    {
        // 所有波向下 -> 使用上方通量
        g_rho = gT_rho;
        g_rho_u = gT_rho_u;
        g_rho_v = gT_rho_v;
        g_E = gT_E;
    }
    else
    {
        // 跨音速区域 -> 加权平均
        float denom = ST - SB + 1e-10f;
        g_rho = (ST * gB_rho - SB * gT_rho + SB * ST * (rhoT - rhoB)) / denom;
        g_rho_u = (ST * gB_rho_u - SB * gT_rho_u + SB * ST * (rhoT * uT - rhoB * uB)) / denom;
        g_rho_v = (ST * gB_rho_v - SB * gT_rho_v + SB * ST * (rhoT * vT - rhoB * vB)) / denom;
        g_E = (ST * gB_E - SB * gT_E + SB * ST * (ET - EB)) / denom;
    }
}

// 功能:HLLC Riemann 求解器(Y方向)，考虑接触间断
// 输入:界面下方状态(rhoB, uB, vB, pB, EB)和上方状态(rhoT, uT, vT, pT, ET)
// 输出:界面处的Y方向数值通量(g_rho, g_rho_u, g_rho_v, g_E)
__device__ __forceinline__ void hllcFluxY(
    float rhoB, float uB, float vB, float pB, float EB,
    float rhoT, float uT, float vT, float pT, float ET,
    float &g_rho, float &g_rho_u, float &g_rho_v, float &g_E)
{
    // 强制物理约束
    rhoB = fmaxf(rhoB, MIN_DENSITY);
    rhoT = fmaxf(rhoT, MIN_DENSITY);
    pB = fmaxf(pB, MIN_PRESSURE);
    pT = fmaxf(pT, MIN_PRESSURE);

    // 计算声速
    float cB = sqrtf(GAMMA * pB / rhoB);
    float cT = sqrtf(GAMMA * pT / rhoT);

    // Roe 平均(与X方向一致，用于更准确的波速估计)
    float sqrtRhoB = sqrtf(rhoB);
    float sqrtRhoT = sqrtf(rhoT);
    float denom_roe = sqrtRhoB + sqrtRhoT + 1e-10f; // 避免除零，加一个常数

    float u_roe = (sqrtRhoB * uB + sqrtRhoT * uT) / denom_roe;
    float v_roe = (sqrtRhoB * vB + sqrtRhoT * vT) / denom_roe;

    float H_B = (EB + pB) / rhoB;
    float H_T = (ET + pT) / rhoT;
    float H_roe = (sqrtRhoB * H_B + sqrtRhoT * H_T) / denom_roe;

    float c_roe_sq = (GAMMA - 1.0f) * (H_roe - 0.5f * (u_roe * u_roe + v_roe * v_roe));
    float c_roe = (c_roe_sq > 0.0f) ? sqrtf(c_roe_sq) : 0.5f * (cB + cT);

    // Davis 波速估计(结合左右和Roe平均)
    float SB = fminf(vB - cB, v_roe - c_roe);
    float ST = fmaxf(vT + cT, v_roe + c_roe);

    // 熵修正
    float delta = ENTROPY_FIX_FACTOR * fmaxf(cB, cT);
    if (fabsf(SB) < delta)
        SB = -delta;
    if (fabsf(ST) < delta)
        ST = delta;

    // 计算接触波速度(Y方向)，作为分母的物理量必须强制满足下限，防止除零
    float denom_star = rhoB * (SB - vB) - rhoT * (ST - vT);
    // 强制规避除零问题，避免分支退化执行HLL
    float safe_denom_star = copysignf(fmaxf(fabsf(denom_star), 1e-10f), denom_star);
    float S_star = (pT - pB + rhoB * vB * (SB - vB) - rhoT * vT * (ST - vT)) / safe_denom_star;

    // HLLC 通量公式(四种情况)
    if (SB >= 0.0f)
    {
        // 所有波向上 -> 使用下方通量
        computeFluxY(rhoB, uB, vB, pB, EB, g_rho, g_rho_u, g_rho_v, g_E);
    }
    else if (ST <= 0.0f)
    {
        // 所有波向下 -> 使用上方通量
        computeFluxY(rhoT, uT, vT, pT, ET, g_rho, g_rho_u, g_rho_v, g_E);
    }
    // 进入此分支意味着：
    // 1. 最左边的波 SB <= 0 (波已经扫过界面，或者被人为修正为扫过)
    // 2. 最右边的波 ST >= 0 (波还没完全扫过去)
    // 3. 我们现在处于由 SB 和 ST 包围的“中间区域”里
    // 接下来需要判断：我们是在接触间断(S_star)的左边还是右边

    // 情况 A: S_star >= 0
    // 物理含义：接触间断的速度是正的，说明它还在界面的上方
    // 几何位置：界面位于 SB 和 S_star 之间
    // 结论：界面被“下侧流体”覆盖，处于下方星区
    else if (S_star >= 0.0f)
    {
        // 接触波在界面上方 -> 使用下方星区
        float coeff_denom = SB - S_star;
        // 避免除零或非常小的数
        if (fabsf(coeff_denom) < 1e-10f)
            coeff_denom = (coeff_denom >= 0) ? 1e-10f : -1e-10f;
        // 假设星区内状态线性变化，coeff代表了星区密度
        float coeff = rhoB * (SB - vB) / coeff_denom;
        coeff = fmaxf(coeff, MIN_DENSITY);

        float rho_star = coeff;
        float rhoU_star = coeff * uB;     // X速度不变
        float rhoV_star = coeff * S_star; // Y动量用接触波速度

        float pBT = pB + rhoB * (SB - vB) * (S_star - vB);
        pBT = fmaxf(pBT, MIN_PRESSURE);
        float E_star = (EB * (SB - vB) + pBT * S_star - pB * vB) / coeff_denom;

        float gB_rho, gB_rho_u, gB_rho_v, gB_E;
        computeFluxY(rhoB, uB, vB, pB, EB, gB_rho, gB_rho_u, gB_rho_v, gB_E);

        // 返回HLLC修正结果
        g_rho = gB_rho + SB * (rho_star - rhoB);
        g_rho_u = gB_rho_u + SB * (rhoU_star - rhoB * uB);
        g_rho_v = gB_rho_v + SB * (rhoV_star - rhoB * vB);
        g_E = gB_E + SB * (E_star - EB);
    }
    else
    {
        // 接触波在界面下方 -> 使用上方星区
        float coeff_denom = ST - S_star;
        if (fabsf(coeff_denom) < 1e-10f)
            coeff_denom = (coeff_denom >= 0) ? 1e-10f : -1e-10f;
        float coeff = rhoT * (ST - vT) / coeff_denom;
        coeff = fmaxf(coeff, MIN_DENSITY);

        float rho_star = coeff;
        float rhoU_star = coeff * uT;
        float rhoV_star = coeff * S_star;

        float pBT = pT + rhoT * (ST - vT) * (S_star - vT);
        pBT = fmaxf(pBT, MIN_PRESSURE);
        float E_star = (ET * (ST - vT) + pBT * S_star - pT * vT) / coeff_denom;

        float gT_rho, gT_rho_u, gT_rho_v, gT_E;
        computeFluxY(rhoT, uT, vT, pT, ET, gT_rho, gT_rho_u, gT_rho_v, gT_E);

        g_rho = gT_rho + ST * (rho_star - rhoT);
        g_rho_u = gT_rho_u + ST * (rhoU_star - rhoT * uT);
        g_rho_v = gT_rho_v + ST * (rhoV_star - rhoT * vT);
        g_E = gT_E + ST * (E_star - ET);
    }
}

// 功能:使用 MUSCL 方法和 HLLC Riemann 求解器计算网格界面通量
// 输入:守恒变量和原始变量数组，网格类型，网格间距 dx,dy
// 输出:X方向和Y方向的通量数组(包括内能通量)
// 说明:这是求解器的核心，实现了二阶精度的空间离散
__global__ void computeFluxesKernel(
    const float *rho,                               // 密度(守恒量，用于MUSCL重构)
    const float *u, const float *v, const float *p, // 原始变量(用于MUSCL+Riemann)
    const uint8_t *cell_type,
    float *flux_rho_x, float *flux_rho_u_x, float *flux_rho_v_x, float *flux_E_x,
    float *flux_rho_e_x, // 内能通量
    float *flux_rho_y, float *flux_rho_u_y, float *flux_rho_v_y, float *flux_E_y,
    float *flux_rho_e_y, // 内能通量
    float dx, float dy, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // 索引辅助函数(带边界限制)
    auto getIdx = [nx, ny](int ii, int jj)
    {
        // 将索引限制在有效范围内
        ii = max(0, min(nx - 1, ii));
        jj = max(0, min(ny - 1, jj));
        return jj * nx + ii;
    };

    // 获取相邻网格的索引(用于MUSCL重构)
    int idx_im1 = getIdx(i - 1, j); // 左邻居
    int idx_ip1 = getIdx(i + 1, j); // 右邻居
    int idx_jm1 = getIdx(i, j - 1); // 下邻居
    int idx_jp1 = getIdx(i, j + 1); // 上邻居
    int idx_im2 = getIdx(i - 2, j); // 左侧第二个邻居
    int idx_ip2 = getIdx(i + 2, j); // 右侧第二个邻居
    int idx_jm2 = getIdx(i, j - 2); // 下侧第二个邻居
    int idx_jp2 = getIdx(i, j + 2); // 上侧第二个邻居

    // ==================== X方向通量(在 i+1/2 界面处) ====================
    if (i < nx - 1)
    {
        // 检查通量是否应该计算(不穿过固体或虚拟网格)
        bool isSolidOrGhost = (cell_type[idx] == CELL_SOLID || cell_type[idx] == CELL_GHOST);
        bool isSolidOrGhostRight = (cell_type[idx_ip1] == CELL_SOLID || cell_type[idx_ip1] == CELL_GHOST);
        bool validFlux = !isSolidOrGhost && !isSolidOrGhostRight;

        if (validFlux)
        {
            // 检查是否靠近边界(邻居是固体/虚拟网格)
            // 靠近边界时降级为一阶格式(更稳定)
            bool nearBoundary = (cell_type[idx_im1] == CELL_SOLID || cell_type[idx_im1] == CELL_GHOST ||
                                 cell_type[idx_ip2] == CELL_SOLID || cell_type[idx_ip2] == CELL_GHOST);

            // 在低密度区域(如尾流)也使用一阶格式，防止数值不稳定
            bool lowDensity = (rho[idx] < LOW_DENSITY_THRESHOLD || rho[idx_ip1] < LOW_DENSITY_THRESHOLD);

            float rhoL, uL, vL, pL, EL; // 界面左侧重构状态
            float rhoR, uR, vR, pR, ER; // 界面右侧重构状态

            // 算术掩码：消灭重构阶数的 if/else 分支
            float order_mask = (nearBoundary || lowDensity) ? 0.0f : 1.0f;

            // 左侧状态重构(从 i 网格外推到 i+1/2 界面)
            float slope_rho_L = musclSlope(rho[idx_im1], rho[idx], rho[idx_ip1]) * order_mask;
            float slope_u_L = musclSlope(u[idx_im1], u[idx], u[idx_ip1]) * order_mask;
            float slope_v_L = musclSlope(v[idx_im1], v[idx], v[idx_ip1]) * order_mask;
            float slope_p_L = musclSlope(p[idx_im1], p[idx], p[idx_ip1]) * order_mask;

            // 外推到界面: q_L(i+1/2) = q(i) + 0.5 * slope
            rhoL = rho[idx] + 0.5f * slope_rho_L;
            uL = u[idx] + 0.5f * slope_u_L;
            vL = v[idx] + 0.5f * slope_v_L;
            pL = p[idx] + 0.5f * slope_p_L;

            // 右侧状态重构(从 i+1 网格外推到 i+1/2 界面)
            float slope_rho_R = musclSlope(rho[idx], rho[idx_ip1], rho[idx_ip2]) * order_mask;
            float slope_u_R = musclSlope(u[idx], u[idx_ip1], u[idx_ip2]) * order_mask;
            float slope_v_R = musclSlope(v[idx], v[idx_ip1], v[idx_ip2]) * order_mask;
            float slope_p_R = musclSlope(p[idx], p[idx_ip1], p[idx_ip2]) * order_mask;

            // 外推到界面: q_R(i+1/2) = q(i+1) - 0.5 * slope
            rhoR = rho[idx_ip1] - 0.5f * slope_rho_R;
            uR = u[idx_ip1] - 0.5f * slope_u_R;
            vR = v[idx_ip1] - 0.5f * slope_v_R;
            pR = p[idx_ip1] - 0.5f * slope_p_R;

            // 确保重构后的状态仍然物理合理
            rhoL = fmaxf(rhoL, MIN_DENSITY);
            pL = fmaxf(pL, MIN_PRESSURE);
            rhoR = fmaxf(rhoR, MIN_DENSITY);
            pR = fmaxf(pR, MIN_PRESSURE);

            // 从原始变量计算守恒变量(Riemann求解器需要)
            // 左侧能量: E_L = p/(gamma-1) + 0.5*rho*(u^2+v^2)
            float keL = 0.5f * rhoL * (uL * uL + vL * vL);
            EL = pL / (GAMMA - 1.0f) + keL;

            // 右侧能量: E_R = p/(gamma-1) + 0.5*rho*(u^2+v^2)
            float keR = 0.5f * rhoR * (uR * uR + vR * vR);
            ER = pR / (GAMMA - 1.0f) + keR;

            // 调用 Riemann 求解器计算数值通量
            float f_rho, f_rho_u, f_rho_v, f_E;
            // 统一使用更精确的HLLC求解器
            hllcFluxX(rhoL, uL, vL, pL, EL, rhoR, uR, vR, pR, ER,
                      f_rho, f_rho_u, f_rho_v, f_E);

            // 存储守恒变量通量
            flux_rho_x[idx] = f_rho;
            flux_rho_u_x[idx] = f_rho_u;
            flux_rho_v_x[idx] = f_rho_v;
            flux_E_x[idx] = f_E;

            // ===== 内能通量(用于双能量法) =====
            // 使用迎风格式: 根据界面速度方向选择上游状态
            float u_face = 0.5f * (uL + uR);     // 界面平均速度
            float rho_e_L = pL / (GAMMA - 1.0f); // 左侧内能密度
            float rho_e_R = pR / (GAMMA - 1.0f); // 右侧内能密度
            float f_rho_e = (u_face >= 0.0f) ? rho_e_L * uL : rho_e_R * uR;
            flux_rho_e_x[idx] = f_rho_e;
        }
        else
        {
            // ===== 固体壁面或无效通量 =====
            flux_rho_x[idx] = 0.0f;                          // 质量不穿透
            flux_rho_u_x[idx] = fmaxf(p[idx], MIN_PRESSURE); // 压力作用在壁面上
            flux_rho_v_x[idx] = 0.0f;                        // 动量不穿透
            flux_E_x[idx] = 0.0f;                            // 能量不穿透
            flux_rho_e_x[idx] = 0.0f;                        // 内能不穿透
        }
    }

    // ==================== Y方向通量(在 j+1/2 界面处) ====================
    // 算法与X方向相同，只是方向变为Y方向
    if (j < ny - 1)
    {
        bool isSolidOrGhost = (cell_type[idx] == CELL_SOLID || cell_type[idx] == CELL_GHOST);
        bool isSolidOrGhostTop = (cell_type[idx_jp1] == CELL_SOLID || cell_type[idx_jp1] == CELL_GHOST);
        bool validFlux = !isSolidOrGhost && !isSolidOrGhostTop;

        if (validFlux)
        {
            bool nearBoundary = (cell_type[idx_jm1] == CELL_SOLID || cell_type[idx_jm1] == CELL_GHOST ||
                                 cell_type[idx_jp2] == CELL_SOLID || cell_type[idx_jp2] == CELL_GHOST);

            bool lowDensity = (rho[idx] < LOW_DENSITY_THRESHOLD || rho[idx_jp1] < LOW_DENSITY_THRESHOLD);

            float rhoB, uB, vB, pB, EB; // 界面下方(Bottom)状态
            float rhoT, uT, vT, pT, ET; // 界面上方(Top)状态

            float order_mask = (nearBoundary || lowDensity) ? 0.0f : 1.0f;

            // MUSCL二阶重构(Y方向)
            float slope_rho_B = musclSlope(rho[idx_jm1], rho[idx], rho[idx_jp1]) * order_mask;
            float slope_u_B = musclSlope(u[idx_jm1], u[idx], u[idx_jp1]) * order_mask;
            float slope_v_B = musclSlope(v[idx_jm1], v[idx], v[idx_jp1]) * order_mask;
            float slope_p_B = musclSlope(p[idx_jm1], p[idx], p[idx_jp1]) * order_mask;

            rhoB = rho[idx] + 0.5f * slope_rho_B;
            uB = u[idx] + 0.5f * slope_u_B;
            vB = v[idx] + 0.5f * slope_v_B;
            pB = p[idx] + 0.5f * slope_p_B;

            float slope_rho_T = musclSlope(rho[idx], rho[idx_jp1], rho[idx_jp2]) * order_mask;
            float slope_u_T = musclSlope(u[idx], u[idx_jp1], u[idx_jp2]) * order_mask;
            float slope_v_T = musclSlope(v[idx], v[idx_jp1], v[idx_jp2]) * order_mask;
            float slope_p_T = musclSlope(p[idx], p[idx_jp1], p[idx_jp2]) * order_mask;

            rhoT = rho[idx_jp1] - 0.5f * slope_rho_T;
            uT = u[idx_jp1] - 0.5f * slope_u_T;
            vT = v[idx_jp1] - 0.5f * slope_v_T;
            pT = p[idx_jp1] - 0.5f * slope_p_T;

            // 确保正定性
            rhoB = fmaxf(rhoB, MIN_DENSITY);
            pB = fmaxf(pB, MIN_PRESSURE);
            rhoT = fmaxf(rhoT, MIN_DENSITY);
            pT = fmaxf(pT, MIN_PRESSURE);

            // 计算能量
            float keB = 0.5f * rhoB * (uB * uB + vB * vB);
            EB = pB / (GAMMA - 1.0f) + keB;

            float keT = 0.5f * rhoT * (uT * uT + vT * vT);
            ET = pT / (GAMMA - 1.0f) + keT;

            // Riemann求解器(Y方向)
            float g_rho, g_rho_u, g_rho_v, g_E;
            hllcFluxY(rhoB, uB, vB, pB, EB, rhoT, uT, vT, pT, ET,
                      g_rho, g_rho_u, g_rho_v, g_E);

            flux_rho_y[idx] = g_rho;
            flux_rho_u_y[idx] = g_rho_u;
            flux_rho_v_y[idx] = g_rho_v;
            flux_E_y[idx] = g_E;

            // 内能通量(Y方向)
            float v_face = 0.5f * (vB + vT);
            float rho_e_B = pB / (GAMMA - 1.0f);
            float rho_e_T = pT / (GAMMA - 1.0f);
            float g_rho_e = (v_face >= 0.0f) ? rho_e_B * vB : rho_e_T * vT;
            flux_rho_e_y[idx] = g_rho_e;
        }
        else
        {
            // 壁面边界
            flux_rho_y[idx] = 0.0f;
            flux_rho_u_y[idx] = 0.0f;
            flux_rho_v_y[idx] = fmaxf(p[idx], MIN_PRESSURE); // 压力作用
            flux_E_y[idx] = 0.0f;
            flux_rho_e_y[idx] = 0.0f;
        }
    }
}

// 功能:使用有限体积法和双能量法更新守恒变量
// 输入:当前守恒变量，通量数组，时间步长 dt，网格间距 dx,dy
// 输出:更新后的守恒变量(新时间步)
// 说明:实现欧拉前向时间积分和双能量同步
__global__ void updateKernel(
    const float *rho, const float *rho_u, const float *rho_v, const float *E,
    const float *rho_e,
    const float *u, const float *v, const float *p, // 已计算的原始变量(避免重复除法)
    const float *flux_rho_x, const float *flux_rho_u_x, const float *flux_rho_v_x, const float *flux_E_x,
    const float *flux_rho_e_x,
    const float *flux_rho_y, const float *flux_rho_u_y, const float *flux_rho_v_y, const float *flux_E_y,
    const float *flux_rho_e_y,
    const uint8_t *cell_type,
    float *rho_new, float *rho_u_new, float *rho_v_new, float *E_new,
    float *rho_e_new,
    float dt, float dx, float dy, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // 跳过固体和虚拟网格(它们由边界条件核函数处理)
    if (cell_type[idx] == CELL_SOLID || cell_type[idx] == CELL_GHOST)
    {
        rho_new[idx] = rho[idx];
        rho_u_new[idx] = rho_u[idx];
        rho_v_new[idx] = rho_v[idx];
        E_new[idx] = E[idx];
        rho_e_new[idx] = rho_e[idx];
        return;
    }

    // 获取通量索引(左侧和下侧界面)
    int idx_im1 = (i > 0) ? (j * nx + (i - 1)) : idx;
    int idx_jm1 = (j > 0) ? ((j - 1) * nx + i) : idx;

    // ===== 计算通量散度(有限体积法) =====
    // div(F) = (F_{i+1/2} - F_{i-1/2}) / dx
    float dFx_rho = (flux_rho_x[idx] - flux_rho_x[idx_im1]) / dx;
    float dFx_rho_u = (flux_rho_u_x[idx] - flux_rho_u_x[idx_im1]) / dx;
    float dFx_rho_v = (flux_rho_v_x[idx] - flux_rho_v_x[idx_im1]) / dx;
    float dFx_E = (flux_E_x[idx] - flux_E_x[idx_im1]) / dx;
    float dFx_rho_e = (flux_rho_e_x[idx] - flux_rho_e_x[idx_im1]) / dx;

    // div(G) = (G_{j+1/2} - G_{j-1/2}) / dy
    float dFy_rho = (flux_rho_y[idx] - flux_rho_y[idx_jm1]) / dy;
    float dFy_rho_u = (flux_rho_u_y[idx] - flux_rho_u_y[idx_jm1]) / dy;
    float dFy_rho_v = (flux_rho_v_y[idx] - flux_rho_v_y[idx_jm1]) / dy;
    float dFy_E = (flux_E_y[idx] - flux_E_y[idx_jm1]) / dy;
    float dFy_rho_e = (flux_rho_e_y[idx] - flux_rho_e_y[idx_jm1]) / dy;

    // 处理边界通量(域边界处单侧通量为零)
    // i==0: 左边界面通量未定义; i==nx-1: 右边界面(i+1/2)通量未计算
    if (i == 0 || i == nx - 1)
    {
        dFx_rho = dFx_rho_u = dFx_rho_v = dFx_E = dFx_rho_e = 0.0f;
    }
    if (j == 0 || j == ny - 1)
    {
        dFy_rho = dFy_rho_u = dFy_rho_v = dFy_E = dFy_rho_e = 0.0f;
    }

    // ===== 计算内能方程的源项: -p * div(v) =====
    // 这是双能量法的关键:内能方程有压力做功项
    // div(v) = du/dx + dv/dy (速度散度)
    int idx_ip1 = (i < nx - 1) ? (j * nx + (i + 1)) : idx;
    int idx_jp1 = (j < ny - 1) ? ((j + 1) * nx + i) : idx;

    // 直接使用已计算的原始变量:
    // - 避免 5 次除法 (rho_u/rho) 和 5 次 fmaxf 操作
    // - 避免 8 次额外的全局内存读取 (rho_u/rho_v/rho 在邻居处)
    // - 压强使用双能量法选取的值(比 E-KE 更精确)
    float du_dx = (u[idx_ip1] - u[idx_im1]) / (2.0f * dx);
    float dv_dy = (v[idx_jp1] - v[idx_jm1]) / (2.0f * dy);
    float div_v = du_dx + dv_dy;

    float p_c = p[idx];

    // 内能方程源项: d(rho*e)/dt = -p * div(v) + ... (这里只包含压力做功)
    float source_rho_e = -p_c * div_v;

    // ===== 欧拉前向时间积分 =====
    // U^{n+1} = U^n - dt * div(F)
    float new_rho = rho[idx] - dt * (dFx_rho + dFy_rho);
    float new_rho_u = rho_u[idx] - dt * (dFx_rho_u + dFy_rho_u);
    float new_rho_v = rho_v[idx] - dt * (dFx_rho_v + dFy_rho_v);
    float new_E = E[idx] - dt * (dFx_E + dFy_E);

    // 内能方程: d(rho*e)/dt + div(flux_rho_e) = source
    float new_rho_e = rho_e[idx] - dt * (dFx_rho_e + dFy_rho_e) + dt * source_rho_e;

    // ===== 数值稳定性检查 =====
    // 检测 NaN 或 Inf，如果出现则回退到旧值
    if (!isfinite(new_rho) || new_rho < MIN_DENSITY)
        new_rho = fmaxf(rho[idx], MIN_DENSITY);
    if (!isfinite(new_rho_u))
        new_rho_u = rho_u[idx];
    if (!isfinite(new_rho_v))
        new_rho_v = rho_v[idx];
    if (!isfinite(new_E))
        new_E = E[idx];
    if (!isfinite(new_rho_e) || new_rho_e < MIN_PRESSURE / (GAMMA - 1.0f))
        new_rho_e = rho_e[idx];

    // 确保密度正定
    new_rho = fmaxf(new_rho, MIN_DENSITY);

    // 计算并限制速度(防止超光速等非物理现象)
    float u_new = new_rho_u / new_rho;
    float v_new = new_rho_v / new_rho;
    float vel_mag = sqrtf(u_new * u_new + v_new * v_new);
    float max_vel = MAX_VELOCITY_LIMIT; // 最大速度限制 [m/s]
    if (vel_mag > max_vel)
    {
        // 等比例缩放速度到最大值
        float scale = max_vel / vel_mag;
        u_new *= scale;
        v_new *= scale;
        new_rho_u = new_rho * u_new;
        new_rho_v = new_rho * v_new;
    }

    // 计算动能
    float ke = 0.5f * new_rho * (u_new * u_new + v_new * v_new);

    // ========== 双能量同步 ==========
    // 关键思想:
    // - 在激波区(高压强): 使用 E - KE 计算内能(保守性好，准确捕捉激波)
    // - 在膨胀区(低KE占比): 使用 rho_e 直接计算(避免E-KE的精度损失)

    // 从总能量计算内能(E法)
    float e_from_E = new_E - ke;

    // 从双能量方程得到的内能(e法)
    float e_from_e = new_rho_e;

    // 确保两种方法都是正值
    float e_min = new_rho * R_GAS * MIN_TEMPERATURE / (GAMMA - 1.0f);
    e_from_E = fmaxf(e_from_E, e_min);
    e_from_e = fmaxf(e_from_e, e_min);

    // 切换准则: eta = e_internal / E_total
    // 当 eta 小时，KE 占主导，E-KE减法不可靠
    float eta = e_from_e / (new_E + 1e-20f);
    float eta_threshold = DUAL_ENERGY_ETA_THRESHOLD; // 阈值以下使用e法

    float e_internal;
    if (eta > eta_threshold && e_from_E > 0.0f)
    {
        // 情况1: 热能占比显著 - 安全使用E法
        // 这在激波处保持守恒性
        e_internal = e_from_E;

        // 同步: 更新 rho_e 使其与 E 一致
        // 防止两个能量之间漂移
        new_rho_e = e_from_E;
    }
    else
    {
        // 情况2: 动能占主导 - 使用追踪的内能
        // 这避免了"大数减大数"的精度问题
        e_internal = e_from_e;

        // 重构总能量使其一致
        new_E = e_internal + ke;
    }

    // 最终安全限制
    float e_max = new_rho * R_GAS * MAX_TEMPERATURE / (GAMMA - 1.0f);
    e_internal = fminf(e_internal, e_max);
    e_internal = fmaxf(e_internal, e_min);

    // 确保一致性
    new_rho_e = e_internal;
    new_E = e_internal + ke;

    // 写入结果到新数组
    rho_new[idx] = new_rho;
    rho_u_new[idx] = new_rho_u;
    rho_v_new[idx] = new_rho_v;
    E_new[idx] = new_E;
    rho_e_new[idx] = new_rho_e;
}

// 功能:应用边界条件到守恒变量
// 输入:守恒变量数组，网格类型，SDF场，来流参数，粘性/壁面温度设置
// 输出:更新边界处的守恒变量
// 说明:实现流入/流出/上下边界的特征边界条件，以及固体壁面的Ghost Cell方法
__global__ void applyBoundaryConditionsKernel(
    float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
    const uint8_t *cell_type, const float *sdf,
    float rho_inf, float u_inf, float v_inf, float p_inf,
    float dx, float dy, int nx, int ny,
    bool enable_viscosity, bool adiabatic_wall, float T_wall)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;
    uint8_t ctype = cell_type[idx];

    // 来流的能量状态(用于边界条件)
    float ke_inf = 0.5f * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    float e_inf = p_inf / (GAMMA - 1.0f); // 来流内能密度
    float E_inf = e_inf + ke_inf;

    // ========== 流入边界(左侧) ==========
    if (ctype == CELL_INFLOW || i == 0)
    {
        // 直接设置为来流条件
        rho[idx] = rho_inf;
        rho_u[idx] = rho_inf * u_inf;
        rho_v[idx] = rho_inf * v_inf;
        E[idx] = E_inf;
        rho_e[idx] = e_inf;
        return;
    }

    // ========== 流出边界(右侧) - 非反射特征边界条件 ==========
    if (ctype == CELL_OUTFLOW || i == nx - 1)
    {
        float rho_loc = fmaxf(rho[idx], MIN_DENSITY);
        // 从守恒变量计算原始变量
        float u_loc, v_loc, p_loc, c_loc, t_loc;
        computePrimitivesKernelInline(rho[idx], rho_u[idx], rho_v[idx], E[idx], rho_e[idx], u_loc, v_loc, p_loc, t_loc);
        c_loc = sqrtf(GAMMA * p_loc / rho_loc); // 局部声速

        // 判断X方向的特征波是否流出（右边界）
        // 对于右边界，流出意味着特征速度 > 0
        bool is_R1_out = u_loc > 0.0f;         // R1（熵波）特征速度 = u
        bool is_R2_out = u_loc > 0.0f;         // R2（剪切波）特征速度 = u
        bool is_R3_out = u_loc + c_loc > 0.0f; // R3（向右声波）特征速度 = u + c
        bool is_R4_out = u_loc - c_loc > 0.0f; // R4（向左声波）特征速度 = u - c

        // 计算局部的Riemann不变量
        float R1, R2, R3, R4;
        computeRiemannInvarFromPrimitiveX(rho[idx], u_loc, v_loc, p_loc, R1, R2, R3, R4);

        // 计算远场（来流）的Riemann不变量
        float R1_inf, R2_inf, R3_inf, R4_inf;
        computeFarfieldRiemannInvarX(rho_inf, u_inf, v_inf, p_inf, R1_inf, R2_inf, R3_inf, R4_inf);

        // 根据特征波方向选择Riemann不变量
        // 流出的波使用局部值，流入的波使用远场值
        R1 = is_R1_out ? R1 : R1_inf;
        R2 = is_R2_out ? R2 : R2_inf;
        R3 = is_R3_out ? R3 : R3_inf;
        R4 = is_R4_out ? R4 : R4_inf;

        // 从Riemann不变量重构原始变量
        computePrimitiveFromRiemannInvarX(R1, R2, R3, R4, rho[idx], u_loc, v_loc, p_loc);

        // 更新守恒变量
        rho_u[idx] = rho[idx] * u_loc;
        rho_v[idx] = rho[idx] * v_loc;
        rho_e[idx] = p_loc / (GAMMA - 1.0f);
        E[idx] = rho_e[idx] + 0.5f * rho[idx] * (u_loc * u_loc + v_loc * v_loc);
        return;
    }

    // ========== 上下边界 - 非反射特征边界条件 ==========
    // 使用二阶外推配合来流混合，防止上游污染
    if (j == 0) // 下边界
    {
        float rho_loc = fmaxf(rho[idx], MIN_DENSITY);
        // 从动量密度计算速度: u = (rho*u) / rho
        float u_loc, v_loc, p_loc, c_loc, t_loc;
        computePrimitivesKernelInline(rho_loc, rho_u[idx], rho_v[idx], E[idx], rho_e[idx], u_loc, v_loc, p_loc, t_loc);
        c_loc = sqrtf(GAMMA * p_loc / rho_loc); // 局部声速

        // 修复：使用速度v_loc而不是动量rho_v[idx]判断特征波方向
        bool is_R1_out = v_loc < 0.0f;         // R1（熵波）特征速度 = v
        bool is_R2_out = v_loc < 0.0f;         // R2（剪切波）特征速度 = v
        bool is_R3_out = v_loc + c_loc < 0.0f; // R3（向上声波）特征速度 = v + c
        bool is_R4_out = v_loc - c_loc < 0.0f; // R4（向下声波）特征速度 = v - c

        float R1, R2, R3, R4;
        computeRiemannInvarFromPrimitiveY(rho_loc, u_loc, v_loc, p_loc, R1, R2, R3, R4);
        float R1_inf, R2_inf, R3_inf, R4_inf;
        computeFarfieldRiemannInvarY(rho_inf, u_inf, v_inf, p_inf, R1_inf, R2_inf, R3_inf, R4_inf);

        R1 = is_R1_out ? R1 : R1_inf;
        R2 = is_R2_out ? R2 : R2_inf;
        R3 = is_R3_out ? R3 : R3_inf;
        R4 = is_R4_out ? R4 : R4_inf;

        computePrimitiveFromRiemannInvarY(R1, R2, R3, R4, rho_loc, u_loc, v_loc, p_loc);
        rho[idx] = rho_loc;
        rho_u[idx] = rho_loc * u_loc;
        rho_v[idx] = rho_loc * v_loc;
        rho_e[idx] = p_loc / (GAMMA - 1.0f);
        E[idx] = rho_e[idx] + 0.5f * rho_loc * (u_loc * u_loc + v_loc * v_loc);
    }

    if (j == ny - 1) // 上边界(算法与下边界对称)
    {
        float rho_loc = fmaxf(rho[idx], MIN_DENSITY);
        // 从动量密度计算速度: u = (rho*u) / rho
        float u_loc, v_loc, p_loc, c_loc, t_loc;
        computePrimitivesKernelInline(rho_loc, rho_u[idx], rho_v[idx], E[idx], rho_e[idx], u_loc, v_loc, p_loc, t_loc);
        c_loc = sqrtf(GAMMA * p_loc / rho_loc); // 局部声速

        // 修复：使用速度v_loc而不是动量rho_v[idx]判断特征波方向
        bool is_R1_out = v_loc > 0.0f;         // R1（熵波）特征速度 = v
        bool is_R2_out = v_loc > 0.0f;         // R2（剪切波）特征速度 = v
        bool is_R3_out = v_loc + c_loc > 0.0f; // R3（向上声波）特征速度 = v + c
        bool is_R4_out = v_loc - c_loc > 0.0f; // R4（向下声波）特征速度 = v - c

        float R1, R2, R3, R4;
        computeRiemannInvarFromPrimitiveY(rho_loc, u_loc, v_loc, p_loc, R1, R2, R3, R4);
        float R1_inf, R2_inf, R3_inf, R4_inf;
        computeFarfieldRiemannInvarY(rho_inf, u_inf, v_inf, p_inf, R1_inf, R2_inf, R3_inf, R4_inf);

        R1 = is_R1_out ? R1 : R1_inf;
        R2 = is_R2_out ? R2 : R2_inf;
        R3 = is_R3_out ? R3 : R3_inf;
        R4 = is_R4_out ? R4 : R4_inf;

        computePrimitiveFromRiemannInvarY(R1, R2, R3, R4, rho_loc, u_loc, v_loc, p_loc);
        rho[idx] = rho_loc;
        rho_u[idx] = rho_loc * u_loc;
        rho_v[idx] = rho_loc * v_loc;
        rho_e[idx] = p_loc / (GAMMA - 1.0f);
        E[idx] = rho_e[idx] + 0.5f * rho_loc * (u_loc * u_loc + v_loc * v_loc);
    }

    // ========== 固体单元 ==========
    if (ctype == CELL_SOLID)
    {
        // 固体内部设为来流状态(这些值不参与计算，只是为了防止NaN传播)
        rho[idx] = rho_inf;
        rho_u[idx] = 0.0f; // 固体内无速度
        rho_v[idx] = 0.0f;
        E[idx] = e_inf; // 只有内能，无动能
        rho_e[idx] = e_inf;
        return;
    }

    // ========== 虚拟单元(Ghost Cell，用于浸入边界法) ==========
    if (ctype == CELL_GHOST)
    {
        // 虚拟网格法的核心思想:
        // 1. 通过SDF场确定壁面法向
        // 2. 找到最近的流体邻居
        // 3. 根据壁面边界条件(无滑移/等温/绝热)镜像或外推

        // ===== 步骤1: 计算SDF梯度得到壁面法向 =====
        // 法向指向流体(SDF正值方向)
        float sdf_xp = (i < nx - 1) ? sdf[(j)*nx + (i + 1)] : sdf[idx];
        float sdf_xm = (i > 0) ? sdf[(j)*nx + (i - 1)] : sdf[idx];
        float sdf_yp = (j < ny - 1) ? sdf[(j + 1) * nx + i] : sdf[idx];
        float sdf_ym = (j > 0) ? sdf[(j - 1) * nx + i] : sdf[idx];

        // 中心差分计算梯度: grad(SDF) = (dSDF/dx, dSDF/dy)
        float _nxsdf = (sdf_xp - sdf_xm) / (2.0f * dx);
        float _nysdf = (sdf_yp - sdf_ym) / (2.0f * dy);
        float norm_len = sqrtf(_nxsdf * _nxsdf + _nysdf * _nysdf) + 1e-10f;
        _nxsdf /= norm_len; // 单位法向量的X分量
        _nysdf /= norm_len; // 单位法向量的Y分量

        // ===== 步骤2: 找到最佳流体邻居 =====
        // 最佳邻居定义: 沿法向方向(进入流体)的最近有效流体网格
        int best_idx = -1;
        float best_dot = -1.0f; // 与法向的最大点积

        // 搜索8邻域
        int neighbors[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}};

        for (int n = 0; n < 8; n++)
        {
            int ni = i + neighbors[n][0];
            int nj = j + neighbors[n][1];
            if (ni >= 0 && ni < nx && nj >= 0 && nj < ny)
            {
                int nidx = nj * nx + ni;
                // 检查是否是有效流体网格(SDF>0 且密度合理)
                if (sdf[nidx] > 0.0f && rho[nidx] >= MIN_DENSITY)
                {
                    // 计算邻居方向与法向的点积
                    float dir_x = (float)neighbors[n][0];
                    float dir_y = (float)neighbors[n][1];
                    float len = sqrtf(dir_x * dir_x + dir_y * dir_y);
                    float dot = (dir_x * _nxsdf + dir_y * _nysdf) / len;

                    // 选择与法向最接近的邻居
                    if (dot > best_dot)
                    {
                        best_dot = dot;
                        best_idx = nidx;
                    }
                }
            }
        }

        // 如果仍然没有有效邻居，使用来流条件
        if (best_idx < 0)
        {
            rho[idx] = rho_inf;
            rho_u[idx] = 0.0f;
            rho_v[idx] = 0.0f;
            E[idx] = e_inf;
            rho_e[idx] = e_inf;
            return;
        }

        // ===== 步骤3: 从流体邻居获取状态 =====
        float rho_f = rho[best_idx];
        float u_f = rho_u[best_idx] / rho_f;
        float v_f = rho_v[best_idx] / rho_f;
        float E_f = E[best_idx];
        float ke_f = 0.5f * rho_f * (u_f * u_f + v_f * v_f);
        float p_f = (GAMMA - 1.0f) * (E_f - ke_f);

        // 物理限制
        rho_f = fmaxf(rho_f, MIN_DENSITY);
        p_f = fmaxf(p_f, MIN_PRESSURE);

        // 分解速度为法向和切向分量
        // vn = v · n (正值表示流体流向壁面)
        // vt = |v - (v·n)n| (切向速度大小)
        float vn = u_f * _nxsdf + v_f * _nysdf;  // 法向速度
        float vt = -v_f * _nxsdf + u_f * _nysdf; // 切向速度(垂直于法向)

        // 计算温度
        float T_f = p_f / (rho_f * R_GAS);
        T_f = fminf(T_f, MAX_TEMPERATURE);
        T_f = fmaxf(T_f, MIN_TEMPERATURE);
        p_f = rho_f * R_GAS * T_f; // 重新确保一致性

        // ===== 步骤4: 设置虚拟网格的速度边界条件 =====
        float u_g, v_g; // 虚拟网格的速度

        if (enable_viscosity)
        {
            // ===== 粘性流动: 严格无滑移边界条件 =====
            // 要求壁面处速度为零: u_wall = (u_fluid + u_ghost) / 2 = 0
            // 因此 u_ghost = -u_fluid (镜像反射)
            u_g = -u_f;
            v_g = -v_f;
        }
        else
        {
            // ===== 无粘流动: 平滑滑移/无滑移过渡 =====
            // 纯无粘应该是完全滑移(法向速度为零，切向速度不变)
            // 但为了数值稳定性，引入部分无滑移
            float slip_factor = 0.0f; // 0=无滑移, 1=完全滑移

            // 根据法向速度自适应调整滑移因子
            // 当流体高速撞击壁面时，使用更多滑移减少数值振荡
            if (vn > 0.0f)
            {
                float vn_ref = SLIP_REFERENCE_VELOCITY; // 参考速度
                slip_factor = fminf(SLIP_FACTOR_MAX, vn / (vn + vn_ref));
            }

            // 无滑移条件(镜像)
            float u_noslip = -u_f;
            float v_noslip = -v_f;

            // 滑移条件(只反射法向速度)
            // u_slip = u_f - 2*(v_f·n)*n
            float u_slip = -vn * _nxsdf + vt * _nysdf;
            float v_slip = -vn * _nysdf - vt * _nxsdf;

            // 线性混合
            u_g = (1.0f - slip_factor) * u_noslip + slip_factor * u_slip;
            v_g = (1.0f - slip_factor) * v_noslip + slip_factor * v_slip;
        }

        // 限制虚拟网格速度大小，防止数值不稳定
        float vel_g_mag = sqrtf(u_g * u_g + v_g * v_g);
        float max_ghost_vel = MAX_GHOST_VELOCITY;
        if (vel_g_mag > max_ghost_vel)
        {
            float scale = max_ghost_vel / vel_g_mag;
            u_g *= scale;
            v_g *= scale;
        }

        // ===== 步骤5: 设置虚拟网格的温度边界条件 =====
        float rho_g = rho_f; // 密度初始与流体相同
        float p_g = p_f;     // 压强初始与流体相同
        float T_g;

        if (adiabatic_wall)
        {
            // ===== 绝热壁面: 零温度梯度 =====
            // dT/dn = 0 => T_ghost = T_fluid (外推)
            T_g = T_f;
        }
        else
        {
            // ===== 等温壁面: 指定壁面温度 =====
            // 要求壁面温度为 T_wall: T_wall = (T_fluid + T_ghost) / 2
            // 因此 T_ghost = 2*T_wall - T_fluid (镜像)
            T_g = 2.0f * T_wall - T_f;
            T_g = fmaxf(T_g, MIN_TEMPERATURE);
            T_g = fminf(T_g, MAX_TEMPERATURE);
        }

        // 从温度重新计算热力学状态
        // 保持压强连续(p_g = p_f)，由理想气体定律调整密度
        // p = rho * R * T => rho_g = p_g / (R * T_g)
        rho_g = p_g / (R_GAS * T_g);
        rho_g = fmaxf(rho_g, MIN_DENSITY);

        // 内能密度
        float e_g = p_g / (GAMMA - 1.0f);

        // 虚拟网格总能量 = 内能 + 动能
        float ke_g = 0.5f * rho_g * (u_g * u_g + v_g * v_g);
        float E_g = e_g + ke_g;

        // 写入虚拟网格
        rho[idx] = rho_g;
        rho_u[idx] = rho_g * u_g;
        rho_v[idx] = rho_g * v_g;
        E[idx] = E_g;
        rho_e[idx] = e_g;
    }
}
#pragma endregion

#pragma region 有粘性逻辑

// ===== 辅助设备函数：在任意网格点 (i,j) 计算粘性项 =====
// 用于融合核函数中按需重算邻居粘性量，避免中间全局内存往返
__device__ __forceinline__ void computeViscousAtPoint(
    const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ T,
    const uint8_t *__restrict__ cell_type,
    int i, int j, int nx, int ny, float dx, float dy,
    float &tau_xx_out, float &tau_yy_out, float &tau_xy_out,
    float &qx_out, float &qy_out, float &mu_val_out)
{
    int idx = j * nx + i;

    if (cell_type[idx] == CELL_SOLID)
    {
        mu_val_out = tau_xx_out = tau_yy_out = tau_xy_out = qx_out = qy_out = 0.0f;
        return;
    }

    // Sutherland 粘性 + 热导率
    float T_val = fmaxf(T[idx], MIN_TEMPERATURE);
    T_val = fminf(T_val, MAX_TEMPERATURE);
    float T_ratio = T_val / T_REF;
    float mu_val = MU_REF * (T_ratio * sqrtf(T_ratio)) * (T_REF + S_SUTHERLAND) / (T_val + S_SUTHERLAND);
    float k_val = mu_val * CP / PRANDTL_NUMBER;
    mu_val_out = mu_val;

    // 邻居索引
    int idx_im1 = (i > 0) ? (j * nx + (i - 1)) : idx;
    int idx_ip1 = (i < nx - 1) ? (j * nx + (i + 1)) : idx;
    int idx_jm1 = (j > 0) ? ((j - 1) * nx + i) : idx;
    int idx_jp1 = (j < ny - 1) ? ((j + 1) * nx + i) : idx;

    // 速度梯度
    float du_dx = (u[idx_ip1] - u[idx_im1]) / (2.0f * dx);
    float du_dy = (u[idx_jp1] - u[idx_jm1]) / (2.0f * dy);
    float dv_dx = (v[idx_ip1] - v[idx_im1]) / (2.0f * dx);
    float dv_dy = (v[idx_jp1] - v[idx_jm1]) / (2.0f * dy);

    if (i == 0)
    {
        du_dx = (u[idx_ip1] - u[idx]) / dx;
        dv_dx = (v[idx_ip1] - v[idx]) / dx;
    }
    else if (i == nx - 1)
    {
        du_dx = (u[idx] - u[idx_im1]) / dx;
        dv_dx = (v[idx] - v[idx_im1]) / dx;
    }
    if (j == 0)
    {
        du_dy = (u[idx_jp1] - u[idx]) / dy;
        dv_dy = (v[idx_jp1] - v[idx]) / dy;
    }
    else if (j == ny - 1)
    {
        du_dy = (u[idx] - u[idx_jm1]) / dy;
        dv_dy = (v[idx] - v[idx_jm1]) / dy;
    }

    float div_v = du_dx + dv_dy;
    tau_xx_out = mu_val * (2.0f * du_dx - (2.0f / 3.0f) * div_v);
    tau_yy_out = mu_val * (2.0f * dv_dy - (2.0f / 3.0f) * div_v);
    tau_xy_out = mu_val * (du_dy + dv_dx);

    // 温度梯度 → 热通量
    float dT_dx = (T[idx_ip1] - T[idx_im1]) / (2.0f * dx);
    float dT_dy = (T[idx_jp1] - T[idx_jm1]) / (2.0f * dy);
    if (i == 0)
        dT_dx = (T[idx_ip1] - T[idx]) / dx;
    else if (i == nx - 1)
        dT_dx = (T[idx] - T[idx_im1]) / dx;
    if (j == 0)
        dT_dy = (T[idx_jp1] - T[idx]) / dy;
    else if (j == ny - 1)
        dT_dy = (T[idx] - T[idx_jm1]) / dy;

    qx_out = -k_val * dT_dx;
    qy_out = -k_val * dT_dy;
}

// ===== 融合核函数：粘性项计算 + 扩散步一次完成 =====
// 原理：每个线程按需重算中心及4邻居的粘性量(tau/q)，结果保持在寄存器中，
//       消除 5 个中间数组(tau_xx/yy/xy, qx, qy)的全局内存写+读往返。
//       因原核函数 Compute Throughput 仅 17.84%（严重 memory-bound），
//       5x 重算的额外计算量在 SM 空闲算力内完全消化。
// 输出：mu_out(供CFL条件使用)，直接更新 rho_u/rho_v/E/rho_e
__global__ void fusedViscousDiffusionKernel(
    const float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ T,
    float *__restrict__ mu_out,
    float *__restrict__ rho_u, float *__restrict__ rho_v,
    float *__restrict__ E, float *__restrict__ rho_e,
    const uint8_t *__restrict__ cell_type,
    float dt, float dx, float dy, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // ===== Phase 1: 计算本单元粘性项，写出 mu(始终需要) =====
    float tau_xx_c, tau_yy_c, tau_xy_c, qx_c, qy_c, mu_c;
    computeViscousAtPoint(u, v, T, cell_type, i, j, nx, ny, dx, dy,
                          tau_xx_c, tau_yy_c, tau_xy_c, qx_c, qy_c, mu_c);
    mu_out[idx] = mu_c;

    // ===== Phase 2: 扩散步(跳过边界/固体/虚拟网格) =====
    if (cell_type[idx] == CELL_SOLID || cell_type[idx] == CELL_GHOST)
        return;
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
        return;

    // --- 重算 i+1 邻居粘性项 ---
    float tau_xx_ip1, tau_yy_ip1, tau_xy_ip1, qx_ip1, qy_ip1, mu_tmp;
    computeViscousAtPoint(u, v, T, cell_type, i + 1, j, nx, ny, dx, dy,
                          tau_xx_ip1, tau_yy_ip1, tau_xy_ip1, qx_ip1, qy_ip1, mu_tmp);

    // --- 重算 i-1 邻居粘性项 ---
    float tau_xx_im1, tau_yy_im1, tau_xy_im1, qx_im1, qy_im1;
    computeViscousAtPoint(u, v, T, cell_type, i - 1, j, nx, ny, dx, dy,
                          tau_xx_im1, tau_yy_im1, tau_xy_im1, qx_im1, qy_im1, mu_tmp);

    // --- 重算 j+1 邻居粘性项 ---
    float tau_xx_jp1, tau_yy_jp1, tau_xy_jp1, qx_jp1, qy_jp1;
    computeViscousAtPoint(u, v, T, cell_type, i, j + 1, nx, ny, dx, dy,
                          tau_xx_jp1, tau_yy_jp1, tau_xy_jp1, qx_jp1, qy_jp1, mu_tmp);

    // --- 重算 j-1 邻居粘性项 ---
    float tau_xx_jm1, tau_yy_jm1, tau_xy_jm1, qx_jm1, qy_jm1;
    computeViscousAtPoint(u, v, T, cell_type, i, j - 1, nx, ny, dx, dy,
                          tau_xx_jm1, tau_yy_jm1, tau_xy_jm1, qx_jm1, qy_jm1, mu_tmp);

    // ===== 应力张量散度(动量方程右端项) =====
    float dtau_xx_dx = (tau_xx_ip1 - tau_xx_im1) / (2.0f * dx);
    float dtau_xy_dy = (tau_xy_jp1 - tau_xy_jm1) / (2.0f * dy);
    float dtau_xy_dx = (tau_xy_ip1 - tau_xy_im1) / (2.0f * dx);
    float dtau_yy_dy = (tau_yy_jp1 - tau_yy_jm1) / (2.0f * dy);

    // ===== 热通量散度 =====
    float dqx_dx = (qx_ip1 - qx_im1) / (2.0f * dx);
    float dqy_dy = (qy_jp1 - qy_jm1) / (2.0f * dy);

    // ===== 速度梯度(中心单元，用于粘性耗散) =====
    int idx_im1 = j * nx + (i - 1);
    int idx_ip1 = j * nx + (i + 1);
    int idx_jm1 = (j - 1) * nx + i;
    int idx_jp1 = (j + 1) * nx + i;

    float du_dx = (u[idx_ip1] - u[idx_im1]) / (2.0f * dx);
    float du_dy = (u[idx_jp1] - u[idx_jm1]) / (2.0f * dy);
    float dv_dx = (v[idx_ip1] - v[idx_im1]) / (2.0f * dx);
    float dv_dy = (v[idx_jp1] - v[idx_jm1]) / (2.0f * dy);

    // ===== 粘性耗散 Phi = tau : grad(v) =====
    float Phi = tau_xx_c * du_dx + tau_yy_c * dv_dy + tau_xy_c * (du_dy + dv_dx);

    // ===== 动量右端项 =====
    float rhs_rho_u = dtau_xx_dx + dtau_xy_dy;
    float rhs_rho_v = dtau_xy_dx + dtau_yy_dy;

    // ===== 粘性应力做功项 div(tau·v) =====
    float work_x_ip1 = u[idx_ip1] * tau_xx_ip1 + v[idx_ip1] * tau_xy_ip1;
    float work_x_im1 = u[idx_im1] * tau_xx_im1 + v[idx_im1] * tau_xy_im1;
    float work_y_jp1 = u[idx_jp1] * tau_xy_jp1 + v[idx_jp1] * tau_yy_jp1;
    float work_y_jm1 = u[idx_jm1] * tau_xy_jm1 + v[idx_jm1] * tau_yy_jm1;

    float dwork_dx = (work_x_ip1 - work_x_im1) / (2.0f * dx);
    float dwork_dy = (work_y_jp1 - work_y_jm1) / (2.0f * dy);

    // ===== 总能量右端项 = -div(q) + div(tau·v) =====
    float rhs_E = -dqx_dx - dqy_dy + dwork_dx + dwork_dy;

    // ===== 内能右端项 = -div(q) + Phi =====
    float rhs_rho_e = -dqx_dx - dqy_dy + Phi;

    // ===== 欧拉前向时间积分 =====
    rho_u[idx] += dt * rhs_rho_u;
    rho_v[idx] += dt * rhs_rho_v;
    E[idx] += dt * rhs_E;
    rho_e[idx] += dt * rhs_rho_e;
}
#pragma endregion
#pragma endregion

#pragma region 并行计算任务发射函数
#pragma region 无条件初始化
// 功能:启动初始化核函数
// 说明:将所有网格初始化为来流条件
void CFDSolver::launchInitializeKernel(float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
                                       const SimParams &params, int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    initializeKernel<<<grid, block>>>(rho, rho_u, rho_v, E, rho_e,
                                      params.rho_inf, params.u_inf, params.v_inf, params.p_inf,
                                      nx, ny);
    CUDA_CHECK(cudaGetLastError());
}

// 功能:启动SDF计算核函数
// 说明:计算带符号距离场并初始化网格类型
void CFDSolver::launchComputeSDFKernel(float *sdf, uint8_t *cell_type,
                                       const SimParams &params, int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    computeSDFKernel<<<grid, block>>>(sdf, cell_type,
                                      params.obstacle_x, params.obstacle_y, params.obstacle_r,
                                      params.obstacle_rotation, params.obstacle_shape,
                                      params.dx, params.dy, nx, ny, params.wing_rotation);
    CUDA_CHECK(cudaGetLastError());
}
#pragma endregion

#pragma region 无粘性条件
// 功能:启动原始变量计算核函数
// 说明:从守恒变量计算原始变量(使用双能量法)
void CFDSolver::launchComputePrimitivesKernel(const float *rho, const float *rho_u,
                                              const float *rho_v, const float *E,
                                              const float *rho_e,
                                              float *u, float *v, float *p, float *T,
                                              int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    computePrimitivesKernel<<<grid, block>>>(rho, rho_u, rho_v, E, rho_e, u, v, p, T, nx, ny);
    CUDA_CHECK(cudaGetLastError());
}

// 功能:启动通量计算核函数
// 说明:使用MUSCL重构和HLLC Riemann求解器计算数值通量
//       仅需密度(rho)和原始变量(u,v,p)，不再传入未使用的守恒量指针
void CFDSolver::launchComputeFluxesKernel(const float *rho,
                                          const float *u, const float *v,
                                          const float *p,
                                          const uint8_t *cell_type,
                                          float *flux_rho_x, float *flux_rho_u_x,
                                          float *flux_rho_v_x, float *flux_E_x,
                                          float *flux_rho_e_x,
                                          float *flux_rho_y, float *flux_rho_u_y,
                                          float *flux_rho_v_y, float *flux_E_y,
                                          float *flux_rho_e_y,
                                          const SimParams &params, int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    computeFluxesKernel<<<grid, block>>>(rho, u, v, p, cell_type,
                                         flux_rho_x, flux_rho_u_x, flux_rho_v_x, flux_E_x, flux_rho_e_x,
                                         flux_rho_y, flux_rho_u_y, flux_rho_v_y, flux_E_y, flux_rho_e_y,
                                         params.dx, params.dy, nx, ny);
    CUDA_CHECK(cudaGetLastError());
}

// 功能:启动更新核函数
// 说明:使用有限体积法和双能量法更新守恒变量，复用已计算的原始变量(u,v,p)
void CFDSolver::launchUpdateKernel(const float *rho, const float *rho_u,
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
                                   const SimParams &params, int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    updateKernel<<<grid, block>>>(rho, rho_u, rho_v, E, rho_e,
                                  u, v, p,
                                  flux_rho_x, flux_rho_u_x, flux_rho_v_x, flux_E_x, flux_rho_e_x,
                                  flux_rho_y, flux_rho_u_y, flux_rho_v_y, flux_E_y, flux_rho_e_y,
                                  cell_type,
                                  rho_new, rho_u_new, rho_v_new, E_new, rho_e_new,
                                  params.dt, params.dx, params.dy, nx, ny);
    CUDA_CHECK(cudaGetLastError());
}

// 功能:启动边界条件应用核函数
// 说明:处理所有类型的边界条件(流入/流出/上下边界/固体壁面)
void CFDSolver::launchApplyBoundaryConditionsKernel(float *rho, float *rho_u, float *rho_v, float *E,
                                                    float *rho_e,
                                                    const uint8_t *cell_type, const float *sdf,
                                                    const SimParams &params, int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    applyBoundaryConditionsKernel<<<grid, block>>>(rho, rho_u, rho_v, E, rho_e,
                                                   cell_type, sdf,
                                                   params.rho_inf, params.u_inf, params.v_inf, params.p_inf,
                                                   params.dx, params.dy, nx, ny,
                                                   params.enable_viscosity, params.adiabatic_wall, params.T_wall);
    CUDA_CHECK(cudaGetLastError());
}
#pragma endregion

#pragma region 有粘性条件
// 功能:启动融合粘性-扩散核函数(替代 ViscousTerms + DiffusionStep 双核函数调用)
// 说明:每线程按需重算邻居粘性量，消除 tau/q 5个中间数组的全局内存往返
//       节省约 40-50% 的显存带宽
void CFDSolver::launchFusedViscousDiffusionKernel(
    const float *u, const float *v, const float *T,
    float *mu,
    float *rho_u, float *rho_v, float *E, float *rho_e,
    const uint8_t *cell_type,
    float dt, float dx, float dy, int nx, int ny)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    fusedViscousDiffusionKernel<<<grid, block>>>(u, v, T, mu,
                                                 rho_u, rho_v, E, rho_e,
                                                 cell_type,
                                                 dt, dx, dy, nx, ny);
    CUDA_CHECK(cudaGetLastError());
}
#pragma endregion

#pragma region 统计工具

// 核函数:计算运动粘性系数 nu = mu/rho
__global__ void computeViscousNumberKernel(const float *mu, const float *rho,
                                           float *nu_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        nu_out[i] = mu[i] / (rho[i] + 1e-10f);
    }
}

// 功能:计算最大运动粘性系数(用于粘性CFL条件)
// 优化:使用预分配缓冲区,无动态分配,无显式同步
float CFDSolver::launchComputeMaxViscousNumber(const float *mu, const float *rho, int nx, int ny)
{
    int n = nx * ny;

    // 使用预分配的临时缓冲区
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    computeViscousNumberKernel<<<grid, block>>>(mu, rho, d_scratch_, n);
    CUDA_CHECK(cudaGetLastError());

    // CUB归约(在默认流上,自动等待上面的kernel完成)
    void *d_temp = d_reduction_buffer_;
    size_t temp_bytes = reduction_buffer_size_;
    float *d_out = d_reduction_output_;

    CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, d_scratch_, d_out, n));

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

// 核函数:计算波速 |v| + c
__global__ void computeWaveSpeedKernel(const float *u, const float *v,
                                       const float *p, const float *rho,
                                       float *speed_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float c = sqrtf(GAMMA * p[i] / (rho[i] + 1e-10f));
        speed_out[i] = sqrtf(u[i] * u[i] + v[i] * v[i]) + c;
    }
}

// 功能:计算最大波速(用于对流CFL条件)
// 优化:使用预分配缓冲区,无动态分配,无显式同步
float CFDSolver::launchComputeMaxWaveSpeed(const float *u, const float *v, const float *p,
                                           const float *rho, int nx, int ny)
{
    int n = nx * ny;

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    computeWaveSpeedKernel<<<grid, block>>>(u, v, p, rho, d_scratch_, n);
    CUDA_CHECK(cudaGetLastError());

    void *d_temp = d_reduction_buffer_;
    size_t temp_bytes = reduction_buffer_size_;
    float *d_out = d_reduction_output_;

    CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, d_scratch_, d_out, n));

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

// 核函数:计算马赫数 |v| / c
__global__ void computeMachNumberKernel(const float *u, const float *v,
                                        const float *p, const float *rho,
                                        float *mach_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float rho_val = rho[i] + 1e-10f;
        float speed = sqrtf(u[i] * u[i] + v[i] * v[i]);
        float c = sqrtf(GAMMA * fmaxf(p[i], MIN_PRESSURE) / rho_val);
        mach_out[i] = speed / (c + 1e-10f);
    }
}

// 功能:计算最大马赫数
// 优化:使用预分配缓冲区,无动态分配,无显式同步
float CFDSolver::launchComputeMaxMach(const float *u, const float *v, const float *p,
                                      const float *rho, int nx, int ny)
{
    int n = nx * ny;

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    computeMachNumberKernel<<<grid, block>>>(u, v, p, rho, d_scratch_, n);
    CUDA_CHECK(cudaGetLastError());

    void *d_temp = d_reduction_buffer_;
    size_t temp_bytes = reduction_buffer_size_;
    float *d_out = d_reduction_output_;

    CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, d_scratch_, d_out, n));

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

// 功能:使用CUB库计算最大温度(直接对原始数组归约)
// 输入:温度场T，网格尺寸
// 输出:最大温度值
float CFDSolver::launchComputeMaxTemperature(const float *T, int nx, int ny)
{
    int n = nx * ny;

    // 缓冲区布局: 独立的临时存储和输出空间
    void *d_temp = d_reduction_buffer_;
    size_t temp_bytes = reduction_buffer_size_;
    float *d_out = d_reduction_output_;

    // 查询CUB所需临时存储大小
    size_t required_bytes = 0;
    cub::DeviceReduce::Max(nullptr, required_bytes, T, d_out, n);

    if (required_bytes > temp_bytes)
    {
        std::cerr << "Warning: CUB需要 " << required_bytes << " 字节，但只有 " << temp_bytes << " 字节可用" << std::endl;
        // 极端情况:预分配空间不足，回退到动态分配
        void *d_temp_alloc;
        CUDA_CHECK(cudaMalloc(&d_temp_alloc, required_bytes));
        CUDA_CHECK(cub::DeviceReduce::Max(d_temp_alloc, required_bytes, T, d_out, n));
        CUDA_CHECK(cudaFree(d_temp_alloc));
    }
    else
    {
        CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, T, d_out, n));
    }

    // 拷贝结果回主机
    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    return result;
}
#pragma endregion
#pragma endregion

#pragma region solver类的公开接口实现
#pragma region solver类的初始化
// 功能:求解器构造函数
// 说明:初始化为空状态，需要调用initialize()分配内存
CFDSolver::CFDSolver() {}

// 功能:求解器析构函数
// 说明:自动释放所有显存
CFDSolver::~CFDSolver()
{
    freeMemory();
}

// 功能:分配所有GPU显存
// 说明:根据网格尺寸 _nx, _ny 分配守恒变量、通量、辅助变量等数组
void CFDSolver::allocateMemory()
{
    if (_nx <= 0 || _ny <= 0)
        return;

    size_t size = _nx * _ny * sizeof(float);
    size_t size_byte = _nx * _ny * sizeof(uint8_t);

    // 1. 分配守恒变量(当前时间步)
    CUDA_CHECK(cudaMalloc(&d_rho_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_u_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_v_, size));
    CUDA_CHECK(cudaMalloc(&d_E_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_e_, size)); // 内能(双能量法)

    // 2. 分配守恒变量(下一时间步，用于双缓冲)
    CUDA_CHECK(cudaMalloc(&d_rho_new_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_u_new_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_v_new_, size));
    CUDA_CHECK(cudaMalloc(&d_E_new_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_e_new_, size));

    // 3. 分配原始变量
    CUDA_CHECK(cudaMalloc(&d_u_, size));
    CUDA_CHECK(cudaMalloc(&d_v_, size));
    CUDA_CHECK(cudaMalloc(&d_p_, size));
    CUDA_CHECK(cudaMalloc(&d_T_, size));

    // 4. 分配网格类型和SDF
    CUDA_CHECK(cudaMalloc(&d_cell_type_, size_byte));
    CUDA_CHECK(cudaMalloc(&d_sdf_, size));

    // 5. 分配通量数组
    CUDA_CHECK(cudaMalloc(&d_flux_rho_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_u_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_v_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_E_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_e_x_, size)); // 内能通量

    CUDA_CHECK(cudaMalloc(&d_flux_rho_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_u_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_v_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_E_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_e_y_, size));

    // 6. 动态计算并分配归约缓冲区（根据网格大小查询CUB所需空间）
    // 说明：CUB库需要临时存储空间进行多级归约，空间需求与元素数量n相关
    //       对于float类型的Max归约，所有归约操作（温度、马赫数、波速、粘性数）
    //       需要的临时空间大小相同，因此只需查询一次
    // 策略：分离临时存储和输出缓冲区，避免内存重叠
    int n = _nx * _ny;
    size_t temp_bytes = 0;

    // 创建临时指针用于查询（nullptr查询模式表示只查询不执行）
    float *dummy_in = nullptr;
    float *dummy_out = nullptr;

    // 查询CUB::DeviceReduce::Max所需的临时存储大小
    // 所有规约操作（温度/马赫数/波速/粘性数）都是对n个float元素求最大值
    // CUB的临时空间需求仅与元素数量和数据类型有关，因此一次查询即可覆盖所有情况
    cub::DeviceReduce::Max(nullptr, temp_bytes, dummy_in, dummy_out, n);

    // 分配临时存储空间（完全按需）
    reduction_buffer_size_ = temp_bytes;
    CUDA_CHECK(cudaMalloc(&d_reduction_buffer_, reduction_buffer_size_));

    // 分配独立的输出缓冲区（单个float）
    CUDA_CHECK(cudaMalloc(&d_reduction_output_, sizeof(float)));

    std::cout << "[CFD规约缓冲区] 网格=" << _nx << "x" << _ny << " (" << n << "元素), CUB临时存储="
              << reduction_buffer_size_ << "字节 (" << std::fixed << std::setprecision(2)
              << (reduction_buffer_size_ / 1024.0f) << "KB), 输出缓冲=4字节" << std::endl;

    // 6b. 分配归约用临时计算缓冲区（波速/马赫数/粘性数等中间结果）
    // 避免在热路径中反复 cudaMalloc/cudaFree
    CUDA_CHECK(cudaMalloc(&d_scratch_, size));

    // 7. 分配粘性相关数组(Navier-Stokes)
    CUDA_CHECK(cudaMalloc(&d_mu_, size)); // 动力粘性系数

    // 8. 分配原子计数器（用于矢量箭头生成，避免每帧cudaMalloc/cudaFree）
    CUDA_CHECK(cudaMalloc(&d_atomic_counter_, sizeof(int)));

    // 9. 分配锁页主机内存（用于computeStableTimeStep异步归约流水线）
    // 允许cudaMemcpyAsync将归约结果直接DMA到锁页内存，消除GPU空闲气泡
    CUDA_CHECK(cudaMallocHost(&h_pinnedReduction_, 2 * sizeof(float)));
    h_pinnedReduction_[0] = 0.0f;
    h_pinnedReduction_[1] = 0.0f;
}

// 功能:释放所有GPU显存
void CFDSolver::freeMemory()
{
    // 释放守恒变量
    if (d_rho_)
        cudaFree(d_rho_);
    if (d_rho_u_)
        cudaFree(d_rho_u_);
    if (d_rho_v_)
        cudaFree(d_rho_v_);
    if (d_E_)
        cudaFree(d_E_);
    if (d_rho_e_)
        cudaFree(d_rho_e_);

    // 释放双缓冲区
    if (d_rho_new_)
        cudaFree(d_rho_new_);
    if (d_rho_u_new_)
        cudaFree(d_rho_u_new_);
    if (d_rho_v_new_)
        cudaFree(d_rho_v_new_);
    if (d_E_new_)
        cudaFree(d_E_new_);
    if (d_rho_e_new_)
        cudaFree(d_rho_e_new_);

    // 释放原始变量
    if (d_u_)
        cudaFree(d_u_);
    if (d_v_)
        cudaFree(d_v_);
    if (d_p_)
        cudaFree(d_p_);
    if (d_T_)
        cudaFree(d_T_);

    // 释放辅助数据
    if (d_cell_type_)
        cudaFree(d_cell_type_);
    if (d_sdf_)
        cudaFree(d_sdf_);

    // 释放通量数组
    if (d_flux_rho_x_)
        cudaFree(d_flux_rho_x_);
    if (d_flux_rho_u_x_)
        cudaFree(d_flux_rho_u_x_);
    if (d_flux_rho_v_x_)
        cudaFree(d_flux_rho_v_x_);
    if (d_flux_E_x_)
        cudaFree(d_flux_E_x_);
    if (d_flux_rho_e_x_)
        cudaFree(d_flux_rho_e_x_);

    if (d_flux_rho_y_)
        cudaFree(d_flux_rho_y_);
    if (d_flux_rho_u_y_)
        cudaFree(d_flux_rho_u_y_);
    if (d_flux_rho_v_y_)
        cudaFree(d_flux_rho_v_y_);
    if (d_flux_E_y_)
        cudaFree(d_flux_E_y_);
    if (d_flux_rho_e_y_)
        cudaFree(d_flux_rho_e_y_);

    if (d_reduction_buffer_)
        cudaFree(d_reduction_buffer_);
    if (d_reduction_output_)
        cudaFree(d_reduction_output_);
    if (d_scratch_)
        cudaFree(d_scratch_);

    // 释放粘性相关数组
    if (d_mu_)
        cudaFree(d_mu_);

    // 释放原子计数器
    if (d_atomic_counter_)
        cudaFree(d_atomic_counter_);

    // 释放锁页主机内存
    if (h_pinnedReduction_)
        cudaFreeHost(h_pinnedReduction_);

    // 置空所有指针(防止重复释放)
    d_rho_ = d_rho_u_ = d_rho_v_ = d_E_ = d_rho_e_ = nullptr;
    d_rho_new_ = d_rho_u_new_ = d_rho_v_new_ = d_E_new_ = d_rho_e_new_ = nullptr;
    d_u_ = d_v_ = d_p_ = d_T_ = nullptr;
    d_cell_type_ = nullptr;
    d_sdf_ = nullptr;
    d_flux_rho_x_ = d_flux_rho_u_x_ = d_flux_rho_v_x_ = d_flux_E_x_ = d_flux_rho_e_x_ = nullptr;
    d_flux_rho_y_ = d_flux_rho_u_y_ = d_flux_rho_v_y_ = d_flux_E_y_ = d_flux_rho_e_y_ = nullptr;
    d_reduction_buffer_ = nullptr;
    d_reduction_output_ = nullptr;
    d_scratch_ = nullptr;
    reduction_buffer_size_ = 0; // 重置缓冲区大小
    d_atomic_counter_ = nullptr;
    h_pinnedReduction_ = nullptr;
}

// 功能:初始化求解器
// 输入:仿真参数 params(包含网格尺寸、来流条件等)
// 说明:设置网格尺寸，分配显存，调用reset初始化流场
void CFDSolver::initialize(const SimParams &params)
{
    _nx = params.nx;
    _ny = params.ny;

    freeMemory();     // 先释放旧内存
    allocateMemory(); // 分配新内存

    reset(params); // 初始化流场
}

// 功能:调整网格分辨率
// 输入:新的网格尺寸 nx, ny
// 说明:如果尺寸改变，重新分配显存
void CFDSolver::resize(int nx, int ny)
{
    if (nx == _nx && ny == _ny)
        return; // 尺寸未变，无需操作

    _nx = nx;
    _ny = ny;

    freeMemory();
    allocateMemory();
}

// 功能:重置流场到初始状态
// 输入:仿真参数 params
// 说明:将所有网格初始化为来流条件，计算SDF，应用边界条件
void CFDSolver::reset(const SimParams &params)
{
    // 1. 初始化为来流(包括内能)
    launchInitializeKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_, params, _nx, _ny);

    // 2. 计算SDF和网格类型
    launchComputeSDFKernel(d_sdf_, d_cell_type_, params, _nx, _ny);

    // 3. 应用边界条件
    launchApplyBoundaryConditionsKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                        d_cell_type_, d_sdf_, params, _nx, _ny);

    // 4. 计算初始原始变量(使用双能量法)
    launchComputePrimitivesKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                  d_u_, d_v_, d_p_, d_T_, _nx, _ny);

    // 5. 初始化粘性场(确保 computeStableTimeStep 首次调用时 d_mu_ 有效)
    //    使用融合核函数(dt=0): 仅计算并写出 mu，不更新守恒变量
    launchFusedViscousDiffusionKernel(d_u_, d_v_, d_T_,
                                      d_mu_,
                                      d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                      d_cell_type_,
                                      0.0f, params.dx, params.dy, _nx, _ny);

    // 同步确保完成
    CUDA_CHECK(cudaDeviceSynchronize());
}

// 功能:动态更新障碍物几何（不重置仿真状态）
// 输入:仿真参数 params（可包含位置、大小、旋转、形状、翼角度的任意变化）
// 说明:重新计算SDF和网格类型，对新暴露的流体区域填充真空，
//      原本在障碍物内部的流体区域守恒变量保持不变（边界条件内核会正确处理），
//      然后重新应用边界条件和计算原始变量，确保下一步仿真正常进行
void CFDSolver::updateObstacleGeometry(const SimParams &params)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((_nx + block.x - 1) / block.x, (_ny + block.y - 1) / block.y);

    // 1. 重新计算SDF和网格类型，同时修正新暴露区域的守恒变量
    updateSDFWithFixupKernel<<<grid, block>>>(
        d_sdf_, d_cell_type_,
        d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
        params.obstacle_x, params.obstacle_y, params.obstacle_r,
        params.obstacle_rotation, params.obstacle_shape,
        params.dx, params.dy, _nx, _ny, params.wing_rotation,
        params.rho_inf, params.u_inf, params.v_inf, params.p_inf);
    CUDA_CHECK(cudaGetLastError());

    // 2. 重新应用边界条件（Ghost Cell等需要使用新的SDF和cell_type）
    launchApplyBoundaryConditionsKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                        d_cell_type_, d_sdf_, params, _nx, _ny);

    // 3. 重新计算原始变量（确保 u, v, p, T 与新的守恒变量一致）
    launchComputePrimitivesKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                  d_u_, d_v_, d_p_, d_T_, _nx, _ny);

    // 注意：不再调用 cudaDeviceSynchronize()，后续操作（PBO映射或cudaMemcpy）
    // 在同一默认流上会隐式等待上述核函数完成
}
#pragma endregion

#pragma region solver类计算链顶层的计算函数
// 功能:将归约核函数+异步D→H回传排入GPU流，不阻塞CPU
// 设计:所有GPU工作背靠背排入流中，消除原来两次同步cudaMemcpy导致的GPU空闲气泡
//       调用方负责在读取结果前确保GPU已同步
void CFDSolver::queueTimeStepComputation(const SimParams &params)
{
    int n = _nx * _ny;
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    void *d_temp = d_reduction_buffer_;
    size_t temp_bytes = reduction_buffer_size_;

    // 1. 计算波速 |v|+c → d_scratch_
    computeWaveSpeedKernel<<<grid, block>>>(d_u_, d_v_, d_p_, d_rho_, d_scratch_, n);
    CUDA_CHECK(cudaGetLastError());

    // 2. CUB归约求最大波速 → d_reduction_output_
    CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, d_scratch_, d_reduction_output_, n));

    // 3. 异步拷贝到锁页内存（不阻塞CPU，由流顺序保证正确性）
    CUDA_CHECK(cudaMemcpyAsync(h_pinnedReduction_, d_reduction_output_,
                               sizeof(float), cudaMemcpyDeviceToHost, 0));

    if (params.enable_viscosity)
    {
        // 4. 计算粘性数 nu=mu/rho → d_scratch_
        //    安全：CUB已在同一流上读完d_scratch_，流顺序保证重写安全
        computeViscousNumberKernel<<<grid, block>>>(d_mu_, d_rho_, d_scratch_, n);
        CUDA_CHECK(cudaGetLastError());

        // 5. CUB归约求最大粘性数 → d_reduction_output_
        //    安全：步骤3的异步拷贝已在同一流上读完d_reduction_output_
        CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_bytes, d_scratch_, d_reduction_output_, n));

        // 6. 异步拷贝粘性数结果
        CUDA_CHECK(cudaMemcpyAsync(h_pinnedReduction_ + 1, d_reduction_output_,
                                   sizeof(float), cudaMemcpyDeviceToHost, 0));
    }
    // 不同步 — GPU继续执行，CPU立即返回
}

// 功能:从Pinned Memory读取已完成的异步归约结果，计算CFL时间步长
// 前置条件：queueTimeStepComputation()已调用且GPU已同步
float CFDSolver::readTimeStepResult(const SimParams &params) const
{
    float max_speed = h_pinnedReduction_[0];
    float min_dx = fminf(params.dx, params.dy);
    float dt_conv = params.cfl * min_dx / (max_speed + 1e-10f);
    float dt = dt_conv;

    if (params.enable_viscosity)
    {
        float max_nu = h_pinnedReduction_[1];
        float dt_visc = params.cfl_visc * min_dx * min_dx / (max_nu + 1e-10f);
        dt = fminf(dt_conv, dt_visc);
    }

    return dt;
}

// 功能:计算稳定时间步长（同步版本，用于首帧或重置后的一次性计算）
// 输入:仿真参数 params
// 输出:满足CFL条件的时间步长 dt
// 说明:同时考虑对流CFL和扩散CFL条件
float CFDSolver::computeStableTimeStep(const SimParams &params)
{
    queueTimeStepComputation(params);
    CUDA_CHECK(cudaDeviceSynchronize());
    return readTimeStepResult(params);
}

// 功能:执行单步时间推进
// 输入输出:仿真参数 params(会更新时间和步数)
// 说明:根据是否启用粘性选择不同的求解策略
void CFDSolver::step(SimParams &params)
{
    // 注意: 原始变量(d_u_, d_v_, d_p_, d_T_)在进入此函数时已由
    //       reset() 或上一步 step() 结尾的 computePrimitives 计算完毕，
    //       因此无需在函数开头重复计算

    if (params.enable_viscosity)
    {
        // ========== Navier-Stokes求解器(Strang算子分裂) ==========
        // Strang分裂将一个时间步分为三部分:
        // 1. 扩散半步(dt/2)
        // 2. 对流全步(dt)
        // 3. 扩散半步(dt/2)
        // 这样做二阶精度且保持稳定性

        // 步骤1: 扩散半步(使用已有的原始变量)
        // 融合核函数: 单次启动完成粘性项计算+扩散积分，消除中间数组的全局内存往返
        launchFusedViscousDiffusionKernel(d_u_, d_v_, d_T_,
                                          d_mu_,
                                          d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                          d_cell_type_,
                                          params.dt * 0.5f, params.dx, params.dy, _nx, _ny);

        // 步骤2: 对流全步(扩散半步修改了守恒变量，需重新计算原始变量)
        launchComputePrimitivesKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                      d_u_, d_v_, d_p_, d_T_, _nx, _ny);
        launchComputeFluxesKernel(d_rho_, d_u_, d_v_, d_p_, d_cell_type_,
                                  d_flux_rho_x_, d_flux_rho_u_x_, d_flux_rho_v_x_, d_flux_E_x_, d_flux_rho_e_x_,
                                  d_flux_rho_y_, d_flux_rho_u_y_, d_flux_rho_v_y_, d_flux_E_y_, d_flux_rho_e_y_,
                                  params, _nx, _ny);
        launchUpdateKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                           d_u_, d_v_, d_p_,
                           d_flux_rho_x_, d_flux_rho_u_x_, d_flux_rho_v_x_, d_flux_E_x_, d_flux_rho_e_x_,
                           d_flux_rho_y_, d_flux_rho_u_y_, d_flux_rho_v_y_, d_flux_E_y_, d_flux_rho_e_y_,
                           d_cell_type_,
                           d_rho_new_, d_rho_u_new_, d_rho_v_new_, d_E_new_, d_rho_e_new_,
                           params, _nx, _ny);
        // 交换指针(双缓冲)
        std::swap(d_rho_, d_rho_new_);
        std::swap(d_rho_u_, d_rho_u_new_);
        std::swap(d_rho_v_, d_rho_v_new_);
        std::swap(d_E_, d_E_new_);
        std::swap(d_rho_e_, d_rho_e_new_);

        // 步骤3: 扩散半步(对流步修改了守恒变量，需重新计算原始变量)
        launchComputePrimitivesKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                      d_u_, d_v_, d_p_, d_T_, _nx, _ny);
        // 融合核函数: 单次启动完成粘性项计算+扩散积分
        launchFusedViscousDiffusionKernel(d_u_, d_v_, d_T_,
                                          d_mu_,
                                          d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                          d_cell_type_,
                                          params.dt * 0.5f, params.dx, params.dy, _nx, _ny);
    }
    else
    {
        // ========== 原始欧拉求解器(无粘性) ==========
        // 原始变量已由上一步结尾计算完毕，直接使用

        // 1. 计算通量(包括内能通量)
        launchComputeFluxesKernel(d_rho_, d_u_, d_v_, d_p_, d_cell_type_,
                                  d_flux_rho_x_, d_flux_rho_u_x_, d_flux_rho_v_x_, d_flux_E_x_, d_flux_rho_e_x_,
                                  d_flux_rho_y_, d_flux_rho_u_y_, d_flux_rho_v_y_, d_flux_E_y_, d_flux_rho_e_y_,
                                  params, _nx, _ny);

        // 2. 更新守恒变量(包括内能)
        launchUpdateKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                           d_u_, d_v_, d_p_,
                           d_flux_rho_x_, d_flux_rho_u_x_, d_flux_rho_v_x_, d_flux_E_x_, d_flux_rho_e_x_,
                           d_flux_rho_y_, d_flux_rho_u_y_, d_flux_rho_v_y_, d_flux_E_y_, d_flux_rho_e_y_,
                           d_cell_type_,
                           d_rho_new_, d_rho_u_new_, d_rho_v_new_, d_E_new_, d_rho_e_new_,
                           params, _nx, _ny);

        // 3. 交换缓冲区指针(双缓冲技术，避免数据拷贝)
        std::swap(d_rho_, d_rho_new_);
        std::swap(d_rho_u_, d_rho_u_new_);
        std::swap(d_rho_v_, d_rho_v_new_);
        std::swap(d_E_, d_E_new_);
        std::swap(d_rho_e_, d_rho_e_new_);
    }

    // 应用边界条件
    launchApplyBoundaryConditionsKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                        d_cell_type_, d_sdf_, params, _nx, _ny);

    // 计算原始变量：为下一次 computeStableTimeStep 和可视化准备最新的原始变量
    // 这样 computeStableTimeStep 和 step() 之间不再有冗余的 computePrimitives 调用
    launchComputePrimitivesKernel(d_rho_, d_rho_u_, d_rho_v_, d_E_, d_rho_e_,
                                  d_u_, d_v_, d_p_, d_T_, _nx, _ny);

    // 更新时间和步数
    params.t_current += params.dt;
    params.step++;
}
#pragma endregion

#pragma region 非零拷贝模式下的统计工具
// 功能:获取最大温度(使用CUB库的归约)
float CFDSolver::getMaxTemperature()
{
    return launchComputeMaxTemperature(d_T_, _nx, _ny);
}

// 功能:获取最大马赫数
float CFDSolver::getMaxMach()
{
    return launchComputeMaxMach(d_u_, d_v_, d_p_, d_rho_, _nx, _ny);
}

// 功能:获取网格类型
void CFDSolver::getCellTypes(uint8_t *host_types)
{
    CUDA_CHECK(cudaMemcpy(host_types, d_cell_type_, _nx * _ny * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

// 功能:在GPU上将uint8网格类型转换为float并写入设备指针（用于CUDA-GL互操作零拷贝更新）
// 输入:目标设备指针（已映射的PBO地址），大小至少 _nx*_ny*sizeof(float)
// 说明:消除了原来 getCellTypes(D→H) + CPU uint8→float转换 + glTexImage2D 的CPU瓶颈
static __global__ void cellTypeToFloatKernel(const uint8_t *__restrict__ cellType,
                                             float *__restrict__ output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = static_cast<float>(cellType[idx]);
}

void CFDSolver::convertCellTypesToDevice(float *devOutput)
{
    int n = _nx * _ny;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    cellTypeToFloatKernel<<<gridSize, blockSize>>>(d_cell_type_, devOutput, n);
    CUDA_CHECK(cudaGetLastError());
}
#pragma endregion

#pragma region 内存占用统计
// 静态函数:获取GPU内存信息
void CFDSolver::getGPUMemoryInfo(size_t &freeMem, size_t &totalMem)
{
    cudaMemGetInfo(&freeMem, &totalMem);
}

// 获取本程序占用的显存
size_t CFDSolver::getSimulationMemoryUsage()
{
    size_t totalMemory = 0;
    size_t floatSize = sizeof(float);
    size_t uint8Size = sizeof(uint8_t);
    size_t gridSize = (size_t)_nx * _ny;

    // 守恒变量（当前时间步）- 5个数组
    totalMemory += gridSize * floatSize * 5; // rho, rho_u, rho_v, E, rho_e

    // 守恒变量（下一时间步）- 5个数组
    totalMemory += gridSize * floatSize * 5; // rho_new, rho_u_new, rho_v_new, E_new, rho_e_new

    // 原始变量 - 4个数组
    totalMemory += gridSize * floatSize * 4; // u, v, p, T

    // 网格类型和SDF - 1个uint8数组 + 1个float数组
    totalMemory += gridSize * uint8Size; // cell_type
    totalMemory += gridSize * floatSize; // sdf

    // 通量数组 - X方向5个 + Y方向5个
    totalMemory += gridSize * floatSize * 10; // flux_rho_x, flux_rho_u_x, flux_rho_v_x, flux_E_x, flux_rho_e_x
                                              // flux_rho_y, flux_rho_u_y, flux_rho_v_y, flux_E_y, flux_rho_e_y

    // 归约缓冲区（动态分配，根据网格大小自动计算）
    totalMemory += reduction_buffer_size_;

    // 归约用临时计算缓冲区
    totalMemory += gridSize * floatSize; // d_scratch_

    // 粘性相关数组 - 6个数组
    totalMemory += gridSize * floatSize * 6; // mu, tau_xx, tau_yy, tau_xy, qx, qy

    // 原子计数器
    totalMemory += sizeof(int); // d_atomic_counter_

    return totalMemory;
}
#pragma endregion
#pragma endregion

#pragma region cuda-OpenGL互操作
// CUDA-OpenGL 互操作实现 - 零拷贝核函数
// 从保守变量直接计算温度到目标指针（避免中间拷贝）
__global__ void computeTemperatureDirectKernel(
    const float *rho, const float *rho_e, float *T_out, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;
    float rho_val = rho[idx];
    float rho_e_val = rho_e[idx];

    // 理想气体状态方程: p = (gamma - 1) * rho * e
    // T = p / (rho * R) = (gamma - 1) * e / R
    float e = rho_e_val / (rho_val + 1e-10f);
    T_out[idx] = (GAMMA - 1.0f) * e / R_GAS;
}

// 从保守变量直接计算压强到目标指针
__global__ void computePressureDirectKernel(
    const float *rho, const float *rho_e, float *p_out, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;
    float rho_e_val = rho_e[idx];

    // p = (gamma - 1) * rho * e = (gamma - 1) * rho_e
    p_out[idx] = (GAMMA - 1.0f) * rho_e_val;
}

// 从保守变量直接计算速度大小和马赫数到目标指针
__global__ void computeVelocityMagDirectKernel(
    const float *rho, const float *rho_u, const float *rho_v,
    float *vmag_out, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;
    float rho_val = rho[idx] + 1e-10f;
    float u = rho_u[idx] / rho_val;
    float v = rho_v[idx] / rho_val;

    vmag_out[idx] = sqrtf(u * u + v * v);
}

// 从保守变量直接计算马赫数到目标指针
__global__ void computeMachDirectKernel(
    const float *rho, const float *rho_u, const float *rho_v,
    const float *rho_e, float *mach_out, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;
    float rho_val = rho[idx] + 1e-10f;
    float u = rho_u[idx] / rho_val;
    float v = rho_v[idx] / rho_val;
    float speed = sqrtf(u * u + v * v);

    // 计算温度和声速
    float e = rho_e[idx] / rho_val;
    float T = (GAMMA - 1.0f) * e / R_GAS;
    float c = sqrtf(GAMMA * R_GAS * T);

    mach_out[idx] = speed / (c + 1e-10f);
}

// 直接计算温度到目标指针
void CFDSolver::computeTemperatureToDevice(float *dev_dst)
{
    dim3 block(16, 16);
    dim3 grid((_nx + block.x - 1) / block.x, (_ny + block.y - 1) / block.y);
    computeTemperatureDirectKernel<<<grid, block>>>(d_rho_, d_rho_e_, dev_dst, _nx, _ny);
    CUDA_CHECK(cudaGetLastError());
}

// 直接计算压强到目标指针
void CFDSolver::computePressureToDevice(float *dev_dst)
{
    dim3 block(16, 16);
    dim3 grid((_nx + block.x - 1) / block.x, (_ny + block.y - 1) / block.y);
    computePressureDirectKernel<<<grid, block>>>(d_rho_, d_rho_e_, dev_dst, _nx, _ny);
    CUDA_CHECK(cudaGetLastError());
}

// 直接计算密度到目标指针
void CFDSolver::computeDensityToDevice(float *dev_dst)
{
    CUDA_CHECK(cudaMemcpy(dev_dst, d_rho_, _nx * _ny * sizeof(float), cudaMemcpyDeviceToDevice));
}

// 直接计算速度大小到目标指针
void CFDSolver::computeVelocityMagToDevice(float *dev_dst)
{
    dim3 block(16, 16);
    dim3 grid((_nx + block.x - 1) / block.x, (_ny + block.y - 1) / block.y);
    computeVelocityMagDirectKernel<<<grid, block>>>(d_rho_, d_rho_u_, d_rho_v_, dev_dst, _nx, _ny);
    CUDA_CHECK(cudaGetLastError());
}

// 直接计算马赫数到目标指针
void CFDSolver::computeMachToDevice(float *dev_dst)
{
    dim3 block(16, 16);
    dim3 grid((_nx + block.x - 1) / block.x, (_ny + block.y - 1) / block.y);
    computeMachDirectKernel<<<grid, block>>>(d_rho_, d_rho_u_, d_rho_v_, d_rho_e_, dev_dst, _nx, _ny);
    CUDA_CHECK(cudaGetLastError());
}
#pragma endregion

#pragma region 矢量箭头生成（CUDA-OpenGL互操作）
// CUDA kernel：GPU并行生成速度矢量箭头的顶点数据
// 输入：速度场 u, v；网格尺寸 nx, ny；箭头参数
// 输出：直接写入映射的OpenGL VBO
__global__ void generateVectorArrowsKernel(
    const float *u, const float *v,
    float *vertexData,
    int *atomicCounter,
    int nx, int ny,
    int step,
    float u_inf,
    float maxArrowLength,
    float arrowHeadAngle,
    float arrowHeadLength)
{
    // 计算当前线程对应的网格位置
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 应用步长偏移，使箭头从网格中心位置开始
    i = i * step + step / 2;
    j = j * step + step / 2;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // 获取该点的速度分量
    float u_val = u[idx];
    float v_val = v[idx];

    // 计算速度大小
    float speed = sqrtf(u_val * u_val + v_val * v_val);
    if (speed < 1e-6f * u_inf)
        return; // 跳过速度接近零的点

    // 归一化速度
    float normalizedSpeed = fminf(speed / (u_inf * 1.5f), 1.0f);

    // 计算箭头起点（NDC坐标）
    float startX = (float)i / nx * 2.0f - 1.0f;
    float startY = (float)j / ny * 2.0f - 1.0f;

    // 计算箭头方向和长度
    float dirX = u_val / speed;
    float dirY = v_val / speed;
    float arrowLength = maxArrowLength * normalizedSpeed;

    // 箭头终点
    float endX = startX + dirX * arrowLength;
    float endY = startY + dirY * arrowLength;

    // 使用黑色箭头
    float r = 0.0f, g = 0.0f, b = 0.0f;

    // 计算箭头头部的两个点
    float headLen = arrowLength * arrowHeadLength;
    float cosA = cosf(arrowHeadAngle);
    float sinA = sinf(arrowHeadAngle);

    // 旋转箭头方向得到头部两个边
    float head1X = endX - headLen * (dirX * cosA - dirY * sinA);
    float head1Y = endY - headLen * (dirX * sinA + dirY * cosA);
    float head2X = endX - headLen * (dirX * cosA + dirY * sinA);
    float head2Y = endY - headLen * (-dirX * sinA + dirY * cosA);

    // 原子地获取顶点写入位置（每个箭头需要8个顶点：箭身2个 + 头部4个）
    int vertexOffset = atomicAdd(atomicCounter, 8);

    // 写入顶点数据（格式：x, y, r, g, b）
    int baseIdx = vertexOffset * 5;

    // 箭身线段
    vertexData[baseIdx + 0] = startX;
    vertexData[baseIdx + 1] = startY;
    vertexData[baseIdx + 2] = r;
    vertexData[baseIdx + 3] = g;
    vertexData[baseIdx + 4] = b;

    vertexData[baseIdx + 5] = endX;
    vertexData[baseIdx + 6] = endY;
    vertexData[baseIdx + 7] = r;
    vertexData[baseIdx + 8] = g;
    vertexData[baseIdx + 9] = b;

    // 箭头头部第一条线段
    vertexData[baseIdx + 10] = endX;
    vertexData[baseIdx + 11] = endY;
    vertexData[baseIdx + 12] = r;
    vertexData[baseIdx + 13] = g;
    vertexData[baseIdx + 14] = b;

    vertexData[baseIdx + 15] = head1X;
    vertexData[baseIdx + 16] = head1Y;
    vertexData[baseIdx + 17] = r;
    vertexData[baseIdx + 18] = g;
    vertexData[baseIdx + 19] = b;

    // 箭头头部第二条线段
    vertexData[baseIdx + 20] = endX;
    vertexData[baseIdx + 21] = endY;
    vertexData[baseIdx + 22] = r;
    vertexData[baseIdx + 23] = g;
    vertexData[baseIdx + 24] = b;

    vertexData[baseIdx + 25] = head2X;
    vertexData[baseIdx + 26] = head2Y;
    vertexData[baseIdx + 27] = r;
    vertexData[baseIdx + 28] = g;
    vertexData[baseIdx + 29] = b;
}

// 生成速度矢量箭头，直接写入OpenGL VBO
int CFDSolver::generateVectorArrows(float *dev_vertexData, int maxVertices,
                                    int step, float u_inf,
                                    float maxArrowLength, float arrowHeadAngle, float arrowHeadLength)
{
    // 使用预分配的原子计数器（避免每帧cudaMalloc/cudaFree）
    int *d_counter = d_atomic_counter_;
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // 计算线程块和网格尺寸
    // 每个线程负责一个可能的箭头位置
    int numArrowsX = (_nx + step - 1) / step;
    int numArrowsY = (_ny + step - 1) / step;

    dim3 block(16, 16);
    dim3 grid((numArrowsX + block.x - 1) / block.x,
              (numArrowsY + block.y - 1) / block.y);

    // 启动kernel
    generateVectorArrowsKernel<<<grid, block>>>(
        d_u_, d_v_,
        dev_vertexData,
        d_counter,
        _nx, _ny,
        step,
        u_inf,
        maxArrowLength,
        arrowHeadAngle,
        arrowHeadLength);

    CUDA_CHECK(cudaGetLastError());

    // 获取实际生成的顶点数量
    int h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    return h_counter; // 返回生成的顶点数量
}
#pragma endregion