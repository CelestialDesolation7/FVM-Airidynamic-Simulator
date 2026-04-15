cmake_minimum_required(VERSION 3.21)

# =========================================================
# 项目配置
# =========================================================

# 启用详细的构建输出
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "Verbose build output")
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)

message(STATUS "========================================")
message(STATUS "  FVM Aerodynamic Simulator 构建配置")
message(STATUS "========================================")
message(STATUS "CMake 版本: ${CMAKE_VERSION}")
message(STATUS "系统: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")
message(STATUS "处理器: ${CMAKE_SYSTEM_PROCESSOR}")

# =========================================================
# Vcpkg 工具链设置
# =========================================================

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # 尝试自动检测 vcpkg 路径
    set(VCPKG_CANDIDATES
        "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        "D:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    )
    
    foreach(VCPKG_PATH ${VCPKG_CANDIDATES})
        if(EXISTS "${VCPKG_PATH}")
            set(CMAKE_TOOLCHAIN_FILE "${VCPKG_PATH}" CACHE STRING "Vcpkg toolchain file")
            message(STATUS "✓ 找到 vcpkg 工具链: ${VCPKG_PATH}")
            break()
        endif()
    endforeach()
    
    if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
        message(WARNING "⚠ 未找到 vcpkg 工具链，请设置环境变量 VCPKG_ROOT 或通过 -DCMAKE_TOOLCHAIN_FILE 指定")
    endif()
else()
    message(STATUS "✓ 使用指定的 vcpkg 工具链: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# =========================================================
# 项目声明
# =========================================================

project(FVM-Aerodynamic-Simulator 
    VERSION 1.0.0
    DESCRIPTION "Real-time GPU-accelerated aerodynamic simulator using FVM"
    LANGUAGES CXX CUDA
)


# =========================================================
# 语言标准设置
# =========================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "C++ 标准: C++${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA 标准: C++${CMAKE_CUDA_STANDARD}")

# =========================================================
# GPU 架构设置
# =========================================================

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89+PTX" CACHE STRING "CUDA 目标架构")
endif()

message(STATUS "CUDA 目标架构: ${CMAKE_CUDA_ARCHITECTURES}")

# =========================================================
# CUDA 运行时链接设置
# =========================================================

# 静态链接 CUDA 运行时库
# 优点：用户无需安装 CUDA Toolkit，只需要兼容的显卡驱动
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
message(STATUS "CUDA 运行时库: 静态链接 (Static)")

# =========================================================
# 查找依赖库
# =========================================================

message(STATUS "========================================")
message(STATUS "  查找依赖库")
message(STATUS "========================================")

# OpenGL
find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
    message(STATUS "✓ OpenGL: ${OPENGL_LIBRARIES}")
else()
    message(FATAL_ERROR "✗ OpenGL 未找到")
endif()

# GLAD (OpenGL 加载器)
find_package(glad CONFIG REQUIRED)
message(STATUS "✓ GLAD: ${glad_VERSION}")

# GLFW (窗口管理)
find_package(glfw3 CONFIG REQUIRED)
message(STATUS "✓ GLFW3: ${glfw3_VERSION}")

# CUDA Toolkit (自动检测系统中的 CUDA)
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "✓ CUDA Toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "  - CUDA 安装路径: ${CUDAToolkit_TARGET_DIR}")
    message(STATUS "  - CUDA 编译器: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  - CUDA 编译器版本: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "  - CUDA 包含目录: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "  - CUDA 库目录: ${CUDAToolkit_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "✗ CUDA Toolkit 未找到，请安装 NVIDIA CUDA Toolkit 12.0+")
endif()

# =========================================================
# 串流功能（可选）
# =========================================================

option(ENABLE_STREAMING "Enable LAN browser streaming via WebRTC + NVENC" ON)

if(ENABLE_STREAMING)
    message(STATUS "========================================")
    message(STATUS "  串流功能: 已启用")
    message(STATUS "========================================")

    find_package(LibDataChannel CONFIG REQUIRED)
    message(STATUS "✓ libdatachannel: found")

    find_package(httplib CONFIG REQUIRED)
    message(STATUS "✓ cpp-httplib: found")

    find_path(FFNVCODEC_INCLUDE_DIRS "ffnvcodec/nvEncodeAPI.h")
    if(FFNVCODEC_INCLUDE_DIRS)
        message(STATUS "✓ ffnvcodec (NVENC headers): ${FFNVCODEC_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "✗ ffnvcodec 未找到，请通过 vcpkg install ffnvcodec 安装")
    endif()
else()
    message(STATUS "串流功能: 未启用 (使用 -DENABLE_STREAMING=ON 启用)")
endif()

# =========================================================
# 配置源文件
# =========================================================

message(STATUS "========================================")
message(STATUS "  配置源文件")
message(STATUS "========================================")

# ImGui 源文件
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/imgui)
if(NOT EXISTS ${IMGUI_DIR})
    message(FATAL_ERROR "✗ ImGui 目录不存在: ${IMGUI_DIR}")
endif()

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
message(STATUS "✓ ImGui 源文件: ${IMGUI_DIR}")

# 项目源文件（自动收集 src 目录下的 .cpp 和 .cu 文件）
file(GLOB PROJECT_CPP_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB PROJECT_CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)
set(PROJECT_SOURCES ${PROJECT_CPP_SOURCES} ${PROJECT_CUDA_SOURCES})

if(ENABLE_STREAMING)
    file(GLOB STREAMING_SOURCES ${CMAKE_SOURCE_DIR}/src/streaming/*.cpp)
    list(APPEND PROJECT_SOURCES ${STREAMING_SOURCES})
    list(LENGTH STREAMING_SOURCES STREAM_COUNT)
    message(STATUS "✓ 找到 ${STREAM_COUNT} 个串流模块源文件")
endif()

list(LENGTH PROJECT_CPP_SOURCES CPP_COUNT)
list(LENGTH PROJECT_CUDA_SOURCES CUDA_COUNT)
message(STATUS "✓ 找到 ${CPP_COUNT} 个 C++ 源文件")
message(STATUS "✓ 找到 ${CUDA_COUNT} 个 CUDA 源文件")

# =========================================================
# 创建可执行文件
# =========================================================

message(STATUS "========================================")
message(STATUS "  创建目标")
message(STATUS "========================================")

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${IMGUI_SOURCES})
message(STATUS "✓ 可执行文件目标: ${PROJECT_NAME}")

# =========================================================
# 配置包含目录
# =========================================================

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
)

if(ENABLE_STREAMING)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/streaming
        ${FFNVCODEC_INCLUDE_DIRS}
    )
endif()

message(STATUS "包含目录:")
message(STATUS "  - ${CMAKE_SOURCE_DIR}/src")
message(STATUS "  - ${CMAKE_SOURCE_DIR}/src/include")
message(STATUS "  - ${IMGUI_DIR}")
message(STATUS "  - ${CUDAToolkit_INCLUDE_DIRS}")

# =========================================================
# 链接库
# =========================================================

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    glad::glad
    glfw
    # CUDA::cudart_static 会根据 CMAKE_CUDA_RUNTIME_LIBRARY 自动链接
)

if(ENABLE_STREAMING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_STREAMING)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        LibDataChannel::LibDataChannel
        httplib::httplib
        CUDA::cuda_driver
    )
    message(STATUS "✓ 串流依赖库: libdatachannel, cpp-httplib, CUDA Driver API")
endif()

message(STATUS "链接库:")
message(STATUS "  - OpenGL::GL")
message(STATUS "  - glad::glad")
message(STATUS "  - glfw")
message(STATUS "  - CUDA 静态运行时 (自动)")

# =========================================================
# 目标属性
# =========================================================

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "CUDA 编译选项:")
message(STATUS "  - 分离编译: 启用")
message(STATUS "  - 设备符号解析: 启用")

# =========================================================
# 编译选项
# =========================================================

# CUDA 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math                    # 启用快速数学运算
        --expt-relaxed-constexpr           # 放宽 constexpr 限制
        -Xcompiler=/utf-8                  # MSVC UTF-8 支持
        
        # 真正用于日常断点找 Bug 的 Debug 模式 (慢，但能查错)
        $<$<CONFIG:Debug>:
            -O0                            # 禁用优化
            -G                             # 开启设备端调试信息
            -lineinfo                      # 行信息
        >
        
        $<$<CONFIG:Release>:
            -O3                            # 最高优化级别
        >
        
        $<$<CONFIG:RelWithDebInfo>:
            -O3
            -lineinfo                      # 在 Release 中保留行号供 Nsight 分析
        >
    >
)

# C++ 编译选项
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            /utf-8                         # UTF-8 字符集
            /W3                            # 警告级别 3
            
            # C++ 的 Debug 和 Release 区分
            $<$<CONFIG:Debug>:/Od /Zi>     # Debug: 无优化 + 调试信息 (系统默认带 /RTC1)
            $<$<CONFIG:Release>:/O2 /Ob2 /Zi> # Release: 优化 + 调试信息库(PDB)
        >
    )
endif()

message(STATUS "编译选项:")
message(STATUS "  - CUDA: --use_fast_math, UTF-8 支持")
message(STATUS "  - C++: UTF-8 支持, 警告级别 3")

# =========================================================
# 核函数性能基准测试目标
# =========================================================

add_executable(KernelBenchmark
    ${CMAKE_SOURCE_DIR}/src/test/kernel_benchmark.cu
    ${CMAKE_SOURCE_DIR}/src/solver.cu
)

target_include_directories(KernelBenchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(KernelBenchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(KernelBenchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --expt-relaxed-constexpr
        -Xcompiler=/utf-8
        -Xcompiler=/Zc:preprocessor
        $<$<CONFIG:Release>:-O3 -lineinfo>
        $<$<CONFIG:RelWithDebInfo>:-O3 -lineinfo>
    >
)

if(MSVC)
    target_compile_options(KernelBenchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            /utf-8
            /Zc:preprocessor
            /W1
        >
    )
endif()

if(MSVC)
    target_link_options(KernelBenchmark PRIVATE
        "LINKER:/NODEFAULTLIB:LIBCMT"
        "LINKER:/NODEFAULTLIB:LIBCMTD"
    )
endif()

message(STATUS "✓ 核函数基准测试目标: KernelBenchmark (无OpenGL依赖)")

# =========================================================
# MSVC 特定配置
# =========================================================

if(MSVC)
    # 解决静态 CUDA 运行时与 vcpkg 动态运行时的冲突
    # 使用 LINKER: 前缀确保选项只传递给链接器，而不是编译器
    target_link_options(${PROJECT_NAME} PRIVATE 
        "LINKER:/NODEFAULTLIB:LIBCMT"
        "LINKER:/NODEFAULTLIB:LIBCMTD"
    )
    message(STATUS "MSVC 链接器选项: 忽略默认 LIBCMT(D)")
endif()

# =========================================================
# 构建后处理
# =========================================================

message(STATUS "========================================")
message(STATUS "  配置构建后处理")
message(STATUS "========================================")

# 1. 复制 assets 资源目录
if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ">> 复制 assets 目录..."
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
        COMMENT "复制资源文件到输出目录"
    )
    message(STATUS "✓ 将在构建后复制 assets 目录")
else()
    message(WARNING "⚠ assets 目录不存在: ${CMAKE_SOURCE_DIR}/assets")
endif()

# 1b. 串流功能：复制 web 客户端目录
if(ENABLE_STREAMING)
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/web)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo ">> 复制 web 目录..."
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/src/web
                $<TARGET_FILE_DIR:${PROJECT_NAME}>/web
            COMMENT "复制 Web 客户端文件到输出目录"
        )
        message(STATUS "✓ 将在构建后复制 web 目录")
    endif()
endif()

# 2. Windows 平台：复制依赖 DLL
if(WIN32)
    # 复制 vcpkg 管理的依赖库 DLL（GLFW, GLAD 等）
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ">> 复制 vcpkg 依赖 DLL..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS
        COMMENT "复制运行时依赖库"
    )
    message(STATUS "✓ 将在构建后复制 vcpkg 依赖 DLL")
    
    # 注意：静态链接 CUDA 运行时，无需复制 cudart DLL
    message(STATUS "ℹ CUDA 运行时已静态链接，无需复制 DLL")
endif()

# =========================================================
# 安装配置（用于创建发布包）
# =========================================================

message(STATUS "========================================")
message(STATUS "  配置安装规则")
message(STATUS "========================================")

include(GNUInstallDirs)

# 设置默认安装路径为 dist 目录
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/dist" CACHE PATH "安装路径" FORCE)
endif()

message(STATUS "安装前缀: ${CMAKE_INSTALL_PREFIX}")

# 1. 安装可执行文件
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME DESTINATION .
    COMPONENT application
)

# 2. 安装依赖库 DLL（Windows）
if(WIN32)
    install(FILES $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
        DESTINATION .
        COMPONENT dependencies
    )
    message(STATUS "✓ 将安装运行时依赖库")
endif()

# 3. 安装资源文件
if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/assets/
        DESTINATION assets
        COMPONENT assets
    )
    message(STATUS "✓ 将安装 assets 资源文件")
endif()

# 4. 串流功能：安装 Web 客户端
if(ENABLE_STREAMING AND EXISTS ${CMAKE_SOURCE_DIR}/src/web)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/web/
        DESTINATION web
        COMPONENT streaming
    )
    message(STATUS "✓ 将安装 web 客户端文件")
endif()

# =========================================================
# 构建配置总结
# =========================================================

message(STATUS "========================================")
message(STATUS "  构建配置总结")
message(STATUS "========================================")
message(STATUS "项目: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ 编译器: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA 编译器: ${CMAKE_CUDA_COMPILER} (v${CMAKE_CUDA_COMPILER_VERSION})")
message(STATUS "输出目录: ${CMAKE_BINARY_DIR}")
message(STATUS "安装路径: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "提示：")
message(STATUS "  - 构建：cmake --build . --config Release")
message(STATUS "  - 安装：cmake --install . --config Release")
message(STATUS "  - 运行：cd Release && ./${PROJECT_NAME}.exe")
message(STATUS "========================================")