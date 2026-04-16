#pragma once
// Thin wrapper: includes the official NVIDIA Video Codec SDK header (from ffnvcodec)
// and provides a runtime DLL loader for NvEncodeAPICreateInstance.

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <ffnvcodec/nvEncodeAPI.h>

#include <cstdint>

#ifdef _WIN32
#define NVENC_DLL "nvEncodeAPI64.dll"
#else
#include <dlfcn.h>
#define NVENC_DLL "libnvidia-encode.so.1"
#endif

typedef NVENCSTATUS(NVENCAPI *NvEncodeAPICreateInstance_t)(NV_ENCODE_API_FUNCTION_LIST *);
typedef NVENCSTATUS(NVENCAPI *NvEncodeAPIGetMaxSupportedVersion_t)(uint32_t *);

inline bool loadNvEncApi(NV_ENCODE_API_FUNCTION_LIST &fnList) {
    memset(&fnList, 0, sizeof(fnList));
    fnList.version = NV_ENCODE_API_FUNCTION_LIST_VER;

#ifdef _WIN32
    HMODULE hModule = LoadLibraryA(NVENC_DLL);
    if (!hModule) return false;

    auto getMaxVer = (NvEncodeAPIGetMaxSupportedVersion_t)GetProcAddress(hModule, "NvEncodeAPIGetMaxSupportedVersion");
    if (getMaxVer) {
        uint32_t maxVer = 0;
        if (getMaxVer(&maxVer) == NV_ENC_SUCCESS) {
            uint32_t driverMajor = maxVer >> 0 & 0xFF;
            uint32_t driverMinor = maxVer >> 8 & 0xFF;
            if (driverMajor < NVENCAPI_MAJOR_VERSION ||
                (driverMajor == NVENCAPI_MAJOR_VERSION && driverMinor < NVENCAPI_MINOR_VERSION)) {
                return false;
            }
        }
    }

    auto createInstance = (NvEncodeAPICreateInstance_t)GetProcAddress(hModule, "NvEncodeAPICreateInstance");
#else
    void *lib = dlopen(NVENC_DLL, RTLD_LAZY);
    if (!lib) return false;
    auto createInstance = (NvEncodeAPICreateInstance_t)dlsym(lib, "NvEncodeAPICreateInstance");
#endif

    if (!createInstance) return false;
    return createInstance(&fnList) == NV_ENC_SUCCESS;
}
