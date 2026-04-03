#pragma once
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <glm/glm.hpp>
#include "../cpu/camera.h"
#include "../cpu/sphere.h"
#include "../cpu/light.h"

struct CameraGPU {
    cl_float4 origin;
    cl_float4 lower_left_corner;
    cl_float4 horizontal;
    cl_float4 vertical;
};

struct SphereGPU {
    cl_float4 center;
    cl_float  radius;
    cl_float  probability;
    cl_float  pad[2];
};

struct LightGPU {
    cl_float4 position;
    cl_float4 color;
    cl_float intensity;
};

inline CameraGPU to_gpu(const Camera& cam) {
    return CameraGPU{
        { cam.origin.x, cam.origin.y, cam.origin.z, 0.0f },
        { cam.lower_left_corner.x, cam.lower_left_corner.y, cam.lower_left_corner.z, 0.0f },
        { cam.horizontal.x, cam.horizontal.y, cam.horizontal.z, 0.0f },
        { cam.vertical.x, cam.vertical.y, cam.vertical.z, 0.0f }
    };
}

inline SphereGPU to_gpu(const Sphere& s) {
    return SphereGPU{
        { s.center.x, s.center.y, s.center.z, 0.0f },
        s.radius,
        s.probability,
        { 0.0f, 0.0f }
    };
}

inline LightGPU to_gpu(const Light& l) {
    return LightGPU{
        { l.position.x, l.position.y, l.position.z, 0.0f },
        { l.color.x, l.color.y, l.color.z, 0.0f },
        l.intensity
    };
}