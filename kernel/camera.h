#ifndef CAMERA_H
#define CAMERA_H

typedef struct {
    float4 origin;
    float4 lower_left_corner;
    float4 horizontal;
    float4 vertical;
} Camera;

Ray get_ray(__constant Camera* cam, float u, float v) {
    float3 dir = cam->lower_left_corner.xyz
               + u * cam->horizontal.xyz
               + v * cam->vertical.xyz
               - cam->origin.xyz;
    Ray r;
    r.origin    = cam->origin;
    r.direction = (float4)(normalize(dir), 0.0f);
    return r;
}

#endif