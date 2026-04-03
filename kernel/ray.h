#ifndef RAY_H
#define RAY_H

typedef struct {
    float4 origin;
    float4 direction;
} Ray;

static float4 at(Ray r, float t) {
    return r.origin + t * r.direction;
}

#endif



