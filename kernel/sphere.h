#ifndef SPHERE_H
#define SPHERE_H

typedef struct {
    float4 center;
    float radius;
    float probability;
} Sphere;

float intersect_sphere(Sphere sphere, Ray ray) {
    // Origin to center vector
    float4 oc = ray.origin - sphere.center;

    // Quadratic coefficients from |P(t) - C|^2 = r^2 = 
    // (O + tD - C)·(O + tD - C) = r^2 expands to:
    // a = D·D, b = 2*oc·D, c = oc·oc - r^2
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;

    // No intersection if no real roots
    if (discriminant < 0) return -1.0f;

    // Smallest root within acceptable range
    float sqrt_d = sqrt(discriminant);
    float t0 = (-b - sqrt_d) / (2.0f * a);
    if (t0 > 0.001f) return t0;

    // Check second root if first is too small
    float t1 = (-b + sqrt_d) / (2.0f * a);
    if (t1 > 0.001f) return t1;

    // If both are too small to be acceptable
    return -1.0f;
}

float intersect_axis(Ray ray, float3 axis_dir, float radius, float max_len) {
    float3 d = ray.direction.xyz;
    float3 o = ray.origin.xyz;
    
    // Project ray onto the plane perpendicular to the axis
    float3 d_proj = d - dot(d, axis_dir) * axis_dir;
    float3 o_proj = o - dot(o, axis_dir) * axis_dir;

    float a = dot(d_proj, d_proj);
    float b = 2.0f * dot(o_proj, d_proj);
    float c = dot(o_proj, o_proj) - (radius * radius);

    // If a is near zero, the ray is parallel to the axis
    if (fabs(a) < 0.0001f) return -1.0f;

    float disc = b * b - 4.0f * a * c;
    if (disc < 0) return -1.0f;

    float t = (-b - sqrt(disc)) / (2.0f * a);
    
    // Capping the length: Check if the hit point is within [0, max_len]
    float3 hit_point = o + t * d;
    float projection = dot(hit_point, axis_dir);
    
    if (projection > max_len) return -1.0f;

    return t;
}


#endif