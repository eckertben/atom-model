#ifndef SPHERE_H
#define SPHERE_H

typedef struct {
    float4 center;
    float radius;
    float probability;
} Sphere;

float intersect(Sphere sphere, Ray ray) {
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

#endif