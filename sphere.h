#pragma once
#include <glm/glm.hpp>

struct Sphere {
    glm::vec3 center;
    float radius;
    float probability;

    // Ray-sphere intersection test: returns t of hit or -1 if no hit
    float intersect(const Ray& ray) const {
        // Origin to center vector
        glm::vec3 oc = ray.origin - center;

        // Quadratic coefficients from |P(t) - C|^2 = r^2 = 
        // (O + tD - C)·(O + tD - C) = r^2 expands to:
        // a = D·D, b = 2*oc·D, c = oc·oc - r^2
        float a = glm::dot(ray.direction, ray.direction);
        float b = 2.0f * glm::dot(oc, ray.direction);
        float c = glm::dot(oc, oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;

        // No intersection if no real roots
        if (discriminant < 0) return -1.0f;

        // Smallest root within acceptable range
        float sqrt_d = glm::sqrt(discriminant);
        float t0 = (-b - sqrt_d) / (2.0f * a);
        if (t0 > 0.001f) return t0;

        // Check second root if first is too small
        float t1 = (-b + sqrt_d) / (2.0f * a);
        if (t1 > 0.001f) return t1;

        // If both are too small to be acceptable
        return -1.0f;
    }
};