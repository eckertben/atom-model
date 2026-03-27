#pragma once
#include <glm/glm.hpp>

struct Light {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;

    Light(const glm::vec3& pos, const glm::vec3& col, float inten)
        : position(pos), color(col), intensity(inten) {}
};