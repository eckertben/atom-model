#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include "ray.h"

struct Camera {
    glm::vec3 origin;
    glm::vec3 lower_left_corner;
    glm::vec3 horizontal;
    glm::vec3 vertical;

    Camera(glm::vec3 position, glm::vec3 target, glm::vec3 up,
           float vfov_deg, float aspect_ratio)
    {
        // Vertical fov degrees -> rads 
        float theta      = glm::radians(vfov_deg);
        // Viewport height fov/2 -> height/2 at distance 1
        float h          = glm::tan(theta * 0.5f);
        float vp_height  = 2.0f * h;
        // Viewport width = height * aspect ratio
        float vp_width   = aspect_ratio * vp_height;

        glm::vec3 w = glm::normalize(position - target);
        glm::vec3 u = glm::normalize(glm::cross(up, w));
        glm::vec3 v = glm::cross(w, u);

        origin = position;
        horizontal = vp_width  * u;
        vertical = vp_height * v;
        lower_left_corner = origin 
                            - horizontal * 0.5f 
                            - vertical   * 0.5f 
                            - w;
    }

    Ray get_ray(float u, float v) const {
        glm::vec3 target = lower_left_corner
                         + u * horizontal
                         + v * vertical;
        return Ray{ origin, glm::normalize(target - origin) };
    }
};