// Convert a normalized floating-point rgb value to a packed ARGB pixel.
// Clamps each component to [0,1], scales to [0,255], and packs into 0xAARRGGBB format.
static uint vec3_to_argb(float3 c) {
    uchar r = (uchar)(clamp(c.x, 0.0f, 1.0f) * 255.0f);
    uchar g = (uchar)(clamp(c.y, 0.0f, 1.0f) * 255.0f);
    uchar b = (uchar)(clamp(c.z, 0.0f, 1.0f) * 255.0f);
    return (0xFFu << 24) | (r << 16) | (g << 8) | b;
}

static int get_local_linear_id() {
    return get_local_id(1) * get_local_size(0) + get_local_id(0);
}

static void process_tile(const Ray ray, __local Sphere* tile, int tile_count, float* closest_t, int* hit_idx, int offset) {
    // Find closest hit in tile group and return the distance, probability, and index (global)
    for (int i = 0; i < tile_count; i++) {
        float temp_t = intersect(tile[i], ray);
        if (temp_t > 0.0f && (*closest_t < 0.0f || temp_t < *closest_t)) {
            *closest_t = temp_t;
            *hit_idx = offset + i;
        }
    }
}

static float3 shade(const Ray ray, float3 color, float4 normal, __constant Light* light, float4 light_dir) {
    float diffuse = max(dot(normal, light_dir), 0.0f); // ABS of portion that hits surface
    float3 shading = (float3)(light->color.x, light->color.y, light->color.z) 
                        * diffuse * light->intensity;
    return color * shading;
}

// Render the current scene into pixel buffer by casting primary rays through each pixel.
__kernel void render(__global uint* pixels, int width, int height, __constant Camera* cam, __constant Sphere* spheres, int num_spheres, __constant Light* light, __local Sphere* tile) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int local_id = get_local_linear_id();

    float u = (x + 0.5f) / (float)(width);
    float v = 1.0f - (y + 0.5f) / (float)(height);
    Ray ray = get_ray(cam, u, v);

    float t = -1.0f;
    int hit_idx = -1;

    for (int i = 0; i < num_spheres; i += TILE_SIZE) {
        // Each thread (as long as there are spheres left) is utilized
        if (i + local_id < num_spheres) {
            tile[local_id] = spheres[i + local_id];
        }

        // Wait until all spheres are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Pass to pixel color func
        if (x < width && y < height) {
            int current_tile_size = min(TILE_SIZE, num_spheres - i);
            process_tile(ray, tile, current_tile_size, &t, &hit_idx, i);
        }

        // Wait until all spheres are processed
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float3 final_color;
    if (t > 0 && hit_idx != -1) {
        Sphere hit_sphere = spheres[hit_idx];

        float3 color = mix((float3)(0.0f, 0.0f, 1.0f),  // blue
                            (float3)(1.0f, 0.0f, 0.0f),  // red
                            hit_sphere.probability);

        // Calculate hit point and normal for shading
        float4 hit_point = at(ray, t);
        float4 normal = normalize(hit_point - hit_sphere.center);
        float4 light_direction = normalize(light->position - hit_point);
        final_color = shade(ray, color, normal, light, light_direction);
    } else {
        final_color = (float3)(0.0f, 0.0f, 0.0f);
    }

    pixels[y * width + x] = vec3_to_argb(final_color);
}

