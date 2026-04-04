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

static void process_tile(const Ray ray, __local float4* tile, int tile_count, float* closest_t, int* hit_idx, int offset, float radius) {
    for (int i = 0; i < tile_count; i++) {
        float4 data = tile[i];
        // Create a temporary sphere using the xyz as center and w as probability
        Sphere s;
        s.center = (float4)(data.xyz, 0.0f); 
        s.radius = radius;
        s.probability = data.w;

        float temp_t = intersect_sphere(s, ray);
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
__kernel void render(__global uint* pixels, int width, int height, __constant Camera* cam, __global float4* points, int num_points, float sphere_radius, __constant Light* light, __local float4* tile, __global int* visible) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int local_id = get_local_linear_id();

    float u = (x + 0.5f) / (float)(width);
    float v = 1.0f - (y + 0.5f) / (float)(height);
    Ray ray = get_ray(cam, u, v);

    float t = -1.0f;
    int hit_idx = -1;

    float axis_t = -1.0f;
    int axis_id = -1; // 0=X, 1=Y, 2=Z

    float tx = intersect_axis(ray, (float3)(1,0,0), 0.02f, 15.0f);
    float ty = intersect_axis(ray, (float3)(0,1,0), 0.02f, 15.0f);
    float tz = intersect_axis(ray, (float3)(0,0,1), 0.02f, 15.0f);

    // Check which axis is closest
    if (tx > 0 && (axis_t < 0 || tx < axis_t)) { axis_t = tx; axis_id = 0; }
    if (ty > 0 && (axis_t < 0 || ty < axis_t)) { axis_t = ty; axis_id = 1; }
    if (tz > 0 && (axis_t < 0 || tz < axis_t)) { axis_t = tz; axis_id = 2; }

    for (int i = 0; i < num_points; i += TILE_SIZE) {
        // Each thread (as long as there are points left) is utilized
        if (i + local_id < num_points && visible[i + local_id]) {
            tile[local_id] = points[i + local_id];
        } else {
            tile[local_id] = (float4)(0.0f, 0.0f, 0.0f, -1.0f);  // invalid filler point
        }

        // Wait until all points are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Pass to pixel color func
        if (x < width && y < height) {
            int current_tile_size = min(TILE_SIZE, num_points - i);
            process_tile(ray, tile, current_tile_size, &t, &hit_idx, i, sphere_radius);
        }

        // Wait until all spheres are processed
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float3 final_color;
    if (axis_t > 0 && (t < 0 || axis_t < t)) {
        if (axis_id == 0) final_color = (float3)(1, 0, 0); // Red X
        if (axis_id == 1) final_color = (float3)(0, 1, 0); // Green Y
        if (axis_id == 2) final_color = (float3)(0, 0, 1); // Blue Z
    } else if (t > 0 && hit_idx != -1) {
        float4 hit = points[hit_idx];

        float3 color = mix((float3)(0.0f, 0.0f, 1.0f),  // blue
                            (float3)(1.0f, 0.0f, 0.0f),  // red
                            hit.w);

        // Calculate hit point and normal for shading
        float4 hit_point = at(ray, t);
        float4 normal = normalize(hit_point - (float4)(hit.xyz, 0.0));
        float4 light_direction = normalize(light->position - hit_point);
        final_color = shade(ray, color, normal, light, light_direction);
    } else {
        final_color = (float3)(0.0f, 0.0f, 0.0f);
    }

    pixels[y * width + x] = vec3_to_argb(final_color);
}

__kernel void update(__global float4* base, __global float4* points, int num_points, float t,
                float energy, int m, float scale) {
    int i = get_global_id(0);
    if (i >= num_points) return;
    if (i == 0) return; // skips nucleus

    int bi = (i - 1) * 2;
    float psi = base[bi+1].y;
    float phi = base[bi+1].x;
    float prob = points[i].w;
    float3 pos  = base[bi].xyz;
    float3 dir_radial = normalize(pos);

    // tangential direction — perpendicular to radial and z-axis
    float3 z_axis = (float3)(0.0f, 0.0f, 1.0f);
    float3 dir_tangent = normalize(cross(z_axis, dir_radial));

    float phase = (float)m * phi - energy * t;
    float disp  = psi * cos(phase) * scale;

    float3 new_pos = pos + dir_tangent * disp;
    points[i] = (float4)(new_pos, prob); 
}

