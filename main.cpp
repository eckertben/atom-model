#include <OpenCL/opencl.h>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <vector>

#include <fstream>
#include <sstream>
#include <string>

#include <cstdint>
#include <cstdio>

#include "shared/bridge.h"
#include "cpu/camera.h"
#include "cpu/sphere.h"
#include "cpu/light.h"
#include "cpu/wavefunction.h"

constexpr int TILE_SIZE = 64;

constexpr int WIDTH  = 1280;
constexpr int HEIGHT = 720;

constexpr int NUM_CANDIDATES = 400000; // Number of random points to sample for electron cloud

static void save_ppm(const std::vector<uint32_t>& pixels) {
    std::ofstream f("output.ppm", std::ios::binary);
    f << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (uint32_t p : pixels) {
        uint8_t rgb[3] = {
            static_cast<uint8_t>((p >> 16) & 0xFF),
            static_cast<uint8_t>((p >>  8) & 0xFF),
            static_cast<uint8_t>( p        & 0xFF)
        };
        f.write(reinterpret_cast<char*>(rgb), 3);
    }
    std::puts("Saved output.ppm");
}

// Helper function that builds a program using multiple
// filenames and returns the cl_program obj.
cl_program create_program_from_sources(cl_context context, cl_device_id device, 
                                        const std::vector<std::string>& sources) {
    // concatenate all files into one string
    std::string combined;
    for (const auto& path : sources) {
        std::ifstream ifs(path);
        if (!ifs) {
            fprintf(stderr, "Failed to open: %s\n", path.c_str());
            return nullptr;
        }
        std::stringstream buffer;
        buffer << ifs.rdbuf();
        combined += buffer.str();
        combined += "\n";
    }

    const char* src = combined.c_str();
    size_t len      = combined.size();

    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program: %d\n", err);
        return nullptr;
    }

    char build_opts[64];
    snprintf(build_opts, sizeof(build_opts), "-DTILE_SIZE=64");
    err = clBuildProgram(program, 1, &device, build_opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, ' ');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
        fprintf(stderr, "Build error: %d\nLog:\n%s\n", err, log.c_str());
        return nullptr;
    }
    return program;
}

// Main application loop: setup SDL, render once, present texture, handle input, cleanup.
int main(int argc, char* argv[]) {
    // Capture quantum numbers from command line
    int n = std::stoi(argv[1]); 
    int l = std::stoi(argv[2]);
    int m = std::stoi(argv[3]);
    if (argc >= 4) {
        printf("n=%d, l=%d, m=%d\n", n, l, m);
    }
    
    // Initialize SDL and create window, renderer,
    // and texture for the scene.
    // Basic initialization with error handling.
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "Atom Ray Tracer",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIDTH, HEIGHT,
        SDL_WINDOW_SHOWN
    );
    if (!window) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(
        window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!renderer) {
        SDL_Log("SDL_CreateRenderer failed: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        WIDTH, HEIGHT
    );
    if (!texture) {
        SDL_Log("SDL_CreateTexture failed: %s", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Gets the first available OpenCL platform (GPU)
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // Gets a GPU device from platform
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Create context, could pass NULLPTR as properties
    // if just using MacOS
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
    };
    cl_context context = clCreateContext(properties, 1, &device, nullptr, nullptr, nullptr);

    // This is for OpenCL 1.2 (macOS standard)
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, nullptr);

    // Load program forom source files and build
    std::vector<std::string> sources = {
        "kernel/ray.h",
        "kernel/camera.h",
        "kernel/sphere.h",
        "kernel/light.h",
        "kernel/render.cl"
    };

    cl_program program = create_program_from_sources(context, device, sources);

    // Create kernel from the built program
    cl_kernel kernel = clCreateKernel(program, "render", nullptr);

    // allocate GPU pixel buffer
    cl_mem gpu_pixels = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    WIDTH * HEIGHT * sizeof(uint32_t), nullptr, nullptr);
    // Link the 'gpu_pixel_buffer' to the 0th argument of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_pixels);
    int width = WIDTH; int height = HEIGHT;
    clSetKernelArg(kernel, 1, sizeof(int), &width);
    clSetKernelArg(kernel, 2, sizeof(int), &height);
    Camera cam(
        glm::vec3(0.0f, 0.0f, 5.0f * (float)n), // position
        glm::vec3(0.0f, 0.0f, 0.0f), // target
        glm::vec3(0.0f, 1.0f, 0.0f), // up
        45.0f,                        // vertical fov in degrees
        (float)WIDTH / (float)HEIGHT  // aspect ratio
    );
    CameraGPU cam_gpu = to_gpu(cam);
    // Copies the camera data to a GPU buffer
    cl_mem d_cam = clCreateBuffer(context, 
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(CameraGPU), &cam_gpu, nullptr);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_cam);
    
    // Initialized the wavefunction and samples points for the electron cloud
    Wavefunction wf(n, l, m);
    // Sample points and evaluate the wavefunction
    float psi_max = 0.0f;
    std::vector<std::pair<glm::vec3, float>> candidates;
    candidates.reserve(NUM_CANDIDATES);
    float bound = 4.0f * n * n;
    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        // Sample points in a cube around the origin
        glm::vec3 p = glm::vec3(
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * bound,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * bound,
            (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * bound
        );
        float prob = wf.evaluate(p);
        candidates.push_back({p, prob});
        if (prob > psi_max) psi_max = prob;
    }
    //float gamma = 0.2f;
    // Monte Carlo sampling to select points based on probabolity
    std::vector<std::pair<glm::vec3, float>> points;
    for (const auto& c : candidates) {
        if ((static_cast<float>(rand()) / RAND_MAX) < c.second / psi_max) {
            float normalized = c.second / psi_max;
            //float visual_prob = glm::pow(normalized, gamma);
            points.push_back({c.first, normalized});
        }
    }
    std::vector<Sphere> spheres;
    spheres.push_back({glm::vec3(0.0f, 0.0f, 0.0f), 0.08f, 1.0f}); // nucleus
    for (const auto& p : points) {
        spheres.push_back({p.first, 0.08f, std::pow(p.second, 0.2f)}); // Points in electron cloud
    }
    printf("Sampled %zu points for electron cloud\n", points.size());
    printf("acceptance rate: %.2f%%\n", 100.0f * points.size() / NUM_CANDIDATES);
    printf("bound: %f\n", bound);
    printf("psi_max: %f\n", psi_max);
    std::vector<SphereGPU> spheres_gpu;
    for (const auto& s : spheres) {
        spheres_gpu.push_back(to_gpu(s));
    }

    cl_mem d_spheres = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(SphereGPU) * spheres_gpu.size(),
                                    spheres_gpu.data(), nullptr);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_spheres);
    int num_spheres = spheres_gpu.size();
    clSetKernelArg(kernel, 5, sizeof(int), &num_spheres);

    Light light{ glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(1.0f), 1.0f };
    LightGPU light_gpu = to_gpu(light);
    cl_mem d_light = clCreateBuffer(context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(LightGPU), &light_gpu, nullptr);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_light);
    clSetKernelArg(kernel, 7, TILE_SIZE * sizeof(SphereGPU), nullptr);

    

    // Launch the kernel
    size_t tile = 8;  // 16x16 = 256 = TILE_SIZE
    size_t global[2] = {
        ((WIDTH  + tile - 1) / tile) * tile,
        ((HEIGHT + tile - 1) / tile) * tile
    };
    size_t local[2] = { tile, tile };
    clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, 
                            global, local, 0, nullptr, nullptr);
    //printf("Kernel launch error code: %d\n", err1);

    
    // Read back rendered pixels from GPU to CPU
    std::vector<uint32_t> pixels(WIDTH * HEIGHT);
    clEnqueueReadBuffer(queue, gpu_pixels, CL_TRUE, 0, 
                            WIDTH * HEIGHT * sizeof(uint32_t), pixels.data(), 
                            0, nullptr, nullptr);

    bool running = true;
    SDL_Event event;
    bool changed = false;

    float yaw = 0.0f;
    float pitch = 0.0f;
    float radius = glm::length(cam.origin);
    bool is_dragging = false;

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE: running = false; break;
                    case SDLK_p:      save_ppm(pixels); break;
                    default: break;
                }
            }
            // Start dragging
            else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) is_dragging = true;
            } 
            // Stop dragging
            else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) is_dragging = false;
            } 
            // Move camera while dragging
            else if (event.type == SDL_MOUSEMOTION && is_dragging) {
                // 1. Update angles (tune 0.005f for speed)
                yaw   -= event.motion.xrel * 0.005f;
                pitch += event.motion.yrel * 0.005f;

                // 2. Clamp pitch to prevent flipping at the poles
                pitch = glm::clamp(pitch, -glm::half_pi<float>() + 0.1f, glm::half_pi<float>() - 0.1f);

                // 3. Calculate new position on the sphere
                glm::vec3 new_pos;
                new_pos.x = radius * cos(pitch) * sin(yaw);
                new_pos.y = radius * sin(pitch);
                new_pos.z = radius * cos(pitch) * cos(yaw);

                // 4. Rebuild the camera looking at (0,0,0)
                // Use the same VFOV and Aspect Ratio as your initial setup
                cam = Camera(new_pos, glm::vec3(0,0,0), glm::vec3(0,1,0), 45.0f, float(WIDTH)/HEIGHT);
                
                changed = true;
            }
            else if (event.type == SDL_MOUSEWHEEL) {
                // Check if Ctrl is held (common way trackpads send pinch-to-zoom)
                bool is_ctrl_held = (SDL_GetModState() & KMOD_CTRL);
                
                if (is_ctrl_held) {
                    // Pinching: adjust radius (zoom)
                    radius -= event.wheel.y * 0.2f; 
                } else {
                    // Standard scroll: could be used for something else or radius as well
                    radius -= event.wheel.y * 0.5f;
                }

                if (radius < 0.1f) radius = 0.1f;
                
                // Recalculate camera position
                glm::vec3 new_pos;
                new_pos.x = radius * cos(pitch) * sin(yaw);
                new_pos.y = radius * sin(pitch);
                new_pos.z = radius * cos(pitch) * cos(yaw);
                
                cam = Camera(new_pos, glm::vec3(0,0,0), glm::vec3(0,1,0), 45.0f, float(WIDTH)/HEIGHT);
                changed = true;
            }

        }

        if (changed) {
            cam_gpu = to_gpu(cam);
            clEnqueueWriteBuffer(queue, d_cam, CL_TRUE, 0, sizeof(CameraGPU), 
                                &cam_gpu, 0, nullptr, nullptr);
            
            cl_event render_event;
            clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, 
                                    global, local, 0, nullptr, &render_event);
            
            cl_event read_event;
            clEnqueueReadBuffer(queue, gpu_pixels, CL_TRUE, 0, 
                                WIDTH * HEIGHT * sizeof(uint32_t), pixels.data(), 
                                0, nullptr, &read_event);
            
            clFinish(queue);  // wait for everything before querying
            
            cl_ulong start, end;
            clGetEventProfilingInfo(render_event, CL_PROFILING_COMMAND_START, 
                                    sizeof(cl_ulong), &start, nullptr);
            clGetEventProfilingInfo(render_event, CL_PROFILING_COMMAND_END,   
                                    sizeof(cl_ulong), &end,   nullptr);
            //printf("render: %.2f ms\n", (end - start) / 1e6f);
            clReleaseEvent(render_event);

            clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, 
                                    sizeof(cl_ulong), &start, nullptr);
            clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END,   
                                    sizeof(cl_ulong), &end,   nullptr);
            //printf("readback: %.2f ms\n", (end - start) / 1e6f);
            clReleaseEvent(read_event);

            changed = false;
        }

        SDL_UpdateTexture(texture, nullptr, pixels.data(), WIDTH * sizeof(uint32_t));
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}