#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Wrap CUDA calls so errors fail early with file and line information

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(err__) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__)             \
            );                                                                  \
        }                                                                       \
    } while (0)

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
};

struct Point {
    float x = 0.0f;
    float y = 0.0f;
};

class CpuTimer {
public:
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double stopMilliseconds() const {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = endTime - startTime;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point startTime;
};

// Stores CUDA timing results for memory transfers and each Lloyd stage

struct CudaTimingBreakdown {
    float h2dTimeMs = 0.0f;
    float d2hTimeMs = 0.0f;

    float totalLloydTimeMs = 0.0f;
    float totalMemsetTimeMs = 0.0f;
    float totalAssignKernelTimeMs = 0.0f;
    float totalUpdateKernelTimeMs = 0.0f;
};

Image loadImage(const std::string& path) {
    Image image;

    const int requestedChannels = 3;
    unsigned char* data = stbi_load(
        path.c_str(),
        &image.width,
        &image.height,
        &image.channels,
        requestedChannels
    );

    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    image.channels = requestedChannels;
    const int totalBytes = image.width * image.height * image.channels;
    image.pixels.assign(data, data + totalBytes);

    stbi_image_free(data);
    return image;
}

void saveRGBImage(
    const std::string& path,
    int width,
    int height,
    const std::vector<unsigned char>& pixels
) {
    const int channels = 3;
    const int strideBytes = width * channels;

    int success = stbi_write_png(
        path.c_str(),
        width,
        height,
        channels,
        pixels.data(),
        strideBytes
    );

    if (!success) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

std::vector<float> computeBrightness(const Image& image) {
    std::vector<float> brightness(image.width * image.height, 0.0f);

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            int pixelIndex = y * image.width + x;
            int byteIndex = pixelIndex * image.channels;

            float r = static_cast<float>(image.pixels[byteIndex + 0]) / 255.0f;
            float g = static_cast<float>(image.pixels[byteIndex + 1]) / 255.0f;
            float b = static_cast<float>(image.pixels[byteIndex + 2]) / 255.0f;

            brightness[pixelIndex] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    return brightness;
}

std::vector<float> computeDensity(const std::vector<float>& brightness, float gamma) {
    std::vector<float> density(brightness.size(), 0.0f);

    for (size_t i = 0; i < brightness.size(); ++i) {
        float value = 1.0f - brightness[i];
        value = std::clamp(value, 0.0f, 1.0f);
        density[i] = std::pow(value, gamma);
    }

    return density;
}

std::vector<Point> initializePointsDensityAware(
    const std::vector<float>& density,
    int width,
    int height,
    int numberOfPoints,
    unsigned int seed = 12345
) {
    std::vector<Point> points;
    points.reserve(numberOfPoints);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> randomX(0, width - 1);
    std::uniform_int_distribution<int> randomY(0, height - 1);
    std::uniform_real_distribution<float> random01(0.0f, 1.0f);

    const int maxAttempts = numberOfPoints * 10000;
    int attempts = 0;

    while (static_cast<int>(points.size()) < numberOfPoints && attempts < maxAttempts) {
        ++attempts;

        int x = randomX(rng);
        int y = randomY(rng);
        int index = y * width + x;

        float acceptProbability = density[index];
        float r = random01(rng);

        if (r <= acceptProbability) {
            points.push_back(Point{static_cast<float>(x), static_cast<float>(y)});
        }
    }


    while (static_cast<int>(points.size()) < numberOfPoints) {
        int x = randomX(rng);
        int y = randomY(rng);
        points.push_back(Point{static_cast<float>(x), static_cast<float>(y)});
    }

    return points;
}

// One thread handles one pixel.
// Each pixel finds its closest point and adds its weighted contribution.

__global__ void assignPixelsAndAccumulateKernel(
    const float* density,
    int width,
    int height,
    const Point* points,
    int numberOfPoints,
    float* sumX,
    float* sumY,
    float* sumW
) {
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int numberOfPixels = width * height;

    // Ignore extra threads outside the image.
    if (pixelIndex >= numberOfPixels) {
        return;
    }

    float weight = density[pixelIndex];
    // Skip pixels with almost no density contribution.
    if (weight <= 1e-6f) {
        return;
    }

    int x = pixelIndex % width;
    int y = pixelIndex / width;

    float xf = static_cast<float>(x);
    float yf = static_cast<float>(y);

    int bestIndex = 0;
    float bestDistanceSquared = FLT_MAX;

    // Search all points and keep the closest one.
    for (int i = 0; i < numberOfPoints; ++i) {
        float dx = xf - points[i].x;
        float dy = yf - points[i].y;
        float distanceSquared = dx * dx + dy * dy;

        if (distanceSquared < bestDistanceSquared) {
            bestDistanceSquared = distanceSquared;
            bestIndex = i;
        }
    }

    // Multiple pixels may update the same point, so these writes must be atomic.
    atomicAdd(&sumX[bestIndex], weight * xf);
    atomicAdd(&sumY[bestIndex], weight * yf);
    atomicAdd(&sumW[bestIndex], weight);
}

// Update each point from its accumulated weighted centroid.
__global__ void updatePointsKernel(
    Point* points,
    const float* sumX,
    const float* sumY,
    const float* sumW,
    int numberOfPoints
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numberOfPoints) {
        return;
    }

    float w = sumW[i];

    // Points with no assigned weight keep their previous position.
    if (w > 1e-12f) {
        points[i].x = sumX[i] / w;
        points[i].y = sumY[i] / w;
    }
}

// Run the weighted Lloyd loop on the GPU and return the final points.
std::vector<Point> runWeightedLloydCUDA(
    const std::vector<float>& density,
    int width,
    int height,
    const std::vector<Point>& initialPoints,
    int iterations,
    int blockSize,
    CudaTimingBreakdown& timing
) {
    const int numberOfPoints = static_cast<int>(initialPoints.size());
    const int numberOfPixels = width * height;

    const size_t densityBytes = static_cast<size_t>(numberOfPixels) * sizeof(float);
    const size_t pointsBytes = static_cast<size_t>(numberOfPoints) * sizeof(Point);
    const size_t accumBytes = static_cast<size_t>(numberOfPoints) * sizeof(float);

    // Device buffers for density, points, and centroid sums.
    float* d_density = nullptr;
    Point* d_points = nullptr;
    float* d_sumX = nullptr;
    float* d_sumY = nullptr;
    float* d_sumW = nullptr;

    // Allocate all GPU buffers used during the iterations.
    CUDA_CHECK(cudaMalloc(&d_density, densityBytes));
    CUDA_CHECK(cudaMalloc(&d_points, pointsBytes));
    CUDA_CHECK(cudaMalloc(&d_sumX, accumBytes));
    CUDA_CHECK(cudaMalloc(&d_sumY, accumBytes));
    CUDA_CHECK(cudaMalloc(&d_sumW, accumBytes));

    // CUDA events are used to time transfers and kernel stages.
    cudaEvent_t h2dStart, h2dStop;
    cudaEvent_t d2hStart, d2hStop;
    cudaEvent_t iterStart, iterStop;
    cudaEvent_t memsetStart, memsetStop;
    cudaEvent_t assignStart, assignStop;
    cudaEvent_t updateStart, updateStop;

    // Copy the fixed density map and initial points to the GPU.
    CUDA_CHECK(cudaEventCreate(&h2dStart));
    CUDA_CHECK(cudaEventCreate(&h2dStop));
    CUDA_CHECK(cudaEventCreate(&d2hStart));
    CUDA_CHECK(cudaEventCreate(&d2hStop));
    CUDA_CHECK(cudaEventCreate(&iterStart));
    CUDA_CHECK(cudaEventCreate(&iterStop));
    CUDA_CHECK(cudaEventCreate(&memsetStart));
    CUDA_CHECK(cudaEventCreate(&memsetStop));
    CUDA_CHECK(cudaEventCreate(&assignStart));
    CUDA_CHECK(cudaEventCreate(&assignStop));
    CUDA_CHECK(cudaEventCreate(&updateStart));
    CUDA_CHECK(cudaEventCreate(&updateStop));

    CUDA_CHECK(cudaEventRecord(h2dStart));
    CUDA_CHECK(cudaMemcpy(d_density, density.data(), densityBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points, initialPoints.data(), pointsBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2dStop));
    CUDA_CHECK(cudaEventSynchronize(h2dStop));
    CUDA_CHECK(cudaEventElapsedTime(&timing.h2dTimeMs, h2dStart, h2dStop));

    // Use separate grids because pixels and points have different counts.
    const int pixelGridSize = (numberOfPixels + blockSize - 1) / blockSize;
    const int pointGridSize = (numberOfPoints + blockSize - 1) / blockSize;

    timing.totalLloydTimeMs = 0.0f;
    timing.totalMemsetTimeMs = 0.0f;
    timing.totalAssignKernelTimeMs = 0.0f;
    timing.totalUpdateKernelTimeMs = 0.0f;

    std::cout << "CUDA pixel grid size: " << pixelGridSize
              << ", point grid size: " << pointGridSize
              << ", block size: " << blockSize << "\n";

    for (int iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(cudaEventRecord(iterStart));

        CUDA_CHECK(cudaEventRecord(memsetStart));
        // Clear centroid sums before assigning pixels for this iteration.
        CUDA_CHECK(cudaMemset(d_sumX, 0, accumBytes));
        CUDA_CHECK(cudaMemset(d_sumY, 0, accumBytes));
        CUDA_CHECK(cudaMemset(d_sumW, 0, accumBytes));
        CUDA_CHECK(cudaEventRecord(memsetStop));

        CUDA_CHECK(cudaEventRecord(assignStart));

        // Assign pixels to the nearest point and accumulate weighted sums.
        assignPixelsAndAccumulateKernel<<<pixelGridSize, blockSize>>>(
            d_density,
            width,
            height,
            d_points,
            numberOfPoints,
            d_sumX,
            d_sumY,
            d_sumW
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(assignStop));

        CUDA_CHECK(cudaEventRecord(updateStart));

        // Move points to their new centroid positions.
        updatePointsKernel<<<pointGridSize, blockSize>>>(
            d_points,
            d_sumX,
            d_sumY,
            d_sumW,
            numberOfPoints
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(updateStop));

        CUDA_CHECK(cudaEventRecord(iterStop));
        CUDA_CHECK(cudaEventSynchronize(iterStop));

        // Read timing values after the iteration has finished on the GPU.

        float iterationMs = 0.0f;
        float memsetMs = 0.0f;
        float assignMs = 0.0f;
        float updateMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(&iterationMs, iterStart, iterStop));
        CUDA_CHECK(cudaEventElapsedTime(&memsetMs, memsetStart, memsetStop));
        CUDA_CHECK(cudaEventElapsedTime(&assignMs, assignStart, assignStop));
        CUDA_CHECK(cudaEventElapsedTime(&updateMs, updateStart, updateStop));

        timing.totalLloydTimeMs += iterationMs;
        timing.totalMemsetTimeMs += memsetMs;
        timing.totalAssignKernelTimeMs += assignMs;
        timing.totalUpdateKernelTimeMs += updateMs;

        std::cout << "CUDA iteration " << (iteration + 1) << "/" << iterations
                  << " total: " << iterationMs << " ms"
                  << " | memset: " << memsetMs << " ms"
                  << " | assign: " << assignMs << " ms"
                  << " | update: " << updateMs << " ms\n";
    }

    std::vector<Point> finalPoints(numberOfPoints);

    // Copy the relaxed points back to the CPU for rendering.

    CUDA_CHECK(cudaEventRecord(d2hStart));
    CUDA_CHECK(cudaMemcpy(finalPoints.data(), d_points, pointsBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2hStop));
    CUDA_CHECK(cudaEventSynchronize(d2hStop));
    CUDA_CHECK(cudaEventElapsedTime(&timing.d2hTimeMs, d2hStart, d2hStop));

    // Release timing events.

    CUDA_CHECK(cudaEventDestroy(h2dStart));
    CUDA_CHECK(cudaEventDestroy(h2dStop));
    CUDA_CHECK(cudaEventDestroy(d2hStart));
    CUDA_CHECK(cudaEventDestroy(d2hStop));
    CUDA_CHECK(cudaEventDestroy(iterStart));
    CUDA_CHECK(cudaEventDestroy(iterStop));
    CUDA_CHECK(cudaEventDestroy(memsetStart));
    CUDA_CHECK(cudaEventDestroy(memsetStop));
    CUDA_CHECK(cudaEventDestroy(assignStart));
    CUDA_CHECK(cudaEventDestroy(assignStop));
    CUDA_CHECK(cudaEventDestroy(updateStart));
    CUDA_CHECK(cudaEventDestroy(updateStop));

    // Release GPU memory.
    CUDA_CHECK(cudaFree(d_density));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_sumX));
    CUDA_CHECK(cudaFree(d_sumY));
    CUDA_CHECK(cudaFree(d_sumW));

    return finalPoints;
}


void drawFilledCircle(
    std::vector<unsigned char>& output,
    int width,
    int height,
    int centerX,
    int centerY,
    int radius
) {
    int radiusSquared = radius * radius;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radiusSquared) {
                continue;
            }

            int x = centerX + dx;
            int y = centerY + dy;

            if (x < 0 || x >= width || y < 0 || y >= height) {
                continue;
            }

            int index = (y * width + x) * 3;
            output[index + 0] = 0;
            output[index + 1] = 0;
            output[index + 2] = 0;
        }
    }
}

std::vector<unsigned char> renderStipples(
    int width,
    int height,
    const std::vector<Point>& points,
    int dotRadius
) {
    std::vector<unsigned char> output(width * height * 3, 255);

    for (const Point& point : points) {
        int x = static_cast<int>(std::round(point.x));
        int y = static_cast<int>(std::round(point.y));
        drawFilledCircle(output, width, height, x, y, dotRadius);
    }

    return output;
}

void printUsage(const char* programName) {
    std::cerr << "Usage:\n"
              << "  " << programName
              << " input_image output_image num_points iterations [gamma] [dot_radius] [block_size]\n\n"
              << "Examples:\n"
              << "  " << programName << " input.png output_cuda.png 1000 10\n"
              << "  " << programName << " input.png output_cuda.png 5000 25 1.0 1 256\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 5) {
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }

        std::string inputPath = argv[1];
        std::string outputPath = argv[2];
        int numberOfPoints = std::stoi(argv[3]);
        int iterations = std::stoi(argv[4]);
        float gamma = 1.0f;
        int dotRadius = 1;
        int blockSize = 256;

        if (argc >= 6) {
            gamma = std::stof(argv[5]);
        }

        if (argc >= 7) {
            dotRadius = std::stoi(argv[6]);
        }

        if (argc >= 8) {
            blockSize = std::stoi(argv[7]);
        }

        if (numberOfPoints <= 0) {
            throw std::runtime_error("Number of points must be positive.");
        }

        if (iterations <= 0) {
            throw std::runtime_error("Number of iterations must be positive.");
        }

        if (gamma <= 0.0f) {
            throw std::runtime_error("Gamma must be positive.");
        }

        if (dotRadius < 1) {
            throw std::runtime_error("Dot radius must be at least 1.");
        }

        if (blockSize <= 0 || blockSize > 1024) {
            throw std::runtime_error("CUDA block size must be in the range 1..1024.");
        }

        CpuTimer totalTimer;
        totalTimer.start();

        std::cout << "Loading image...\n";
        Image image = loadImage(inputPath);

        std::cout << "Image size: " << image.width << " x " << image.height << "\n";
        std::cout << "Stipple points: " << numberOfPoints << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Gamma: " << gamma << "\n";
        std::cout << "Dot radius: " << dotRadius << "\n";
        std::cout << "CUDA block size: " << blockSize << "\n";

        // Make sure a CUDA device is available before starting GPU work.

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA-capable GPU was found.");
        }

        // Print the selected device for reproducible timing logs.
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using CUDA device 0: " << prop.name << "\n";

        // Preprocessing is kept on the CPU in this naive version.

        std::cout << "Computing brightness map on CPU...\n";
        std::vector<float> brightness = computeBrightness(image);

        std::cout << "Computing density map on CPU...\n";
        std::vector<float> density = computeDensity(brightness, gamma);

        // Initial points are generated on the CPU, then copied to the GPU.
        std::cout << "Initializing stipple points on CPU...\n";
        std::vector<Point> initialPoints = initializePointsDensityAware(
            density,
            image.width,
            image.height,
            numberOfPoints
        );

        std::cout << "Running CUDA Weighted Lloyd iterations...\n";
        CudaTimingBreakdown cudaTiming;

        // Run the main Lloyd optimization on the GPU.
        std::vector<Point> finalPoints = runWeightedLloydCUDA(
            density,
            image.width,
            image.height,
            initialPoints,
            iterations,
            blockSize,
            cudaTiming
        );

        // Rendering is done on the CPU after the GPU returns the final points.
        std::cout << "Rendering output on CPU...\n";
        std::vector<unsigned char> output = renderStipples(
            image.width,
            image.height,
            finalPoints,
            dotRadius
        );

        std::cout << "Saving output image...\n";
        saveRGBImage(outputPath, image.width, image.height, output);

        double totalTimeMs = totalTimer.stopMilliseconds();

        std::cout << "\nTiming summary:\n";
        std::cout << "  H2D copy time: " << cudaTiming.h2dTimeMs << " ms\n";
        std::cout << "  CUDA Lloyd loop time: " << cudaTiming.totalLloydTimeMs << " ms\n";
        std::cout << "  Average CUDA Lloyd iteration time: "
                  << cudaTiming.totalLloydTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total accumulator clear time: " << cudaTiming.totalMemsetTimeMs << " ms\n";
        std::cout << "  Average accumulator clear time: "
                  << cudaTiming.totalMemsetTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total assignment kernel time: " << cudaTiming.totalAssignKernelTimeMs << " ms\n";
        std::cout << "  Average assignment kernel time: "
                  << cudaTiming.totalAssignKernelTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total update kernel time: " << cudaTiming.totalUpdateKernelTimeMs << " ms\n";
        std::cout << "  Average update kernel time: "
                  << cudaTiming.totalUpdateKernelTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  D2H copy time: " << cudaTiming.d2hTimeMs << " ms\n";
        std::cout << "  Total program time: " << totalTimeMs << " ms\n";
        std::cout << "Done. Output saved to: " << outputPath << "\n";

        CUDA_CHECK(cudaDeviceReset());
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
