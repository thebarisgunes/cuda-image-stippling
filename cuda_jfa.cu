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

// Stores timing for memory copies and each stage of the JFA based Lloyd loop.
struct CudaTimingBreakdown {
    float h2dTimeMs = 0.0f;
    float d2hTimeMs = 0.0f;

    float totalLloydTimeMs = 0.0f;
    float totalMemsetTimeMs = 0.0f;
    float totalAssignKernelTimeMs = 0.0f;
    float totalUpdateKernelTimeMs = 0.0f;

    float totalSeedKernelTimeMs = 0.0f;
    float totalJfaPassKernelTimeMs = 0.0f;
    float totalJfaAccumKernelTimeMs = 0.0f;
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

// Small helper used by JFA passes when comparing candidate owners.
__device__ __forceinline__ float pointDistanceSquaredToPixel(
    const Point& p,
    float x,
    float y
) {
    const float dx = x - p.x;
    const float dy = y - p.y;
    return dx * dx + dy * dy;
}

// Place each point into the owner map as an initial seed.
// Empty pixels remain marked as -1.
__global__ void seedOwnerMapKernel(
    const Point* __restrict__ points,
    int numberOfPoints,
    int width,
    int height,
    int* owner
) {
    const int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIndex >= numberOfPoints) {
        return;
    }

    int x = static_cast<int>(roundf(points[pointIndex].x));
    int y = static_cast<int>(roundf(points[pointIndex].y));

    if (x < 0) {
        x = 0;
    } else if (x >= width) {
        x = width - 1;
    }

    if (y < 0) {
        y = 0;
    } else if (y >= height) {
        y = height - 1;
    }

    const int pixelIndex = y * width + x;
    // If two points round to the same pixel, keep the first one that writes.
    atomicCAS(&owner[pixelIndex], -1, pointIndex);
}

// Run one jump flooding pass.
// Each pixel checks nearby owner candidates at the current jump distance.
__global__ void jfaPassKernel(
    const int* __restrict__ inputOwner,
    int* __restrict__ outputOwner,
    int width,
    int height,
    const Point* __restrict__ points,
    int step
) {
    const int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int numberOfPixels = width * height;

    if (pixelIndex >= numberOfPixels) {
        return;
    }

    const int x = pixelIndex % width;
    const int y = pixelIndex / width;
    const float xf = static_cast<float>(x);
    const float yf = static_cast<float>(y);

    // Start with the owner already known for this pixel.
    int bestOwner = inputOwner[pixelIndex];
    float bestDistanceSquared = FLT_MAX;

    if (bestOwner >= 0) {
        bestDistanceSquared = pointDistanceSquaredToPixel(points[bestOwner], xf, yf);
    }

    // Check the 3x3 neighborhood at the current step size.
    for (int oy = -1; oy <= 1; ++oy) {
        const int ny = y + oy * step;

        if (ny < 0 || ny >= height) {
            continue;
        }

        for (int ox = -1; ox <= 1; ++ox) {
            const int nx = x + ox * step;

            if (nx < 0 || nx >= width) {
                continue;
            }

            const int neighborPixelIndex = ny * width + nx;
            const int candidateOwner = inputOwner[neighborPixelIndex];

            // Ignore neighbors that do not have an owner yet.
            if (candidateOwner < 0) {
                continue;
            }

            const float candidateDistanceSquared =
                pointDistanceSquaredToPixel(points[candidateOwner], xf, yf);

            if (candidateDistanceSquared < bestDistanceSquared) {
                bestDistanceSquared = candidateDistanceSquared;
                bestOwner = candidateOwner;
            }
        }
    }

    // Store the best owner found in this pass.
    outputOwner[pixelIndex] = bestOwner;
}

// Use the completed owner map to accumulate weighted centroid sums.
__global__ void accumulateFromOwnerMapKernel(
    const float* __restrict__ density,
    int width,
    int height,
    const int* __restrict__ owner,
    float* sumX,
    float* sumY,
    float* sumW
) {
    const int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int numberOfPixels = width * height;

    if (pixelIndex >= numberOfPixels) {
        return;
    }

    const float weight = density[pixelIndex];

    // Pixels with very small density do not affect the centroid.
    if (weight <= 1e-6f) {
        return;
    }

    const int pointIndex = owner[pixelIndex];

    // Skip pixels that were not assigned to any point.
    if (pointIndex < 0) {
        return;
    }

    const int x = pixelIndex % width;
    const int y = pixelIndex / width;
    const float xf = static_cast<float>(x);
    const float yf = static_cast<float>(y);

    // Many pixels can contribute to the same point, so accumulation is atomic.
    atomicAdd(&sumX[pointIndex], weight * xf);
    atomicAdd(&sumY[pointIndex], weight * yf);
    atomicAdd(&sumW[pointIndex], weight);
}

// Move each point to the centroid accumulated for its region.
__global__ void updatePointsKernel(
    Point* points,
    const float* sumX,
    const float* sumY,
    const float* sumW,
    int numberOfPoints
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numberOfPoints) {
        return;
    }

    const float w = sumW[i];

    // Keep points unchanged if they received no weight.
    if (w > 1e-12f) {
        points[i].x = sumX[i] / w;
        points[i].y = sumY[i] / w;
    }
}

// Choose the largest power-of-two jump needed to cover the image.
int computeInitialJfaStep(int width, int height) {
    const int maxDimension = std::max(width, height);
    int step = 1;

    while (step < maxDimension) {
        step <<= 1;
    }

    step >>= 1;
    return std::max(step, 1);
}

// Count how many jump passes are needed until the step reaches one pixel.
int computeJfaPassCount(int width, int height) {
    int step = computeInitialJfaStep(width, height);
    int count = 0;

    while (step >= 1) {
        ++count;
        step >>= 1;
    }

    return count;
}

// Run weighted Lloyd iterations using JFA for the assignment step.
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

    // Owner maps store the nearest point index for each pixel.
    const size_t ownerBytes = static_cast<size_t>(numberOfPixels) * sizeof(int);

    float* d_density = nullptr;
    Point* d_points = nullptr;
    float* d_sumX = nullptr;
    float* d_sumY = nullptr;
    float* d_sumW = nullptr;

    // Two owner buffers are used for ping pong JFA passes.
    int* d_ownerA = nullptr;
    int* d_ownerB = nullptr;

    // Allocate device buffers for image data, points, sums, and owner maps.
    CUDA_CHECK(cudaMalloc(&d_density, densityBytes));
    CUDA_CHECK(cudaMalloc(&d_points, pointsBytes));
    CUDA_CHECK(cudaMalloc(&d_sumX, accumBytes));
    CUDA_CHECK(cudaMalloc(&d_sumY, accumBytes));
    CUDA_CHECK(cudaMalloc(&d_sumW, accumBytes));
    CUDA_CHECK(cudaMalloc(&d_ownerA, ownerBytes));
    CUDA_CHECK(cudaMalloc(&d_ownerB, ownerBytes));

    // Events measure each stage without forcing CPU timers around GPU work.
    cudaEvent_t h2dStart, h2dStop;
    cudaEvent_t d2hStart, d2hStop;
    cudaEvent_t iterStart, iterStop;
    cudaEvent_t clearStart, clearStop;
    cudaEvent_t seedStart, seedStop;
    cudaEvent_t jfaStart, jfaStop;
    cudaEvent_t accumStart, accumStop;
    cudaEvent_t updateStart, updateStop;

    CUDA_CHECK(cudaEventCreate(&h2dStart));
    CUDA_CHECK(cudaEventCreate(&h2dStop));
    CUDA_CHECK(cudaEventCreate(&d2hStart));
    CUDA_CHECK(cudaEventCreate(&d2hStop));
    CUDA_CHECK(cudaEventCreate(&iterStart));
    CUDA_CHECK(cudaEventCreate(&iterStop));
    CUDA_CHECK(cudaEventCreate(&clearStart));
    CUDA_CHECK(cudaEventCreate(&clearStop));
    CUDA_CHECK(cudaEventCreate(&seedStart));
    CUDA_CHECK(cudaEventCreate(&seedStop));
    CUDA_CHECK(cudaEventCreate(&jfaStart));
    CUDA_CHECK(cudaEventCreate(&jfaStop));
    CUDA_CHECK(cudaEventCreate(&accumStart));
    CUDA_CHECK(cudaEventCreate(&accumStop));
    CUDA_CHECK(cudaEventCreate(&updateStart));
    CUDA_CHECK(cudaEventCreate(&updateStop));

    // Copy fixed input data to the GPU once before the iteration loop.
    CUDA_CHECK(cudaEventRecord(h2dStart));
    CUDA_CHECK(cudaMemcpy(d_density, density.data(), densityBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points, initialPoints.data(), pointsBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2dStop));
    CUDA_CHECK(cudaEventSynchronize(h2dStop));
    CUDA_CHECK(cudaEventElapsedTime(&timing.h2dTimeMs, h2dStart, h2dStop));

    const int pixelGridSize = (numberOfPixels + blockSize - 1) / blockSize;
    const int pointGridSize = (numberOfPoints + blockSize - 1) / blockSize;

    // JFA starts with a large jump and halves it each pass.
    const int initialJfaStep = computeInitialJfaStep(width, height);
    const int jfaPassCount = computeJfaPassCount(width, height);

    // Reset timing totals before collecting per-iteration measurements.
    timing.totalLloydTimeMs = 0.0f;
    timing.totalMemsetTimeMs = 0.0f;
    timing.totalAssignKernelTimeMs = 0.0f;
    timing.totalSeedKernelTimeMs = 0.0f;
    timing.totalJfaPassKernelTimeMs = 0.0f;
    timing.totalJfaAccumKernelTimeMs = 0.0f;
    timing.totalUpdateKernelTimeMs = 0.0f;

    std::cout << "CUDA JFA pixel grid size: " << pixelGridSize
              << ", point grid size: " << pointGridSize
              << ", block size: " << blockSize
              << ", initial JFA step: " << initialJfaStep
              << ", JFA passes: " << jfaPassCount << "\n";

    // Each Lloyd iteration rebuilds the owner map from the current point set.          
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CUDA_CHECK(cudaEventRecord(iterStart));
        // Clear centroid sums and reset the owner map to -1.
        CUDA_CHECK(cudaEventRecord(clearStart));
        CUDA_CHECK(cudaMemset(d_sumX, 0, accumBytes));
        CUDA_CHECK(cudaMemset(d_sumY, 0, accumBytes));
        CUDA_CHECK(cudaMemset(d_sumW, 0, accumBytes));
        CUDA_CHECK(cudaMemset(d_ownerA, 0xff, ownerBytes));
        CUDA_CHECK(cudaEventRecord(clearStop));

        CUDA_CHECK(cudaEventRecord(seedStart));
        // Write current point positions into the owner map as seeds.
        seedOwnerMapKernel<<<pointGridSize, blockSize>>>(
            d_points,
            numberOfPoints,
            width,
            height,
            d_ownerA
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(seedStop));

        // Ping pong between owner buffers during JFA passes.
        int* currentOwner = d_ownerA;
        int* nextOwner = d_ownerB;

        CUDA_CHECK(cudaEventRecord(jfaStart));

        // Propagate seed ownership across the image with decreasing jump sizes.
        for (int step = initialJfaStep; step >= 1; step >>= 1) {
            jfaPassKernel<<<pixelGridSize, blockSize>>>(
                currentOwner,
                nextOwner,
                width,
                height,
                d_points,
                step
            );
            CUDA_CHECK(cudaGetLastError());
            // The output of this pass becomes the input for the next pass.
            std::swap(currentOwner, nextOwner);
        }
        CUDA_CHECK(cudaEventRecord(jfaStop));

        CUDA_CHECK(cudaEventRecord(accumStart));
        // Convert the final owner map into weighted sums for each point.
        accumulateFromOwnerMapKernel<<<pixelGridSize, blockSize>>>(
            d_density,
            width,
            height,
            currentOwner,
            d_sumX,
            d_sumY,
            d_sumW
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(accumStop));

        CUDA_CHECK(cudaEventRecord(updateStart));
        // Update point positions after accumulation is complete.
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

        float iterationMs = 0.0f;
        float clearMs = 0.0f;
        float seedMs = 0.0f;
        float jfaMs = 0.0f;
        float accumMs = 0.0f;
        float updateMs = 0.0f;
        float assignmentMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(&iterationMs, iterStart, iterStop));
        CUDA_CHECK(cudaEventElapsedTime(&clearMs, clearStart, clearStop));
        CUDA_CHECK(cudaEventElapsedTime(&seedMs, seedStart, seedStop));
        CUDA_CHECK(cudaEventElapsedTime(&jfaMs, jfaStart, jfaStop));
        CUDA_CHECK(cudaEventElapsedTime(&accumMs, accumStart, accumStop));
        CUDA_CHECK(cudaEventElapsedTime(&updateMs, updateStart, updateStop));

        // Treat seed, JFA passes, and accumulation as the full assignment cost.
        assignmentMs = seedMs + jfaMs + accumMs;

        timing.totalLloydTimeMs += iterationMs;
        timing.totalMemsetTimeMs += clearMs;
        timing.totalAssignKernelTimeMs += assignmentMs;
        timing.totalSeedKernelTimeMs += seedMs;
        timing.totalJfaPassKernelTimeMs += jfaMs;
        timing.totalJfaAccumKernelTimeMs += accumMs;
        timing.totalUpdateKernelTimeMs += updateMs;

        std::cout << "CUDA JFA iteration " << (iteration + 1) << "/" << iterations
                  << " total: " << iterationMs << " ms"
                  << " | clear: " << clearMs << " ms"
                  << " | seed: " << seedMs << " ms"
                  << " | jfa: " << jfaMs << " ms"
                  << " | accum: " << accumMs << " ms"
                  << " | update: " << updateMs << " ms\n";
    }

    std::vector<Point> finalPoints(numberOfPoints);

    // Copy final points back to the CPU for rendering.
    CUDA_CHECK(cudaEventRecord(d2hStart));
    CUDA_CHECK(cudaMemcpy(finalPoints.data(), d_points, pointsBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2hStop));
    CUDA_CHECK(cudaEventSynchronize(d2hStop));
    CUDA_CHECK(cudaEventElapsedTime(&timing.d2hTimeMs, d2hStart, d2hStop));

    CUDA_CHECK(cudaEventDestroy(h2dStart));
    CUDA_CHECK(cudaEventDestroy(h2dStop));
    CUDA_CHECK(cudaEventDestroy(d2hStart));
    CUDA_CHECK(cudaEventDestroy(d2hStop));
    CUDA_CHECK(cudaEventDestroy(iterStart));
    CUDA_CHECK(cudaEventDestroy(iterStop));
    CUDA_CHECK(cudaEventDestroy(clearStart));
    CUDA_CHECK(cudaEventDestroy(clearStop));
    CUDA_CHECK(cudaEventDestroy(seedStart));
    CUDA_CHECK(cudaEventDestroy(seedStop));
    CUDA_CHECK(cudaEventDestroy(jfaStart));
    CUDA_CHECK(cudaEventDestroy(jfaStop));
    CUDA_CHECK(cudaEventDestroy(accumStart));
    CUDA_CHECK(cudaEventDestroy(accumStop));
    CUDA_CHECK(cudaEventDestroy(updateStart));
    CUDA_CHECK(cudaEventDestroy(updateStop));

    // Release GPU buffers.
    CUDA_CHECK(cudaFree(d_density));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_sumX));
    CUDA_CHECK(cudaFree(d_sumY));
    CUDA_CHECK(cudaFree(d_sumW));
    CUDA_CHECK(cudaFree(d_ownerA));
    CUDA_CHECK(cudaFree(d_ownerB));

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
              << "  " << programName << " input.png output_jfa.png 1000 10\n"
              << "  " << programName << " input.png output_jfa.png 5000 25 1.0 1 256\n";
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

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA-capable GPU was found.");
        }

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using CUDA device 0: " << prop.name << "\n";

        // Preprocessing remains on the CPU in this version.
        std::cout << "Computing brightness map on CPU...\n";
        std::vector<float> brightness = computeBrightness(image);

        std::cout << "Computing density map on CPU...\n";
        std::vector<float> density = computeDensity(brightness, gamma);

        // Initial points are generated on the CPU and copied to the GPU.
        std::cout << "Initializing stipple points on CPU...\n";
        std::vector<Point> initialPoints = initializePointsDensityAware(
            density,
            image.width,
            image.height,
            numberOfPoints
        );

        std::cout << "Running CUDA Weighted Lloyd iterations with JFA assignment...\n";
        CudaTimingBreakdown cudaTiming;

        // Main optimization uses JFA to approximate the nearest point assignment.
        std::vector<Point> finalPoints = runWeightedLloydCUDA(
            density,
            image.width,
            image.height,
            initialPoints,
            iterations,
            blockSize,
            cudaTiming
        );

        // The final point set is rendered on the CPU.
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
        std::cout << "  Total clear time (accumulators + owner map): " << cudaTiming.totalMemsetTimeMs << " ms\n";
        std::cout << "  Average clear time: "
                  << cudaTiming.totalMemsetTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total JFA assignment time (seed + JFA + accumulation): " << cudaTiming.totalAssignKernelTimeMs << " ms\n";
        std::cout << "  Average JFA assignment time: "
                  << cudaTiming.totalAssignKernelTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total JFA seed time: " << cudaTiming.totalSeedKernelTimeMs << " ms\n";
        std::cout << "  Average JFA seed time: "
                  << cudaTiming.totalSeedKernelTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total JFA pass time: " << cudaTiming.totalJfaPassKernelTimeMs << " ms\n";
        std::cout << "  Average JFA pass time: "
                  << cudaTiming.totalJfaPassKernelTimeMs / static_cast<float>(iterations) << " ms\n";
        std::cout << "  Total JFA accumulation time: " << cudaTiming.totalJfaAccumKernelTimeMs << " ms\n";
        std::cout << "  Average JFA accumulation time: "
                  << cudaTiming.totalJfaAccumKernelTimeMs / static_cast<float>(iterations) << " ms\n";
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
