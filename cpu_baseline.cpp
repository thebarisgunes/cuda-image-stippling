#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

// Stores weighted sums used to update point positions
// after each Lloyd iteration

struct Accumulators {
    std::vector<double> sumX;
    std::vector<double> sumY;
    std::vector<double> sumW;

    explicit Accumulators(int n) : sumX(n), sumY(n), sumW(n) {}

    void clear() {
        std::fill(sumX.begin(), sumX.end(), 0.0);
        std::fill(sumY.begin(), sumY.end(), 0.0);
        std::fill(sumW.begin(), sumW.end(), 0.0);
    }
};

// Simple timer used for performance measurements

class Timer {
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

Image loadImage(const std::string& path) {
    Image image;

    int requestedChannels = 3;
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
    int totalBytes = image.width * image.height * image.channels;
    image.pixels.assign(data, data + totalBytes);

    stbi_image_free(data);
    return image;
}

void saveRGBImage(const std::string& path, int width, int height, const std::vector<unsigned char>& pixels) {
    int channels = 3;
    int strideBytes = width * channels;

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
            
            // Convert RGB values to grayscale brightness.

            brightness[pixelIndex] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    return brightness;
}

// Convert brightness to a density map
// Darker regions receive higher density values

std::vector<float> computeDensity(const std::vector<float>& brightness, float gamma) {
    std::vector<float> density(brightness.size(), 0.0f);

    for (size_t i = 0; i < brightness.size(); ++i) {
        float value = 1.0f - brightness[i];
        value = std::clamp(value, 0.0f, 1.0f);
        density[i] = std::pow(value, gamma);
    }

    return density;
}

// Place initial points using rejection sampling
// guided by the density map

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

    // Fill any remaining points with uniform sampling.
    while (static_cast<int>(points.size()) < numberOfPoints) {
        int x = randomX(rng);
        int y = randomY(rng);
        points.push_back(Point{static_cast<float>(x), static_cast<float>(y)});
    }

    return points;
}

// Assign each pixel to its nearest point and
// accumulate weighted centroid information

void assignPixelsAndAccumulateCPU(
    const std::vector<float>& density,
    int width,
    int height,
    const std::vector<Point>& points,
    Accumulators& accumulators
) {
    const int numberOfPoints = static_cast<int>(points.size());
    const float epsilon = 1e-6f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int pixelIndex = y * width + x;
            float weight = density[pixelIndex];
            
            // Skip pixels that contribute almost no weight

            if (weight <= epsilon) {
                continue;
            }

            int bestIndex = 0;
            float bestDistanceSquared = std::numeric_limits<float>::max();

            // Find the closest point
            for (int i = 0; i < numberOfPoints; ++i) {
                float dx = static_cast<float>(x) - points[i].x;
                float dy = static_cast<float>(y) - points[i].y;
                float distanceSquared = dx * dx + dy * dy;

                if (distanceSquared < bestDistanceSquared) {
                    bestDistanceSquared = distanceSquared;
                    bestIndex = i;
                }
            }

            accumulators.sumX[bestIndex] += static_cast<double>(weight) * static_cast<double>(x);
            accumulators.sumY[bestIndex] += static_cast<double>(weight) * static_cast<double>(y);
            accumulators.sumW[bestIndex] += static_cast<double>(weight);
        }
    }
}

// Move each point to its weighted centroid
void updatePointsCPU(std::vector<Point>& points, const Accumulators& accumulators) {
    const int numberOfPoints = static_cast<int>(points.size());

    for (int i = 0; i < numberOfPoints; ++i) {
        if (accumulators.sumW[i] > 0.0) {
            points[i].x = static_cast<float>(accumulators.sumX[i] / accumulators.sumW[i]);
            points[i].y = static_cast<float>(accumulators.sumY[i] / accumulators.sumW[i]);
        }
        
    }
}


struct CpuLloydMetrics {
    double totalLloydLoopMs = 0.0;
    double totalClearMs = 0.0;
    double totalAssignmentMs = 0.0;
    double totalUpdateMs = 0.0;

    double averageIterationMs(int iterations) const {
        return totalLloydLoopMs / static_cast<double>(iterations);
    }

    double averageClearMs(int iterations) const {
        return totalClearMs / static_cast<double>(iterations);
    }

    double averageAssignmentMs(int iterations) const {
        return totalAssignmentMs / static_cast<double>(iterations);
    }

    double averageUpdateMs(int iterations) const {
        return totalUpdateMs / static_cast<double>(iterations);
    }
};

// Run weighted Lloyd relaxation and collect timing data

CpuLloydMetrics runWeightedLloydCPU(
    const std::vector<float>& density,
    int width,
    int height,
    std::vector<Point>& points,
    int iterations
) {
    Accumulators accumulators(static_cast<int>(points.size()));
    CpuLloydMetrics metrics;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        Timer iterationTimer;
        Timer stageTimer;

        iterationTimer.start();

        stageTimer.start();
        accumulators.clear();
        double clearTimeMs = stageTimer.stopMilliseconds();

        stageTimer.start();
        assignPixelsAndAccumulateCPU(density, width, height, points, accumulators);
        double assignmentTimeMs = stageTimer.stopMilliseconds();

        stageTimer.start();
        updatePointsCPU(points, accumulators);
        double updateTimeMs = stageTimer.stopMilliseconds();

        double iterationTimeMs = iterationTimer.stopMilliseconds();

        metrics.totalClearMs += clearTimeMs;
        metrics.totalAssignmentMs += assignmentTimeMs;
        metrics.totalUpdateMs += updateTimeMs;
        metrics.totalLloydLoopMs += iterationTimeMs;

        std::cout << "CPU iteration " << (iteration + 1) << "/" << iterations
                  << " total: " << iterationTimeMs << " ms"
                  << " | clear: " << clearTimeMs << " ms"
                  << " | assign: " << assignmentTimeMs << " ms"
                  << " | update: " << updateTimeMs << " ms\n";
    }

    std::cout << "\nCPU Lloyd timing summary:\n";
    std::cout << "  CPU Lloyd loop time: "
              << metrics.totalLloydLoopMs << " ms\n";
    std::cout << "  Average CPU Lloyd iteration time: "
              << metrics.averageIterationMs(iterations) << " ms\n";
    std::cout << "  Total accumulator clear time: "
              << metrics.totalClearMs << " ms\n";
    std::cout << "  Average accumulator clear time: "
              << metrics.averageClearMs(iterations) << " ms\n";
    std::cout << "  Total CPU assignment time: "
              << metrics.totalAssignmentMs << " ms\n";
    std::cout << "  Average CPU assignment time: "
              << metrics.averageAssignmentMs(iterations) << " ms\n";
    std::cout << "  Total CPU update time: "
              << metrics.totalUpdateMs << " ms\n";
    std::cout << "  Average CPU update time: "
              << metrics.averageUpdateMs(iterations) << " ms\n";

    return metrics;
}

// Draw a filled circle into the output image

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

// Render all stipple points onto a white canvas
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

// Print command line usage information
void printUsage(const char* programName) {
    std::cerr << "Usage:\n"
              << "  " << programName << " input_image output_image num_points iterations [gamma] [dot_radius]\n\n"
              << "Example:\n"
              << "  " << programName << " input.png output.png 5000 25\n"
              << "  " << programName << " input.png output.png 5000 25 1.0 1\n";
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

        if (argc >= 6) {
            gamma = std::stof(argv[5]);
        }

        if (argc >= 7) {
            dotRadius = std::stoi(argv[6]);
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

        Timer totalTimer;
        totalTimer.start();

        std::cout << "Loading image...\n";
        Image image = loadImage(inputPath);

        std::cout << "Image size: " << image.width << " x " << image.height << "\n";
        std::cout << "Stipple points: " << numberOfPoints << "\n";
        std::cout << "Iterations: " << iterations << "\n";
        std::cout << "Gamma: " << gamma << "\n";
        std::cout << "Dot radius: " << dotRadius << "\n";

        // Build brightness and density maps used by the optimizer

        std::cout << "Computing brightness map...\n";
        std::vector<float> brightness = computeBrightness(image);

        std::cout << "Computing density map...\n";
        std::vector<float> density = computeDensity(brightness, gamma);

        std::cout << "Initializing stipple points...\n";
        // Generate the initial point distribution
        std::vector<Point> points = initializePointsDensityAware(
            density,
            image.width,
            image.height,
            numberOfPoints
        );

        std::cout << "Running CPU Weighted Lloyd iterations...\n";
        // Refine point positions using weighted Lloyd iterations
        CpuLloydMetrics lloydMetrics = runWeightedLloydCPU(
            density,
            image.width,
            image.height,
            points,
            iterations
        );

        std::cout << "Rendering output...\n";
        // Convert the final point set into an image
        std::vector<unsigned char> output = renderStipples(
            image.width,
            image.height,
            points,
            dotRadius
        );

        std::cout << "Saving output image...\n";
        saveRGBImage(outputPath, image.width, image.height, output);

        double totalTimeMs = totalTimer.stopMilliseconds();
        std::cout << "\nEnd-to-end timing summary:\n";
        std::cout << "  CPU Lloyd loop time: " << lloydMetrics.totalLloydLoopMs << " ms\n";
        std::cout << "  Non-Lloyd program time: " << (totalTimeMs - lloydMetrics.totalLloydLoopMs) << " ms\n";
        std::cout << "  Total program time: " << totalTimeMs << " ms\n";
        std::cout << "Done. Output saved to: " << outputPath << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}