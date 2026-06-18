# cuda-image-stippling
CUDA-accelerated weighted Voronoi image stippling with CPU, naive GPU, tiled GPU, and Jump Flooding Algorithm (JFA) implementations.

## Build

Compile the CPU baseline with:

g++ -O2 cpu_baseline.cpp -o stippling_cpu

## Run

Basic usage:

./stippling_cpu input.png output.png 5000 25

Usage with optional parameters:

./stippling_cpu input.png output.png 5000 25 1.0 1

Arguments
argv[1] = input image path
argv[2] = output image path
argv[3] = number of stipple points
argv[4] = number of Lloyd iterations
argv[5] = optional gamma value, default = 1.0
argv[6] = optional dot radius, default = 1

Example:

./stippling_cpu input.png output.png 5000 25 1.0 1

This command generates a stippled version of input.png using 5000 points and 25 Lloyd iterations.

Dependencies

This project uses the single-header stb image libraries:

stb_image.h
stb_image_write.h

Download them from the official stb repository and place both files in the same folder as cpu_baseline.cpp.
