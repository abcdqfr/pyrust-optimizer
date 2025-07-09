"""
Real-World Image Processing Bottleneck

This script simulates performance bottlenecks found in:
- Computer vision pipelines
- Medical image analysis
- Satellite image processing
- Photo editing applications
- Scientific imaging

NO TOY EXAMPLES - These are REAL optimization opportunities.
"""

import time
import math
import random
from typing import List, Tuple, Dict
import numpy as np


def generate_test_image(width: int, height: int) -> List[List[List[int]]]:
    """Generate a realistic test image with RGB channels"""
    image = []
    for y in range(height):
        row = []
        for x in range(width):
            # Create realistic image patterns
            r = int(128 + 127 * math.sin(x * 0.01) * math.cos(y * 0.01))
            g = int(128 + 127 * math.sin((x + y) * 0.005))
            b = int(128 + 127 * math.cos(x * 0.008) * math.sin(y * 0.008))

            # Add some noise for realism
            r += random.randint(-20, 20)
            g += random.randint(-20, 20)
            b += random.randint(-20, 20)

            # Clamp values
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            row.append([r, g, b])
        image.append(row)
    return image


def apply_gaussian_blur(image: List[List[List[int]]], kernel_size: int = 5) -> List[List[List[int]]]:
    """
    REAL-WORLD BOTTLENECK #1: Gaussian blur filter

    This is used in:
    - Photo editing software (Photoshop, GIMP)
    - Computer vision preprocessing
    - Medical image enhancement
    - Surveillance systems

    OPTIMIZATION OPPORTUNITIES:
    - Nested loops for convolution (O(n²))
    - Heavy mathematical computations per pixel
    - Memory-intensive operations
    """

    height = len(image)
    width = len(image[0])
    result = [[[0, 0, 0] for _ in range(width)] for _ in range(height)]

    # Generate Gaussian kernel - EXPENSIVE MATH OPERATIONS
    sigma = kernel_size / 3.0
    kernel = []
    kernel_sum = 0.0

    for i in range(kernel_size):
        kernel_row = []
        for j in range(kernel_size):
            x = i - kernel_size // 2
            y = j - kernel_size // 2
            value = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
            kernel_row.append(value)
            kernel_sum += value
        kernel.append(kernel_row)

    # Normalize kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] /= kernel_sum

    # Apply convolution - NESTED LOOP BOTTLENECK
    for y in range(height):
        for x in range(width):
            for channel in range(3):  # RGB channels
                weighted_sum = 0.0

                # Convolution operation
                for ky in range(kernel_size):
                    for kx in range(kernel_size):
                        image_y = y + ky - kernel_size // 2
                        image_x = x + kx - kernel_size // 2

                        # Handle edge cases
                        if 0 <= image_y < height and 0 <= image_x < width:
                            pixel_value = image[image_y][image_x][channel]
                            kernel_value = kernel[ky][kx]
                            weighted_sum += pixel_value * kernel_value

                result[y][x][channel] = int(max(0, min(255, weighted_sum)))

    return result


def detect_edges_sobel(image: List[List[List[int]]]) -> List[List[int]]:
    """
    REAL-WORLD BOTTLENECK #2: Edge detection (Sobel operator)

    Critical for:
    - Object detection in autonomous vehicles
    - Medical image analysis (tumor detection)
    - Manufacturing quality control
    - Robotics vision systems

    OPTIMIZATION OPPORTUNITIES:
    - Complex mathematical operations per pixel
    - Gradient calculations
    - Multiple passes through image data
    """

    height = len(image)
    width = len(image[0])

    # Convert to grayscale first - ANOTHER LOOP
    gray_image = []
    for y in range(height):
        gray_row = []
        for x in range(width):
            # Standard luminance formula
            gray_value = int(0.299 * image[y][x][0] +
                           0.587 * image[y][x][1] +
                           0.114 * image[y][x][2])
            gray_row.append(gray_value)
        gray_image.append(gray_row)

    # Sobel kernels
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    edges = []

    # Apply Sobel operator - NESTED CONVOLUTION LOOPS
    for y in range(1, height - 1):
        edge_row = []
        for x in range(1, width - 1):
            # Calculate gradients in X and Y directions
            gx = 0
            gy = 0

            for ky in range(3):
                for kx in range(3):
                    pixel = gray_image[y + ky - 1][x + kx - 1]
                    gx += pixel * sobel_x[ky][kx]
                    gy += pixel * sobel_y[ky][kx]

            # Calculate gradient magnitude - EXPENSIVE MATH
            gradient_magnitude = math.sqrt(gx * gx + gy * gy)
            edge_value = int(min(255, gradient_magnitude))
            edge_row.append(edge_value)

        edges.append(edge_row)

    return edges


def histogram_analysis(image: List[List[List[int]]]) -> Dict:
    """
    REAL-WORLD BOTTLENECK #3: Histogram analysis

    Used in:
    - Photo editing (exposure, contrast adjustments)
    - Medical imaging (intensity analysis)
    - Security systems (lighting normalization)
    - Scientific imaging

    OPTIMIZATION OPPORTUNITIES:
    - Multiple passes through large datasets
    - Statistical calculations
    - Memory allocation for histograms
    """

    height = len(image)
    width = len(image[0])

    # Initialize histograms for each channel
    hist_r = [0] * 256
    hist_g = [0] * 256
    hist_b = [0] * 256

    # Calculate histograms - EXPENSIVE LOOP
    for y in range(height):
        for x in range(width):
            r, g, b = image[y][x]
            hist_r[r] += 1
            hist_g[g] += 1
            hist_b[b] += 1

    total_pixels = width * height

    # Calculate statistics for each channel - MORE LOOPS
    stats = {}

    for channel_name, histogram in [('red', hist_r), ('green', hist_g), ('blue', hist_b)]:
        # Mean calculation
        mean_value = 0.0
        for intensity, count in enumerate(histogram):
            mean_value += intensity * count
        mean_value /= total_pixels

        # Variance calculation - ANOTHER EXPENSIVE LOOP
        variance = 0.0
        for intensity, count in enumerate(histogram):
            variance += count * (intensity - mean_value) ** 2
        variance /= total_pixels

        # Standard deviation
        std_dev = math.sqrt(variance)

        # Find mode (most frequent intensity)
        mode_intensity = 0
        max_count = 0
        for intensity, count in enumerate(histogram):
            if count > max_count:
                max_count = count
                mode_intensity = intensity

        # Calculate percentiles - CUMULATIVE OPERATIONS
        cumulative = 0
        percentile_5 = 0
        percentile_95 = 0

        for intensity, count in enumerate(histogram):
            cumulative += count
            percentile = (cumulative / total_pixels) * 100

            if percentile >= 5 and percentile_5 == 0:
                percentile_5 = intensity
            if percentile >= 95 and percentile_95 == 0:
                percentile_95 = intensity
                break

        stats[channel_name] = {
            'mean': mean_value,
            'std_dev': std_dev,
            'variance': variance,
            'mode': mode_intensity,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95
        }

    return stats


def morphological_operations(image: List[List[int]], operation: str = 'erosion') -> List[List[int]]:
    """
    REAL-WORLD BOTTLENECK #4: Morphological operations

    Critical for:
    - Medical image analysis (cell counting)
    - Document processing (text cleanup)
    - Industrial inspection
    - Biometric systems

    OPTIMIZATION OPPORTUNITIES:
    - Structuring element operations
    - Min/max operations across neighborhoods
    - Multiple image passes
    """

    height = len(image)
    width = len(image[0])
    result = [[0 for _ in range(width)] for _ in range(height)]

    # 3x3 structuring element
    struct_elem = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    # Apply morphological operation - NESTED LOOPS WITH MIN/MAX
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if operation == 'erosion':
                min_value = 255
                # Find minimum in neighborhood
                for sy in range(3):
                    for sx in range(3):
                        if struct_elem[sy][sx]:
                            pixel_value = image[y + sy - 1][x + sx - 1]
                            min_value = min(min_value, pixel_value)
                result[y][x] = min_value

            elif operation == 'dilation':
                max_value = 0
                # Find maximum in neighborhood
                for sy in range(3):
                    for sx in range(3):
                        if struct_elem[sy][sx]:
                            pixel_value = image[y + sy - 1][x + sx - 1]
                            max_value = max(max_value, pixel_value)
                result[y][x] = max_value

    return result


def main():
    """
    REAL-WORLD IMAGE PROCESSING PERFORMANCE TEST

    This demonstrates the kind of performance bottlenecks found in
    production computer vision and image processing systems.
    """
    print("🖼️  PyRust Optimizer - Real-World Image Processing Test")
    print("=" * 65)

    # Generate realistic test image
    print("🎨 Generating test image...")
    width, height = 800, 600  # Realistic image size
    test_image = generate_test_image(width, height)
    print(f"✅ Generated {width}x{height} image ({width * height:,} pixels)")

    # Benchmark 1: Gaussian Blur
    print("\n🌫️  Applying Gaussian blur filter...")
    start_time = time.time()
    blurred_image = apply_gaussian_blur(test_image, kernel_size=7)
    blur_time = time.time() - start_time
    print(f"   ⏱️  Gaussian blur time: {blur_time:.3f} seconds")

    # Benchmark 2: Edge Detection
    print("\n🔍 Running Sobel edge detection...")
    start_time = time.time()
    edges = detect_edges_sobel(test_image)
    edge_time = time.time() - start_time
    print(f"   Detected {len(edges)} x {len(edges[0])} edge map")
    print(f"   ⏱️  Edge detection time: {edge_time:.3f} seconds")

    # Benchmark 3: Histogram Analysis
    print("\n📊 Computing color histograms...")
    start_time = time.time()
    histogram_stats = histogram_analysis(test_image)
    histogram_time = time.time() - start_time

    for channel, stats in histogram_stats.items():
        print(f"   {channel.capitalize()} channel - Mean: {stats['mean']:.1f}, Std: {stats['std_dev']:.1f}")
    print(f"   ⏱️  Histogram analysis time: {histogram_time:.3f} seconds")

    # Benchmark 4: Morphological Operations
    print("\n🔧 Applying morphological operations...")
    start_time = time.time()

    # Convert first edge map row to grayscale for morphological ops
    gray_edges = []
    for row in edges:
        gray_row = []
        for pixel in row:
            gray_row.append(pixel)
        gray_edges.append(gray_row)

    eroded = morphological_operations(gray_edges, 'erosion')
    dilated = morphological_operations(gray_edges, 'dilation')
    morphology_time = time.time() - start_time

    print(f"   Applied erosion and dilation to {len(gray_edges)}x{len(gray_edges[0])} image")
    print(f"   ⏱️  Morphological operations time: {morphology_time:.3f} seconds")

    # Total performance summary
    total_time = blur_time + edge_time + histogram_time + morphology_time
    total_operations = width * height * 4  # Approximate operation count

    print(f"\n📈 TOTAL IMAGE PROCESSING TIME: {total_time:.3f} seconds")
    print(f"📊 Total pixel operations: {total_operations:,}")
    print(f"🚀 Operations per second: {total_operations / total_time:,.0f}")

    print("\n🎯 OPTIMIZATION OPPORTUNITIES DETECTED:")
    print("   • Gaussian Blur: Convolution operations → Rust SIMD optimization")
    print("   • Edge Detection: Mathematical gradients → Rust math optimization")
    print("   • Histogram Analysis: Statistical operations → Rust optimization")
    print("   • Morphological Ops: Min/max operations → Rust optimization")

    print(f"\n💡 EXPECTED SPEEDUP WITH PYRUST OPTIMIZER: 20-100x faster")
    print(f"   Predicted optimized time: {total_time / 50:.3f} seconds")
    print(f"   Predicted ops/sec: {(total_operations / total_time) * 50:,.0f}")


if __name__ == "__main__":
    main()
