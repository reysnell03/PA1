#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip> // For precision formatting

// CUDA Kernel for matrix-vector transformation
__global__ void quantum_transform(const float *input, float *output, const float *matrix, int length, int qubit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int mask = 1 << qubit;
    int paired_idx = idx ^ mask;

    // Ensure index is within bounds before accessing memory
    if (idx < length && paired_idx < length) {
        if ((idx / mask) % 2 == 0) {
            output[idx] = matrix[0] * input[idx] + matrix[1] * input[paired_idx];
            output[paired_idx] = matrix[2] * input[idx] + matrix[3] * input[paired_idx];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream file(argv[1]);
    if (!file) {
        std::cerr << "Error: Unable to open file " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    // Read 2x2 transformation matrix
    float transformation_matrix[4];
    for (int i = 0; i < 4; ++i) {
        file >> transformation_matrix[i];
    }

    // Read input vector and qubit index
    std::vector<float> input_data;
    float value;
    while (file >> value) {
        input_data.push_back(value);
    }
    file.close();

    // Extract qubit index
    int qubit_index = static_cast<int>(input_data.back());
    input_data.pop_back();
    int vector_size = input_data.size();

    // Allocate unified memory for CPU-GPU shared access
    float *d_input, *d_output, *d_matrix;
    cudaMallocManaged(&d_input, vector_size * sizeof(float));
    cudaMallocManaged(&d_output, vector_size * sizeof(float));
    cudaMallocManaged(&d_matrix, 4 * sizeof(float));

    // Check for CUDA memory allocation errors
    if (!d_input || !d_output || !d_matrix) {
        std::cerr << "CUDA memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Copy data to unified memory
    std::copy(input_data.begin(), input_data.end(), d_input);
    std::copy(transformation_matrix, transformation_matrix + 4, d_matrix);

    // Define CUDA kernel execution parameters
    int threads_per_block = 256;
    int blocks_per_grid = (vector_size + threads_per_block - 1) / threads_per_block;

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Execute kernel
    quantum_transform<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, d_matrix, vector_size, qubit_index);
    cudaDeviceSynchronize();

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Ensure event has completed

    // Measure elapsed time
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Print formatted output
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < vector_size; ++i) {
        std::cout << d_output[i] << std::endl;
    }

    // Print execution time
    std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_matrix);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}
