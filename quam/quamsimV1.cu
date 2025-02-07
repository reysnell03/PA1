#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

// CUDA kernel for matrix-vector transformation
__global__ void apply_transformation(const float *input, float *output, const float *transformation, int length, int qubit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int mask = 1 << qubit;
    int paired_idx = idx ^ mask;

    // Ensure index is within bounds
    if (idx < length && paired_idx < length) {
        if ((idx / mask) % 2 == 0) { 
            output[idx] = transformation[0] * input[idx] + transformation[1] * input[paired_idx];
            output[paired_idx] = transformation[2] * input[idx] + transformation[3] * input[paired_idx];
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

    // Load transformation matrix
    float transform_matrix[4];
    for (int i = 0; i < 4; ++i) {
        file >> transform_matrix[i];
    }

    // Load input vector and qubit index
    std::vector<float> input_data;
    float value;
    while (file >> value) {
        input_data.push_back(value);
    }
    file.close();

    // Extract qubit index
    int qubit_idx = static_cast<int>(input_data.back());
    input_data.pop_back();
    
    int vector_size = input_data.size();
    size_t bytes = vector_size * sizeof(float);
    size_t matrix_bytes = 4 * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *h_matrix = (float*)malloc(matrix_bytes);

    // Copy data to host arrays
    std::copy(input_data.begin(), input_data.end(), h_input);
    std::copy(transform_matrix, transform_matrix + 4, h_matrix);

    // Allocate device memory
    float *d_input, *d_output, *d_matrix;
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMalloc((void**)&d_matrix, matrix_bytes);

    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice);

    // CUDA event timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch CUDA kernel
    int threads_per_block = 256;
    int blocks_per_grid = (vector_size + threads_per_block - 1) / threads_per_block;

    // Record start event
    cudaEventRecord(start, 0);

    apply_transformation<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, d_matrix, vector_size, qubit_idx);

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop); // Ensure event has completed

    // Measure elapsed time
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Check for CUDA kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Copy results back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Output results
    for (int i = 0; i < vector_size; ++i) {
        std::cout << std::fixed << std::setprecision(3) << h_output[i] << std::endl;
    }

    // Report kernel execution time
    std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;

    // Free GPU and CPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_matrix);
    free(h_input);
    free(h_output);
    free(h_matrix);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
