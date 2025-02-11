#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void quantum_gate_transform(const float *input, float *output, const float *gate, int size, int target) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bit_flip = 1 << target;
    int paired_idx = idx ^ bit_flip;

    if (idx < size && paired_idx < size) {
        if ((idx / bit_flip) % 2 == 0) { 
            output[idx] = gate[0] * input[idx] + gate[1] * input[paired_idx];
            output[paired_idx] = gate[2] * input[idx] + gate[3] * input[paired_idx];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return EXIT_FAILURE;
    }

    std::ifstream input_file(argv[1]);
    if (!input_file) {
        std::cerr << "Error: Unable to open file " << argv[1] << "\n";
        return EXIT_FAILURE;
    }

    float gate_matrix[4];
    for (int i = 0; i < 4; ++i) {
        input_file >> gate_matrix[i];
    }

    std::vector<float> state_vector;
    float val;
    while (input_file >> val) {
        state_vector.push_back(val);
    }
    input_file.close();

    int qubit_index = static_cast<int>(state_vector.back());
    state_vector.pop_back();
    
    int vector_length = state_vector.size();
    size_t state_bytes = vector_length * sizeof(float);
    size_t gate_bytes = 4 * sizeof(float);

    float *host_input = (float*)malloc(state_bytes);
    float *host_output = (float*)malloc(state_bytes);
    float *host_gate = (float*)malloc(gate_bytes);

    std::copy(state_vector.begin(), state_vector.end(), host_input);
    std::copy(gate_matrix, gate_matrix + 4, host_gate);

    float *device_input, *device_output, *device_gate;
    cudaMalloc((void**)&device_input, state_bytes);
    cudaMalloc((void**)&device_output, state_bytes);
    cudaMalloc((void**)&device_gate, gate_bytes);

    cudaMemcpy(device_input, host_input, state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gate, host_gate, gate_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start_time, stop_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    int threads_per_block = 256;
    int blocks_per_grid = (vector_length + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start_time, 0);
    quantum_gate_transform<<<blocks_per_grid, threads_per_block>>>(device_input, device_output, device_gate, vector_length, qubit_index);
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);

    float execution_duration;
    cudaEventElapsedTime(&execution_duration, start_time, stop_time);

    cudaError_t error_status = cudaGetLastError();
    if (error_status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error_status) << "\n";
        return EXIT_FAILURE;
    }

    cudaMemcpy(host_output, device_output, state_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < vector_length; ++i) {
        std::cout << std::fixed << std::setprecision(3) << host_output[i] << "\n";
    }

    std::cout << "Execution Time: " << execution_duration << " ms\n";

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_gate);
    free(host_input);
    free(host_output);
    free(host_gate);

    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
