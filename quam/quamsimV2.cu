#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

using namespace std;

__global__ void apply_quantum_gate(const float *input, float *output, const float *matrix, int size, int qubit) {
    int in1 = blockIdx.x * blockDim.x + threadIdx.x;
    int mask = 1 << qubit;
    int in2 = in1 ^ mask;
    bool is_lower_half = (in1 & mask) == 0;
    float a, b, res1, res2;

    if (in1 < size && in2 < size && is_lower_half) { 
        a = input[in1];
        b = input[in2];

        res1 = fmaf(matrix[0], a, matrix[1] * b);  
        res2 = fmaf(matrix[2], a, matrix[3] * b);

        output[in1] = res1;
        output[in2] = res2;
    }
}

int main(int argc, char *argv[]) {
    ifstream input_file;
    float gate_matrix[4];
    vector<float> state_vector;
    float val;
    int qubit_index, vector_length, threads_per_block, blocks_per_grid;
    size_t state_bytes, gate_bytes;
    float *host_input, *host_output, *host_gate;
    cudaEvent_t start_time, stop_time;
    float execution_duration;
    cudaError_t error_status;

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return EXIT_FAILURE;
    }

    input_file.open(argv[1]);
    if (!input_file) {
        cerr << "Error: Unable to open file " << argv[1] << "\n";
        return EXIT_FAILURE;
    }

    for (int i = 0; i < 4; ++i) {
        input_file >> gate_matrix[i];
    }

    while (input_file >> val) {
        state_vector.push_back(val);
    }
    input_file.close();

    qubit_index = static_cast<int>(state_vector.back());
    state_vector.pop_back();

    vector_length = state_vector.size();
    state_bytes = vector_length * sizeof(float);
    gate_bytes = 4 * sizeof(float);

    cudaMallocManaged(&host_input, state_bytes);
    cudaMallocManaged(&host_output, state_bytes);
    cudaMallocManaged(&host_gate, gate_bytes);

    copy(state_vector.begin(), state_vector.end(), host_input);
    copy(gate_matrix, gate_matrix + 4, host_gate);

    threads_per_block = 256;
    blocks_per_grid = (vector_length + threads_per_block - 1) / threads_per_block;

    cudaEventCreate(&start_time);
    cudaEventCreate(&stop_time);

    cudaEventRecord(start_time, 0);
    apply_quantum_gate<<<blocks_per_grid, threads_per_block>>>(host_input, host_output, host_gate, vector_length, qubit_index);
    cudaEventRecord(stop_time, 0);
    cudaEventSynchronize(stop_time);

    cudaEventElapsedTime(&execution_duration, start_time, stop_time);

    error_status = cudaGetLastError();
    if (error_status != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(error_status) << "\n";
        return EXIT_FAILURE;
    }

    for (int i = 0; i < vector_length; ++i) {
        cout << fixed << setprecision(3) << host_output[i] << "\n";
    }

  //  cout << "Execution Time: " << execution_duration << " ms\n";

    cudaFree(host_input);
    cudaFree(host_output);
    cudaFree(host_gate);

    cudaEventDestroy(start_time);
    cudaEventDestroy(stop_time);

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
