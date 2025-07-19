// Muhid Qaiser, 22i-0472, AI-B, Assignment-4

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;

constexpr int IMG_WIDTH = 2048;
constexpr int IMG_HEIGHT = 2048;

// * 3x3 filter Sobel Filter X
static const float filter[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1 };

#define CHECK_ERR(err, msg)                        \
    if (err != CL_SUCCESS)                         \
    {                                              \
        cerr << msg << " (" << err << ")" << endl; \
        exit(EXIT_FAILURE);                        \
    }

// * Round up total to nearest multiple of block
size_t round_up(size_t total, size_t block)
{
    return ((total + block - 1) / block) * block;
}

// * Function to load kernel source from file
string load_Kernel_Source(const string& filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Failed to open kernel file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

// * CPU reference convolution
void cpu_convolution(const vector<float>& in, vector<float>& out, const float* filt)
{
    for (int y = 1; y < IMG_HEIGHT - 1; ++y)
    {
        for (int x = 1; x < IMG_WIDTH - 1; ++x)
        {
            float sum = 0.0f;
            for (int fy = -1; fy <= 1; ++fy)
                for (int fx = -1; fx <= 1; ++fx)
                    sum += in[(y + fy) * IMG_WIDTH + (x + fx)] * filt[(fy + 1) * 3 + (fx + 1)];
            out[y * IMG_WIDTH + x] = sum;
        }
    }
}

int main()
{
    // * Prepare image and output buffers
    vector<float> image(IMG_WIDTH * IMG_HEIGHT),
        out_cpu(IMG_WIDTH * IMG_HEIGHT),
        out_naive(IMG_WIDTH * IMG_HEIGHT),
        out_opt(IMG_WIDTH * IMG_HEIGHT);
    for (auto& v : image)
        v = rand() / (float)RAND_MAX;

    // * Measure CPU convolution time
    auto start_cpu = chrono::high_resolution_clock::now();
    cpu_convolution(image, out_cpu, filter);
    auto end_cpu = chrono::high_resolution_clock::now();
    double t_cpu = chrono::duration<double, milli>(end_cpu - start_cpu).count();
    cout << "CPU convolution: " << t_cpu << " ms\n\n";

    // * OpenCL platform/device
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    CHECK_ERR(clGetPlatformIDs(1, &platform, nullptr), "Platform");
    CHECK_ERR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "Device");

    // * Query device limits
    size_t maxWG;
    CHECK_ERR(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWG), &maxWG, nullptr), "Max WG");
    size_t maxDims[3];
    CHECK_ERR(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxDims), maxDims, nullptr), "Max item sizes");
    cout << "Device limits: maxWG=" << maxWG
        << ", maxItems=[" << maxDims[0] << "," << maxDims[1] << "]\n\n";

    // * Context and queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERR(err, "Context");
#if defined(CL_VERSION_2_0)
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CHECK_ERR(err, "Queue");

    // * Load Kernel Source from kernels.cl file
    string kernelSourceStr = load_Kernel_Source("kernels.cl");

    // * Build program
    const char* kernelSourceCStr = kernelSourceStr.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, nullptr, &err);
    CHECK_ERR(err, "Program create");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS)
    {
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        vector<char> log(len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        cerr << "Build log:\n"
            << log.data() << endl;
        CHECK_ERR(err, "Build");
    }

    // * Kernels
    cl_kernel kn = clCreateKernel(program, "conv_naive", &err);
    CHECK_ERR(err, "Kernel naive");
    cl_kernel ko = clCreateKernel(program, "conv_opt", &err);
    CHECK_ERR(err, "Kernel opt");

    // * Buffers
    cl_mem buf_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        image.size() * sizeof(float), (void*)image.data(), &err);
    CHECK_ERR(err, "Buf in");
    cl_mem buf_out_n = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        out_naive.size() * sizeof(float), nullptr, &err);
    CHECK_ERR(err, "Buf out n");
    cl_mem buf_out_o = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        out_opt.size() * sizeof(float), nullptr, &err);
    CHECK_ERR(err, "Buf out o");
    cl_mem buf_filt = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(filter), (void*)filter, &err);
    CHECK_ERR(err, "Buf filt");

    // * Kernel args for conv_naive
    int w = IMG_WIDTH, h = IMG_HEIGHT;
    auto setArgs = [&](cl_kernel k, cl_mem out)
        {
            int i = 0;
            CHECK_ERR(clSetKernelArg(k, i++, sizeof(buf_in), &buf_in), "Arg in");
            CHECK_ERR(clSetKernelArg(k, i++, sizeof(out), &out), "Arg out");
            CHECK_ERR(clSetKernelArg(k, i++, sizeof(buf_filt), &buf_filt), "Arg filt");
            CHECK_ERR(clSetKernelArg(k, i++, sizeof(w), &w), "Arg w");
            CHECK_ERR(clSetKernelArg(k, i++, sizeof(h), &h), "Arg h");
        };

    setArgs(kn, buf_out_n);

    // * For conv_opt, arguments 0-4 are the same; local memory (argument-5) will be set dynamically.
    setArgs(ko, buf_out_o);

    // * Helper to execute a kernel and measure its time
    auto run = [&](cl_kernel k, size_t gw[2], size_t lw[2], double& t)
        {
            auto t0 = chrono::high_resolution_clock::now();
            err = clEnqueueNDRangeKernel(queue, k, 2, nullptr, gw, lw, 0, nullptr, nullptr);
            if (err == CL_INVALID_WORK_GROUP_SIZE)
            {
                cerr << "Invalid work-group size (" << lw[0] << "x" << lw[1] << ")" << endl;
                return false;
            }
            CHECK_ERR(err, "Enqueue");
            clFinish(queue);
            auto t1 = chrono::high_resolution_clock::now();
            t = chrono::duration<double, milli>(t1 - t0).count();
            return true;
        };

    // * Run naive GPU convolution (work-group size = 16x16)
    size_t gw[2] = { round_up(IMG_WIDTH, 16), round_up(IMG_HEIGHT, 16) };
    size_t lw16[2] = { 16, 16 };
    double t_naive;
    run(kn, gw, lw16, t_naive);
    clEnqueueReadBuffer(queue, buf_out_n, CL_TRUE, 0, out_naive.size() * sizeof(float), out_naive.data(), 0, nullptr, nullptr);
    bool correct_naive = true;
    for (size_t i = 0; i < out_cpu.size(); ++i)
    {
        if (fabs(out_cpu[i] - out_naive[i]) > 1e-3f)
        {
            correct_naive = false;
            cerr << "Mismatch in naive output at index " << i << endl;
            break;
        }
    }
    cout << "Naive GPU (16x16): " << t_naive << " ms, \n\tspeedup (CPU/naive): " << (t_cpu / t_naive)
        << (correct_naive ? " \n\t[Correct]" : " \n\t[Incorrect]") << "\n\n";


    // * Run optimized GPU convolution with varying work-group sizes (e.g., 8, 16, 32)
    for (int g : {8, 16, 32})
    {
        size_t lx = min<size_t>(g, maxDims[0]);
        size_t ly = min<size_t>(g, maxDims[1]);
        if (lx * ly > maxWG)
            ly = maxWG / lx;
        size_t lw[2] = { lx, ly };
        size_t gw2[2] = { round_up(IMG_WIDTH, lx), round_up(IMG_HEIGHT, ly) };


        // * Set local memory argument for conv_opt:
        // * (lw[0]+2) * (lw[1]+2) * sizeof(float)
        size_t localMemSize = (lw[0] + 2) * (lw[1] + 2) * sizeof(float);
        CHECK_ERR(clSetKernelArg(ko, 5, localMemSize, nullptr), "Set local mem");
        double t_opt;
        if (!run(ko, gw2, lw, t_opt))
            continue;
        clEnqueueReadBuffer(queue, buf_out_o, CL_TRUE, 0, out_opt.size() * sizeof(float), out_opt.data(), 0, nullptr, nullptr);

        bool correct_opt = true;
        for (size_t i = 0; i < out_cpu.size(); ++i)
        {
            if (fabs(out_cpu[i] - out_opt[i]) > 1e-3f)
            {
                correct_opt = false;
                cerr << "Mismatch in optimized output at index " << i << endl;
                break;
            }
        }
        cout << "Optimized GPU (" << lx << "x" << ly << "): " << t_opt
            << " ms, \n\tspeedup (CPU/opt): " << (t_cpu / t_opt)
            << ", \n\tspeedup (naive/opt): " << (t_naive / t_opt)
            << (correct_opt ? " \n\t[Correct]" : " \n\t[Incorrect]") << "\n\n";
    }

    // * Cleanup Memory
    clReleaseMemObject(buf_in);
    clReleaseMemObject(buf_out_n);
    clReleaseMemObject(buf_out_o);
    clReleaseMemObject(buf_filt);
    clReleaseKernel(kn);
    clReleaseKernel(ko);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
