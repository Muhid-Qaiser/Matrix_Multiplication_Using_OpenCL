__kernel void conv_naive(__global const float* input, __global float* output,
                          __constant float* filt, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x > 0 && y > 0 && x < width-1 && y < height-1) {
        float sum = 0.0f;
        for (int fy = -1; fy <= 1; ++fy)
            for (int fx = -1; fx <= 1; ++fx)
                sum += input[(y+fy)*width + (x+fx)] * filt[(fy+1)*3 + (fx+1)];
        output[y*width + x] = sum;
    }
}

__kernel void conv_opt(__global const float* input, __global float* output,
                       __constant float* filt, int width, int height,
                       __local float* tile) {
    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int blockW = get_local_size(0);
    const int blockH = get_local_size(1);

    int gx = bx * blockW + lx;
    int gy = by * blockH + ly;

    // Load tile with halo
    for (int dy = ly; dy < blockH + 2; dy += blockH)
        for (int dx = lx; dx < blockW + 2; dx += blockW) {
            int ix = bx * blockW + dx - 1;
            int iy = by * blockH + dy - 1;
            tile[dy * (blockW + 2) + dx] = (ix >= 0 && ix < width && iy >= 0 && iy < height)
                                           ? input[iy * width + ix]
                                           : 0.0f;
        }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx > 0 && gy > 0 && gx < width - 1 && gy < height - 1) {
        float sum = 0.0f;
        for (int fy = 0; fy < 3; ++fy)
            for (int fx = 0; fx < 3; ++fx)
                sum += tile[(ly + fy) * (blockW + 2) + (lx + fx)] * filt[fy * 3 + fx];
        output[gy * width + gx] = sum;
    }
}
