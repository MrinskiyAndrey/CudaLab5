
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE 5

//@@ Место для вставки кода
__global__ void blur(float* in, float* out, int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h)
    {
        float pixVal = 0;
        int pixels = 0;
        for (int bRow = -BLUR_SIZE; bRow < BLUR_SIZE + 1; bRow++)
        {
            for (int bCol = -BLUR_SIZE; bCol < BLUR_SIZE + 1; bCol++)
            {
                int curRow = row + bRow;
                int curCol = col + bCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
                {
                    pixVal += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }
        out[row * w + col] = pixVal / pixels;
    }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    char* inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    float* deviceInputImageData;
    float* deviceOutputImageData;

    args = wbArg_read(argc, argv); /* получение входных аргументов */

    inputImageFile = wbArg_getInputFile(args, 0);

    inputImage = wbImport(inputImageFile);

    // Входное изображение в оттенках серого,
    // поэтому количество каналов равно 1
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);

    // Так как изображение монохромное, оно содержит только 1 канал
    outputImage = wbImage_new(imageWidth, imageHeight, 1);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void**)&deviceInputImageData,
        imageWidth * imageHeight * sizeof(float));
    cudaMalloc((void**)&deviceOutputImageData,
        imageWidth * imageHeight * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
        imageWidth * imageHeight * sizeof(float),
        cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ Вставьте сюда ваш код
    dim3 blockSize(16, 16);
    dim3 grid(ceil(static_cast<float>(imageWidth) / blockSize.x), ceil(static_cast<float>(imageHeight) / blockSize.y));
    blur << <grid, blockSize >> > (deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

    wbTime_stop(Compute, "Doing the computation on the GPU");

    ///////////////////////////////////////////////////////
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
        imageWidth * imageHeight * sizeof(float),
        cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
