//Ella Wilkins and Emma Frampton

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void fill(vector3 *values, double *hPos, double *hVel, double *mass);
__global__ void sum_clmn(vector3 *values, double *hPos, double *hVel, double *mass);

void compute()
{
    vector3 *d_values;
    double *d_hPos, *d_hVel, *d_mass;

    size_t values_size = sizeof(vector3) * NUMENTITIES * NUMENTITIES;
    size_t pos_vel_size = sizeof(double) * NUMENTITIES * 3;
    size_t mass_size = sizeof(double) * NUMENTITIES;

    cudaMalloc(&d_values, values_size);
    cudaMalloc(&d_hPos, pos_vel_size);
    cudaMalloc(&d_hVel, pos_vel_size);
    cudaMalloc(&d_mass, mass_size);

    // copy to device
    cudaMemcpy(d_values, d_values, values_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hPos, hPos, pos_vel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, pos_vel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, mass_size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NUMENTITIES + 15) / 16, (NUMENTITIES + 15) / 16);

    // launch kernels
    fill<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_hPos, d_hVel, d_mass);
    cudaDeviceSynchronize();
    sum_clmn<<<(NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_values, d_hPos, d_hVel, d_mass);
    cudaDeviceSynchronize();

    // copy back to host
    cudaMemcpy(hPos, d_hPos, pos_vel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, pos_vel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_values, d_values, values_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_mass, mass, mass_size, cudaMemcpyDeviceToHost);

    // free
    cudaFree(d_values);
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}

__global__ void fill(vector3 *values, double *hPos, double *hVel, double *mass)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES)
    {
        if (i == j)
        {
            FILL_VECTOR(values[i * NUMENTITIES + j], 0, 0, 0);
        }
        else
        {
            vector3 distance;
            for (int k = 0; k < 3; k++)
            {
                distance[k] = hPos[i * 3 + k] - hPos[j * 3 + k];
            }

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];

            // Check for potential issues
            double accelmag = 0.0; // Default value
            if (magnitude_sq != 0.0)
            {
                double magnitude = sqrt(magnitude_sq);
                accelmag = -GRAV_CONSTANT * mass[j] / magnitude_sq;
                FILL_VECTOR(values[i * NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
            }
        }
    }
}

__global__ void sum_clmn(vector3 *values, double *hPos, double *hVel, double *mass)
{
    // i = x's index * dimension * index in the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // if i is greater than # of entities, stop
    if (i >= NUMENTITIES)
    {
        return;
    }
    else
    {
        // set total to empty (init)
        vector3 total = {0, 0, 0};

        // fill for each entity
        for (int j = 0; j < NUMENTITIES; j++)
        {
            for (int k = 0; k < 3; k++)
                total[k] += values[i * NUMENTITIES + j][k];
        }

        for (int k = 0; k < 3; k++)
        {
            hVel[i * 3 + k] += total[k] * INTERVAL;
            hPos[i * 3 + k] += hVel[i * 3 + k] * INTERVAL;
        }
    }
}