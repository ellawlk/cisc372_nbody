#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void fill_vector(vector3 *vals, double *hPos, double *hVel, double *mass)
{
    // i = x's index * dimension * index in the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // j = y's index * dimension * index in the thread
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // k = thread index of z
    int k = threadIdx.z;

    // if i and j are equal, set vector to 0,0,0
    if (i == j)
    {
        FILL_VECTOR(vals[(i * NUMENTITIES) + j], 0, 0, 0);
    }
    else
    {
        // init distance
        vector3 distance;

        // calculate distance for each
        for (int k = 0; k < 3; k++)
        {
            distance[k] = hPos[i * 3 + k] - hPos[j * 3 + k];
        };

        // calculating the magnitude
        double magnitude_sq = (distance[0] * distance[0]) + (distance[1] * distance[1]) + ((distance[2] * distance[2]));
        double magnitude = sqrt(magnitude_sq);

        // calculating magnitude of the acceleration
        double accel_mag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;

        // fill x, y, z
        FILL_VECTOR(vals[i * NUMENTITIES + j],
                    accel_mag * distance[0] / magnitude, accel_mag * distance[1] / magnitude,
                    accel_mag * distance[2] / magnitude);
    }
}

__global__ void sum_clmn(vector3 *accels, double *hPos, double *hVel, double *mass)
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
                total[k] += accels[i * NUMENTITIES + j][k];
        }
        
        for (int k = 0; k < 3; k++)
        {
            hVel[i * 3 + k] += total[k] * INTERVAL;
            hPos[i * 3 + k] += hVel[i * 3 + k] * INTERVAL;
        }
    }
}

void compute(vector3 *accels, double *hPos, double *hVel, double *mass)
{
    dim3 size(16, 16, 3);
    int a = ((NUMENTITIES + 15) / 16);
}