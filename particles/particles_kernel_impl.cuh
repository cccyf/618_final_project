/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

#include <cuda.h>
#include <device_functions.h>
#include "Eigen/Core"

// simulation parameters in constant memory
__constant__ SimParams params;


struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        vel += params.gravity * deltaTime;
        vel *= params.globalDamping;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        // set this to zero to disable collisions with cube sides
#if 1

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > 1.0f - params.particleRadius)
        {
            pos.z = 1.0f - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif

        if (pos.y < -1.0f + params.particleRadius)
        {
            pos.y = -1.0f + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.worldOrigin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.worldOrigin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.worldOrigin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float4 *oldPos,           // input: sorted position array
                                  float4 *oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = oldPos[sortedIndex];
        float4 vel = oldVel[sortedIndex];

        sortedPos[index] = pos;
        sortedVel[index] = vel;
    }


}

// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        
        //printf("%f %f\n", dist, collideDist);
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}

__device__ __inline__
bool is_neighbor(int gx, int gy, int ix, int iy) {
    /*
    if (ind1 > ind2) {
        uint t = ind1;
        ind1 = ind2;
        ind2 = t;
    }
    uint diff = ind2 - ind1;
    if (diff == 1 || diff == 2 || diff == side || diff == 2 * side || diff == side - 1 || diff == side + 1) {
        return true;
    }
    */
    if (abs(gx - ix) + abs(gy - iy) <= 2) return true;

    return false;
}

// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  pos,
                   float3  vel,
                   float4 *oldPos,
                   float4 *oldVel,
                   uint   *cellStart,
                   uint   *cellEnd,
                   uint   *gridParticleIndex)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    uint sideLength = sqrt((float)params.numBodies);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];
        uint oind1 = gridParticleIndex[index];
        for (uint j=startIndex; j<endIndex; j++)
        {
            uint oind2 = gridParticleIndex[j];
            if (!is_neighbor(oind1%sideLength, oind1 /sideLength, oind2 %sideLength, oind2 /sideLength))                // check not colliding with self
            {
                
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(pos, pos2, vel, vel2, params.particleRadius, params.particleRadius, params.attraction);
                    
            }
        }
    }

    return force;
}


__global__
void collideD(float4 *prevPos,
              float4 *newPos,
              float4 *newVel,               // output: new velocity
              float4 *oldPos,               // input: sorted positions
              float4 *oldVel,               // input: sorted velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    //bool collide = false;
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd, gridParticleIndex);
            }

        }

    }
    
    // collide with cursor sphere
    // force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);
    //if (collide) {
        //printf("collide\n");
        // write new velocity back to original unsorted location
        uint originalIndex = gridParticleIndex[index];
        newVel[originalIndex] = make_float4(vel + force * 0.01f, 0.0f);
        //newVel[originalIndex] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        //newPos[originalIndex] = prevPos[originalIndex];
    //}
    
    
}


__device__ __inline__
void compute_force(float4* p1, float4* p2, float4* v1, float4* v2, float m_dist, Eigen::Vector3f* res) {
    float m_ks = 50.f;
    float m_kd = 0.98f;
    Eigen::Vector3f pos_diff = { p1->x - p2->x, p1->y - p2->y, p1->z - p2->z };
    Eigen::Vector3f vel_diff = { v1->x - v2->x, v1->y - v2->y, v1->z - v2->z };
    float dot_product = vel_diff.dot(pos_diff);
    *res = -(m_ks * (pos_diff.norm() - m_dist) + m_kd * dot_product / pos_diff.norm()) * pos_diff / pos_diff.norm();
}

//Round a / b to nearest higher integer value
__device__ __inline__
uint iDiv(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


__device__ __inline__
bool check_collision(float* p1, float* p2, float dist) {
    Eigen::Vector3f pos_diff = { p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2] };
    return pos_diff.norm() < dist;
}

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

__global__ void parallel_kernel(float* prevPos, float* pos, float* vel, float deltaTime, uint sideLength, float mass, float offset, float damp) {
	//__shared__ float4 block_pos[144];
    //__shared__ float4 block_vel[144];
	__shared__ float4 block_pos[(1*BLOCK_DIM_X + 4)*(1*BLOCK_DIM_Y + 4)];
	__shared__ float4 block_vel[(1*BLOCK_DIM_X + 4)*(1*BLOCK_DIM_Y + 4)];

	uint col = 1;
	uint row = 1;
	uint width = col * BLOCK_DIM_X + 4;
	uint height = row * BLOCK_DIM_Y + 4;

    uint xstart = blockIdx.x * blockDim.x * col - 2, ystart = blockIdx.y * blockDim.y * row - 2;
    uint threadInd = threadIdx.y * blockDim.x + threadIdx.x;
    for (uint i = threadInd; i < width * height; i += blockDim.x * blockDim.y) {
        uint x = xstart + i % width;
        uint y = ystart + i / width;
		if (x < 0 || x > sideLength - 1 || y < 0 || y > sideLength - 1) {
			continue;
		}

        //x = x > 0 ? (x < sideLength - 1 ? x : sideLength - 1) : 0;
        //y = y > 0 ? (y < sideLength - 1 ? y : sideLength - 1) : 0;
        block_pos[i] = *(float4*)&pos[4 * (x + y * sideLength)];
        block_vel[i] = *(float4*)&vel[4 * (x + y * sideLength)];
    }

    __syncthreads();
	for (uint xi = 0; xi < col; xi++) {
		for (uint yi = 0; yi < row; yi++) {
			uint globalx = col * blockDim.x * blockIdx.x + blockDim.x * xi+ threadIdx.x;
			uint globaly = row * blockDim.y * blockIdx.y + blockDim.y * yi + threadIdx.y;
			if (globalx >= sideLength || globaly >= sideLength) {
				continue;
			}
			
			// save prev positions
			for (uint i = 0; i < 4; i++) {
				prevPos[globaly * sideLength + globalx + i] = pos[globaly * sideLength + globalx + i];
			}

			if ((globalx == 0 && globaly == 0) || (globalx == sideLength - 1 && globaly == sideLength - 1)) continue;

			uint xind = blockDim.x * xi + threadIdx.x + 2;
			uint yind = blockDim.y * yi + threadIdx.y + 2;

			Eigen::Vector3f force_accumulator;
			Eigen::Vector3f cur_force;

			force_accumulator = { 0.0f, -0.098f * mass, 0.0f };

			float4 *cPos = &block_pos[xind + yind * width];
			float4 *cVel = &block_vel[xind + yind * width];

			float dist = offset;

			if (globalx > 0) {
				float4* nPos = &block_pos[xind - 1 + yind * width];
				float4* nVel = &block_vel[xind - 1 + yind * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globaly > 0) {
				float4* nPos = &block_pos[xind + (yind - 1) * width];
				float4* nVel = &block_vel[xind + (yind - 1) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globalx < sideLength - 1) {
				float4* nPos = &block_pos[xind + 1 + yind * width];
				float4* nVel = &block_vel[xind + 1 + yind * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globaly < sideLength - 1) {
				float4* nPos = &block_pos[xind + (yind + 1) * width];
				float4* nVel = &block_vel[xind + (yind + 1) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			dist *= sqrt(2.0f);

			if (globalx > 0 && globaly > 0) {
				float4* nPos = &block_pos[xind - 1 + (yind - 1) * width];
				float4* nVel = &block_vel[xind - 1 + (yind - 1) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globalx < sideLength - 1 && globaly > 0) {
				float4* nPos = &block_pos[xind + 1 + (yind - 1) * width];
				float4* nVel = &block_vel[xind + 1 + (yind - 1) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globalx > 0 && globaly < sideLength - 1) {
				float4* nPos = &block_pos[xind - 1 + (yind + 1) * width];
				float4* nVel = &block_vel[xind - 1 + (yind + 1) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globalx < sideLength - 1 && globaly < sideLength - 1) {
				float4* nPos = &block_pos[xind + 1 + (yind + 1) * width];
				float4* nVel = &block_vel[xind + 1 + (yind + 1) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			dist *= sqrt(2.0f);

			if (globalx > 1) {
				float4* nPos = &block_pos[xind - 2 + yind * width];
				float4* nVel = &block_vel[xind - 2 + yind * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globaly > 1) {
				float4* nPos = &block_pos[xind + (yind - 2) * width];
				float4* nVel = &block_vel[xind + (yind - 2) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globalx < sideLength - 2) {
				float4* nPos = &block_pos[xind + 2 + yind * width];
				float4* nVel = &block_vel[xind + 2 + yind * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			if (globaly < sideLength - 2) {
				float4* nPos = &block_pos[xind + (yind + 2) * width];
				float4* nVel = &block_vel[xind + (yind + 2) * width];
				compute_force(cPos, nPos, cVel, nVel, dist, &cur_force);
				force_accumulator += cur_force;
			}

			Eigen::Vector3f vPos = { cPos->x, cPos->y, cPos->z };
			Eigen::Vector3f vVel = { cVel->x, cVel->y, cVel->z };


			vPos += deltaTime * vVel;
			vVel = damp * vVel + deltaTime * force_accumulator / mass;
			
			uint start_ind = (globalx + globaly * sideLength) * 4;

			Eigen::Vector3f collider_pos = { params.colliderPos.x, params.colliderPos.y, params.colliderPos.z };
			if ((vPos - collider_pos).norm() > params.colliderRadius) {
				pos[start_ind] = vPos.x();
				//pos[start_ind + 1] = vPos.y() > -0.5f ? vPos.y() : -0.5f;
				pos[start_ind + 1] = vPos.y();
				pos[start_ind + 2] = vPos.z();
				vel[start_ind] = vVel.x();
				vel[start_ind + 1] = vVel.y();
				vel[start_ind + 2] = vVel.z();
			}
			else {
				vel[start_ind] = 0.f;
				vel[start_ind + 1] = 0.f;
				vel[start_ind + 2] = 0.f;
			}
		}
	}
   
}

#endif
