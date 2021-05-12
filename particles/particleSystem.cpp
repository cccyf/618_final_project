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

// OpenGL Graphics includes
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include "omp_sim.h"
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

StopWatchInterface* integrate_t = NULL;
StopWatchInterface* collide_t = NULL;

ParticleSystem::ParticleSystem(uint numParticles, uint3 gridSize, bool bUseOpenGL) :
    m_bInitialized(false),
    m_bUseOpenGL(bUseOpenGL),
    m_numParticles(numParticles),
    m_hPos(0),
    m_hVel(0),
    m_dPos(0),
    m_dVel(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    //    float3 worldSize = make_float3(2.0f, 2.0f, 2.0f);

    m_gridSortBits = 18;    // increase this for larger grids

    // set simulation parameters
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numParticles = m_numParticles;

    m_params.particleRadius = 0.025f;
    m_params.colliderPos = make_float3(1.5f, -0.25f, 1.5f);
    m_params.colliderRadius = 0.2f;

    m_params.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
    //    m_params.cellSize = make_float3(worldSize.x / m_gridSize.x, worldSize.y / m_gridSize.y, worldSize.z / m_gridSize.z);
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);

    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    m_params.gravity = make_float3(0.0f, -0.0003f, 0.0f);
    m_params.globalDamping = 1.0f;

	m_params.dt = 0.01f;
	m_params.offset = 0.05f;
	m_params.damp = 0.98f;
	m_params.mass = 1.f;

    sdkCreateTimer(&integrate_t);
    sdkCreateTimer(&collide_t);

    _initialize(numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

uint
ParticleSystem::createVBO(uint size)
{
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    return vbo;
}

inline float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);
    int type = 0;

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos = new float[m_numParticles*4];
    m_hVel = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
    

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;

    if (m_bUseOpenGL)
    {
        m_posVbo = createVBO(memSize);
        registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
    }

    allocateArray((void **)&m_prevPos, memSize);
    allocateArray((void **)&m_dVel, memSize);

    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    if (m_bUseOpenGL)
    {
        m_colorVBO = createVBO(m_numParticles*4*sizeof(float));
        registerGLBufferObject(m_colorVBO, &m_cuda_colorvbo_resource);

        // fill color buffer
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        float *ptr = data;

        for (uint i=0; i<m_numParticles; i++)
        {
            float t = i / (float) m_numParticles;
#if 0
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
            *ptr++ = rand() / (float) RAND_MAX;
#else
            colorRamp(t, ptr);
            ptr+=3;
#endif
            *ptr++ = 1.0f;
        }

        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    else
    {
        checkCudaErrors(cudaMalloc((void **)&m_cudaColorVBO, sizeof(float)*numParticles*4));
    }

    sdkCreateTimer(&m_timer);

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dVel);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);

    if (m_bUseOpenGL)
    {
        unregisterGLBufferObject(m_cuda_colorvbo_resource);
        unregisterGLBufferObject(m_cuda_posvbo_resource);
        glDeleteBuffers(1, (const GLuint *)&m_posVbo);
        glDeleteBuffers(1, (const GLuint *)&m_colorVBO);
    }
    else
    {
        checkCudaErrors(cudaFree(m_cudaPosVBO));
        checkCudaErrors(cudaFree(m_cudaColorVBO));
    }
}



Vector3f ParticleSystem::compute_force(float* p1, float* p2, float* v1, float* v2, float m_dist) {
    float m_ks = 50.f;
    float m_kd = 0.98f;
    Vector3f pos_diff = make_vector(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
	Vector3f force = -(m_ks * (length(pos_diff) - m_dist) + m_kd * (make_vector(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]) * pos_diff) / length(pos_diff)) * pos_diff / length(pos_diff);
    return force;
}


bool is_neighbor(uint ind1, uint ind2, uint side) {
	int x1 = ind1 % side;
	int y1 = ind1 / side;
	int x2 = ind2 % side;
	int y2 = ind2 / side;

	return (abs(x1 - x2) + abs(y1 - y2) <= 2);
}

bool check_collision(float* p1, float*p2, float dist) {
	Vector3f pos_diff = make_vector(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
	return length(pos_diff) < dist;
}

void ParticleSystem::seq_sim() {
	sdkStartTimer(&integrate_t);
	float* dPos = m_hPos;
	float* dVel = m_hVel;
	float* prevPos = (float*)malloc(sizeof(float) * 4 * m_numParticles);
	memcpy(prevPos, dPos, 4 * m_numParticles * sizeof(float));
	uint side = sqrt(m_numParticles);
	Vector3f force_accumulator;

	for (int i = 1; i < m_numParticles - 1; i++) {
		force_accumulator = make_vector(0.0f, -9.8f * m_params.mass, 0.0f);
		uint xind = i % side;
		uint yind = i / side;
		float* cPos = &dPos[(xind + yind * side) * 4];
		float* cVel = &dVel[(xind + yind * side) * 4];
		float dist = m_params.offset;
		if (xind > 0) {
			float* nPos = &dPos[(xind - 1 + yind * side) * 4];
			float* nVel = &dVel[(xind - 1 + yind * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind > 0) {
			float* nPos = &dPos[(xind + (yind - 1) * side) * 4];
			float* nVel = &dVel[(xind + (yind - 1) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 1) {
			float* nPos = &dPos[(xind + 1 + yind * side) * 4];
			float* nVel = &dVel[(xind + 1 + yind * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind < side - 1) {
			float* nPos = &dPos[(xind + (yind + 1) * side) * 4];
			float* nVel = &dVel[(xind + (yind + 1) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		dist *= sqrt(2);

		if (xind > 0 && yind > 0) {
			float* nPos = &dPos[(xind - 1 + (yind - 1) * side) * 4];
			float* nVel = &dVel[(xind - 1 + (yind - 1) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 1 && yind > 0) {
			float* nPos = &dPos[(xind + 1 + (yind - 1) * side) * 4];
			float* nVel = &dVel[(xind + 1 + (yind - 1) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind > 0 && yind < side - 1) {
			float* nPos = &dPos[(xind - 1 + (yind + 1) * side) * 4];
			float* nVel = &dVel[(xind - 1 + (yind + 1) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 1 && yind < side - 1) {
			float* nPos = &dPos[(xind + 1 + (yind + 1) * side) * 4];
			float* nVel = &dVel[(xind + 1 + (yind + 1) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		dist *= sqrt(2);

		if (xind > 1) {
			float* nPos = &dPos[(xind - 2 + yind * side) * 4];
			float* nVel = &dVel[(xind - 2 + yind * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind > 1) {
			float* nPos = &dPos[(xind + (yind - 2) * side) * 4];
			float* nVel = &dVel[(xind + (yind - 2) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 2) {
			float* nPos = &dPos[(xind + 2 + yind * side) * 4];
			float* nVel = &dVel[(xind + 2 + yind * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind < side - 2) {
			float* nPos = &dPos[(xind + (yind + 2) * side) * 4];
			float* nVel = &dVel[(xind + (yind + 2) * side) * 4];
			force_accumulator += compute_force(cPos, nPos, cVel, nVel, dist);
		}
		Vector3f pos = make_vector(cPos[0], cPos[1], cPos[2]);
		Vector3f vel = make_vector(cVel[0], cVel[1], cVel[2]);
		pos += m_params.dt * vel;
		vel = m_params.damp * vel + m_params.dt * force_accumulator / m_params.mass;
		cPos[0] = pos.x;
		cPos[1] = std::max(-10.f, pos.y);
		cPos[2] = pos.z;
		cVel[0] = vel.x;
		cVel[1] = vel.y;
		cVel[2] = vel.z;
	}
	sdkStopTimer(&integrate_t);

	// collission detection with sphere
	Vector3f sphere = make_vector(m_params.colliderPos.x, m_params.colliderPos.y, m_params.colliderPos.z);
	for (int i = 0; i < m_numParticles; i++) {
		Vector3f p = make_vector(dPos[4 * i], dPos[4 * i + 1], dPos[4 * i + 2]);
		Vector3f diff = p - sphere;
		if (length(diff) <= m_params.colliderRadius) {
			dVel[4 * i] = 0.f;
			dVel[4 * i + 1] = 0.f;
			dVel[4 * i + 2] = 0.f;
		}
	}

	sdkStartTimer(&collide_t);

	for (int i = 0; i < m_numParticles; i++) {
		for (int j = 0; j < m_numParticles; j++) {
			if (i <= j || is_neighbor(i, j, side) || !(check_collision(&dPos[4*i], &dPos[4*j], m_params.offset))) {
				continue;
			}

			// reset position
			memcpy(&dPos[4 * i], &prevPos[4 * i], 4*sizeof(float));
			memcpy(&dPos[4 * j], &prevPos[4 * j], 4*sizeof(float));
			// reset velocity
			dVel[4 * i] = 0.f;
			dVel[4 * i + 1] = 0.f;
			dVel[4 * i + 2] = 0.f;
			dVel[4 * j] = 0.f;
			dVel[4 * j + 1] = 0.f;
			dVel[4 * j + 2] = 0.f;
		}

	}
	delete prevPos;
	sdkStopTimer(&collide_t);
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{   
    int type = 2; // 0/1 for sequential or omp, 2 for cuda
	if (type == 0) {
		seq_sim();
		setArray(POSITION, m_hPos, 0, m_numParticles);
	}else
    if (type == 1) {
		omp_sim(m_hPos, m_hVel, m_numParticles, m_params);
		setArray(POSITION, m_hPos, 0, m_numParticles);
	}
    else {
        float* dPos = (float*)mapGLBufferObject(&m_cuda_posvbo_resource);
        setParameters(&m_params);
        sdkStartTimer(&integrate_t);
        parallel_sim(m_prevPos, dPos, m_dVel, m_params.numParticles);
        sdkStopTimer(&integrate_t);
        sdkStartTimer(&collide_t);
        calcHash(
            m_dGridParticleHash,
            m_dGridParticleIndex,
            dPos,
            m_numParticles);
        sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);
        reorderDataAndFindCellStart(
            m_dCellStart,
            m_dCellEnd,
            m_dSortedPos,
            m_dSortedVel,
            m_dGridParticleHash,
            m_dGridParticleIndex,
            dPos,
            m_dVel,
            m_numParticles,
            m_numGridCells);
        collide(
            m_prevPos,
            dPos,
            m_dVel,
            m_dSortedPos,
            m_dSortedVel,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            m_numParticles,
            m_numGridCells);
        sdkStopTimer(&collide_t);
        unmapGLBufferObject(m_cuda_posvbo_resource);
    }
    

    //printf("integration time: %.3f; collide time: %.3f\n", sdkGetAverageTimerValue(&integrate_t), sdkGetAverageTimerValue(&collide_t));
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

            //            printf("cell: %d, %d particles\n", i, cellSize);
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    // debug
    copyArrayFromDevice(m_hPos, 0, &m_cuda_posvbo_resource, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

    for (uint i=start; i<start+count; i++)
    {
        //        printf("%d: ", i);
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            cuda_vbo_resource = m_cuda_posvbo_resource;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            {
                if (m_bUseOpenGL)
                {
                    unregisterGLBufferObject(m_cuda_posvbo_resource);
                    glBindBuffer(GL_ARRAY_BUFFER, m_posVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(float), count*4*sizeof(float), data);
                    glBindBuffer(GL_ARRAY_BUFFER, 0);
                    registerGLBufferObject(m_posVbo, &m_cuda_posvbo_resource);
                }
                else
                {
                    copyArrayToDevice(m_cudaPosVBO, data, start*4*sizeof(float), count*4*sizeof(float));
                }
            }
            break;

        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void
ParticleSystem::initGrid(uint *size, float spacing, float jitter, uint numParticles)
{
    srand(1973);

    for (uint z=0; z<size[2]; z++)
    {
        for (uint y=0; y<size[1]; y++)
        {
            for (uint x=0; x<size[0]; x++)
            {
                uint i = (z*size[1]*size[0]) + (y*size[0]) + x;

                if (i < numParticles)
                {
                    m_hPos[i*4] = (spacing * x) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+1] = (spacing * y) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+2] = (spacing * z) + m_params.particleRadius - 1.0f + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[i*4+3] = 1.0f;

                    m_hVel[i*4] = 0.0f;
                    m_hVel[i*4+1] = 0.0f;
                    m_hVel[i*4+2] = 0.0f;
                    m_hVel[i*4+3] = 0.0f;
                }
            }
        }
    }
}

void
ParticleSystem::reset(ParticleConfig config)
{
    switch (config)
    {
        default:
        case CONFIG_RANDOM:
            {
                int p = 0, v = 0;

                for (uint i=0; i < m_numParticles; i++)
                {
                    float point[3];
                    point[0] = frand();
                    point[1] = frand();
                    point[2] = frand();
                    m_hPos[p++] = 2 * (point[0] - 0.5f);
                    m_hPos[p++] = 2 * (point[1] - 0.5f);
                    m_hPos[p++] = 2 * (point[2] - 0.5f);
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
            }
            break;

        case CONFIG_GRID:
            {
                float jitter = m_params.particleRadius*0.01f;
                uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
                uint gridSize[3];
                gridSize[0] = gridSize[1] = gridSize[2] = s;
                initGrid(gridSize, m_params.particleRadius*2.0f, jitter, m_numParticles);
            }
            break;

        case CONFIG_CLOTH:
            {
                float start_x = -1.f;
                float start_y = -1.f;
                
                int p = 0, v = 0;
                uint side = sqrt(m_numParticles);
                for (uint i = 0; i < m_numParticles; i++) {
                    uint xind = i % side;
                    uint yind = i / side;
                    m_hPos[p++] = start_x + xind * m_params.offset;
                    m_hPos[p++] = 0.f;
                    m_hPos[p++] = start_y + yind * m_params.offset;
                    m_hPos[p++] = 1.0f; // radius
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                    m_hVel[v++] = 0.0f;
                }
            }
            break;
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;

                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
                {
                    m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
