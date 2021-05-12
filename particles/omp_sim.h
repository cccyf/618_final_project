#pragma once
#include <helper_gl.h>

#include "particles_kernel.cuh"

void omp_sim(float* m_hPos, float* m_hVel, uint numParticles, SimParams m_params);
