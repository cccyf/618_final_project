#pragma once
#include <helper_gl.h>

#include "particles_kernel.cuh"
#include <helper_functions.h>

void omp_sim(float* m_hPos, float* m_hVel, uint numParticles, SimParams m_params,
	StopWatchInterface* omp_integrate_t,
	StopWatchInterface* omp_collide_t,
	StopWatchInterface* omp_total_t);
