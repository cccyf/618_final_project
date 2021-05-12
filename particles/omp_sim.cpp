
#include "omp_sim.h"
#include <Vector.hpp>
#include "vector_functions.h"

#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>


Vector3f omp_compute_force(float* p1, float* p2, float* v1, float* v2, float m_dist) {
	float m_ks = 50.f;
	float m_kd = 0.98f;
	Vector3f pos_diff = make_vector(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
	Vector3f force = -(m_ks * (length(pos_diff) - m_dist) + m_kd * (make_vector(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]) * pos_diff) / length(pos_diff)) * pos_diff / length(pos_diff);
	return force;
}

bool omp_is_neighbor(uint ind1, uint ind2, uint side) {
	int x1 = ind1 % side;
	int y1 = ind1 / side;
	int x2 = ind2 % side;
	int y2 = ind2 / side;

	return (abs(x1 - x2) + abs(y1 - y2) <= 2);
}

bool omp_check_collision(float* p1, float*p2, float dist) {
	Vector3f pos_diff = make_vector(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]);
	return length(pos_diff) < dist;
}


void omp_sim(float* m_hPos, float* m_hVel, uint m_numParticles, SimParams m_params) {
	float* dPos = m_hPos;
	float* dVel = m_hVel;
	float* prevPos = (float*)malloc(sizeof(float) * 4 * m_numParticles);
	memcpy(prevPos, dPos, 4 * m_numParticles * sizeof(float));
	uint side = sqrt(m_numParticles);
	Vector3f force_accumulator;

#pragma omp parallel for
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
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind > 0) {
			float* nPos = &dPos[(xind + (yind - 1) * side) * 4];
			float* nVel = &dVel[(xind + (yind - 1) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 1) {
			float* nPos = &dPos[(xind + 1 + yind * side) * 4];
			float* nVel = &dVel[(xind + 1 + yind * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind < side - 1) {
			float* nPos = &dPos[(xind + (yind + 1) * side) * 4];
			float* nVel = &dVel[(xind + (yind + 1) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		dist *= sqrt(2);

		if (xind > 0 && yind > 0) {
			float* nPos = &dPos[(xind - 1 + (yind - 1) * side) * 4];
			float* nVel = &dVel[(xind - 1 + (yind - 1) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 1 && yind > 0) {
			float* nPos = &dPos[(xind + 1 + (yind - 1) * side) * 4];
			float* nVel = &dVel[(xind + 1 + (yind - 1) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind > 0 && yind < side - 1) {
			float* nPos = &dPos[(xind - 1 + (yind + 1) * side) * 4];
			float* nVel = &dVel[(xind - 1 + (yind + 1) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 1 && yind < side - 1) {
			float* nPos = &dPos[(xind + 1 + (yind + 1) * side) * 4];
			float* nVel = &dVel[(xind + 1 + (yind + 1) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		dist *= sqrt(2);

		if (xind > 1) {
			float* nPos = &dPos[(xind - 2 + yind * side) * 4];
			float* nVel = &dVel[(xind - 2 + yind * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind > 1) {
			float* nPos = &dPos[(xind + (yind - 2) * side) * 4];
			float* nVel = &dVel[(xind + (yind - 2) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (xind < side - 2) {
			float* nPos = &dPos[(xind + 2 + yind * side) * 4];
			float* nVel = &dVel[(xind + 2 + yind * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
		}

		if (yind < side - 2) {
			float* nPos = &dPos[(xind + (yind + 2) * side) * 4];
			float* nVel = &dVel[(xind + (yind + 2) * side) * 4];
			force_accumulator += omp_compute_force(cPos, nPos, cVel, nVel, dist);
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

	// collission detection with sphere
	Vector3f sphere = make_vector(m_params.colliderPos.x, m_params.colliderPos.y, m_params.colliderPos.z);
	//#pragma omp parallel for
	for (int i = 0; i < m_numParticles; i++) {
		Vector3f p = make_vector(dPos[4 * i], dPos[4 * i + 1], dPos[4 * i + 2]);
		Vector3f diff = p - sphere;
		if (length(diff) <= m_params.colliderRadius) {
			dVel[4 * i] = 0.f;
			dVel[4 * i + 1] = 0.f;
			dVel[4 * i + 2] = 0.f;
		}
	}

#pragma omp parallel for schedule(dynamic, 64)
	//for (int ij = 0; ij < m_numParticles*m_numParticles; ij++) {
	for (int i = 0; i < m_numParticles; i++) {
		for (int j = 0; j < m_numParticles; j++) {

			//	}
			//}
			//int i = ij / m_numParticles;
			//int j = ij % m_numParticles;
			if (i <= j || omp_is_neighbor(i, j, side) || !(omp_check_collision(&dPos[4 * i], &dPos[4 * j], m_params.offset))) {
				continue;
			}
			// reset position
			memcpy(&dPos[4 * i], &prevPos[4 * i], 4 * sizeof(float));
			memcpy(&dPos[4 * j], &prevPos[4 * j], 4 * sizeof(float));

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
}
