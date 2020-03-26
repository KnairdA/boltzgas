#define RADIUS      $radius
#define N_PARTICLES $n_particles
#define DELTA_T     $delta_t

#define ENABLE_3D

typedef float  scalar_t;

#ifdef ENABLE_3D
typedef float3 vec_t;
typedef float4 data_vec_t;
#define N_WALLS 6
#else
typedef float2 vec_t;
typedef float2 data_vec_t;
#define N_WALLS 4
#endif

#define SCALAR_MAX FLT_MAX

__constant float4 wall_normals[N_WALLS] = {
	(float4)(-1., 0., 0., 0.),
	(float4)( 1., 0., 0., 0.),
	(float4)( 0., 1., 0., 0.),
	(float4)( 0.,-1., 0., 0.),
	(float4)( 0., 0.,-1., 0.),
	(float4)( 0., 0., 1., 0.),
};


__constant scalar_t wall_loc[N_WALLS] = {
	-RADIUS, 1.-RADIUS, 1.-RADIUS, -RADIUS, -RADIUS, 1.-RADIUS
};

vec_t wall_normal(unsigned int iWall) {
#ifdef ENABLE_3D
	return wall_normals[iWall].xyz;
#else
	return wall_normals[iWall].xy;
#endif
}

scalar_t solve_wall_collision(vec_t n, scalar_t loc, vec_t p, vec_t v) {
	if (dot(n,v) > 0.) {
		vec_t wall_v = dot(n,v) * n;
		return (loc - dot(p,n)) / dot(wall_v,n);
	} else {
		return SCALAR_MAX;
	}
}

scalar_t pos_min_or_infty(scalar_t t0, scalar_t t1) {
	if (t0 >= 0.) {
		if (t1 >= 0.) {
			return min(min(t0, t1), SCALAR_MAX);
		} else {
			return min(t0, SCALAR_MAX);
		}
	} else {
		if (t1 >= 0.) {
			return min(t1, SCALAR_MAX);
		} else {
			return SCALAR_MAX;
		}
	}
}

scalar_t solve_collision(vec_t p, vec_t v, vec_t p_, vec_t v_) {
	scalar_t a = dot(v-v_, v-v_);
	scalar_t b  = 2.*dot(p-p_, v-v_);
	scalar_t c = dot(p-p_, p-p_) - 4.*RADIUS*RADIUS;
	scalar_t d = b*b - 4.*a*c;

	if (d >= 0.) {
		scalar_t t0 = (-b + sqrt(d))/(2.*a);
		scalar_t t1 = (-b - sqrt(d))/(2.*a);

		return pos_min_or_infty(t0, t1);
	} else {
		return SCALAR_MAX;
	}
}

vec_t getPosition(__global data_vec_t* pos,
                  unsigned int iParticle)
{
#ifdef ENABLE_3D
	return pos[iParticle].xyz;
#else
	return pos[iParticle];
#endif
}

vec_t getVelocity(__global data_vec_t* vel,
                  unsigned int iParticle)
{
#ifdef ENABLE_3D
	return vel[iParticle].xyz;
#else
	return vel[iParticle];
#endif
}

void set(__global data_vec_t* pos,
         __global data_vec_t* vel,
         unsigned int iParticle,
         vec_t p,
         vec_t v)
{
#ifdef ENABLE_3D
	pos[iParticle].xyz = p;
	vel[iParticle].xyz = v;
#else
	pos[iParticle] = p;
	vel[iParticle] = v;
#endif
}

__kernel void evolve(__global data_vec_t* pos_a,
                     __global data_vec_t* vel_a,
                     __global data_vec_t* pos_b,
                     __global data_vec_t* vel_b,
                     __global volatile unsigned int* last_collide)
{
	unsigned int i = get_global_id(0);

	vec_t p = getPosition(pos_a, i);
	vec_t v = getVelocity(vel_a, i);

	unsigned int jParticle = N_PARTICLES;
	scalar_t min2intersect = SCALAR_MAX;

	for (unsigned int iParticle=0; iParticle < N_PARTICLES; ++iParticle) {
		if (iParticle != i && !(last_collide[i] == iParticle && last_collide[iParticle] == i)) {
			vec_t p_ = getPosition(pos_a, iParticle);
			vec_t v_ = getVelocity(vel_a, iParticle);

			scalar_t time2intersect = solve_collision(p, v, p_, v_);

			if (time2intersect < min2intersect) {
				min2intersect = time2intersect;
				jParticle = iParticle;
			}
		}
	}

	unsigned int jWall = N_PARTICLES;
	scalar_t min2wall = SCALAR_MAX;

	for (unsigned int iWall=0; iWall < N_WALLS; ++iWall) {
		scalar_t time2wall = solve_wall_collision(wall_normal(iWall), wall_loc[iWall], p, v);
		if (time2wall < min2wall) {
			min2wall = time2wall;
			jWall = iWall;
		}
	}

	if (min2intersect < DELTA_T) {
		if (min2wall < min2intersect) {
			p += min2wall * v;
			v -= 2*dot(v,wall_normal(jWall)) * wall_normal(jWall);
			p += (DELTA_T - min2wall) * v;
			last_collide[i] = N_PARTICLES;

			set(pos_b, vel_b, i, p, v);
		} else {
			if (i < jParticle) {
				vec_t p_ = getPosition(pos_a, jParticle);
				vec_t v_ = getVelocity(vel_a, jParticle);

				p  += min2intersect * v;
				p_ += min2intersect * v_;

				vec_t omega = normalize(p - p_);

				v  -= dot(getVelocity(vel_a, i) - getVelocity(vel_a, jParticle), omega) * omega;
				v_ -= dot(getVelocity(vel_a, jParticle) - getVelocity(vel_a, i), omega) * omega;

				p  += (DELTA_T - min2intersect) * v;
				p_ += (DELTA_T - min2intersect) * v_;

				set(pos_b, vel_b, i,         p,  v);
				set(pos_b, vel_b, jParticle, p_, v_);

				last_collide[i] = jParticle;
				last_collide[jParticle] = i;
			}
		}
	} else {
		if (min2wall < DELTA_T) {
			p += min2wall * v;
			v -= 2*dot(v,wall_normal(jWall)) * wall_normal(jWall);
			p += (DELTA_T - min2wall) * v;
			last_collide[i] = N_PARTICLES;
		} else {
			p += DELTA_T * v;
		}

		set(pos_b, vel_b, i, p, v);
	}
}

__kernel void get_velocity_norms(__global data_vec_t* velocities, __global scalar_t* norms)
{
	unsigned int i = get_global_id(0);

	norms[i] = length(getVelocity(velocities, i));
}
