/* Fused CartPole-v1 and Acrobot-v1 batches, loaded by control_clock/native.py through ctypes.
 *
 * Resets reproduce gymnasium.make(task).reset(seed=s) bit for bit by repeating NumPy 2.2.6:
 * SeedSequence (numpy/random/bit_generator.pyx), PCG64 seeding and XSL-RR output
 * (numpy/random/src/pcg64/pcg64.{h,c}), next_double (numpy/random/_common.pxd) and
 * Generator.uniform (numpy/random/_generator.pyx, src/distributions/distributions.c).
 * Acrobot's float32 reset observation repeats NumPy's float32 sin/cos for FMA3 targets
 * (numpy/_core/src/umath/loops_trigonometric.dispatch.cpp).
 *
 * Steps repeat the float64 operations of control_clock/vector_env.py in the same order, with
 * Gymnasium's constants. native.py builds with -ffp-contract=off so that no multiply-add is
 * fused unless written as fmaf, and with -fno-builtin for sin, cos, sinf and cosf so that GCC
 * cannot merge a sin and a cos of one argument into sincos: each stays a separate libm call,
 * as in NumPy's loops.
 *
 * Dynamics are adapted from Gymnasium 1.2.1 (MIT) and, for Acrobot, RLPy (BSD-3-Clause).
 * The seeding and float32 trigonometry follow NumPy (BSD-3-Clause, third_party/NumPy-LICENSE).
 * NumPy's SeedSequence derives from Melissa E. O'Neill's seed_seq and carries its own MIT
 * notice (third_party/SeedSequence-LICENSE); its PCG64 comes from Melissa O'Neill's PCG
 * library (MIT, third_party/PCG64-LICENSE). Other notices: docs/ENVIRONMENTS.md.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifndef COLDSTART_SOURCE_SHA256
#define COLDSTART_SOURCE_SHA256 "unknown"
#endif

/* native.py refuses a library whose source digest differs from classic.c on disk. */
const char *coldstart_source_sha256(void) { return COLDSTART_SOURCE_SHA256; }

/* ---- SeedSequence(seed): default pool of four words, no spawn key ---- */

#define INIT_A 0x43b0d7e5u
#define MULT_A 0x931e8875u
#define INIT_B 0x8b51f9ddu
#define MULT_B 0x58f38dedu
#define MIX_MULT_L 0xca01f9ddu
#define MIX_MULT_R 0x4973f715u
#define XSHIFT 16
#define POOL_SIZE 4

static uint32_t hashmix(uint32_t value, uint32_t *hash_const) {
    value ^= *hash_const;
    *hash_const *= MULT_A;
    value *= *hash_const;
    value ^= value >> XSHIFT;
    return value;
}

static uint32_t mix(uint32_t x, uint32_t y) {
    uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
    result ^= result >> XSHIFT;
    return result;
}

/* ---- PCG64 (XSL-RR 128/64) ---- */

typedef unsigned __int128 u128;
#define PCG_MULTIPLIER (((u128)2549297995355413924ULL << 64) | 4865540595714422341ULL)

typedef struct {
    u128 state;
    u128 inc;
} pcg64;

static void pcg64_step(pcg64 *rng) { rng->state = rng->state * PCG_MULTIPLIER + rng->inc; }

/* np.random.PCG64(np.random.SeedSequence(seed)) for a seed below 2**64. */
static void pcg64_seed(uint64_t seed, pcg64 *rng) {
    /* _int_to_uint32_array: lowest word first; zero is one zero word. */
    const uint32_t entropy[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};
    const int words = (seed >> 32) ? 2 : 1;
    uint32_t pool[POOL_SIZE];
    uint32_t hash_const = INIT_A;
    /* mix_entropy; with at most two words, no entropy is left after the pool is filled. */
    for (int i = 0; i < POOL_SIZE; i++) {
        pool[i] = hashmix(i < words ? entropy[i] : 0, &hash_const);
    }
    for (int src = 0; src < POOL_SIZE; src++) {
        for (int dst = 0; dst < POOL_SIZE; dst++) {
            if (src != dst) {
                pool[dst] = mix(pool[dst], hashmix(pool[src], &hash_const));
            }
        }
    }
    /* generate_state(4, np.uint64): eight words cycling over the pool, paired little-endian. */
    uint32_t state[8];
    uint32_t hash_b = INIT_B;
    for (int i = 0; i < 8; i++) {
        uint32_t value = pool[i % POOL_SIZE];
        value ^= hash_b;
        hash_b *= MULT_B;
        value *= hash_b;
        value ^= value >> XSHIFT;
        state[i] = value;
    }
    uint64_t val[4];
    for (int i = 0; i < 4; i++) {
        val[i] = (uint64_t)state[2 * i] | ((uint64_t)state[2 * i + 1] << 32);
    }
    /* pcg64_set_seed: val[0:2] is the initial state, val[2:4] the stream (high word first). */
    const u128 initstate = ((u128)val[0] << 64) | val[1];
    const u128 initseq = ((u128)val[2] << 64) | val[3];
    /* pcg_setseq_128_srandom_r */
    rng->state = 0;
    rng->inc = (initseq << 1) | 1;
    pcg64_step(rng);
    rng->state += initstate;
    pcg64_step(rng);
}

static uint64_t pcg64_next64(pcg64 *rng) {
    pcg64_step(rng);
    const uint64_t value = (uint64_t)(rng->state >> 64) ^ (uint64_t)rng->state;
    const unsigned rot = (unsigned)(rng->state >> 122);
    return (value >> rot) | (value << ((-rot) & 63));
}

/* uint64_to_double, then random_uniform: lower + range * next_double. */
static double uniform(pcg64 *rng, double low, double high) {
    const double range = high - low;
    const double next_double = (pcg64_next64(rng) >> 11) * (1.0 / 9007199254740992.0);
    return low + range * next_double;
}

/* ---- NumPy's float32 sin/cos on FMA3 targets, one lane ---- */

static float numpy_sincosf(float x_in, int cosine) {
    if (isnan(x_in)) {
        return NAN;
    }
    /* Outside Cody-Waite's range NumPy calls libm. */
    const float max_cody = cosine ? 71476.0625f : 117435.992f;
    if (!(fabsf(x_in) <= max_cody)) {
        return cosine ? cosf(x_in) : sinf(x_in);
    }
    /* The source multiplies by 2/pi and then adds the rounding constant, but GCC fused the two
     * into one FMA in the manylinux x86-64 wheel; this matches that binary. The forms can
     * differ only where x * 2/pi lies close to a half-integer, so never for |x| <= 0.1. */
    float quadrant = fmaf(x_in, 0x1.45f306p-1f, 0x1.800000p+23f);
    quadrant = quadrant - 0x1.800000p+23f;
    float x = fmaf(quadrant, -0x1.921fb0p+00f, x_in);
    x = fmaf(quadrant, -0x1.5110b4p-22f, x);
    x = fmaf(quadrant, -0x1.846988p-48f, x);
    const float x2 = x * x;

    float cos_poly = fmaf(0x1.98e616p-16f, x2, -0x1.6c06dcp-10f);
    cos_poly = fmaf(cos_poly, x2, 0x1.55553cp-05f);
    cos_poly = fmaf(cos_poly, x2, -0x1.000000p-01f);
    cos_poly = fmaf(cos_poly, x2, 0x1.000000p+00f);

    float sin_poly = fmaf(0x1.7d3bbcp-19f, x2, -0x1.a06bbap-13f);
    sin_poly = fmaf(sin_poly, x2, 0x1.11119ap-07f);
    sin_poly = fmaf(sin_poly, x2, -0x1.555556p-03f);
    sin_poly = fmaf(sin_poly, x2, 0.0f);
    sin_poly = fmaf(sin_poly, x, x);

    int iquadrant = (int)quadrant;
    if (cosine) {
        iquadrant += 1;
    }
    float result = (iquadrant & 1) == 0 ? sin_poly : cos_poly;
    if ((iquadrant & 2) == 2) {
        result = 0.0f - result;
    }
    return result;
}

/* Exposed so tests can compare it with np.cos and np.sin over the whole float32 range. */
void coldstart_numpy_trig_f32(int64_t count, const float *x, float *out, int cosine) {
    for (int64_t i = 0; i < count; i++) {
        out[i] = numpy_sincosf(x[i], cosine);
    }
}

/* ---- CartPole-v1 ---- */

#define CARTPOLE_GRAVITY 9.8
#define CARTPOLE_MASSCART 1.0
#define CARTPOLE_MASSPOLE 0.1
#define CARTPOLE_TOTAL_MASS (CARTPOLE_MASSPOLE + CARTPOLE_MASSCART)
#define CARTPOLE_LENGTH 0.5
#define CARTPOLE_POLEMASS_LENGTH (CARTPOLE_MASSPOLE * CARTPOLE_LENGTH)
#define CARTPOLE_FORCE_MAG 10.0
#define CARTPOLE_TAU 0.02
#define CARTPOLE_THETA_THRESHOLD (12 * 2 * 3.141592653589793 / 360)
#define CARTPOLE_X_THRESHOLD 2.4

/* Reset every slot whose mask entry is set (every slot when mask is NULL), taking one seed
 * per reset slot in slot order, and clear its elapsed counter and done flag. */
void coldstart_cartpole_reset(int64_t n, const uint8_t *mask, const uint64_t *seeds,
                              double *state, float *obs, int64_t *steps, uint8_t *done) {
    for (int64_t i = 0, k = 0; i < n; i++) {
        if (mask != NULL && !mask[i]) {
            continue;
        }
        pcg64 rng;
        pcg64_seed(seeds[k++], &rng);
        double *s = state + 4 * i;
        float *o = obs + 4 * i;
        for (int j = 0; j < 4; j++) {
            s[j] = uniform(&rng, -0.05, 0.05);
            o[j] = (float)s[j];
        }
        steps[i] = 0;
        done[i] = 0;
    }
}

/* Validate, then step every active slot; returns 1 for an action out of range and 2 for an
 * active slot that already finished, before changing anything. active may be NULL. */
int coldstart_cartpole_step(int64_t n, const int64_t *actions, const uint8_t *active,
                            double *state, float *obs, int64_t *steps, uint8_t *done,
                            double *reward, uint8_t *terminated, uint8_t *truncated,
                            int64_t max_steps) {
    for (int64_t i = 0; i < n; i++) {
        if (actions[i] < 0 || actions[i] >= 2) {
            return 1;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        if ((active == NULL || active[i]) && done[i]) {
            return 2;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        if (active != NULL && !active[i]) {
            reward[i] = 0.0;
            terminated[i] = 0;
            truncated[i] = 0;
            continue;
        }
        double *s = state + 4 * i;
        double x = s[0], x_dot = s[1], theta = s[2], theta_dot = s[3];
        const double force = actions[i] == 1 ? CARTPOLE_FORCE_MAG : -CARTPOLE_FORCE_MAG;
        const double costheta = cos(theta);
        const double sintheta = sin(theta);
        const double temp =
            (force + CARTPOLE_POLEMASS_LENGTH * (theta_dot * theta_dot) * sintheta) /
            CARTPOLE_TOTAL_MASS;
        const double thetaacc =
            (CARTPOLE_GRAVITY * sintheta - costheta * temp) /
            (CARTPOLE_LENGTH *
             (4.0 / 3.0 - CARTPOLE_MASSPOLE * (costheta * costheta) / CARTPOLE_TOTAL_MASS));
        const double xacc =
            temp - CARTPOLE_POLEMASS_LENGTH * thetaacc * costheta / CARTPOLE_TOTAL_MASS;
        x = x + CARTPOLE_TAU * x_dot;
        x_dot = x_dot + CARTPOLE_TAU * xacc;
        theta = theta + CARTPOLE_TAU * theta_dot;
        theta_dot = theta_dot + CARTPOLE_TAU * thetaacc;
        s[0] = x;
        s[1] = x_dot;
        s[2] = theta;
        s[3] = theta_dot;
        float *o = obs + 4 * i;
        for (int j = 0; j < 4; j++) {
            o[j] = (float)s[j];
        }
        const int ended = x < -CARTPOLE_X_THRESHOLD || x > CARTPOLE_X_THRESHOLD ||
                          theta < -CARTPOLE_THETA_THRESHOLD || theta > CARTPOLE_THETA_THRESHOLD;
        reward[i] = 1.0;
        terminated[i] = (uint8_t)ended;
        steps[i] += 1;
        truncated[i] = steps[i] >= max_steps;
        done[i] = terminated[i] | truncated[i];
    }
    return 0;
}

/* ---- Acrobot-v1, default "book" equations ---- */

#define PI 3.141592653589793
#define ACROBOT_DT 0.2
#define LINK_LENGTH_1 1.0
#define LINK_MASS_1 1.0
#define LINK_MASS_2 1.0
#define LINK_COM_POS_1 0.5
#define LINK_COM_POS_2 0.5
#define LINK_MOI 1.0
#define MAX_VEL_1 (4 * PI)
#define MAX_VEL_2 (9 * PI)

void coldstart_acrobot_reset(int64_t n, const uint8_t *mask, const uint64_t *seeds,
                             double *state, float *obs, int64_t *steps, uint8_t *done) {
    for (int64_t i = 0, k = 0; i < n; i++) {
        if (mask != NULL && !mask[i]) {
            continue;
        }
        pcg64 rng;
        pcg64_seed(seeds[k++], &rng);
        float s[4];
        for (int j = 0; j < 4; j++) {
            /* Gymnasium rounds the initial state to float32. */
            s[j] = (float)uniform(&rng, -0.1, 0.1);
            state[4 * i + j] = s[j];
        }
        float *o = obs + 6 * i;
        o[0] = numpy_sincosf(s[0], 1);
        o[1] = numpy_sincosf(s[0], 0);
        o[2] = numpy_sincosf(s[1], 1);
        o[3] = numpy_sincosf(s[1], 0);
        o[4] = s[2];
        o[5] = s[3];
        steps[i] = 0;
        done[i] = 0;
    }
}

/* AcrobotEnv._dsdt; m2 * l1 * lc2 * x**2 is written ((m2 * l1) * lc2) * (x * x), as in Python. */
static void acrobot_derivative(const double *s, double a, double *out) {
    const double m1 = LINK_MASS_1, m2 = LINK_MASS_2, l1 = LINK_LENGTH_1;
    const double lc1 = LINK_COM_POS_1, lc2 = LINK_COM_POS_2, I1 = LINK_MOI, I2 = LINK_MOI;
    const double g = 9.8;
    const double theta1 = s[0], theta2 = s[1], dtheta1 = s[2], dtheta2 = s[3];
    const double c2 = cos(theta2);
    const double s2 = sin(theta2);
    const double d1 =
        m1 * (lc1 * lc1) + m2 * (l1 * l1 + lc2 * lc2 + 2 * l1 * lc2 * c2) + I1 + I2;
    const double d2 = m2 * (lc2 * lc2 + l1 * lc2 * c2) + I2;
    const double phi2 = m2 * lc2 * g * cos(theta1 + theta2 - PI / 2.0);
    const double phi1 = -m2 * l1 * lc2 * (dtheta2 * dtheta2) * s2 -
                        2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * s2 +
                        (m1 * lc1 + m2 * l1) * g * cos(theta1 - PI / 2) + phi2;
    const double ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * (dtheta1 * dtheta1) * s2 - phi2) /
                            (m2 * (lc2 * lc2) + I2 - (d2 * d2) / d1);
    const double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
    out[0] = dtheta1;
    out[1] = dtheta2;
    out[2] = ddtheta1;
    out[3] = ddtheta2;
}

/* Gymnasium's inclusive wrap into [-pi, pi]. */
static double wrap(double x) {
    const double diff = PI - -PI;
    while (x > PI) {
        x = x - diff;
    }
    while (x < -PI) {
        x = x + diff;
    }
    return x;
}

/* Gymnasium's bound: min(max(x, low), high) with Python's comparison order. */
static double bound(double x, double low, double high) {
    x = low > x ? low : x;
    return high < x ? high : x;
}

int coldstart_acrobot_step(int64_t n, const int64_t *actions, const uint8_t *active,
                           double *state, float *obs, int64_t *steps, uint8_t *done,
                           double *reward, uint8_t *terminated, uint8_t *truncated,
                           int64_t max_steps) {
    for (int64_t i = 0; i < n; i++) {
        if (actions[i] < 0 || actions[i] >= 3) {
            return 1;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        if ((active == NULL || active[i]) && done[i]) {
            return 2;
        }
    }
    /* rk4 takes dt = t[1] - t[0] with t = [0, self.dt]. */
    const double dt = ACROBOT_DT;
    const double dt2 = dt / 2.0;
    for (int64_t i = 0; i < n; i++) {
        if (active != NULL && !active[i]) {
            reward[i] = 0.0;
            terminated[i] = 0;
            truncated[i] = 0;
            continue;
        }
        double *s = state + 4 * i;
        const double torque = (double)actions[i] - 1.0;
        double k1[4], k2[4], k3[4], k4[4], y[4];
        /* gymnasium.envs.classic_control.acrobot.rk4 over t = [0, dt]. */
        acrobot_derivative(s, torque, k1);
        for (int j = 0; j < 4; j++) {
            y[j] = s[j] + dt2 * k1[j];
        }
        acrobot_derivative(y, torque, k2);
        for (int j = 0; j < 4; j++) {
            y[j] = s[j] + dt2 * k2[j];
        }
        acrobot_derivative(y, torque, k3);
        for (int j = 0; j < 4; j++) {
            y[j] = s[j] + dt * k3[j];
        }
        acrobot_derivative(y, torque, k4);
        for (int j = 0; j < 4; j++) {
            s[j] = s[j] + dt / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        }
        s[0] = wrap(s[0]);
        s[1] = wrap(s[1]);
        s[2] = bound(s[2], -MAX_VEL_1, MAX_VEL_1);
        s[3] = bound(s[3], -MAX_VEL_2, MAX_VEL_2);
        const double cos1 = cos(s[0]);
        const int ended = -cos1 - cos(s[0] + s[1]) > 1.0;
        float *o = obs + 6 * i;
        o[0] = (float)cos1;
        o[1] = (float)sin(s[0]);
        o[2] = (float)cos(s[1]);
        o[3] = (float)sin(s[1]);
        o[4] = (float)s[2];
        o[5] = (float)s[3];
        reward[i] = ended ? 0.0 : -1.0;
        terminated[i] = (uint8_t)ended;
        steps[i] += 1;
        truncated[i] = steps[i] >= max_steps;
        done[i] = terminated[i] | truncated[i];
    }
    return 0;
}
