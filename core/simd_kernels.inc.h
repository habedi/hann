// simd_kernels.inc.h holds the vector body of every SIMD kernel, written
// once against a small macro vocabulary and instantiated per ISA by a
// wrapper file (simd_kernels_avx.inc.h, simd_kernels_avx2.inc.h, and
// simd_kernels_neon.inc.h) that defines the vocabulary and then includes
// this file. cgo compiles every .c file in the package directory, so the
// kernel bodies live here instead of in a .c file. The .inc.h suffix marks the file as a textual include while
// keeping the .h extension the Go build cache tracks.
//
// The vocabulary an instantiation must define before including this file:
//
//   HANN_SUFFIX          the ISA name appended to each kernel, such as avx
//   HANN_TARGET          the per-function target attribute for the ISA, or
//                        empty for an ISA that is a baseline feature of the
//                        architecture, such as NEON on aarch64
//   HANN_UNROLL          independent accumulator chains in the main loop,
//                        1 or 4
//   HANN_VEC             the vector type
//   HANN_LANES           float lanes per vector
//   HANN_ZERO()          a zero vector
//   HANN_LOAD(p)         an unaligned load of HANN_LANES floats from p
//   HANN_STORE(p, v)     an unaligned store of v to p
//   HANN_SET1(x)         a vector with every lane set to x
//   HANN_ADD(x, y)       lanewise addition
//   HANN_SUB(x, y)       lanewise subtraction
//   HANN_MUL(x, y)       lanewise multiplication
//   HANN_DIV(x, y)       lanewise division
//   HANN_FMA(x, y, acc)  acc plus the product of x and y; an ISA without a
//                        fused instruction expands it to
//                        HANN_ADD(acc, HANN_MUL(x, y))
//   HANN_ABS(v)          lanewise absolute value
//   HANN_REDUCE(v)       the horizontal sum of v as a float
//
// The NEON instantiation defines HANN_VEC as float32x4_t and HANN_LANES as
// 4, maps the operations to the vld1q_f32 family, and uses vfmaq_f32 for
// HANN_FMA, vabsq_f32 for HANN_ABS, and vaddvq_f32 for HANN_REDUCE.
//
// The including .c file selects the kernels to emit:
//
//   HANN_EMIT_DISTANCE   euclidean, squared_euclidean, and cosine_distance
//   HANN_EMIT_MANHATTAN  manhattan
//   HANN_EMIT_NORMALIZE  normalize
//
// With HANN_UNROLL set to 4, an accumulation runs a main loop over four
// independent accumulators, because a single accumulator chains every fused
// multiply-add through the previous one and the loop then runs at FMA
// latency instead of FMA throughput. A cleanup loop one vector wide and a
// scalar tail follow, and the accumulators combine pairwise before the
// reduction, so the accumulation order is fixed by the unroll factor. With
// HANN_UNROLL set to 1, only the one-vector loop and the scalar tail
// remain.
//
// This file #undefs the vocabulary at the end, so a translation unit can
// instantiate it several times. The emit flags stay defined, because the
// including .c file owns them.

#if HANN_UNROLL != 1 && HANN_UNROLL != 4
#error "HANN_UNROLL must be 1 or 4"
#endif

#ifdef HANN_EMIT_DISTANCE
// The accumulation shared by euclidean and squared_euclidean: the sum of
// squared differences over the whole input, vector part and scalar tail.
HANN_TARGET
static inline float HANN_FN(sum_squared_diff)(const float* a, const float* b, size_t n) {
    HANN_VEC acc0 = HANN_ZERO();
#if HANN_UNROLL == 4
    HANN_VEC acc1 = HANN_ZERO();
    HANN_VEC acc2 = HANN_ZERO();
    HANN_VEC acc3 = HANN_ZERO();
#endif
    size_t i = 0;
#if HANN_UNROLL == 4
    size_t limit_unrolled = n - (n % (4 * HANN_LANES));
    for (; i < limit_unrolled; i += 4 * HANN_LANES) {
        HANN_VEC d0 = HANN_SUB(HANN_LOAD(a + i), HANN_LOAD(b + i));
        HANN_VEC d1 = HANN_SUB(HANN_LOAD(a + i + HANN_LANES), HANN_LOAD(b + i + HANN_LANES));
        HANN_VEC d2 = HANN_SUB(HANN_LOAD(a + i + 2 * HANN_LANES), HANN_LOAD(b + i + 2 * HANN_LANES));
        HANN_VEC d3 = HANN_SUB(HANN_LOAD(a + i + 3 * HANN_LANES), HANN_LOAD(b + i + 3 * HANN_LANES));
        acc0 = HANN_FMA(d0, d0, acc0);
        acc1 = HANN_FMA(d1, d1, acc1);
        acc2 = HANN_FMA(d2, d2, acc2);
        acc3 = HANN_FMA(d3, d3, acc3);
    }
#endif
    size_t limit = n - (n % HANN_LANES);
    for (; i < limit; i += HANN_LANES) {
        HANN_VEC d = HANN_SUB(HANN_LOAD(a + i), HANN_LOAD(b + i));
        acc0 = HANN_FMA(d, d, acc0);
    }
#if HANN_UNROLL == 4
    HANN_VEC sum_vec = HANN_ADD(HANN_ADD(acc0, acc1), HANN_ADD(acc2, acc3));
#else
    HANN_VEC sum_vec = acc0;
#endif
    float sum = HANN_REDUCE(sum_vec);
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

HANN_TARGET
float HANN_FN(euclidean)(const float* a, const float* b, size_t n) {
    return sqrtf(HANN_FN(sum_squared_diff)(a, b, n));
}

HANN_TARGET
float HANN_FN(squared_euclidean)(const float* a, const float* b, size_t n) {
    return HANN_FN(sum_squared_diff)(a, b, n);
}
#endif // HANN_EMIT_DISTANCE

#ifdef HANN_EMIT_MANHATTAN
HANN_TARGET
float HANN_FN(manhattan)(const float* a, const float* b, size_t n) {
    HANN_VEC acc0 = HANN_ZERO();
#if HANN_UNROLL == 4
    HANN_VEC acc1 = HANN_ZERO();
    HANN_VEC acc2 = HANN_ZERO();
    HANN_VEC acc3 = HANN_ZERO();
#endif
    size_t i = 0;
#if HANN_UNROLL == 4
    size_t limit_unrolled = n - (n % (4 * HANN_LANES));
    for (; i < limit_unrolled; i += 4 * HANN_LANES) {
        acc0 = HANN_ADD(acc0, HANN_ABS(HANN_SUB(HANN_LOAD(a + i), HANN_LOAD(b + i))));
        acc1 = HANN_ADD(acc1, HANN_ABS(HANN_SUB(HANN_LOAD(a + i + HANN_LANES), HANN_LOAD(b + i + HANN_LANES))));
        acc2 = HANN_ADD(acc2, HANN_ABS(HANN_SUB(HANN_LOAD(a + i + 2 * HANN_LANES), HANN_LOAD(b + i + 2 * HANN_LANES))));
        acc3 = HANN_ADD(acc3, HANN_ABS(HANN_SUB(HANN_LOAD(a + i + 3 * HANN_LANES), HANN_LOAD(b + i + 3 * HANN_LANES))));
    }
#endif
    size_t limit = n - (n % HANN_LANES);
    for (; i < limit; i += HANN_LANES) {
        HANN_VEC diff = HANN_SUB(HANN_LOAD(a + i), HANN_LOAD(b + i));
        acc0 = HANN_ADD(acc0, HANN_ABS(diff));
    }
#if HANN_UNROLL == 4
    HANN_VEC sum_vec = HANN_ADD(HANN_ADD(acc0, acc1), HANN_ADD(acc2, acc3));
#else
    HANN_VEC sum_vec = acc0;
#endif
    float sum = HANN_REDUCE(sum_vec);
    for (; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return sum;
}
#endif // HANN_EMIT_MANHATTAN

#ifdef HANN_EMIT_DISTANCE
HANN_TARGET
float HANN_FN(cosine_distance)(const float* a, const float* b, size_t n) {
    HANN_VEC dot0 = HANN_ZERO();
    HANN_VEC na0 = HANN_ZERO();
    HANN_VEC nb0 = HANN_ZERO();
#if HANN_UNROLL == 4
    HANN_VEC dot1 = HANN_ZERO();
    HANN_VEC na1 = HANN_ZERO();
    HANN_VEC nb1 = HANN_ZERO();
    HANN_VEC dot2 = HANN_ZERO();
    HANN_VEC na2 = HANN_ZERO();
    HANN_VEC nb2 = HANN_ZERO();
    HANN_VEC dot3 = HANN_ZERO();
    HANN_VEC na3 = HANN_ZERO();
    HANN_VEC nb3 = HANN_ZERO();
#endif
    size_t i = 0;
#if HANN_UNROLL == 4
    size_t limit_unrolled = n - (n % (4 * HANN_LANES));
    for (; i < limit_unrolled; i += 4 * HANN_LANES) {
        HANN_VEC va0 = HANN_LOAD(a + i);
        HANN_VEC vb0 = HANN_LOAD(b + i);
        dot0 = HANN_FMA(va0, vb0, dot0);
        na0 = HANN_FMA(va0, va0, na0);
        nb0 = HANN_FMA(vb0, vb0, nb0);
        HANN_VEC va1 = HANN_LOAD(a + i + HANN_LANES);
        HANN_VEC vb1 = HANN_LOAD(b + i + HANN_LANES);
        dot1 = HANN_FMA(va1, vb1, dot1);
        na1 = HANN_FMA(va1, va1, na1);
        nb1 = HANN_FMA(vb1, vb1, nb1);
        HANN_VEC va2 = HANN_LOAD(a + i + 2 * HANN_LANES);
        HANN_VEC vb2 = HANN_LOAD(b + i + 2 * HANN_LANES);
        dot2 = HANN_FMA(va2, vb2, dot2);
        na2 = HANN_FMA(va2, va2, na2);
        nb2 = HANN_FMA(vb2, vb2, nb2);
        HANN_VEC va3 = HANN_LOAD(a + i + 3 * HANN_LANES);
        HANN_VEC vb3 = HANN_LOAD(b + i + 3 * HANN_LANES);
        dot3 = HANN_FMA(va3, vb3, dot3);
        na3 = HANN_FMA(va3, va3, na3);
        nb3 = HANN_FMA(vb3, vb3, nb3);
    }
#endif
    size_t limit = n - (n % HANN_LANES);
    for (; i < limit; i += HANN_LANES) {
        HANN_VEC va = HANN_LOAD(a + i);
        HANN_VEC vb = HANN_LOAD(b + i);
        dot0 = HANN_FMA(va, vb, dot0);
        na0 = HANN_FMA(va, va, na0);
        nb0 = HANN_FMA(vb, vb, nb0);
    }
#if HANN_UNROLL == 4
    HANN_VEC dot_vec = HANN_ADD(HANN_ADD(dot0, dot1), HANN_ADD(dot2, dot3));
    HANN_VEC norm_a_vec = HANN_ADD(HANN_ADD(na0, na1), HANN_ADD(na2, na3));
    HANN_VEC norm_b_vec = HANN_ADD(HANN_ADD(nb0, nb1), HANN_ADD(nb2, nb3));
#else
    HANN_VEC dot_vec = dot0;
    HANN_VEC norm_a_vec = na0;
    HANN_VEC norm_b_vec = nb0;
#endif
    float dot = HANN_REDUCE(dot_vec);
    float norm_a = HANN_REDUCE(norm_a_vec);
    float norm_b = HANN_REDUCE(norm_b_vec);
    for (; i < n; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float normA = sqrtf(norm_a);
    float normB = sqrtf(norm_b);
    if (normA == 0.0f || normB == 0.0f) {
        return 1.0f;
    }
    float cosine_similarity = dot / (normA * normB);
    if (cosine_similarity > 1.0f) cosine_similarity = 1.0f;
    if (cosine_similarity < -1.0f) cosine_similarity = -1.0f;
    return 1.0f - cosine_similarity;
}
#endif // HANN_EMIT_DISTANCE

#ifdef HANN_EMIT_NORMALIZE
HANN_TARGET
void HANN_FN(normalize)(float *vec, size_t len) {
    HANN_VEC acc0 = HANN_ZERO();
#if HANN_UNROLL == 4
    HANN_VEC acc1 = HANN_ZERO();
    HANN_VEC acc2 = HANN_ZERO();
    HANN_VEC acc3 = HANN_ZERO();
#endif
    size_t i = 0;
#if HANN_UNROLL == 4
    size_t limit_unrolled = len - (len % (4 * HANN_LANES));
    for (; i < limit_unrolled; i += 4 * HANN_LANES) {
        HANN_VEC v0 = HANN_LOAD(&vec[i]);
        HANN_VEC v1 = HANN_LOAD(&vec[i + HANN_LANES]);
        HANN_VEC v2 = HANN_LOAD(&vec[i + 2 * HANN_LANES]);
        HANN_VEC v3 = HANN_LOAD(&vec[i + 3 * HANN_LANES]);
        acc0 = HANN_FMA(v0, v0, acc0);
        acc1 = HANN_FMA(v1, v1, acc1);
        acc2 = HANN_FMA(v2, v2, acc2);
        acc3 = HANN_FMA(v3, v3, acc3);
    }
#endif
    size_t limit = len - (len % HANN_LANES);
    for (; i < limit; i += HANN_LANES) {
        HANN_VEC v = HANN_LOAD(&vec[i]);
        acc0 = HANN_FMA(v, v, acc0);
    }
#if HANN_UNROLL == 4
    HANN_VEC sum = HANN_ADD(HANN_ADD(acc0, acc1), HANN_ADD(acc2, acc3));
#else
    HANN_VEC sum = acc0;
#endif
    float total = HANN_REDUCE(sum);
    for (; i < len; i++) {
        total += vec[i] * vec[i];
    }
    float norm = sqrtf(total);
    if (norm == 0.0f) return;
    HANN_VEC norm_vec = HANN_SET1(norm);
    i = 0;
    for (; i < limit; i += HANN_LANES) {
        HANN_VEC v = HANN_LOAD(&vec[i]);
        v = HANN_DIV(v, norm_vec);
        HANN_STORE(&vec[i], v);
    }
    for (; i < len; i++) {
        vec[i] /= norm;
    }
}
#endif // HANN_EMIT_NORMALIZE

#undef HANN_SUFFIX
#undef HANN_TARGET
#undef HANN_UNROLL
#undef HANN_VEC
#undef HANN_LANES
#undef HANN_ZERO
#undef HANN_LOAD
#undef HANN_STORE
#undef HANN_SET1
#undef HANN_ADD
#undef HANN_SUB
#undef HANN_MUL
#undef HANN_DIV
#undef HANN_FMA
#undef HANN_ABS
#undef HANN_REDUCE
