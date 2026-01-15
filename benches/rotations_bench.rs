use criterion::{black_box, criterion_group, criterion_main, Criterion};
use math_bench::generators::{random_euler, random_matrix, random_quat};
use nalgebra::{Rotation3, UnitQuaternion};
use rand::Rng;

fn bench_rotation_ops(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    let mut group_comp = c.benchmark_group("Composition (Single)");

    let q1 = random_quat(&mut rng);
    let q2 = random_quat(&mut rng);
    let m1 = random_matrix(&mut rng);
    let m2 = random_matrix(&mut rng);
    let e1 = random_euler(&mut rng);
    let e2 = random_euler(&mut rng);

    group_comp.bench_function("Quaternion", |b| {
        b.iter(|| black_box(q1) * black_box(q2))
    });

    group_comp.bench_function("Matrix", |b| {
        b.iter(|| black_box(m1) * black_box(m2))
    });

    group_comp.bench_function("Euler (via Quat)", |b| {
        b.iter(|| {
            let q_a = UnitQuaternion::from_euler_angles(e1.x, e1.y, e1.z);
            let q_b = UnitQuaternion::from_euler_angles(e2.x, e2.y, e2.z);
            let result = q_a * q_b;
            result.euler_angles()
        })
    });
    group_comp.finish();

    let mut group_batch = c.benchmark_group("Composition (Batch 10k)");
    let size = 10_000;

    let quats_vec: Vec<_> = (0..size).map(|_| random_quat(&mut rng)).collect();
    let matrices_vec: Vec<_> = (0..size).map(|_| random_matrix(&mut rng)).collect();
    let quats_vec2 = quats_vec.clone();
    let matrices_vec2 = matrices_vec.clone();

    group_batch.bench_function("Quaternion Vector", |b| {
        b.iter(|| {
            quats_vec.iter().zip(quats_vec2.iter()).for_each(|(a, b)| {
                let _ = black_box(*a * *b);
            })
        })
    });

    group_batch.bench_function("Matrix Vector", |b| {
        b.iter(|| {
            matrices_vec.iter().zip(matrices_vec2.iter()).for_each(|(a, b)| {
                let _ = black_box(*a * *b);
            })
        })
    });
    group_batch.finish();

    let mut group_interp = c.benchmark_group("Interpolation (Single)");
    let t = 0.5;

    group_interp.bench_function("Quaternion SLERP", |b| {
        b.iter(|| black_box(q1).slerp(&black_box(q2), black_box(t)))
    });

    group_interp.bench_function("Quaternion NLERP", |b| {
        b.iter(|| black_box(q1).nlerp(&black_box(q2), black_box(t)))
    });

    group_interp.bench_function("Matrix (via Quat)", |b| {
        b.iter(|| {
            let qa = UnitQuaternion::from_rotation_matrix(&m1);
            let qb = UnitQuaternion::from_rotation_matrix(&m2);
            let q_res = qa.slerp(&qb, t);
            q_res.to_rotation_matrix()
        })
    });
    group_interp.finish();

    let mut group_interp_batch = c.benchmark_group("Interpolation (Batch 10k)");

    group_interp_batch.bench_function("Quaternion SLERP Vector", |b| {
        b.iter(|| {
            quats_vec.iter().zip(quats_vec2.iter()).for_each(|(a, b)| {
                let _ = black_box(a.slerp(b, t));
            })
        })
    });

    group_interp_batch.bench_function("Matrix Linear Mix (Wrong but fast)", |b| {
        b.iter(|| {
            matrices_vec.iter().zip(matrices_vec2.iter()).for_each(|(a, b)| {
                let _ = black_box(
                    Rotation3::from_matrix_unchecked(
                        a.matrix() * (1.0 - t) + b.matrix() * t
                    )
                );
            })
        })
    });
    group_interp_batch.finish();

    let mut group_accum = c.benchmark_group("Accumulation & Normalization");
    let steps = 100;
    let delta_q = UnitQuaternion::from_euler_angles(0.01, 0.01, 0.01);
    let delta_m = Rotation3::from_euler_angles(0.01, 0.01, 0.01);

    group_accum.bench_function("Quaternion (Mul + Normalize)", |b| {
        b.iter(|| {
            let mut q = q1;
            for _ in 0..steps {
                q = q * delta_q;
                let raw = q.into_inner();
                q = UnitQuaternion::new_normalize(raw);
            }
            black_box(q)
        })
    });

    group_accum.bench_function("Matrix (Mul + Ortho-normalize)", |b| {
        b.iter(|| {
            let mut m = m1;
            for _ in 0..steps {
                m = m * delta_m;
                m = UnitQuaternion::from_rotation_matrix(&m).to_rotation_matrix();
            }
            black_box(m)
        })
    });
    group_accum.finish();

    let mut group_conv = c.benchmark_group("Conversions");

    group_conv.bench_function("Euler to Quaternion", |b| {
        b.iter(|| {
            UnitQuaternion::from_euler_angles(
                black_box(e1.x),
                black_box(e1.y),
                black_box(e1.z)
            )
        })
    });

    group_conv.bench_function("Quaternion to Matrix", |b| {
        b.iter(|| {
            black_box(q1).to_rotation_matrix()
        })
    });

    group_conv.bench_function("Matrix to Quaternion", |b| {
        b.iter(|| {
            UnitQuaternion::from_rotation_matrix(&black_box(m1))
        })
    });

    group_conv.bench_function("Matrix to Euler", |b| {
        b.iter(|| {
            black_box(m1).euler_angles()
        })
    });
    group_conv.finish();

    let mut group_cache = c.benchmark_group("Cache Locality (Random Access)");
    let huge_size = 100_000;

    let indices: Vec<usize> = (0..10_000)
        .map(|_| rng.gen_range(0..huge_size))
        .collect();

    let huge_quats: Vec<_> = (0..huge_size).map(|_| random_quat(&mut rng)).collect();
    let huge_matrices: Vec<_> = (0..huge_size).map(|_| random_matrix(&mut rng)).collect();

    group_cache.bench_function("Quaternion Random Read", |b| {
        b.iter(|| {
            let mut acc = UnitQuaternion::identity();
            for &idx in &indices {
                acc = acc * huge_quats[idx];
            }
            black_box(acc)
        })
    });

    group_cache.bench_function("Matrix Random Read", |b| {
        b.iter(|| {
            let mut acc = Rotation3::identity();
            for &idx in &indices {
                acc = acc * huge_matrices[idx];
            }
            black_box(acc)
        })
    });
    group_cache.finish();
}

criterion_group!(benches, bench_rotation_ops);
criterion_main!(benches);