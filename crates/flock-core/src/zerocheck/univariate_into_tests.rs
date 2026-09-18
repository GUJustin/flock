use super::*;

#[test]
fn into_matches_frozen_original_poisoned_dense_and_padded() {
    for workers in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for (m, k_log, useful) in [
                (7usize, 7usize, 128usize),
                (10, 10, 1024),
                (17, 14, 15409),
                (18, 15, 31401),
                (10, 10, 0),
                (10, 10, 128),
            ] {
                let padding = PaddingSpec {
                    k_log,
                    useful_bits_per_block: useful,
                };
                let bits = 1usize << m;
                let mut a = vec![false; bits];
                let mut b = vec![false; bits];
                for i in 0..bits {
                    if i % (1usize << k_log) < useful {
                        a[i] = i % 7 < 3;
                        b[i] = i % 11 < 5;
                    }
                }
                let ap = pack_bits(&a);
                let bp = pack_bits(&b);
                let n = 1usize << (m - 6);
                for z in [F128::ZERO, F128::ONE, F128::new(u64::MAX, 0x715)] {
                    let table = UniSkipFoldTable::new(6, z);
                    for challenge in [F128::ZERO, F128::ONE, F128::new(0xdeadbeef, u64::MAX)] {
                        let challenges = vec![challenge; m - 6];
                        let expected = original_univariate_fold_oracle(
                            &ap,
                            &bp,
                            m,
                            6,
                            &table,
                            &challenges,
                            &padding,
                        );
                        for poison in [F128::new(u64::MAX, u64::MAX), F128::new(0x123456, 0xabcdef)]
                        {
                            let mut ao = vec![poison; n];
                            let mut bo = vec![poison; n];
                            let msg = uni_skip_fold_and_round_pair_optimized_packed_padded_into(
                                &ap,
                                &bp,
                                m,
                                6,
                                &table,
                                &challenges,
                                &padding,
                                &mut ao,
                                &mut bo,
                            );
                            assert_eq!(ao, expected.0);
                            assert_eq!(bo, expected.1);
                            assert_eq!(msg, (expected.2, expected.3));
                        }
                        let wrapper = uni_skip_fold_and_round_pair_optimized_packed_padded(
                            &ap,
                            &bp,
                            m,
                            6,
                            &table,
                            &challenges,
                            &padding,
                        );
                        assert_eq!(wrapper, expected);
                    }
                }
            }
        });
    }
}

#[test]
fn invalid_shapes_reject_before_destination_write() {
    let m = 10usize;
    let n = 1usize << (m - 6);
    let packed = vec![0u8; 1usize << (m - 3)];
    let poison = F128::new(7, 19);
    for case in 0..11 {
        let mut table = UniSkipFoldTable::new(6, F128::ONE);
        let mut padding = PaddingSpec::dense(m);
        let mut challenges = vec![F128::ONE; m - 6];
        let mut a = vec![poison; if case == 0 { n - 1 } else { n }];
        let mut b = vec![poison; if case == 1 { n + 1 } else { n }];
        let mut input = packed.clone();
        if case == 2 {
            input.pop();
        }
        if case == 3 {
            challenges.pop();
        }
        if case == 4 {
            table.data.pop();
        }
        if case == 5 {
            padding.k_log = usize::MAX;
        }
        if case == 6 {
            padding.useful_bits_per_block = (1usize << m) + 1;
        }
        let before = (a.clone(), b.clone());
        if case == 10 {
            table.n_chunks = 7;
        }
        let bad_k = if case == 9 { 5 } else { 6 };
        let bad_m = if case == 7 {
            5
        } else if case == 8 {
            usize::MAX
        } else {
            m
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            uni_skip_fold_and_round_pair_optimized_packed_padded_into(
                &input,
                &packed,
                bad_m,
                bad_k,
                &table,
                &challenges,
                &padding,
                &mut a,
                &mut b,
            )
        }));
        assert!(result.is_err());
        assert_eq!((a, b), before);
    }
}
