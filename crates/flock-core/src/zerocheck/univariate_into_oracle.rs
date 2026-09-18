// Exact frozen pre-extraction allocating implementation; test only.
fn original_univariate_fold_oracle(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    use rayon::prelude::*;

    assert_eq!(
        k_skip, 6,
        "optimized fold-and-round_pair variant is k_skip=6 only"
    );
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    // Uninit alloc — the parallel loop below writes every slot (dense path)
    // or explicitly writes F128::ZERO at padding holes (padded path).
    // Saves ~22 ms of sequential zero-fill at m=29 (256 MB total) that would
    // otherwise cap the parallel speedup of this phase at ~2.5× on 8 cores.
    let mut a_folded: Vec<F128> = crate::scratch::take_f128(n_out);
    let mut b_folded: Vec<F128> = crate::scratch::take_f128(n_out);

    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    // Parallel: each worker writes one disjoint chunk of a_folded/b_folded
    // and returns its (sum1, sum_inf) contribution. Reduce by F128 XOR.
    let (sum1, sum_inf) = a_folded
        .par_chunks_mut(chunk_size)
        .zip(b_folded.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            let pair_idx_base = x_hi * lo_size;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                let table_ptr = table.data.as_ptr() as *const u8;
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;

                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        // Padding hole: write zero (a_folded/b_folded were alloc'd
                        // uninit, so we have to write every slot we don't fold into).
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;

                    let a0 = fold_one_row_neon_unchecked_8(table_ptr, a_pkt_ptr.add(x0g * 8));
                    let b0 = fold_one_row_neon_unchecked_8(table_ptr, b_pkt_ptr.add(x0g * 8));
                    let a1 = fold_one_row_neon_unchecked_8(table_ptr, a_pkt_ptr.add(x1g * 8));
                    let b1 = fold_one_row_neon_unchecked_8(table_ptr, b_pkt_ptr.add(x1g * 8));

                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;

                    let eq_l = eq_lo[x_lo];
                    let g1 = a1 * b1;
                    p1_acc ^= eq_l.mul_unreduced(g1);
                    let g_inf = (a0 + a1) * (b0 + b1);
                    pinf_acc ^= eq_l.mul_unreduced(g_inf);
                }
            }
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            unsafe {
                let table_ptr = table.data.as_ptr();
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;
                let mut p1_wide = WideGhashX4::zero();
                let mut pinf_wide = WideGhashX4::zero();
                let mut x_lo = 0;

                while x_lo + 4 <= lo_size {
                    let mut a0 = [F128::ZERO; 4];
                    let mut a1 = [F128::ZERO; 4];
                    let mut b0 = [F128::ZERO; 4];
                    let mut b1 = [F128::ZERO; 4];

                    for lane in 0..4 {
                        let pair = x_lo + lane;
                        let x0l = 2 * pair;
                        let x1l = x0l + 1;
                        if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                            a_chunk[x0l] = F128::ZERO;
                            a_chunk[x1l] = F128::ZERO;
                            b_chunk[x0l] = F128::ZERO;
                            b_chunk[x1l] = F128::ZERO;
                            continue;
                        }

                        let x0g = base + x0l;
                        let x1g = x0g + 1;
                        let folded = fold_round2_pair_x86_unchecked_8(
                            table_ptr,
                            a_pkt_ptr.add(x0g * 8),
                            a_pkt_ptr.add(x1g * 8),
                            b_pkt_ptr.add(x0g * 8),
                            b_pkt_ptr.add(x1g * 8),
                        );
                        [a0[lane], a1[lane], b0[lane], b1[lane]] = folded;
                        a_chunk[x0l] = a0[lane];
                        a_chunk[x1l] = a1[lane];
                        b_chunk[x0l] = b0[lane];
                        b_chunk[x1l] = b1[lane];
                    }

                    let a1x4 = f128x4_loadu(a1.as_ptr());
                    let b1x4 = f128x4_loadu(b1.as_ptr());
                    let a_sum_x4 =
                        f128x4_set(a0[0] + a1[0], a0[1] + a1[1], a0[2] + a1[2], a0[3] + a1[3]);
                    let b_sum_x4 =
                        f128x4_set(b0[0] + b1[0], b0[1] + b1[1], b0[2] + b1[2], b0[3] + b1[3]);
                    let g1x4 = ghash_mul_x4(a1x4, b1x4);
                    let g_inf_x4 = ghash_mul_x4(a_sum_x4, b_sum_x4);
                    let eqx4 = f128x4_loadu(eq_lo[x_lo..].as_ptr());
                    p1_wide.mul_acc(eqx4, g1x4);
                    pinf_wide.mul_acc(eqx4, g_inf_x4);
                    x_lo += 4;
                }

                // Small instances can leave a 1- or 2-pair tail.
                while x_lo < lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        x_lo += 1;
                        continue;
                    }

                    let x0g = base + x0l;
                    let x1g = x0g + 1;
                    let [a0, a1, b0, b1] = fold_round2_pair_x86_unchecked_8(
                        table_ptr,
                        a_pkt_ptr.add(x0g * 8),
                        a_pkt_ptr.add(x1g * 8),
                        b_pkt_ptr.add(x0g * 8),
                        b_pkt_ptr.add(x1g * 8),
                    );
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                    x_lo += 1;
                }

                p1_acc ^= p1_wide.fold();
                pinf_acc ^= pinf_wide.fold();
            }
            #[cfg(not(any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                )
            )))]
            {
                let base = x_hi * chunk_size;
                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        // See aarch64 branch above for why this zero write is needed.
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;
                    let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    let g1 = a1 * b1;
                    p1_acc ^= eq_l.mul_unreduced(g1);
                    let g_inf = (a0 + a1) * (b0 + b1);
                    pinf_acc ^= eq_l.mul_unreduced(g_inf);
                }
            }

            let p1 = p1_acc.reduce();
            let pinf = pinf_acc.reduce();
            let eq_h = eq_hi[x_hi];
            (eq_h * p1, eq_h * pinf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
        );

    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}
