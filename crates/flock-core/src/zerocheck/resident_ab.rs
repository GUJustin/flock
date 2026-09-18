//! Experimental fallible external arithmetic path; original CPU entrypoints remain untouched.
use super::*;
/// Trusted arithmetic provider for the resident AB prefix of zerocheck.
///
/// The host owns all transcript operations and supplies the actual sampled fold
/// challenges. Implementations must compute the same field messages as the CPU
/// prover; this interface does not verify arbitrary backend arithmetic. Calls are
/// synchronous and may not retain borrowed inputs beyond their lifetimes.
/// After any error the proof attempt must be abandoned, not replayed on the same
/// challenger. Backend owners are responsible for their own failed-state cleanup.
pub trait AbBackend {
    /// Original number of post-univariate-fold rows, before any `advance`.
    fn rows(&self) -> usize;
    /// Compute and retain both full folded planes, returning the first bare
    /// `(G(1), G(infinity))` message with the supplied equality factors.
    /// Packed padding bits must already be zero according to `padding`.
    /// Called once after round-1 messages and the univariate challenge are bound;
    /// an error therefore does not imply an untouched transcript.
    fn produce(
        &mut self,
        a: &[u8],
        b: &[u8],
        m: usize,
        k: usize,
        table: &UniSkipFoldTable,
        r: &[F128],
        padding: &PaddingSpec,
    ) -> Result<[F128; 2], String>;
    /// Fold the current planes exactly once at host challenge `rho`, halve their
    /// length, and return the next message using `r` (whose first factor is ONE).
    /// No challenge may be sampled internally. An error aborts the proof attempt.
    fn advance(&mut self, rho: F128, r: &[F128]) -> Result<[F128; 2], String>;
    /// Export the already-folded current A/B planes, each exactly 4096 elements.
    /// Do not fold again. Called once at the CPU-tail boundary; malformed lengths
    /// are rejected by the host before CPU continuation. An error is fatal.
    fn finish(&mut self) -> Result<(Vec<F128>, Vec<F128>), String>;
}
/// Prove with an external AB prefix and the original CPU continuation.
///
/// Shape admission errors occur before this function touches the challenger or
/// invokes backend arithmetic. Once admitted, errors from the backend or tail
/// validation may occur after transcript mutation: discard that proof attempt
/// and challenger state. There is no automatic fallback or replay. Packed inputs
/// must satisfy the zero-padding contract; only padding geometry is checked.
#[allow(clippy::too_many_arguments)]
pub fn prove_capture_with_backend<C: Challenger>(
    a: &[u8],
    b: &[u8],
    c: &[u8],
    m: usize,
    padding: &PaddingSpec,
    options: ProverOptions,
    challenger: &mut C,
    backend: &mut dyn AbBackend,
) -> Result<(ZerocheckProof, ZerocheckClaim, Vec<F128>), String> {
    // All admission precedes the first transcript operation.
    if !(19..=33).contains(&m) {
        return Err("resident m admission".into());
    }
    let input = (1usize.checked_shl(m as u32).ok_or("input overflow")?) / 8;
    if a.len() != input
        || b.len() != input
        || c.len() != input
        || backend.rows() != (1usize << (m - K_SKIP))
        || padding.k_log < 7
        || padding.k_log > m
        || padding.useful_bits_per_block > (1usize << padding.k_log)
    {
        return Err("resident shape/padding admission".into());
    }
    let (p, q, c) = resident_inner(
        PackedInputs::Borrowed(a, b),
        c,
        m,
        padding,
        true,
        options,
        challenger,
        backend,
    )?;
    Ok((p, q, c.expect("capture requested")))
}
fn resident_inner<C: Challenger>(
    inputs: PackedInputs<'_>,
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    capture_s_hat_v_c: bool,
    options: ProverOptions,
    challenger: &mut C,
    backend: &mut dyn AbBackend,
) -> Result<(ZerocheckProof, ZerocheckClaim, Option<Vec<F128>>), String> {
    let (a_packed, b_packed) = inputs.bytes();
    let k_skip = K_SKIP;
    const N_INNER: usize = 7; // 3 small + 4 medium fixed-constant eq dims
    assert!(
        m >= k_skip + N_INNER,
        "prove requires m >= k_skip + N_INNER (= {})",
        k_skip + N_INNER
    );
    let expected_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected_bytes);
    assert_eq!(b_packed.len(), expected_bytes);
    assert_eq!(c_packed.len(), expected_bytes);
    let n_mlv = m - k_skip;

    challenger.observe_label(b"flock-zerocheck-v0");

    // ---- 1. Sample r (with protocol-fixed constants in the inner 7 dims) ----
    //
    // r layout:
    //   r[0..k_skip]                — sampled (used by verifier for the
    //                                  final check at S; not by the URM)
    //   r[k_skip..k_skip+3]         — protocol small-eq constants φ_8(0xF7..)
    //   r[k_skip+3..k_skip+7]       — protocol medium-eq constants β_i
    //   r[k_skip+7..m]              — sampled (the "outer" eq weights for
    //                                  the URM and multilinear rounds)
    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    // ---- 3. Round 1: URM (extract_c, parallel) ----
    //
    // The optimized URM drops a `C_s = φ_8(0x1C)` scalar from its accumulators
    // (a prover-side optimization tied to the small-eq trick — see the
    // C_s factor analysis in `univariate_skip_optimized`). The wire format
    // must be in "naive" convention so the verifier doesn't need to know
    // about this internal optimization; we restore the C_s factor here.
    let zc_timing = std::env::var_os("FLOCK_ZC_TIMING").is_some();
    let t_round1 = std::time::Instant::now();
    let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
    let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
    let (round1_ab_opt, round1_c_opt, s_hat_v_c) = if capture_s_hat_v_c {
        use univariate_skip_optimized::{
            round1_shift_reduce_extract_c_packed_padded_with_s_hat_v as reduced,
            round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_deferred as deferred,
        };
        let round1 = match options.round1 {
            Round1Mode::Reduced => reduced,
            Round1Mode::Deferred => deferred,
        };
        let (ab, c, s) = round1(
            a_packed, b_packed, c_packed, m, k_skip, &r, &inv_table, padding,
        );
        (ab, c, Some(s))
    } else {
        let (ab, c) = round1_shift_reduce_extract_c_packed_padded(
            a_packed, b_packed, c_packed, m, k_skip, &r, &inv_table, padding,
        );
        (ab, c, None)
    };
    let c_s = c_s_f128();
    let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();
    let round1_c: Vec<F128> = round1_c_opt.iter().map(|x| c_s * *x).collect();
    if zc_timing {
        eprintln!(
            "[zc-timing] round1 URM: {:.2} ms",
            t_round1.elapsed().as_secs_f64() * 1e3
        );
    }

    // ---- 4. Observe round-1 message, sample z (URM fold point) ----
    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = challenger.sample_f128();

    // ---- 5. c_eval = ĉ(z, r_rest) via interpolation of round1_c at z ----
    //
    // round1_c (now in naive convention) carries `P^C(λ) = Σ_x eq(r_rest, x) · ĉ(λ, x)`
    // as its 2^k_skip evaluations on Λ. Interpolating to λ=z gives
    // `ĉ(z, r_rest)` directly (the eq-weighted sum collapses to the MLE
    // evaluation because ĉ is linear). This is **the c-claim** — at point
    // `(z, r_rest)`, *not* `(z, ρ-values)`. ~64 F128 muls + Lagrange weights.
    let final_c_eval = interpolate_at_z_on_lambda(&round1_c, k_skip, z);

    // ---- 6. Round 2: fused fold + first multilinear message ----
    //
    // Convention A wrapping: pass `mlv_arg[0] = ONE` so the function's output
    // `mlv_arg[0] · G(1)` becomes the bare `G(1)` we send on the wire. The
    // verifier samples ρ_1 after observing this message.
    let t_round2 = std::time::Instant::now();
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);
    let [msg_1, msg_inf] = backend.produce(
        a_packed,
        b_packed,
        m,
        k_skip,
        &fold_table,
        &mlv_arg,
        padding,
    )?;
    let mut resident_n = 1usize << (m - k_skip);
    let (mut a_mlv, mut b_mlv) = (Vec::new(), Vec::new());

    if zc_timing {
        eprintln!(
            "[zc-timing] round2 fused fold: {:.2} ms",
            t_round2.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_tail = std::time::Instant::now();
    let mut multilinear_msgs = Vec::with_capacity(n_mlv);
    multilinear_msgs.push((msg_1, msg_inf));
    challenger.observe_f128(msg_1);
    challenger.observe_f128(msg_inf);
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    mlv_rhos.push(challenger.sample_f128());

    // ---- 7. Rounds 3..(n_mlv + 1) — AB only (c is done) ----
    //
    // Iter i: fold (a, b) at ρ_{i+1}, compute round (i+3) message, sample
    // ρ_{i+2}. Use the fused parallel path while log_n ≥ 10; below that the
    // SplitEqGhash inner can't form lo_size ≥ 2, so we fall back to
    // fold_in_place_pair + round_pair_naive.
    //
    // Ping-pong scratch buffers for the fused path: each fused round folds
    // (a_mlv, b_mlv) of size N into size N/2. Rather than allocating — and,
    // worse, `munmap`-ing, which is single-threaded and caps the tail's
    // parallel speedup — a fresh 64 MB buffer per round, we alternate between
    // two persistent buffers. Scratch capacity = N/2 (the largest fused
    // output); only needed when the first round is actually fused.
    // Borrowed input lifetimes remain with the caller; no full-size CPU spare.
    drop(inputs);
    let (mut a_nxt, mut b_nxt) = (Vec::new(), Vec::new());

    for i in 0..(n_mlv - 1) {
        let rho_prev = mlv_rhos[i];
        let log_n_before = if resident_n > 4096 {
            resident_n.trailing_zeros() as usize
        } else {
            a_mlv.len().trailing_zeros() as usize
        };

        // r_next for the next round's message: length log_n_before - 1.
        // r_next[0] = ONE (Convention A factor); r_next[1..] are the eq
        // weights for the remaining variables = r[k_skip + i + 2..m].
        let mut r_next = vec![F128::ONE; log_n_before - 1];
        r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

        let (m1, mi) = if resident_n > 4096 {
            let msg = backend.advance(rho_prev, &r_next)?;
            resident_n /= 2;
            if resident_n == 4096 {
                (a_mlv, b_mlv) = backend.finish()?;
                if a_mlv.len() != 4096 || b_mlv.len() != 4096 {
                    return Err("backend tail geometry".into());
                }
                a_nxt = crate::scratch::take_f128(2048);
                b_nxt = crate::scratch::take_f128(2048);
            }
            (msg[0], msg[1])
        } else if log_n_before >= 10 {
            let half = a_mlv.len() / 2;
            let fold_and_message = if options.fused_tail {
                multilinear::fold_and_compute_round_pair_fused_tail_into
            } else {
                fold_and_compute_round_pair_into
            };
            let (m1, mi) = fold_and_message(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
            );
            // Swap current <-> scratch, then shrink the new current to the
            // folded size. The old (larger) buffer becomes scratch; we only
            // ever write its leading `half` slots next round, so its stale
            // length is harmless.
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            (m1, mi)
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };

        multilinear_msgs.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        mlv_rhos.push(challenger.sample_f128());
    }

    // ---- 8. Final binding at ρ_{n_mlv} (the last challenge) ----
    let rho_last = *mlv_rhos.last().expect("at least one ρ sampled");
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_last);
    debug_assert_eq!(a_mlv.len(), 1);
    debug_assert_eq!(b_mlv.len(), 1);

    let final_a_eval = a_mlv[0];
    let final_b_eval = b_mlv[0];

    // ---- Fiat–Shamir: bind the final â, b̂ claims into the transcript ----
    //
    // These two claims are reduced downstream by lincheck via a *single*
    // random-linear-combination check with coefficient α (`target = α·v_a + v_b`,
    // see `lincheck`). That batching is only sound if α is sampled *after*
    // (v_a, v_b) are committed to the transcript — otherwise a prover that knows
    // α can pick (v_a, v_b) to satisfy the one batched equation while violating
    // the individual checks. So observe them here, before any later challenge
    // (the next one drawn is lincheck's α). `final_c_eval` needs no observe — the
    // verifier recomputes it from the already-absorbed `round1_c`/`z` and rejects
    // on mismatch (see `verify`), so it is already transcript-bound.
    challenger.observe_f128(final_a_eval);
    challenger.observe_f128(final_b_eval);

    // Recycle the four tail buffers (the two len-1 survivors still own their
    // full round-2 capacity) for the next phase/prove.
    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);

    if zc_timing {
        eprintln!(
            "[zc-timing] rounds 3+ tail: {:.2} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
        );
    }

    let r_rest: Vec<F128> = r[k_skip..].to_vec();

    let proof = ZerocheckProof {
        round1_ab,
        round1_c,
        multilinear_rounds: multilinear_msgs,
        final_a_eval,
        final_b_eval,
        final_c_eval,
    };
    let claim = ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: final_a_eval,
        b_eval: final_b_eval,
        c_eval: final_c_eval,
    };
    Ok((proof, claim, s_hat_v_c))
}

#[cfg(test)]
mod tests;
