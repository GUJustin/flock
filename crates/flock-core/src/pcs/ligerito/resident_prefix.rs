//! Optional arithmetic-only initial PCS prefix; existing CPU bodies are unchanged.
use super::*;
/// Synchronous trusted arithmetic provider. Host owns every transcript operation.
/// An error after admission may follow transcript mutation: abandon that attempt,
/// never replay on the same challenger. Implementations must release failed owners.
pub trait InitialFoldBackend {
    /// Pure presubmit shape admission; called before this function touches transcript.
    fn admit(&self, rows: usize, rounds: usize) -> Result<(), String>;
    /// Consume exact f/basis owners. Ring-switch may already have bound the transcript.
    fn begin(&mut self, f: Vec<F128>, basis: Vec<F128>) -> Result<(), String>;
    /// Fold once at actual host challenge, return unweighted PCS (u0,u2).
    fn advance(&mut self, rho: F128) -> Result<SumcheckMessage, String>;
    /// Return both already-folded planes exactly; no additional folding.
    fn finish(&mut self) -> Result<(Vec<F128>, Vec<F128>), String>;
}
#[allow(clippy::too_many_arguments)]
pub fn prove_with_backend<Ch: Challenger>(
    config: &ProverConfig,
    packed_witness: Vec<F128>,
    b_initial: Vec<F128>,
    target: F128,
    l0_codeword: &[F128],
    l0_tree: &[Hash],
    first_msg: Option<SumcheckMessage>,
    challenger: &mut Ch,
    backend: &mut dyn InitialFoldBackend,
) -> Result<LigeritoProof, String> {
    let rows = packed_witness.len();
    if rows < 4
        || !rows.is_power_of_two()
        || b_initial.len() != rows
        || config.initial_k == 0
        || config.initial_k >= rows.trailing_zeros() as usize
    {
        return Err("resident PCS prefix shape".into());
    }
    backend.admit(rows, config.initial_k)?;
    let log_n = rows.trailing_zeros() as usize;
    let r = config.recursive_steps;
    let initial_k = config.initial_k;

    assert_eq!(packed_witness.len(), 1usize << log_n);
    assert_eq!(b_initial.len(), 1usize << log_n);
    assert_eq!(config.recursive_ks.len(), r);
    assert_eq!(config.log_inv_rates.len(), r + 1);
    assert!(r >= 1);

    let log_inv_rate_0 = config.log_inv_rates[0];
    let log_msg_cols_0 = log_n - initial_k;
    let block_len_0 = 1usize << (log_msg_cols_0 + log_inv_rate_0);
    let num_interleaved_0 = 1usize << initial_k;
    assert_eq!(l0_codeword.len(), block_len_0 * num_interleaved_0);
    assert_eq!(l0_tree.len(), 2 * block_len_0 - 1);

    let trace = std::env::var("LIG_PROVE_TRACE").is_ok();
    let mut t_init_sumcheck = std::time::Duration::ZERO;
    let mut t_commits = std::time::Duration::ZERO;
    let mut t_opens = std::time::Duration::ZERO;
    let mut t_induce = std::time::Duration::ZERO;
    let mut t_sumcheck_folds = std::time::Duration::ZERO;
    let mut t_intro_glue = std::time::Duration::ZERO;
    let mut t_ood = std::time::Duration::ZERO;

    let t_total = std::time::Instant::now();

    challenger.observe_label(b"flock-ligerito-basis-v0");
    challenger.observe_f128(target);

    // L0 codeword + tree are borrowed (reused from upstream `pcs::commit`).
    // wtns_0 access reduces to: root (last tree node), row(q), block_len.
    let initial_root: Hash = l0_tree[l0_tree.len() - 1];
    let l0_block_len = block_len_0;
    let l0_num_interleaved = num_interleaved_0;
    let l0_row = |q: usize| -> &[F128] {
        let start = q * l0_num_interleaved;
        &l0_codeword[start..start + l0_num_interleaved]
    };
    challenger.observe_bytes(&initial_root);

    // L0 takes no explicit OOD samples: it is bound by the opening's own
    // evaluation claim (`target` at the post-commit random point behind
    // `b_initial`), which plays the OOD role with a union over the list
    // instead of over pairs. See `paper_ood_bits`.
    assert_eq!(
        config.ood_samples.first().copied().unwrap_or(0),
        0,
        "L0 must not take explicit OOD samples"
    );
    let mut ood_values: Vec<F128> = Vec::new();
    let mut fold_grinding_nonces: Vec<u64> = Vec::new();
    let fold_bits =
        |lvl: usize| -> u32 { config.fold_grinding_bits.get(lvl).copied().unwrap_or(0) as u32 };
    let ood_count = |lvl: usize| -> usize { config.ood_samples.get(lvl).copied().unwrap_or(0) };

    let _t = std::time::Instant::now();
    let (mut sc_prover, start_msg) = match first_msg {
        Some(msg) => SumcheckProver::new_with_first_msg(packed_witness, b_initial, target, msg),
        None => SumcheckProver::new(packed_witness, b_initial, target),
    };
    // Transfer only planes; keep target, initial message/history and pending glue.
    backend.begin(
        std::mem::take(&mut sc_prover.f),
        std::mem::take(&mut sc_prover.combined_basis),
    )?;
    challenger.observe_f128(start_msg.u_0);
    challenger.observe_f128(start_msg.u_2);

    let mut r_lane_fold = Vec::with_capacity(initial_k);
    for j in 0..initial_k {
        // Fold-challenge grinding: the L0 proximity-gap bad event lives on
        // each of these lane-fold challenges, so each one is individually
        // PoW-guarded (a cheating prover re-rolls a fold challenge by
        // varying the preceding sumcheck message; the grind prices every
        // such attempt). Tapered per round: round j folds a 2^{ℓ-j}-row word
        // whose MCA error carries the factor 2^{ℓ-1-j} (App. C.3 Lemma
        // `mca-commutes`), so it needs (fold_bits − j) bits — one fewer per
        // round than the worst (j=0) round `fold_grinding_bits` is sized for.
        // Derived from fold_grinding_bits + round index; not stored.
        let bits = fold_bits(0).saturating_sub(j as u32);
        if bits > 0 {
            fold_grinding_nonces.push(challenger.grind_pow(bits));
        }
        let r = challenger.sample_f128();
        let msg = backend.advance(r)?;
        sc_prover.transcript.push(msg);
        challenger.observe_f128(msg.u_0);
        challenger.observe_f128(msg.u_2);
        r_lane_fold.push(r);
    }
    if trace {
        t_init_sumcheck += _t.elapsed();
    }

    let expected_tail = rows >> initial_k;
    let (f, basis) = backend.finish()?;
    if f.len() != expected_tail || basis.len() != expected_tail {
        return Err("resident PCS tail shape".into());
    }
    sc_prover.f = f;
    sc_prover.combined_basis = basis;
    debug_assert!(sc_prover.pending_glue.is_none());
    debug_assert_eq!(sc_prover.t_r, target);

    // Commit f^1 = folded packed witness as wtns_1.
    let n1 = log_n - initial_k;
    let log_num_interleaved_1 = config.recursive_ks[0];
    assert!(n1 >= log_num_interleaved_1);
    let log_msg_cols_1 = n1 - log_num_interleaved_1;
    let log_inv_rate_1 = config.log_inv_rates[1];
    let _t = std::time::Instant::now();
    let ntt_1 = AdditiveNttF128::standard(log_msg_cols_1 + log_inv_rate_1);
    let f1 = sc_prover.f().to_vec();
    let wtns_1 = ligero_commit(
        &f1,
        log_msg_cols_1,
        log_num_interleaved_1,
        log_inv_rate_1,
        &ntt_1,
        config.merkle_hash,
    );
    if trace {
        t_commits += _t.elapsed();
    }
    challenger.observe_bytes(&wtns_1.root());

    // OOD binding for the L1 commit: each sample evaluates f1's multilinear
    // extension at a random transcript point z ∈ F^{n1}, sends the claimed
    // value, and folds the claim `Σ_x f1(x)·eq(z,x) = y` into the running
    // sumcheck (introduce + glue). Binds the prover to a single codeword of
    // the interleaved list before any of L0's queries are drawn.
    {
        let _t = std::time::Instant::now();
        for _ in 0..ood_count(1) {
            let z = challenger.sample_f128_vec(n1);
            // Build eq(z, ·) once and fuse the MLE eval `y = f̂1(z)` into the
            // introduce round message (single pass over f1 + eq_z), instead of
            // a separate `mle_eval_inline` fold.
            let eq_z = build_eq_table(&z);
            let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
            challenger.observe_f128(y);
            ood_values.push(y);
            challenger.observe_f128(intro.u_0);
            challenger.observe_f128(intro.u_2);
            let beta = challenger.sample_f128();
            sc_prover.glue(beta);
        }
        if trace {
            t_ood += _t.elapsed();
        }
    }

    // Query-phase PoW grinding for L0: each ground bit substitutes for
    // ~1/log₂(1/(1−γ)) queries at this level (the Slim profile grinds 16
    // bits here). Verifier mirror checks the nonce; both then proceed to
    // sample query positions. (The proximity-gap shortfall is covered
    // separately by the fold-challenge grinds above.)
    let pow_nonce_0 = challenger.grind_pow(config.grinding_bits[0] as u32);
    let mut grinding_nonces: Vec<u64> = vec![pow_nonce_0];

    // Open L0; lane-fold weights = r_lane_fold.
    let num_queries_0 = config.queries[0];
    let queries_0 = sample_distinct_queries(challenger, l0_block_len, num_queries_0);
    let alpha_0 = challenger.sample_f128_vec(ceil_log2(num_queries_0));
    let _t = std::time::Instant::now();
    let opened_rows_0: Vec<Vec<F128>> = queries_0.iter().map(|&q| l0_row(q).to_vec()).collect();
    let merkle_proof_0 = merkle_multi_proof_for(l0_tree, l0_block_len, &queries_0);
    if trace {
        t_opens += _t.elapsed();
    }
    let initial_proof = RecursiveProof {
        opened_rows: opened_rows_0.clone(),
        merkle_proof: merkle_proof_0,
    };

    // Induce basis_0 from wtns_0 opens. L0 dominates the induce phase, where the
    // sparse-prefix Fᵀ-NTT path wins; the dispatcher auto-selects it (deeper
    // levels stay dense).
    let sks_vks_n1 = eval_sk_at_vks(n1);
    let _t = std::time::Instant::now();
    let (basis_0_induced, enforced_sum_0) = induce_sumcheck_poly_auto(
        n1,
        log_inv_rate_0,
        &sks_vks_n1,
        &opened_rows_0,
        &r_lane_fold,
        &queries_0,
        &alpha_0,
    );
    if trace {
        t_induce += _t.elapsed();
    }

    // Introduce + glue basis_0.
    let _t = std::time::Instant::now();
    let intro_msg_0 = sc_prover.introduce_new(basis_0_induced, enforced_sum_0);
    challenger.observe_f128(intro_msg_0.u_0);
    challenger.observe_f128(intro_msg_0.u_2);
    let beta_0 = challenger.sample_f128();
    sc_prover.glue(beta_0);
    if trace {
        t_intro_glue += _t.elapsed();
    }

    // Recursive levels — same as recursive_prover_inner from here.
    let mut wtns_prev = wtns_1;
    let mut recursive_roots: Vec<Hash> = vec![wtns_prev.root()];
    let mut recursive_proofs: Vec<RecursiveProof> = Vec::new();

    for i in 0..r {
        let k_i = config.recursive_ks[i];
        let mut level_rs = Vec::with_capacity(k_i);
        let _t = std::time::Instant::now();
        for j in 0..k_i {
            // These folds fold level i+1's commitment — fold-challenge
            // grinding guards its proximity-gap term. Tapered per round:
            // round j needs (fold_bits − j) bits (see L0 loop).
            let bits = fold_bits(i + 1).saturating_sub(j as u32);
            if bits > 0 {
                fold_grinding_nonces.push(challenger.grind_pow(bits));
            }
            let ri = challenger.sample_f128();
            let msg = sc_prover.fold(ri);
            challenger.observe_f128(msg.u_0);
            challenger.observe_f128(msg.u_2);
            level_rs.push(ri);
        }
        if trace {
            t_sumcheck_folds += _t.elapsed();
        }

        if i == r - 1 {
            let yr = sc_prover.f().to_vec();
            for v in &yr {
                challenger.observe_f128(*v);
            }
            // PoW grinding for the last level before sampling its queries.
            let nonce_last = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
            grinding_nonces.push(nonce_last);
            let num_queries_last = config.queries[i + 1];
            let queries_last =
                sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_last);
            // Mirror the verifier's final-level `yr` binding challenges
            // (`alpha_last` then `beta_last`, see the terminal branch of the
            // verifiers). The prover needs neither value (the verifier
            // computes the enforced sum from the Merkle-bound opened rows on
            // its own), but the challenger must consume them symmetrically,
            // or any protocol stage appended after this open desyncs.
            let _alpha_last = challenger.sample_f128_vec(ceil_log2(num_queries_last));
            let _beta_last = challenger.sample_f128();
            let _t = std::time::Instant::now();
            let opened_rows_last: Vec<Vec<F128>> = queries_last
                .iter()
                .map(|&q| wtns_prev.row(q).to_vec())
                .collect();
            let merkle_proof_last =
                merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_last);
            if trace {
                t_opens += _t.elapsed();
            }
            if trace {
                let total = t_total.elapsed();
                eprintln!("[lig-prove] total = {:.2} ms", total.as_secs_f64() * 1e3);
                eprintln!(
                    "  initial sumcheck (initial_k folds + SC build): {:.2} ms",
                    t_init_sumcheck.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  recursive commits (NTT + merkle):              {:.2} ms",
                    t_commits.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  opens (rows + multi-proof):                    {:.2} ms",
                    t_opens.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  induce_sumcheck_poly:                          {:.2} ms",
                    t_induce.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  sumcheck recursive folds:                      {:.2} ms",
                    t_sumcheck_folds.as_secs_f64() * 1e3
                );
                eprintln!(
                    "  introduce_new + glue:                          {:.2} ms",
                    t_intro_glue.as_secs_f64() * 1e3
                );
                if !ood_values.is_empty() {
                    eprintln!(
                        "  OOD samples ({}): MLE evals + glue:            {:.2} ms",
                        ood_values.len(),
                        t_ood.as_secs_f64() * 1e3
                    );
                }
            }
            return Ok(LigeritoProof {
                initial_root,
                initial_proof,
                recursive_roots,
                recursive_proofs,
                final_proof: FinalProof {
                    yr,
                    opened_rows: opened_rows_last,
                    merkle_proof: merkle_proof_last,
                },
                sumcheck_transcript: sc_prover.transcript().to_vec(),
                grinding_nonces,
                ood_values,
                fold_grinding_nonces,
            });
        }

        let n_next = sc_prover.f().len().trailing_zeros() as usize;
        let log_num_interleaved_next = config.recursive_ks[i + 1];
        assert!(n_next >= log_num_interleaved_next);
        let log_msg_cols_next = n_next - log_num_interleaved_next;
        let log_inv_rate_next = config.log_inv_rates[i + 2];
        let _t = std::time::Instant::now();
        let ntt_next = AdditiveNttF128::standard(log_msg_cols_next + log_inv_rate_next);
        let f_evals = sc_prover.f().to_vec();
        let wtns_next = ligero_commit(
            &f_evals,
            log_msg_cols_next,
            log_num_interleaved_next,
            log_inv_rate_next,
            &ntt_next,
            config.merkle_hash,
        );
        if trace {
            t_commits += _t.elapsed();
        }
        let root_next = wtns_next.root();
        challenger.observe_bytes(&root_next);
        recursive_roots.push(root_next);

        // OOD binding for the L_{i+2} commit (same as the L1 block above).
        {
            let _t = std::time::Instant::now();
            for _ in 0..ood_count(i + 2) {
                let z = challenger.sample_f128_vec(n_next);
                let eq_z = build_eq_table(&z);
                let (intro, y) = sc_prover.introduce_new_with_eval(eq_z);
                challenger.observe_f128(y);
                ood_values.push(y);
                challenger.observe_f128(intro.u_0);
                challenger.observe_f128(intro.u_2);
                let beta = challenger.sample_f128();
                sc_prover.glue(beta);
            }
            if trace {
                t_ood += _t.elapsed();
            }
        }

        // PoW grinding for this iteration's query phase.
        let nonce_i = challenger.grind_pow(config.grinding_bits[i + 1] as u32);
        grinding_nonces.push(nonce_i);
        let num_queries_i = config.queries[i + 1];
        let queries_i = sample_distinct_queries(challenger, wtns_prev.block_len, num_queries_i);
        let alpha_i = challenger.sample_f128_vec(ceil_log2(num_queries_i));
        let _t = std::time::Instant::now();
        let opened_rows_i: Vec<Vec<F128>> = queries_i
            .iter()
            .map(|&q| wtns_prev.row(q).to_vec())
            .collect();
        let merkle_proof_i =
            merkle_multi_proof_for(&wtns_prev.tree, wtns_prev.block_len, &queries_i);
        if trace {
            t_opens += _t.elapsed();
        }
        recursive_proofs.push(RecursiveProof {
            opened_rows: opened_rows_i.clone(),
            merkle_proof: merkle_proof_i,
        });

        let sks_vks_i = eval_sk_at_vks(n_next);
        let _t = std::time::Instant::now();
        let (basis_i_induced, enforced_sum_i) = induce_sumcheck_poly(
            n_next,
            &sks_vks_i,
            &opened_rows_i,
            &level_rs,
            &queries_i,
            &alpha_i,
        );
        if trace {
            t_induce += _t.elapsed();
        }

        let _t = std::time::Instant::now();
        let intro_msg_i = sc_prover.introduce_new(basis_i_induced, enforced_sum_i);
        challenger.observe_f128(intro_msg_i.u_0);
        challenger.observe_f128(intro_msg_i.u_2);
        let beta_i = challenger.sample_f128();
        sc_prover.glue(beta_i);
        if trace {
            t_intro_glue += _t.elapsed();
        }

        wtns_prev = wtns_next;
    }

    unreachable!()
}
