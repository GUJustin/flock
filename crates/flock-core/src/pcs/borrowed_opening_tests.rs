//! Differential oracle retains the exact pre-refactor opening body from726f125.
use super::*;
use crate::challenger::FsChallenger;
use crate::hash::HashKind;
use crate::zerocheck::univariate_skip::build_eq;

fn fixture(
    log_n: usize,
    batch: usize,
    rate: usize,
) -> (Vec<F128>, Commitment, ProverData, ligerito::ProverConfig) {
    let witness = (0..1usize << log_n)
        .map(|i| F128::new((i as u64).wrapping_mul(0x9e3779b97f4a7c15), !(i as u64)))
        .collect::<Vec<_>>();
    let params = PcsParams {
        m: log_n + LOG_PACKING,
        log_inv_rate: rate,
        log_batch_size: batch,
        profile: Default::default(),
        merkle_hash: HashKind::Sha256,
    };
    let (commitment, data) = commit(&witness, &params);
    let cfg = ligerito::default_config(log_n, batch, rate).unwrap();
    (witness, commitment, data, cfg)
}

#[test]
fn borrowed_opening_direct_matches_original_wrapper_and_verifier() {
    for (case, (batch, rate)) in [(2, 1), (3, 2), (4, 1)].into_iter().enumerate() {
        let (witness, commitment, data, cfg) = fixture(12, batch, rate);
        let point = (0..12)
            .map(|i| match case {
                0 => F128::ZERO,
                1 => F128::ONE,
                _ => F128::new(0x715 + i, u64::MAX - i),
            })
            .collect::<Vec<_>>();
        let eq = build_eq(&point);
        let value = witness
            .iter()
            .zip(&eq)
            .fold(F128::ZERO, |a, (x, y)| a + *x * *y);
        let direct = [PackedDirectClaim {
            point: point.clone(),
            value,
            eq_ind: DirectEqInd::Dense(eq),
        }];
        let padding = PaddingSpec::dense(12 + LOG_PACKING);
        let mut original_ch = FsChallenger::new(b"borrowed-direct");
        let original = original_opening(
            witness.clone(),
            &data,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut original_ch,
        );
        let mut wrapper_ch = FsChallenger::new(b"borrowed-direct");
        let wrapper = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
            witness.clone(),
            &data,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut wrapper_ch,
        );
        let mut borrowed_ch = FsChallenger::new(b"borrowed-direct");
        let borrowed = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_codeword(
            witness,
            &data.codeword,
            &data.merkle_tree,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut borrowed_ch,
        )
        .unwrap();
        assert_eq!(
            bincode::serialize(&original).unwrap(),
            bincode::serialize(&wrapper).unwrap()
        );
        assert_eq!(
            bincode::serialize(&original).unwrap(),
            bincode::serialize(&borrowed).unwrap()
        );
        let next = original_ch.sample_f128();
        assert_eq!(wrapper_ch.sample_f128(), next);
        assert_eq!(borrowed_ch.sample_f128(), next);
        let mut verifier = FsChallenger::new(b"borrowed-direct");
        let vc = ligerito::default_verifier_config(12, batch, rate).unwrap();
        verify_opening_batch_ligerito_mixed(
            &commitment,
            &[],
            &[],
            &[],
            &[PackedDirectClaimRef {
                point: &point,
                value,
            }],
            &borrowed,
            &vc,
            &mut verifier,
        )
        .unwrap();
        assert_eq!(verifier.sample_f128(), next);
    }
}

#[test]
fn borrowed_opening_precomputed_ring_switch_matches_original_transcript() {
    let (witness, commitment, data, cfg) = fixture(12, 3, 1);
    let points = [
        (0..13)
            .map(|i| F128::new(i + 17, u64::MAX - i))
            .collect::<Vec<_>>(),
        vec![F128::ONE; 13],
    ];
    let refs = points.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let padding = PaddingSpec::dense(12 + LOG_PACKING);
    let computed = ring_switch::s_hat_v_multi_padded(&witness, &refs, &padding);
    let precomputed = [Some(computed[0].as_slice()), None];
    let mut old_ch = FsChallenger::new(b"borrowed-precomputed");
    let old = original_opening(
        witness.clone(),
        &data,
        &commitment,
        &refs,
        &precomputed,
        &[],
        &padding,
        &cfg,
        &mut old_ch,
    );
    let mut new_ch = FsChallenger::new(b"borrowed-precomputed");
    let new = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_codeword(
        witness,
        &data.codeword,
        &data.merkle_tree,
        &commitment,
        &refs,
        &precomputed,
        &[],
        &padding,
        &cfg,
        &mut new_ch,
    )
    .unwrap();
    assert_eq!(
        bincode::serialize(&old).unwrap(),
        bincode::serialize(&new).unwrap()
    );
    assert_eq!(old_ch.sample_f128(), new_ch.sample_f128());
}

#[test]
fn borrowed_opening_rejects_geometry_before_transcript() {
    let (witness, commitment, data, cfg) = fixture(12, 3, 1);
    for case in 0..12 {
        let mut c = commitment.clone();
        let mut config = cfg.clone();
        let mut w = witness.clone();
        let mut code = data.codeword.as_slice();
        let mut tree = data.merkle_tree.as_slice();
        match case {
            0 => c.params.m = LOG_PACKING - 1,
            1 => c.params.log_batch_size = usize::MAX,
            2 => c.params.log_inv_rate = usize::MAX,
            3 => c.params.m = usize::BITS as usize + LOG_PACKING,
            4 => {
                w.pop();
            }
            5 => code = &code[..code.len() - 1],
            6 => tree = &tree[..tree.len() - 1],
            7 => config.initial_k += 1,
            8 => config.log_inv_rates.clear(),
            9 => config.log_inv_rates[0] += 1,
            10 => config.initial_log_num_interleaved += 1,
            _ => config.initial_log_msg_cols += 1,
        }
        let mut ch = FsChallenger::new(b"invalid-borrowed");
        let mut untouched = ch.clone();
        let result = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_codeword(
            w,
            code,
            tree,
            &c,
            &[],
            &[],
            &[],
            &PaddingSpec::dense(19),
            &config,
            &mut ch,
        );
        assert!(result.is_err(), "case {case}");
        assert_eq!(ch.sample_f128(), untouched.sample_f128(), "case {case}");
    }
}

#[test]
fn borrowed_opening_admits_registered_initial_geometries() {
    use ligerito::LigeritoProfile::{Fast, Grind, Secure, Slim};
    let mut admitted = 0;
    for m in 16..=40 {
        for batch in 2..=8 {
            for profile in [Fast, Grind, Secure, Slim] {
                let params = PcsParams {
                    m,
                    log_inv_rate: profile.log_inv_rate(),
                    log_batch_size: batch,
                    profile,
                    merkle_hash: HashKind::Sha256,
                };
                if let Ok(cfg) = params.ligerito_prover_config() {
                    assert_eq!(cfg.initial_k, batch);
                    assert_eq!(cfg.initial_log_num_interleaved, batch);
                    assert_eq!(cfg.initial_log_msg_cols, m - LOG_PACKING - batch);
                    assert_eq!(
                        cfg.log_inv_rates.first().copied(),
                        Some(params.log_inv_rate)
                    );
                    admitted += 1;
                }
            }
        }
    }
    assert!(admitted > 0);
    let actual = PcsParams {
        m: 33,
        log_inv_rate: 1,
        log_batch_size: 6,
        profile: Grind,
        merkle_hash: HashKind::Sha256,
    };
    assert_eq!(
        actual
            .ligerito_prover_config()
            .unwrap()
            .initial_log_msg_cols,
        20
    );
}

// Exact frozen body and signature; only the function name/visibility differ.
#[allow(clippy::too_many_arguments)]
fn original_opening<Ch: Challenger>(
    packed_witness: Vec<F128>,
    prover_data: &ProverData,
    commitment: &Commitment,
    x_outers: &[&[F128]],
    precomputed_s_hat_v: &[Option<&[F128]>],
    packed_direct: &[PackedDirectClaim],
    padding: &PaddingSpec,
    lig_config: &ligerito::ProverConfig,
    challenger: &mut Ch,
) -> BatchOpeningProofLigerito {
    let trace = std::env::var("PCS_TRACE").is_ok();
    let t_total = std::time::Instant::now();

    assert_eq!(
        lig_config.initial_k, commitment.params.log_batch_size,
        "ligerito initial_k ({}) must match PcsParams.log_batch_size ({}) for L0 reuse",
        lig_config.initial_k, commitment.params.log_batch_size,
    );
    assert_eq!(
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
        "ligerito log_inv_rates[0] ({}) must match PcsParams.log_inv_rate ({}) for L0 reuse",
        lig_config.log_inv_rates[0], commitment.params.log_inv_rate,
    );

    let combined = compute_combined_basis_and_target(
        &packed_witness,
        x_outers,
        precomputed_s_hat_v,
        packed_direct,
        padding,
        challenger,
        trace,
    );

    let t = std::time::Instant::now();
    let ligerito_proof = ligerito::recursive_prover_with_basis_precomputed_round0(
        lig_config,
        packed_witness,
        combined.b_combined,
        combined.target_combined,
        &prover_data.codeword,
        &prover_data.merkle_tree,
        combined.round0_prime,
        challenger,
    );
    if trace {
        eprintln!(
            "  [open_batch] ligerito::recursive_prover_with_basis: {:6.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        eprintln!(
            "  [open_batch] TOTAL: {:6.2} ms",
            t_total.elapsed().as_secs_f64() * 1e3
        );
    }

    BatchOpeningProofLigerito {
        ring_switches: combined.ring_switches,
        ligerito: ligerito_proof,
    }
}

#[cfg(feature = "resident-pcs-prefix")]
#[derive(Default)]
struct TestPrefix {
    a: Vec<F128>,
    b: Vec<F128>,
    advances: usize,
}
#[cfg(feature = "resident-pcs-prefix")]
impl ligerito::resident_prefix::InitialFoldBackend for TestPrefix {
    fn admit(&self, rows: usize, rounds: usize) -> Result<(), String> {
        if !rows.is_power_of_two() || rounds == 0 || rounds >= rows.trailing_zeros() as usize {
            return Err("shape".into());
        }
        Ok(())
    }
    fn begin(&mut self, a: Vec<F128>, b: Vec<F128>) -> Result<(), String> {
        self.a = a;
        self.b = b;
        Ok(())
    }
    fn advance(&mut self, r: F128) -> Result<ligerito::SumcheckMessage, String> {
        let fold = |x: &[F128]| {
            x.chunks_exact(2)
                .map(|p| p[0] + r * (p[0] + p[1]))
                .collect::<Vec<_>>()
        };
        self.a = fold(&self.a);
        self.b = fold(&self.b);
        self.advances += 1;
        let mut u_0 = F128::ZERO;
        let mut u_2 = F128::ZERO;
        for (a, b) in self.a.chunks_exact(2).zip(self.b.chunks_exact(2)) {
            u_0 += a[0] * b[0];
            u_2 += (a[0] + a[1]) * (b[0] + b[1]);
        }
        Ok(ligerito::SumcheckMessage { u_0, u_2 })
    }
    fn finish(&mut self) -> Result<(Vec<F128>, Vec<F128>), String> {
        Ok((std::mem::take(&mut self.a), std::mem::take(&mut self.b)))
    }
}
#[cfg(feature = "resident-pcs-prefix")]
#[test]
fn resident_prefix_cpu_backend_matches_original_proof_and_transcript() {
    for (case, (batch, rate)) in [(2, 1), (3, 2), (4, 1)].into_iter().enumerate() {
        let (witness, commitment, data, mut cfg) = fixture(12, batch, rate);
        cfg.fold_grinding_bits[0] = 2;
        let point = (0..12)
            .map(|i| match case {
                0 => F128::ZERO,
                1 => F128::ONE,
                _ => F128::new(0x715 + i, u64::MAX - i),
            })
            .collect::<Vec<_>>();
        let eq = build_eq(&point);
        let value = witness
            .iter()
            .zip(&eq)
            .fold(F128::ZERO, |a, (x, y)| a + *x * *y);
        let direct = [PackedDirectClaim {
            point: point.clone(),
            value,
            eq_ind: DirectEqInd::Dense(eq),
        }];
        let padding = PaddingSpec::dense(12 + LOG_PACKING);
        let mut original_ch = FsChallenger::new(b"borrowed-direct");
        let original = original_opening(
            witness.clone(),
            &data,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut original_ch,
        );
        let mut wrapper_ch = FsChallenger::new(b"borrowed-direct");
        let wrapper = open_batch_mixed_ligerito_with_precomputed_s_hat_v(
            witness.clone(),
            &data,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut wrapper_ch,
        );
        let mut borrowed_ch = FsChallenger::new(b"borrowed-direct");
        let borrowed = open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_codeword(
            witness.clone(),
            &data.codeword,
            &data.merkle_tree,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut borrowed_ch,
        )
        .unwrap();
        let mut backend = TestPrefix::default();
        let mut resident_ch = FsChallenger::new(b"borrowed-direct");
        let resident = open_batch_mixed_ligerito_with_resident_prefix(
            witness.clone(),
            &data.codeword,
            &data.merkle_tree,
            &commitment,
            &[],
            &[],
            &direct,
            &padding,
            &cfg,
            &mut resident_ch,
            &mut backend,
        )
        .unwrap();
        assert_eq!(
            bincode::serialize(&resident).unwrap(),
            bincode::serialize(&original).unwrap()
        );
        assert_eq!(backend.advances, batch);
        let resident_next = resident_ch.sample_f128();
        assert_eq!(
            bincode::serialize(&original).unwrap(),
            bincode::serialize(&wrapper).unwrap()
        );
        assert_eq!(
            bincode::serialize(&original).unwrap(),
            bincode::serialize(&borrowed).unwrap()
        );
        let next = original_ch.sample_f128();
        assert_eq!(resident_next, next);
        assert_eq!(wrapper_ch.sample_f128(), next);
        assert_eq!(borrowed_ch.sample_f128(), next);
        let mut verifier = FsChallenger::new(b"borrowed-direct");
        let mut vc = ligerito::default_verifier_config(12, batch, rate).unwrap();
        vc.fold_grinding_bits[0] = 2;
        verify_opening_batch_ligerito_mixed(
            &commitment,
            &[],
            &[],
            &[],
            &[PackedDirectClaimRef {
                point: &point,
                value,
            }],
            &resident,
            &vc,
            &mut verifier,
        )
        .unwrap();
        assert_eq!(verifier.sample_f128(), next);
    }
}

#[cfg(feature = "resident-pcs-prefix")]
#[test]
fn resident_prefix_failures_and_malformed_tails_are_fatal_without_replay() {
    use ligerito::resident_prefix::InitialFoldBackend;
    struct Fault {
        inner: TestPrefix,
        phase: u8,
        calls: [usize; 4],
    }
    impl InitialFoldBackend for Fault {
        fn admit(&self, n: usize, rounds: usize) -> Result<(), String> {
            if self.phase == 0 {
                return Err("injected admission".into());
            }
            self.inner.admit(n, rounds)
        }
        fn begin(&mut self, a: Vec<F128>, b: Vec<F128>) -> Result<(), String> {
            self.calls[0] += 1;
            if self.phase == 1 {
                return Err("injected begin".into());
            }
            self.inner.begin(a, b)
        }
        fn advance(&mut self, r: F128) -> Result<ligerito::SumcheckMessage, String> {
            self.calls[1] += 1;
            if self.phase == 2 && self.calls[1] == 2 {
                return Err("injected advance".into());
            }
            self.inner.advance(r)
        }
        fn finish(&mut self) -> Result<(Vec<F128>, Vec<F128>), String> {
            self.calls[2] += 1;
            if self.phase == 3 {
                return Err("injected finish".into());
            }
            let (mut a, mut b) = self.inner.finish()?;
            if self.phase == 4 {
                a.pop();
            }
            if self.phase == 5 {
                b.pop();
            }
            self.calls[3] += 1;
            Ok((a, b))
        }
    }
    let (witness, commitment, data, mut cfg) = fixture(12, 3, 1);
    cfg.fold_grinding_bits[0] = 2;
    let point = vec![F128::ONE; 12];
    let eq = build_eq(&point);
    let value = witness
        .iter()
        .zip(&eq)
        .fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    let direct = [PackedDirectClaim {
        point,
        value,
        eq_ind: DirectEqInd::Dense(eq),
    }];
    for phase in 0..6 {
        let mut final_challenges = Vec::new();
        for _ in 0..2 {
            let mut backend = Fault {
                inner: TestPrefix::default(),
                phase,
                calls: [0; 4],
            };
            let mut ch = FsChallenger::new(b"resident-failure");
            let error = open_batch_mixed_ligerito_with_resident_prefix(
                witness.clone(),
                &data.codeword,
                &data.merkle_tree,
                &commitment,
                &[],
                &[],
                &direct,
                &PaddingSpec::dense(19),
                &cfg,
                &mut ch,
                &mut backend,
            )
            .unwrap_err();
            let (text, calls) = match phase {
                0 => ("injected admission", [0, 0, 0, 0]),
                1 => ("injected begin", [1, 0, 0, 0]),
                2 => ("injected advance", [1, 2, 0, 0]),
                3 => ("injected finish", [1, 3, 1, 0]),
                _ => ("resident PCS tail shape", [1, 3, 1, 1]),
            };
            assert_eq!(error, text);
            assert_eq!(backend.calls, calls);
            // Independent host-only prefix oracle: original combination and CPU fold,
            // no adapter continuation or backend calls. Stop before the failed operation's output.
            let mut expected_ch = FsChallenger::new(b"resident-failure");
            if phase != 0 {
                let combined = compute_combined_basis_and_target(
                    &witness,
                    &[],
                    &[],
                    &direct,
                    &PaddingSpec::dense(19),
                    &mut expected_ch,
                    false,
                );
                expected_ch.observe_label(b"flock-ligerito-basis-v0");
                expected_ch.observe_f128(combined.target_combined);
                expected_ch.observe_bytes(data.merkle_tree.last().unwrap());
                if phase != 1 {
                    let first = ligerito::SumcheckMessage {
                        u_0: combined.round0_prime.0,
                        u_2: combined.round0_prime.1,
                    };
                    let (mut cpu, _) = ligerito::SumcheckProver::new_with_first_msg(
                        witness.clone(),
                        combined.b_combined,
                        combined.target_combined,
                        first,
                    );
                    expected_ch.observe_f128(first.u_0);
                    expected_ch.observe_f128(first.u_2);
                    for j in 0..3 {
                        let bits = 2u32.saturating_sub(j as u32);
                        if bits > 0 {
                            expected_ch.grind_pow(bits);
                        }
                        let rho = expected_ch.sample_f128();
                        if phase == 2 && j == 1 {
                            break;
                        }
                        let msg = cpu.fold(rho);
                        expected_ch.observe_f128(msg.u_0);
                        expected_ch.observe_f128(msg.u_2);
                    }
                }
            }
            let next = ch.sample_f128();
            assert_eq!(
                next,
                expected_ch.sample_f128(),
                "independent stop oracle phase {phase}"
            );
            let untouched = FsChallenger::new(b"resident-failure").sample_f128();
            if phase == 0 {
                assert_eq!(next, untouched);
            } else {
                assert_ne!(next, untouched);
            }
            final_challenges.push(next);
        }
        // A fresh deterministic attempt stops at the same point; never retry the failed challenger.
        assert_eq!(final_challenges[0], final_challenges[1]);
    }
}
