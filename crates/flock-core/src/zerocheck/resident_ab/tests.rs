use super::*;
use crate::challenger::FsChallenger;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    Produce,
    Advance,
    Finish,
    ShortA,
    ShortB,
}
struct CpuBackend {
    rows: usize,
    a: Vec<F128>,
    b: Vec<F128>,
    calls: [usize; 3],
    fault: Fault,
}
impl CpuBackend {
    fn new(m: usize, fault: Fault) -> Self {
        Self {
            rows: 1 << (m - K_SKIP),
            a: Vec::new(),
            b: Vec::new(),
            calls: [0; 3],
            fault,
        }
    }
}
impl AbBackend for CpuBackend {
    fn rows(&self) -> usize {
        self.rows
    }
    fn produce(
        &mut self,
        a: &[u8],
        b: &[u8],
        m: usize,
        k: usize,
        table: &UniSkipFoldTable,
        r: &[F128],
        padding: &PaddingSpec,
    ) -> Result<[F128; 2], String> {
        self.calls[0] += 1;
        if self.fault == Fault::Produce {
            return Err("injected produce".into());
        }
        let (a, b, one, inf) = multilinear::uni_skip_fold_and_round_pair_optimized_packed_padded(
            a, b, m, k, table, r, padding,
        );
        self.a = a;
        self.b = b;
        Ok([one, inf])
    }
    fn advance(&mut self, rho: F128, r: &[F128]) -> Result<[F128; 2], String> {
        self.calls[1] += 1;
        if self.fault == Fault::Advance {
            return Err("injected advance".into());
        }
        // Separate in-place fold and naive message oracle, not the fused backend.
        multilinear::fold_in_place_pair(&mut self.a, &mut self.b, rho);
        let (one, inf) = multilinear::round_pair_naive(&self.a, &self.b, r);
        Ok([one, inf])
    }
    fn finish(&mut self) -> Result<(Vec<F128>, Vec<F128>), String> {
        self.calls[2] += 1;
        if self.fault == Fault::Finish {
            return Err("injected finish".into());
        }
        if self.fault == Fault::ShortA {
            self.a.pop();
        }
        if self.fault == Fault::ShortB {
            self.b.pop();
        }
        Ok((std::mem::take(&mut self.a), std::mem::take(&mut self.b)))
    }
}
fn fixture(m: usize, useful: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>, PaddingSpec) {
    let padding = PaddingSpec {
        k_log: 13,
        useful_bits_per_block: useful,
    };
    let bytes = |mut seed: u64| -> Vec<u8> {
        (0..(1 << (m - 3)))
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect()
    };
    let mut a = bytes(17);
    let mut b = bytes(571);
    for v in [&mut a, &mut b] {
        for bit in 0..v.len() * 8 {
            if bit % (1 << padding.k_log) >= useful {
                v[bit / 8] &= !(1 << (bit % 8));
            }
        }
    }
    let c = a.iter().zip(&b).map(|(a, b)| a & b).collect();
    (a, b, c, padding)
}
#[test]
fn cpu_backend_real_transcript_proof_and_verifier_parity() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap()
        .install(|| {
            for m in [19, 20] {
                for useful in [8192, 3000] {
                    let (a, b, c, padding) = fixture(m, useful);
                    for round1 in [Round1Mode::Reduced, Round1Mode::Deferred] {
                        for fused_tail in [false, true] {
                            let options = ProverOptions { round1, fused_tail };
                            let mut reference = FsChallenger::new(b"portable-resident-ab");
                            let expected = prove_packed_padded_capture_s_hat_v_c_with_options(
                                &a,
                                &b,
                                &c,
                                m,
                                &padding,
                                options,
                                &mut reference,
                            );
                            let mut transcript = FsChallenger::new(b"portable-resident-ab");
                            let mut backend = CpuBackend::new(m, Fault::None);
                            let actual = prove_capture_with_backend(
                                &a,
                                &b,
                                &c,
                                m,
                                &padding,
                                options,
                                &mut transcript,
                                &mut backend,
                            )
                            .unwrap();
                            assert_eq!(actual, expected);
                            assert_eq!(
                                bincode::serialize(&actual.0).unwrap(),
                                bincode::serialize(&expected.0).unwrap()
                            );
                            assert_eq!(backend.calls, [1, m - 18, 1]);
                            let next = transcript.sample_f128_vec(4);
                            assert_eq!(next, reference.sample_f128_vec(4));
                            let mut verifier = FsChallenger::new(b"portable-resident-ab");
                            assert_eq!(verify(m, &actual.0, &mut verifier).unwrap(), actual.1);
                            assert_eq!(next, verifier.sample_f128_vec(4));
                        }
                    }
                }
            }
        });
}
#[test]
fn cpu_backend_errors_and_malformed_tails_are_fatal() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap()
        .install(|| {
            let m = 19;
            let (a, b, c, padding) = fixture(m, 3000);
            for (fault, expected, calls) in [
                (Fault::Produce, "injected produce", [1, 0, 0]),
                (Fault::Advance, "injected advance", [1, 1, 0]),
                (Fault::Finish, "injected finish", [1, 1, 1]),
                (Fault::ShortA, "backend tail geometry", [1, 1, 1]),
                (Fault::ShortB, "backend tail geometry", [1, 1, 1]),
            ] {
                let mut backend = CpuBackend::new(m, fault);
                let mut transcript = FsChallenger::new(b"portable-errors");
                let result = prove_capture_with_backend(
                    &a,
                    &b,
                    &c,
                    m,
                    &padding,
                    ProverOptions::default(),
                    &mut transcript,
                    &mut backend,
                );
                assert_eq!(result.unwrap_err(), expected);
                assert_eq!(backend.calls, calls);
            }
        });
}
#[test]
fn cpu_backend_invalid_admission_preserves_transcript() {
    let m = 19;
    let (a, b, c, padding) = fixture(m, 8192);
    for case in 0..5 {
        let mut backend = CpuBackend::new(m, Fault::Produce);
        if case == 3 {
            backend.rows /= 2;
        }
        let bad_padding = PaddingSpec {
            k_log: 20,
            useful_bits_per_block: 1,
        };
        let mut transcript = FsChallenger::new(b"portable-admission");
        let mut reference = FsChallenger::new(b"portable-admission");
        let result = prove_capture_with_backend(
            if case == 0 { &a[..a.len() - 1] } else { &a },
            &b,
            if case == 4 { &c[..c.len() - 1] } else { &c },
            if case == 1 { 18 } else { m },
            if case == 2 { &bad_padding } else { &padding },
            ProverOptions::default(),
            &mut transcript,
            &mut backend,
        );
        assert!(result.is_err());
        assert_eq!(backend.calls, [0, 0, 0]);
        assert_eq!(transcript.sample_f128_vec(4), reference.sample_f128_vec(4));
    }
}
