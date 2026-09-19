use super::*;
use crate::r1cs_hashes::keccak::lanes_to_state;

fn inputs(seed: u64) -> [Lanes; N_SUB] {
    let mut word = seed;
    std::array::from_fn(|_| {
        std::array::from_fn(|_| {
            word = word.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = word;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        })
    })
}

#[test]
fn direct_block_matches_packed_witness_with_poisoned_outputs() {
    let cases = [
        [[0; 25]; N_SUB],
        [[u64::MAX; 25]; N_SUB],
        inputs(1),
        inputs(0x123456789abcdef0),
    ];
    for lanes in cases {
        let states: Vec<_> = lanes.iter().map(lanes_to_state).collect();
        let (z, a, b, _) = generate_witness_with_ab_packed_and_lincheck(&states, 3);
        for poison in [
            F128::new(0xaaaaaaaaaaaaaaaa, 0x5555555555555555),
            F128::new(u64::MAX, u64::MAX),
        ] {
            let mut actual = [
                vec![poison; K / 128],
                vec![poison; K / 128],
                vec![poison; K / 128],
            ];
            let [az, aa, ab] = &mut actual;
            generate_block_witness_from_lanes_into(&lanes, az, aa, ab);
            assert_eq!(
                (az.as_slice(), aa.as_slice(), ab.as_slice()),
                (&z[..K / 128], &a[..K / 128], &b[..K / 128])
            );
        }
    }
    // Every padding block is a real all-zero-input computation, including its constant pins.
    let states: Vec<_> = inputs(7).iter().map(lanes_to_state).collect();
    let (z, a, b, _) = generate_witness_with_ab_packed_and_lincheck(&states[..1], 3);
    for block in 0..8 {
        let lanes = if block == 0 {
            [inputs(7)[0], [0; 25], [0; 25]]
        } else {
            [[0; 25]; N_SUB]
        };
        let mut az = vec![F128::ONE; K / 128];
        let mut aa = az.clone();
        let mut ab = az.clone();
        generate_block_witness_from_lanes_into(&lanes, &mut az, &mut aa, &mut ab);
        let range = block * (K / 128)..(block + 1) * (K / 128);
        assert_eq!(
            (az.as_slice(), aa.as_slice(), ab.as_slice()),
            (&z[range.clone()], &a[range.clone()], &b[range])
        );
    }
}

#[test]
fn direct_block_invalid_lengths_reject_before_writes() {
    // This gate is invoked alone with --test-threads=1. Restore the process hook
    // even if an assertion fails; only the six caught admission panics are quiet.
    struct RestoreHook(Option<Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>>);
    impl Drop for RestoreHook {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                std::panic::set_hook(self.0.take().unwrap());
            }
        }
    }

    for plane in 0..3 {
        for delta in [-1isize, 1] {
            let mut outputs = [
                vec![F128::ONE; K / 128],
                vec![F128::ONE; K / 128],
                vec![F128::ONE; K / 128],
            ];
            outputs[plane].resize((K as isize / 128 + delta) as usize, F128::ONE);
            let before = outputs.clone();
            let restore = RestoreHook(Some(std::panic::take_hook()));
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let [z, a, b] = &mut outputs;
                generate_block_witness_from_lanes_into(&inputs(9), z, a, b);
            }));
            drop(restore);
            assert!(result.is_err());
            assert_eq!(outputs, before);
        }
    }
}
