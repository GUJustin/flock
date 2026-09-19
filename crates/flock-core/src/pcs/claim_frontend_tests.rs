use super::*;
use crate::challenger::FsChallenger;
use crate::hash::HashKind;
use crate::zerocheck::univariate_skip::build_eq;
fn values(n: usize) -> Vec<F128> {
    (0..n).map(|i| F128::new((i as u64+17).wrapping_mul(0x9e3779b97f4a7c15), !(i as u64+31))).collect()
}
fn direct(w: &[F128], log: usize, sparse: bool) -> PackedDirectClaim {
    let mut point=values(log);
    if sparse { point[0]=F128::ZERO; point[log-1]=F128::ONE; }
    let eq=build_eq(&point);
    let value=w.iter().zip(&eq).fold(F128::ZERO,|a,(x,y)|a+*x**y);
    let eq_ind=if sparse {DirectEqInd::Sparse(ring_switch::build_eq_sparse(&point))} else {DirectEqInd::Dense(eq)};
    PackedDirectClaim {point,value,eq_ind}
}
#[test]
fn frontend_gamma_baked_variants_and_direct_match_original() {
    for log in [3usize,12] {
        let w=values(1<<log); let padding=PaddingSpec::dense(log+LOG_PACKING);
        let mut points=vec![values(log+1);4];
        points[1][log]=F128::ONE; // Windowed, unless all effective coords are Boolean.
        for i in 1..=3 {points[2][i]=F128::ZERO;}
        points[3].fill(F128::ONE); // Single-entry sparse boundary.
        let refs:Vec<_>=points.iter().map(Vec::as_slice).collect();
        for mode in 0..3 {
            let pd=vec![direct(&w,log,false),direct(&w,log,true)];
            let pd=if mode==0 {&[][..]}else{pd.as_slice()};
            let rs=if mode==2 {&[][..]}else{refs.as_slice()};
            let mut old_ch=FsChallenger::new(b"claim-frontend-matrix");
            let old=original_compute_combined_basis_and_target(&w,rs,&[],pd,&padding,&mut old_ch,false);
            let mut ch=FsChallenger::new(b"claim-frontend-matrix");
            let bundle=prepare_claim_bundle(&w,rs,&[],pd,&padding,&mut ch,false);
            if mode!=2 {
                assert!(bundle.rs_results.iter().any(|(_,o)|matches!(o.rs_eq_ind,ring_switch::RsEqInd::Sparse{..})));
                if log==12 {
                    assert!(matches!(bundle.rs_results[0].1.rs_eq_ind,ring_switch::RsEqInd::DeferredDense{..}));
                    assert!(matches!(bundle.rs_results[1].1.rs_eq_ind,ring_switch::RsEqInd::Windowed{..}));
                } else {assert!(matches!(bundle.rs_results[0].1.rs_eq_ind,ring_switch::RsEqInd::Dense(_)));}
            }
            let mut materialized=vec![F128::ZERO;w.len()];
            for (_,o) in &bundle.rs_results {o.rs_eq_ind.add_scaled_into(F128::ONE,&mut materialized);}
            for (p,g) in bundle.packed_direct.iter().zip(&bundle.gammas_pd) {
                let eq=build_eq(&p.point);
                for (b,e) in materialized.iter_mut().zip(eq) {*b+=*g*e;}
            }
            let new=assemble_claim_bundle_cpu(&w,bundle,false);
            assert_eq!(new.b_combined,materialized);
            assert_eq!(new.b_combined,old.b_combined);
            assert_eq!(new.target_combined,old.target_combined);
            assert_eq!(new.round0_prime,old.round0_prime);
            assert_eq!(new.ring_switches,old.ring_switches);
            assert_eq!(ch.sample_f128(),old_ch.sample_f128());
        }
    }
}
#[test]
fn frontend_complete_opening_bytes_and_next_challenge_match_original() {
    let log=12;let w=values(1<<log);let padding=PaddingSpec::dense(log+LOG_PACKING);
    let params=PcsParams {m:log+LOG_PACKING,log_inv_rate:1,log_batch_size:3,profile:Default::default(),merkle_hash:HashKind::Sha256};
    let (commitment,data)=commit(&w,&params);
    let cfg=ligerito::default_config(log,3,1).unwrap();
    for with_rs in [false,true] {
        let points=[values(log+1),vec![F128::ONE;log+1]];
        let refs:Vec<_>=points.iter().map(Vec::as_slice).collect();
        let refs=if with_rs {refs.as_slice()}else{&[]};
        let pre=ring_switch::s_hat_v_multi_padded(&w,refs,&padding);
        let pre:Vec<_>=pre.iter().map(|x|Some(x.as_slice())).collect();
        let pd=[direct(&w,log,false),direct(&w,log,true)];
        let mut old_ch=FsChallenger::new(b"claim-frontend-proof");
        let old=original_frontend_opening(w.clone(),&data.codeword,&data.merkle_tree,&commitment,refs,&pre,&pd,&padding,&cfg,&mut old_ch).unwrap();
        let mut new_ch=FsChallenger::new(b"claim-frontend-proof");
        let new=open_batch_mixed_ligerito_with_precomputed_s_hat_v_and_codeword(w.clone(),&data.codeword,&data.merkle_tree,&commitment,refs,&pre,&pd,&padding,&cfg,&mut new_ch).unwrap();
        assert_eq!(bincode::serialize(&old).unwrap(),bincode::serialize(&new).unwrap());
        let next=old_ch.sample_f128();assert_eq!(new_ch.sample_f128(),next);
        {
            // Independent bit-MLE evaluation binds real RS claims, not proof-derived values.
            let low: [F128; LOG_PACKING] = values(LOG_PACKING).try_into().unwrap();
            let low_eq=build_eq(&low);
            let bits=unpack_witness(&w,log+LOG_PACKING);
            let claims:Vec<_>=refs.iter().map(|point| {
                let high_eq=build_eq(&point[1..]);
                bits.chunks_exact(1<<LOG_PACKING).zip(high_eq).fold(F128::ZERO,|acc,(chunk,high)| {
                    let inner=chunk.iter().zip(&low_eq).fold(F128::ZERO,|v,(bit,weight)| if *bit {v+*weight}else{v});
                    acc+inner*high
                })
            }).collect();
            let bindings:Vec<_>=refs.iter().map(|_|LowBinding::Multilinear{x_low:low}).collect();
            let pdrefs:Vec<_>=pd.iter().map(|p|PackedDirectClaimRef{point:&p.point,value:p.value}).collect();
            let vc=ligerito::default_verifier_config(log,3,1).unwrap();
            let mut verifier=FsChallenger::new(b"claim-frontend-proof");
            verify_opening_batch_ligerito_mixed_bound(&commitment,&claims,&bindings,refs,&pdrefs,&new,&vc,&mut verifier).unwrap();
            assert_eq!(verifier.sample_f128(),next);
        }
    }
}
