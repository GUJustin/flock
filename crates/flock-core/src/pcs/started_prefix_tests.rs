use super::*;
use crate::challenger::FsChallenger;
use crate::hash::HashKind;
use crate::zerocheck::univariate_skip::build_eq;
use ligerito::resident_prefix::{StartedFoldBackend,prove_with_started_backend};
struct StartedCpu {
    a:Vec<F128>, b:Vec<F128>, advances:usize, finishes:usize, fault:u8,
}
impl StartedFoldBackend for StartedCpu {
    fn admit_initialized(&self,n:usize,k:usize)->Result<(),String> {
        if self.fault==1 {return Err("admission".into());}
        if self.a.len()!=n || self.b.len()!=n || self.advances!=0 || self.finishes!=0 || k==0 {return Err("state".into());}
        Ok(())
    }
    fn advance(&mut self,r:F128)->Result<ligerito::SumcheckMessage,String> {
        self.advances+=1;
        if self.fault==2 && self.advances==2 {return Err("advance".into());}
        let fold=|x:&[F128]|x.chunks_exact(2).map(|p|p[0]+r*(p[0]+p[1])).collect::<Vec<_>>();
        self.a=fold(&self.a);self.b=fold(&self.b);
        let (u_0,u_2)=self.a.chunks_exact(2).zip(self.b.chunks_exact(2)).fold((F128::ZERO,F128::ZERO),|(x,y),(a,b)|(x+a[0]*b[0],y+(a[0]+a[1])*(b[0]+b[1])));
        Ok(ligerito::SumcheckMessage{u_0,u_2})
    }
    fn finish(&mut self)->Result<(Vec<F128>,Vec<F128>),String> {
        self.finishes+=1;
        if self.fault==3 {return Err("finish".into());}
        let(mut a,mut b)=(std::mem::take(&mut self.a),std::mem::take(&mut self.b));
        if self.fault==4 {a.pop();} if self.fault==5 {b.pop();}
        Ok((a,b))
    }
}
fn fixture(k:usize)->(Vec<F128>,Commitment,ProverData,ligerito::ProverConfig,Vec<PackedDirectClaim>) {
    let w=(0..1usize<<12).map(|i|F128::new((i as u64+3).wrapping_mul(0x9e3779b97f4a7c15),!(i as u64))).collect::<Vec<_>>();
    let params=PcsParams{m:19,log_inv_rate:1,log_batch_size:k,profile:Default::default(),merkle_hash:HashKind::Sha256};
    let(c,d)=commit(&w,&params);let mut cfg=ligerito::default_config(12,k,1).unwrap();cfg.fold_grinding_bits[0]=2;
    let point=(0..12).map(|i|F128::new(i+31,i+17)).collect::<Vec<_>>();let eq=build_eq(&point);
    let value=w.iter().zip(&eq).fold(F128::ZERO,|a,(x,y)|a+*x**y);
    (w,c,d,cfg,vec![PackedDirectClaim{point,value,eq_ind:DirectEqInd::Dense(eq)}])
}
#[test]
fn started_prefix_matches_original_proof_and_next_challenge() {
    for k in [2,3,4] {for with_rs in [false,true] {
        let(w,c,d,cfg,pd)=fixture(k);let padding=PaddingSpec::dense(19);
        let points=[vec![F128::new(13,27);13],vec![F128::ONE;13]];
        let refs=points.iter().map(Vec::as_slice).collect::<Vec<_>>();let refs=if with_rs{refs.as_slice()}else{&[]};
        let mut old_ch=FsChallenger::new(b"started-prefix");
        let old=original_frontend_opening(w.clone(),&d.codeword,&d.merkle_tree,&c,refs,&[],&pd,&padding,&cfg,&mut old_ch).unwrap();
        let mut ch=FsChallenger::new(b"started-prefix");
        let combined=compute_combined_basis_and_target(&w,refs,&[],&pd,&padding,&mut ch,false);
        let bits=unpack_witness(&w,19);
        let mut backend=StartedCpu{a:w,b:combined.b_combined,advances:0,finishes:0,fault:0};
        let proof=prove_with_started_backend(&cfg,1<<12,combined.target_combined,&d.codeword,&d.merkle_tree,
            ligerito::SumcheckMessage{u_0:combined.round0_prime.0,u_2:combined.round0_prime.1},&mut ch,&mut backend).unwrap();
        let new=BatchOpeningProofLigerito{ring_switches:combined.ring_switches,ligerito:proof};
        assert_eq!(bincode::serialize(&old).unwrap(),bincode::serialize(&new).unwrap());
        let next=old_ch.sample_f128();assert_eq!(ch.sample_f128(),next);
        assert_eq!(backend.advances,k);assert_eq!(backend.finishes,1);assert!(backend.a.is_empty()&&backend.b.is_empty());
        {
            // Independent bit-MLE claims, not values recovered from the proof.
            let low: [F128;LOG_PACKING]=std::array::from_fn(|i|F128::new(i as u64+23,i as u64+41));
            let low_eq=build_eq(&low);
            let claims:Vec<_>=refs.iter().map(|point| {
                let high_eq=build_eq(&point[1..]);
                bits.chunks_exact(1<<LOG_PACKING).zip(high_eq).fold(F128::ZERO,|acc,(chunk,high)| {
                    let inner=chunk.iter().zip(&low_eq).fold(F128::ZERO,|v,(bit,weight)|if *bit{v+*weight}else{v});
                    acc+inner*high
                })
            }).collect();
            let bindings:Vec<_>=refs.iter().map(|_|LowBinding::Multilinear{x_low:low}).collect();
            let mut vc=ligerito::default_verifier_config(12,k,1).unwrap();vc.fold_grinding_bits[0]=2;
            let pdrefs=pd.iter().map(|p|PackedDirectClaimRef{point:&p.point,value:p.value}).collect::<Vec<_>>();
            let mut verifier=FsChallenger::new(b"started-prefix");
            verify_opening_batch_ligerito_mixed_bound(&c,&claims,&bindings,refs,&pdrefs,&new,&vc,&mut verifier).unwrap();
            assert_eq!(verifier.sample_f128(),next);
        }
    }}
}
#[test]
fn started_prefix_admission_and_late_errors_never_restart() {
    let(w,_c,d,cfg,pd)=fixture(3);let padding=PaddingSpec::dense(19);
    for fault in 1..=5 {
        let mut results=Vec::new();
        for _ in 0..2 {
            let mut ch=FsChallenger::new(b"started-prefix-errors");
            let combined=compute_combined_basis_and_target(&w,&[],&[],&pd,&padding,&mut ch,false);
            let mut baseline=FsChallenger::new(b"started-prefix-errors");
            // The frozen original frontend supplies independent transcript prefix
            // and CPU basis. The host stop sequence below never invokes started code.
            let reference=original_compute_combined_basis_and_target(&w,&[],&[],&pd,&padding,&mut baseline,false);
            if fault!=1 {
                baseline.observe_label(b"flock-ligerito-basis-v0");
                baseline.observe_f128(reference.target_combined);
                baseline.observe_bytes(d.merkle_tree.last().unwrap());
                let start=ligerito::SumcheckMessage{u_0:reference.round0_prime.0,u_2:reference.round0_prime.1};
                baseline.observe_f128(start.u_0);baseline.observe_f128(start.u_2);
                let(mut cpu,_)=ligerito::SumcheckProver::new_with_first_msg(
                    w.clone(),reference.b_combined,reference.target_combined,start);
                for j in 0..cfg.initial_k {
                    let bits=(cfg.fold_grinding_bits[0] as u32).saturating_sub(j as u32);
                    if bits>0 {let _=baseline.grind_pow(bits);}
                    let rho=baseline.sample_f128();
                    // Second advance fails AFTER rho sampling, BEFORE its message.
                    if fault==2 && j==1 {break;}
                    let msg=cpu.fold(rho);
                    baseline.observe_f128(msg.u_0);baseline.observe_f128(msg.u_2);
                }
                // finish/tail errors occur before L1 commitment/root observation.
            }
            let mut backend=StartedCpu{a:w.clone(),b:combined.b_combined,advances:0,finishes:0,fault};
            let error=prove_with_started_backend(&cfg,w.len(),combined.target_combined,&d.codeword,&d.merkle_tree,
                ligerito::SumcheckMessage{u_0:combined.round0_prime.0,u_2:combined.round0_prime.1},&mut ch,&mut backend).unwrap_err();
            let expected=match fault {1=>"admission",2=>"advance",3=>"finish",_=>"resident PCS tail shape"};assert_eq!(error,expected);
            assert_eq!(backend.advances,if fault==1{0}else if fault==2{2}else{3});
            assert_eq!(backend.finishes,if fault<=2{0}else{1});
            let next=ch.sample_f128();assert_eq!(next,baseline.sample_f128(),"fault {fault} transcript stop");
            results.push(next);
        }
        assert_eq!(results[0],results[1]);
    }
}
