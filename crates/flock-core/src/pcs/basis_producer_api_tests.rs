use super::*;
use std::cell::Cell;
use crate::{challenger::FsChallenger,hash::HashKind};
use ligerito::resident_prefix::StartedFoldBackend;
struct Mock {
    phase:u8, admits:Cell<usize>, produces:usize, advances:usize, finishes:usize,
}
impl Mock {fn new(phase:u8)->Self{Self{phase,admits:Cell::new(0),produces:0,advances:0,finishes:0}}}
impl BasisProducerBackend for Mock {
    fn admit_producer(&self,_:usize,_:usize)->Result<(),String>{self.admits.set(self.admits.get()+1);if self.phase==1{Err("admit".into())}else{Ok(())}}
    fn produce(&mut self,_:Vec<F128>,d:&[ring_switch::RsEqInd],_:usize)->Result<ligerito::SumcheckMessage,String>{
        self.produces+=1;assert!(!d.is_empty());
        Err(if self.phase==2{"descriptor"}else{"device"}.into())
    }
}
impl StartedFoldBackend for Mock {
    fn admit_initialized(&self,_:usize,_:usize)->Result<(),String>{panic!("producer failure must not reach started admission")}
    fn advance(&mut self,_:F128)->Result<ligerito::SumcheckMessage,String>{self.advances+=1;panic!("unexpected advance")}
    fn finish(&mut self)->Result<(Vec<F128>,Vec<F128>),String>{self.finishes+=1;panic!("unexpected finish")}
}
fn fixture()->(Vec<F128>,Commitment,ProverData,ligerito::ProverConfig){
    let w=(0..1usize<<12).map(|i|F128::new(i as u64+1,!(i as u64))).collect::<Vec<_>>();
    let(c,d)=commit(&w,&PcsParams{m:19,log_inv_rate:1,log_batch_size:3,profile:Default::default(),merkle_hash:HashKind::Sha256});
    (w,c,d,ligerito::default_config(12,3,1).unwrap())
}
#[test]
fn producer_static_rejections_preserve_transcript_and_never_call_backend(){
    let(w,c,d,cfg)=fixture();let padding=PaddingSpec::dense(19);
    for case in 0..6 {
        let mut points=vec![vec![F128::new(7,13);13]];
        let mut pre_owned:Vec<Vec<F128>>=Vec::new();let mut pd=Vec::new();
        match case {
            0=>pd.push(PackedDirectClaim{point:vec![F128::ZERO;12],value:F128::ZERO,eq_ind:DirectEqInd::Dense(vec![F128::ZERO;w.len()])}),
            1=>points.clear(),2=>points=vec![vec![F128::ONE;13];5],
            3=>{points[0].pop();},4=>pre_owned=vec![vec![F128::ONE;128];2],
            5=>pre_owned=vec![vec![F128::ONE;127]],_=>unreachable!(),
        }
        let refs=points.iter().map(Vec::as_slice).collect::<Vec<_>>();let pre=pre_owned.iter().map(|x|Some(x.as_slice())).collect::<Vec<_>>();
        let mut ch=FsChallenger::new(b"producer-static");let mut untouched=FsChallenger::new(b"producer-static");let mut backend=Mock::new(0);
        assert!(open_batch_mixed_ligerito_with_basis_producer(w.clone(),&d.codeword,&d.merkle_tree,&c,&refs,&pre,&pd,&padding,&cfg,&mut ch,&mut backend).is_err());
        assert_eq!(ch.sample_f128(),untouched.sample_f128());
        assert_eq!((backend.admits.get(),backend.produces,backend.advances,backend.finishes),(0,0,0,0));
    }
}
#[test]
fn producer_failure_stops_at_exact_original_frontend_without_retry(){
    let(w,c,d,cfg)=fixture();let padding=PaddingSpec::dense(19);
    let points=[vec![F128::new(7,13);13],vec![F128::ONE;13]];
    let refs=points.iter().map(Vec::as_slice).collect::<Vec<_>>();
    for phase in 1..=3 {
        let mut ch=FsChallenger::new(b"producer-failure");let mut expected=FsChallenger::new(b"producer-failure");let mut backend=Mock::new(phase);
        if phase!=1 {let _=original_compute_combined_basis_and_target(&w,&refs,&[],&[],&padding,&mut expected,false);}
        let error=open_batch_mixed_ligerito_with_basis_producer(w.clone(),&d.codeword,&d.merkle_tree,&c,&refs,&[],&[],&padding,&cfg,&mut ch,&mut backend).unwrap_err();
        assert_eq!(error,match phase{1=>"admit",2=>"descriptor",_=>"device"});
        assert_eq!(ch.sample_f128(),expected.sample_f128());
        assert_eq!((backend.admits.get(),backend.produces,backend.advances,backend.finishes),(1,usize::from(phase!=1),0,0));
    }
}
