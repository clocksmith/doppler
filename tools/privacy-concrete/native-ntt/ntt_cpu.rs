//! Experimental exact-u64 arithmetic adapter; no cryptographic parameter changes.
//! Cache only public evaluation material inside its owning compression key.
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::entities::compressed_modulus_switched_glwe_ciphertext::CompressedModulusSwitchedGlweCiphertext;
use tfhe_ntt::native64::Plan32;
use crate::fhext_classes::CpuCompressionKey;

const N: usize = 2048;
#[derive(Clone)]
struct Rns([Vec<u32>; 5]);
impl Rns {
    fn zero() -> Self { Self(std::array::from_fn(|_| vec![0; N])) }
    fn add(&mut self, rhs: &Self, primes: &[u32; 5]) {
        for (p, (left, right)) in self.0.iter_mut().zip(rhs.0.iter()).enumerate() {
            for (a, b) in left.iter_mut().zip(right.iter()) {
                *a = ((*a as u64 + *b as u64) % primes[p] as u64) as u32;
            }
        }
    }
}
struct Ring {
    plan: Plan32,
    primes: [u32; 5],
    group: usize,
    two64_mod: [i64; 5],
}
impl Ring {
    fn new() -> Self {
        let plan = Plan32::try_new(N).expect("Unsupported exact NTT degree");
        let primes = [plan.ntt_0().modulus(), plan.ntt_1().modulus(), plan.ntt_2().modulus(), plan.ntt_3().modulus(), plan.ntt_4().modulus()];
        // A coefficient of a group sum has magnitude < group*N*2^128.
        // Require that bound to be strictly below half the CRT modulus product.
        // sum(floor(log2(p))) is a conservative lower bound on log2(product(p)).
        let lower: u32 = primes.iter().map(|p| 31 - p.leading_zeros()).sum();
        let available = lower.checked_sub(128 + N.ilog2() + 2).expect("Insufficient CRT headroom");
        let group = 1usize << available.min(7);
        assert!(128 + N.ilog2() + group.ilog2() + 1 < lower);
        let two64_mod=primes.map(|p|((1u128<<64)%p as u128) as i64);
        Self { plan, primes, group, two64_mod }
    }
    fn forward(&self, value: &[u64]) -> Rns {
        assert_eq!(value.len(), N);
        let mut result = Rns::zero();
        let [a,b,c,d,e] = &mut result.0;
        self.plan.fwd(value,a,b,c,d,e);
        result
    }
    fn forward_digits_into(&self, value: &[u64], result: &mut Rns, limit: i64) {
        assert_eq!(value.len(),N);
        assert!(matches!(limit,512|8192));
        assert!(value.iter().all(|&v|checked_digit(v,limit).is_some()),"Packing digit outside the pinned signed decomposition domain");
        for (p,buffer) in result.0.iter_mut().enumerate() {
            let prime=self.primes[p];let correction=self.two64_mod[p];
            for (target,&value) in buffer.iter_mut().zip(value) {
                *target=digit_residue(value as i64,prime,correction);
            }
        }
        // The pinned general fwd overwrites these same five residue arrays
        // before invoking these exact prime transforms. No old scratch survives.
        self.plan.ntt_0().fwd(&mut result.0[0]);
        self.plan.ntt_1().fwd(&mut result.0[1]);
        self.plan.ntt_2().fwd(&mut result.0[2]);
        self.plan.ntt_3().fwd(&mut result.0[3]);
        self.plan.ntt_4().fwd(&mut result.0[4]);
    }
    fn inverse(&self, value: &mut Rns) -> Vec<u64> {
        let mut result = vec![0; N];
        let [a,b,c,d,e] = &mut value.0;
        self.plan.inv(&mut result,a,b,c,d,e);
        result
    }
    fn product(&self, lhs: &Rns, rhs: &Rns) -> Rns {
        let mut result = lhs.clone();
        self.plan.ntt_0().mul_assign_normalize(&mut result.0[0],&rhs.0[0]);
        self.plan.ntt_1().mul_assign_normalize(&mut result.0[1],&rhs.0[1]);
        self.plan.ntt_2().mul_assign_normalize(&mut result.0[2],&rhs.0[2]);
        self.plan.ntt_3().mul_assign_normalize(&mut result.0[3],&rhs.0[3]);
        self.plan.ntt_4().mul_assign_normalize(&mut result.0[4],&rhs.0[4]);
        result
    }
    fn accumulate_product(&self, sum: &mut Rns, lhs: &Rns, rhs: &Rns) {
        // The pinned library supplies SIMD modular multiply/accumulate. Keep
        // products unnormalized only within this same bounded CRT group.
        self.plan.ntt_0().mul_accumulate(&mut sum.0[0],&lhs.0[0],&rhs.0[0]);
        self.plan.ntt_1().mul_accumulate(&mut sum.0[1],&lhs.0[1],&rhs.0[1]);
        self.plan.ntt_2().mul_accumulate(&mut sum.0[2],&lhs.0[2],&rhs.0[2]);
        self.plan.ntt_3().mul_accumulate(&mut sum.0[3],&lhs.0[3],&rhs.0[3]);
        self.plan.ntt_4().mul_accumulate(&mut sum.0[4],&lhs.0[4],&rhs.0[4]);
    }
    fn normalize(&self, value: &mut Rns) {
        self.plan.ntt_0().normalize(&mut value.0[0]);
        self.plan.ntt_1().normalize(&mut value.0[1]);
        self.plan.ntt_2().normalize(&mut value.0[2]);
        self.plan.ntt_3().normalize(&mut value.0[3]);
        self.plan.ntt_4().normalize(&mut value.0[4]);
    }
}

fn checked_digit(value: u64, limit: i64) -> Option<i64> {
    let signed=value as i64;
    if signed>=-limit && signed<=limit { Some(signed) } else { None }
}

#[inline]
fn digit_residue(value: i64, prime: u32, two64_mod: i64) -> u32 {
    // Preserve the UNSIGNED u64 interpretation used by the existing CRT bound.
    // For a negative digit d, u64(d)=2^64+d. The one correction below is exact
    // because |d|<=8192<prime and 0<=two64_mod<prime.
    let reduced=value+if value<0 { two64_mod } else { 0 };
    (reduced+if reduced<0 { prime as i64 } else { 0 }) as u32
}

pub(crate) fn validate_digit_encoding() -> (usize,usize,usize,usize) {
    let ring=Ring::new();let mut scratch=Rns::zero();
    let (mut digits,mut residues,mut polynomials,mut rejected)=(0,0,0,0);
    for bits in [10,14] {
        let limit=1i64<<(bits-1);
        for signed in -limit..=limit {
            assert_eq!(checked_digit(signed as u64,limit),Some(signed));
            for p in 0..5 {
                assert_eq!(digit_residue(signed,ring.primes[p],ring.two64_mod[p]),((signed as u64)%ring.primes[p] as u64) as u32);
                residues+=1;
            }
            digits+=1;
        }
        for value in [limit+1,-limit-1,i64::MIN,i64::MAX] {
            assert_eq!(checked_digit(value as u64,limit),None);rejected+=1;
        }
        for pattern in 0..4 {
            let values: Vec<u64>=(0..N).map(|index| {
                let signed=match pattern {
                    0=>[limit,-limit,limit-1,1-limit,0,1,-1][index%7],
                    1=>(index as i64*391+17)%(2*limit+1)-limit,
                    2=>0,
                    _=>if index%2==0 { -limit } else { limit },
                };
                signed as u64
            }).collect();
            let expected=ring.forward(&values);
            ring.forward_digits_into(&values,&mut scratch,limit);
            assert_eq!(&expected.0,&scratch.0);polynomials+=1;
        }
    }
    (digits,residues,polynomials,rejected)
}

#[derive(Clone, Copy)]
pub(crate) enum PackingMode { SerialReference, ParallelReference, ParallelFused }
#[derive(Clone, Copy)]
pub(crate) enum DotMode { FullGlweReference, ExtractedBody }
#[derive(Clone, Copy)]
pub(crate) enum DigitMode { SerialReference, ParallelBlocks }
#[derive(Clone, Copy)]
pub(crate) enum ForwardMode { AllocatingReference, BoundedScratch }

#[derive(Default)]
pub(crate) struct PackingTimings {
    pub dot_seconds: f64,
    pub digits_seconds: f64,
    pub groups_seconds: f64,
    pub compression_seconds: f64,
}

pub(crate) struct PackingPlan {
    ring: Ring,
    key: Vec<Rns>,
    levels: usize,
    base_log: DecompositionBaseLog,
    glwe_size: GlweSize,
    modulus: CiphertextModulus<u64>,
}
impl PackingPlan {
    fn new(key: &LwePackingKeyswitchKeyOwned<u64>) -> Self {
        assert_eq!(key.input_key_lwe_dimension().0, N);
        assert_eq!(key.output_polynomial_size().0, N);
        assert_eq!(key.output_glwe_size().0, 2);
        assert_eq!(key.ciphertext_modulus(), CiphertextModulus::<u64>::new_native());
        let levels = key.decomposition_level_count().0;
        let base_log = key.decomposition_base_log();
        assert!((levels == 2 && base_log.0 == 14) || (levels == 4 && base_log.0 == 10));
        let ring = Ring::new();
        let transformed = key.as_ref().par_chunks_exact(N).map(|value| ring.forward(value)).collect::<Vec<_>>();
        assert_eq!(transformed.len(), N*levels*2);
        Self { ring, key: transformed, levels, base_log, glwe_size: key.output_glwe_size(), modulus: key.ciphertext_modulus() }
    }
    fn pack<C: Container<Element=u64>>(&self, list: &LweCiphertextList<C>, mode: PackingMode, digit_mode: DigitMode, forward_mode: ForwardMode, mut timings: Option<&mut PackingTimings>) -> GlweCiphertextOwned<u64> {
        let count = list.lwe_ciphertext_count().0;
        assert!(count>0 && count<=N);
        assert_eq!(list.lwe_size().0, N+1);
        assert_eq!(list.ciphertext_modulus(), self.modulus);
        let terms = N*self.levels;
        let started=timings.as_ref().map(|_|Instant::now());
        let mut digits = vec![0u64; terms*N];
        let mut out = GlweCiphertext::new(0u64,self.glwe_size,PolynomialSize(N),self.modulus);
        let decomposer = SignedDecomposer::new(self.base_log,DecompositionLevelCount(self.levels));
        for (column, lwe) in list.iter().enumerate() {
            out.as_mut()[N+column] = *lwe.get_body().data;
            if matches!(digit_mode,DigitMode::SerialReference) {
                for (input, &mask) in lwe.get_mask().as_ref().iter().enumerate() {
                    for (level, digit) in decomposer.decompose(decomposer.closest_representable(mask)).enumerate() {
                        digits[(input*self.levels+level)*N+column] = digit.value();
                    }
                }
            }
        }
        if matches!(digit_mode,DigitMode::ParallelBlocks) {
            // Each worker owns every level/column for eight input masks. The
            // maximum block is 512 KiB, while total storage stays terms*N u64s.
            // Read masks in short contiguous runs; never share mutable digits.
            const INPUTS: usize=8;
            let raw=list.as_ref();
            digits.par_chunks_mut(INPUTS*self.levels*N).enumerate().for_each(|(block,owned)| {
                let first=block*INPUTS;
                for (column,lwe) in raw.chunks_exact(N+1).enumerate() {
                    for (input,&mask) in lwe[first..first+INPUTS].iter().enumerate() {
                        for (level,digit) in decomposer.decompose(decomposer.closest_representable(mask)).enumerate() {
                            owned[(input*self.levels+level)*N+column]=digit.value();
                        }
                    }
                }
            });
        }
        if let Some(value)=timings.as_deref_mut() { value.digits_seconds+=started.unwrap().elapsed().as_secs_f64(); }
        let started=timings.as_ref().map(|_|Instant::now());
        // Each CRT group already has its own strict magnitude bound. Compute
        // those independent groups on the existing pool, then subtract their
        // reconstructions in the original order. No regrouping or parameter,
        // decomposition, rounding, key or compression change is involved.
        let groups: Vec<_> = (0..terms).step_by(self.ring.group).collect();
        let reconstruct = |&start: &usize| {
            let end=(start+self.ring.group).min(terms);
            let mut sums: [Rns; 2] = std::array::from_fn(|_| Rns::zero());
            let mut scratch=if matches!(forward_mode,ForwardMode::BoundedScratch) { Some(Rns::zero()) } else { None };
            for term in start..end {
                let allocated=if scratch.is_none() { Some(self.ring.forward(&digits[term*N..(term+1)*N])) } else { None };
                let value=if let Some(value)=scratch.as_mut() {
                    self.ring.forward_digits_into(&digits[term*N..(term+1)*N],value,1i64<<(self.base_log.0-1));
                    &*value
                } else { allocated.as_ref().unwrap() };
                for (poly,sum) in sums.iter_mut().enumerate() {
                    if matches!(mode,PackingMode::ParallelFused) {
                        self.ring.accumulate_product(sum,value,&self.key[term*2+poly]);
                    } else {
                        sum.add(&self.ring.product(value,&self.key[term*2+poly]),&self.ring.primes);
                    }
                }
            }
            let mut values = vec![0u64; 2*N];
            for (poly,sum) in sums.iter_mut().enumerate() {
                // Linearity over each prime moves the same inverse-N factor
                // after accumulation; the CRT inverse receives identical residues.
                if matches!(mode,PackingMode::ParallelFused) { self.ring.normalize(sum); }
                let reconstructed=self.ring.inverse(sum);
                values[poly*N..(poly+1)*N].copy_from_slice(&reconstructed);
            }
            values
        };
        let reconstructed: Vec<_> = if !matches!(mode,PackingMode::SerialReference) {
            groups.par_iter().map(reconstruct).collect()
        } else {
            groups.iter().map(reconstruct).collect()
        };
        for values in reconstructed {
            for (target,value) in out.as_mut().iter_mut().zip(values) {
                *target=target.wrapping_sub(value);
            }
        }
        if let Some(value)=timings { value.groups_seconds+=started.unwrap().elapsed().as_secs_f64(); }
        out
    }
    pub(crate) fn compress<C: Container<Element=u64>>(&self, key: &CpuCompressionKey, values: &LweCiphertextList<C>, mode: PackingMode, digit_mode: DigitMode, forward_mode: ForwardMode, mut timings: Option<&mut PackingTimings>) -> Vec<CompressedModulusSwitchedGlweCiphertext<u64>> {
        assert_eq!(key.inner.lwe_per_glwe.0,N);
        values.chunks(N).map(|list| {
            let out=self.pack(&list,mode,digit_mode,forward_mode,timings.as_deref_mut());
            let started=timings.as_ref().map(|_|Instant::now());
            let result=CompressedModulusSwitchedGlweCiphertext::compress(&out,key.inner.storage_log_modulus,list.lwe_ciphertext_count());
            if let Some(value)=timings.as_deref_mut() { value.compression_seconds+=started.unwrap().elapsed().as_secs_f64(); }
            result
        }).collect()
    }
}

pub(crate) fn prepare_key(key: &CpuCompressionKey) -> &PackingPlan {
    key.ntt_cache.get_or_init(|| Arc::new(PackingPlan::new(&key.inner.packing_key_switching_key))).as_ref()
}

pub(crate) struct PreparedVector<'a> {
    plan: &'a PackingPlan,
    data: Vec<Vec<Rns>>,
    actual_len: usize,
    bodies: Option<Vec<Vec<u64>>>,
}
impl<'a> PreparedVector<'a> {
    pub(crate) fn new(value: &crate::ml::EncryptedVector<u64>, plan: &'a PackingPlan, mode: DotMode) -> Self {
        let (parts,actual_len)=value.ntt_parts();
        assert_eq!(parts.len(),actual_len.div_ceil(N));
        let extracted_body=matches!(mode,DotMode::ExtractedBody);
        let data=parts.iter().map(|part| {
            assert_eq!(part.polynomial_size().0,N);
            assert_eq!(part.glwe_size().0,2);
            assert_eq!(part.ciphertext_modulus(),plan.modulus);
            part.as_ref().chunks_exact(N).take(if extracted_body { 1 } else { 2 }).map(|p| plan.ring.forward(p)).collect()
        }).collect();
        let bodies=if extracted_body { Some(parts.iter().map(|part|part.as_ref()[N..].to_vec()).collect()) } else { None };
        Self { plan,data,actual_len,bodies }
    }
    pub(crate) fn dot(&self, clear: &[u64]) -> LweCiphertextOwned<u64> {
        assert_eq!(self.actual_len,clear.len());
        let mut out=GlweCiphertext::new(0u64,GlweSize(2),PolynomialSize(N),self.plan.modulus);
        let mut body=0u64;
        for (block,(encrypted,chunk)) in self.data.iter().zip(clear.chunks(N)).enumerate() {
            let mut padded=vec![0u64;N];padded[..chunk.len()].copy_from_slice(chunk);padded.reverse();
            let rhs=self.plan.ring.forward(&padded);
            for (poly,lhs) in encrypted.iter().enumerate() {
                let mut product=self.plan.ring.product(lhs,&rhs);
                let reconstructed=self.plan.ring.inverse(&mut product);
                for (target,value) in out.as_mut()[poly*N..(poly+1)*N].iter_mut().zip(reconstructed) {
                    *target=target.wrapping_add(value);
                }
            }
            if let Some(bodies)=&self.bodies {
                // With reversed clear coefficients, product coefficient N-1
                // is this dot product. No negacyclic wrap contributes there.
                // The pinned extractor ignores every other body coefficient.
                for (&value,&weight) in bodies[block].iter().zip(chunk) {
                    body=body.wrapping_add(value.wrapping_mul(weight));
                }
            }
        }
        if self.bodies.is_some() { out.as_mut()[2*N-1]=body; }
        crate::computations::extract_dot_product_as_lwe_ciphertext(&out)
    }
}
