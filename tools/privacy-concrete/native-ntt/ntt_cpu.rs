//! Experimental exact-u64 arithmetic adapter; no cryptographic parameter changes.
//! Cache only public evaluation material inside its owning compression key.
use rayon::prelude::*;
use std::sync::Arc;
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
        Self { plan, primes, group }
    }
    fn forward(&self, value: &[u64]) -> Rns {
        assert_eq!(value.len(), N);
        let mut result = Rns::zero();
        let [a,b,c,d,e] = &mut result.0;
        self.plan.fwd(value,a,b,c,d,e);
        result
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
    fn pack<C: Container<Element=u64>>(&self, list: &LweCiphertextList<C>, parallel: bool) -> GlweCiphertextOwned<u64> {
        let count = list.lwe_ciphertext_count().0;
        assert!(count>0 && count<=N);
        assert_eq!(list.lwe_size().0, N+1);
        assert_eq!(list.ciphertext_modulus(), self.modulus);
        let terms = N*self.levels;
        let mut digits = vec![0u64; terms*N];
        let mut out = GlweCiphertext::new(0u64,self.glwe_size,PolynomialSize(N),self.modulus);
        let decomposer = SignedDecomposer::new(self.base_log,DecompositionLevelCount(self.levels));
        for (column, lwe) in list.iter().enumerate() {
            out.as_mut()[N+column] = *lwe.get_body().data;
            for (input, &mask) in lwe.get_mask().as_ref().iter().enumerate() {
                for (level, digit) in decomposer.decompose(decomposer.closest_representable(mask)).enumerate() {
                    digits[(input*self.levels+level)*N+column] = digit.value();
                }
            }
        }
        // Each CRT group already has its own strict magnitude bound. Compute
        // those independent groups on the existing pool, then subtract their
        // reconstructions in the original order. No regrouping or parameter,
        // decomposition, rounding, key or compression change is involved.
        let groups: Vec<_> = (0..terms).step_by(self.ring.group).collect();
        let reconstruct = |&start: &usize| {
            let end=(start+self.ring.group).min(terms);
            let mut sums: [Rns; 2] = std::array::from_fn(|_| Rns::zero());
            for term in start..end {
                let value=self.ring.forward(&digits[term*N..(term+1)*N]);
                for (poly,sum) in sums.iter_mut().enumerate() {
                    sum.add(&self.ring.product(&value,&self.key[term*2+poly]),&self.ring.primes);
                }
            }
            let mut values = vec![0u64; 2*N];
            for (poly,sum) in sums.iter_mut().enumerate() {
                let reconstructed=self.ring.inverse(sum);
                values[poly*N..(poly+1)*N].copy_from_slice(&reconstructed);
            }
            values
        };
        let reconstructed: Vec<_> = if parallel {
            groups.par_iter().map(reconstruct).collect()
        } else {
            groups.iter().map(reconstruct).collect()
        };
        for values in reconstructed {
            for (target,value) in out.as_mut().iter_mut().zip(values) {
                *target=target.wrapping_sub(value);
            }
        }
        out
    }
    pub(crate) fn compress<C: Container<Element=u64>>(&self, key: &CpuCompressionKey, values: &LweCiphertextList<C>, parallel: bool) -> Vec<CompressedModulusSwitchedGlweCiphertext<u64>> {
        assert_eq!(key.inner.lwe_per_glwe.0,N);
        values.chunks(N).map(|list| {
            let out=self.pack(&list,parallel);
            CompressedModulusSwitchedGlweCiphertext::compress(&out,key.inner.storage_log_modulus,list.lwe_ciphertext_count())
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
}
impl<'a> PreparedVector<'a> {
    pub(crate) fn new(value: &crate::ml::EncryptedVector<u64>, plan: &'a PackingPlan) -> Self {
        let (parts,actual_len)=value.ntt_parts();
        let data=parts.iter().map(|part| {
            assert_eq!(part.polynomial_size().0,N);
            assert_eq!(part.glwe_size().0,2);
            assert_eq!(part.ciphertext_modulus(),plan.modulus);
            part.as_ref().chunks_exact(N).map(|p| plan.ring.forward(p)).collect()
        }).collect();
        Self { plan,data,actual_len }
    }
    pub(crate) fn dot(&self, clear: &[u64]) -> LweCiphertextOwned<u64> {
        assert_eq!(self.actual_len,clear.len());
        let mut out=GlweCiphertext::new(0u64,GlweSize(2),PolynomialSize(N),self.plan.modulus);
        for (encrypted,chunk) in self.data.iter().zip(clear.chunks(N)) {
            let mut padded=vec![0u64;N];padded[..chunk.len()].copy_from_slice(chunk);padded.reverse();
            let rhs=self.plan.ring.forward(&padded);
            for (poly,lhs) in encrypted.iter().enumerate() {
                let mut product=self.plan.ring.product(lhs,&rhs);
                let reconstructed=self.plan.ring.inverse(&mut product);
                for (target,value) in out.as_mut()[poly*N..(poly+1)*N].iter_mut().zip(reconstructed) {
                    *target=target.wrapping_add(value);
                }
            }
        }
        crate::computations::extract_dot_product_as_lwe_ciphertext(&out)
    }
}
