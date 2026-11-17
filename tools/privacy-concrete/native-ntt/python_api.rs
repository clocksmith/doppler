#[pyfunction]
fn cpu_prepare_ntt_key(compression_key: &CpuCompressionKey) {
    crate::ntt_cpu::prepare_key(compression_key);
}

fn matrix_multiplication_ntt(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
    mode: crate::ntt_cpu::PackingMode,
    dot_mode: crate::ntt_cpu::DotMode,
    digit_mode: crate::ntt_cpu::DigitMode,
    forward_mode: crate::ntt_cpu::ForwardMode,
    mut timings: Option<&mut crate::ntt_cpu::PackingTimings>,
) -> PyResult<CompressedResultEncryptedMatrix> {
    let array=data.as_array();
    assert_eq!(encrypted_matrix.shape.1,array.nrows());
    assert_eq!(encrypted_matrix.shape.0,encrypted_matrix.inner.len());
    assert!(array.ncols()>0);
    let columns: Vec<_>=array.axis_iter(Axis(1)).map(|column|column.to_owned()).collect();
    let plan=crate::ntt_cpu::prepare_key(compression_key);
    let mut raw=LweCiphertextList::new(0u64,LweSize(2049),LweCiphertextCount(columns.len()),compression_key.inner.packing_key_switching_key.ciphertext_modulus());
    let mut result=Vec::new();
    for row in &encrypted_matrix.inner {
        let started=timings.as_ref().map(|_|std::time::Instant::now());
        let row=row.decompress();
        let prepared=crate::ntt_cpu::PreparedVector::new(&row,plan,dot_mode);
        columns.par_iter().zip(raw.par_iter_mut()).for_each(|(column,mut output)| {
            output.as_mut().copy_from_slice(prepared.dot(column.as_slice().unwrap()).as_ref());
        });
        if let Some(value)=timings.as_deref_mut() { value.dot_seconds+=started.unwrap().elapsed().as_secs_f64(); }
        result.push(CompressedResultCipherText { inner:plan.compress(compression_key,&raw,mode,digit_mode,forward_mode,timings.as_deref_mut()) });
    }
    Ok(CompressedResultEncryptedMatrix { inner:result })
}

#[pyfunction]
fn cpu_matrix_multiplication_ntt(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    // Preserve the measured .5 default; scratch conversion was not a clear win.
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelFused,crate::ntt_cpu::DotMode::ExtractedBody,crate::ntt_cpu::DigitMode::ParallelBlocks,crate::ntt_cpu::ForwardMode::AllocatingReference,None)
}

// Rejected performance candidate, retained only for explicit reproduction.
#[pyfunction]
fn cpu_matrix_multiplication_ntt_bounded_diagnostic(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelFused,crate::ntt_cpu::DotMode::ExtractedBody,crate::ntt_cpu::DigitMode::ParallelBlocks,crate::ntt_cpu::ForwardMode::BoundedScratch,None)
}

// Diagnostic arithmetic baseline only; no backend advertisement/registration.
#[pyfunction]
fn cpu_matrix_multiplication_ntt_serial(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::SerialReference,crate::ntt_cpu::DotMode::FullGlweReference,crate::ntt_cpu::DigitMode::SerialReference,crate::ntt_cpu::ForwardMode::AllocatingReference,None)
}

#[pyfunction]
fn cpu_matrix_multiplication_ntt_parallel_reference(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelReference,crate::ntt_cpu::DotMode::FullGlweReference,crate::ntt_cpu::DigitMode::SerialReference,crate::ntt_cpu::ForwardMode::AllocatingReference,None)
}

#[pyfunction]
fn cpu_matrix_multiplication_ntt_fused_reference(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelFused,crate::ntt_cpu::DotMode::FullGlweReference,crate::ntt_cpu::DigitMode::SerialReference,crate::ntt_cpu::ForwardMode::AllocatingReference,None)
}

// Exact .4 arithmetic with the old serial digit layout; diagnostics only.
#[pyfunction]
fn cpu_matrix_multiplication_ntt_body_reference(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelFused,crate::ntt_cpu::DotMode::ExtractedBody,crate::ntt_cpu::DigitMode::SerialReference,crate::ntt_cpu::ForwardMode::AllocatingReference,None)
}

// Public-fixture secondary instrumentation; never called by the backend.
fn profile_ntt_packing(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
    parallel_digits: bool,
    forward: crate::ntt_cpu::ForwardMode,
) -> PyResult<(CompressedResultEncryptedMatrix,String)> {
    let mut timings=crate::ntt_cpu::PackingTimings::default();
    let digits=if parallel_digits { crate::ntt_cpu::DigitMode::ParallelBlocks } else { crate::ntt_cpu::DigitMode::SerialReference };
    let result=matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelFused,crate::ntt_cpu::DotMode::ExtractedBody,digits,forward,Some(&mut timings))?;
    let report=format!("{{\"dotSeconds\":{},\"digitsSeconds\":{},\"groupsSeconds\":{},\"compressionSeconds\":{}}}",
        timings.dot_seconds,timings.digits_seconds,timings.groups_seconds,timings.compression_seconds);
    Ok((result,report))
}

// Exact .5 arithmetic with generic, per-term allocated digit transforms.
#[pyfunction]
fn cpu_matrix_multiplication_ntt_digits_reference(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<CompressedResultEncryptedMatrix> {
    matrix_multiplication_ntt(encrypted_matrix,data,compression_key,crate::ntt_cpu::PackingMode::ParallelFused,crate::ntt_cpu::DotMode::ExtractedBody,crate::ntt_cpu::DigitMode::ParallelBlocks,crate::ntt_cpu::ForwardMode::AllocatingReference,None)
}

#[pyfunction]
fn cpu_profile_ntt_packing(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
    parallel_digits: bool,
) -> PyResult<(CompressedResultEncryptedMatrix,String)> {
    profile_ntt_packing(encrypted_matrix,data,compression_key,parallel_digits,crate::ntt_cpu::ForwardMode::AllocatingReference)
}

#[pyfunction]
fn cpu_profile_ntt_bounded_packing(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
) -> PyResult<(CompressedResultEncryptedMatrix,String)> {
    profile_ntt_packing(encrypted_matrix,data,compression_key,true,crate::ntt_cpu::ForwardMode::BoundedScratch)
}

#[pyfunction]
fn cpu_validate_ntt_digit_encoding() -> String {
    let (digits,residues,polynomials,rejected)=crate::ntt_cpu::validate_digit_encoding();
    format!("{{\"admissibleDigitsChecked\":{},\"residueComparisons\":{},\"fullTransformComparisons\":{},\"outOfDomainValuesRejected\":{}}}",digits,residues,polynomials,rejected)
}
