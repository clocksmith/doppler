#[pyfunction]
fn cpu_prepare_ntt_key(compression_key: &CpuCompressionKey) {
    crate::ntt_cpu::prepare_key(compression_key);
}

#[pyfunction]
fn cpu_matrix_multiplication_ntt(
    encrypted_matrix: &EncryptedMatrix,
    data: PyReadonlyArray<Scalar, Ix2>,
    compression_key: &CpuCompressionKey,
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
        let row=row.decompress();
        let prepared=crate::ntt_cpu::PreparedVector::new(&row,plan);
        columns.par_iter().zip(raw.par_iter_mut()).for_each(|(column,mut output)| {
            output.as_mut().copy_from_slice(prepared.dot(column.as_slice().unwrap()).as_ref());
        });
        result.push(CompressedResultCipherText { inner:plan.compress(compression_key,&raw) });
    }
    Ok(CompressedResultEncryptedMatrix { inner:result })
}
