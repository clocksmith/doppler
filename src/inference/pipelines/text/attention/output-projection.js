export async function prepareAttentionProjectionInput(attnForProjection, matmulOutputDtype, castToF16) {
  if (matmulOutputDtype === 'f16' && attnForProjection.dtype !== 'f16') {
    const casted = await castToF16(attnForProjection);
    return { oProjInput: casted, oProjInputTemp: casted };
  }

  return { oProjInput: attnForProjection, oProjInputTemp: null };
}
