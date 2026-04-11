export async function prepareAttentionProjectionInput(attnForProjection, matmulOutputDtype, castTensor) {
  if (matmulOutputDtype && attnForProjection.dtype !== matmulOutputDtype) {
    const casted = await castTensor(attnForProjection);
    return { oProjInput: casted, oProjInputTemp: casted };
  }

  return { oProjInput: attnForProjection, oProjInputTemp: null };
}
