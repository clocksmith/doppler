/** Internal qualification fixture only: Microsoft SEAL owns all cryptographic arithmetic. */
import initialize from 'node-seal';
import policy from './policy.json' with { type: 'json' };
export async function openExperimentContext() {
  if (policy.parameters.securityLevel !== 128 || policy.parameters.compression !== 'none') throw Error('Unqualified experiment parameter change');
  const seal = await initialize();
  const handles = [];
  const own = value => { handles.push(value); return value; };
  const close = () => {
    const errors = [];
    for (const handle of handles.splice(0).reverse()) {
      try { if (!handle.isDeleted()) handle.delete(); } catch (error) { errors.push(error); }
    }
    if (errors.length) throw new AggregateError(errors, 'Experiment resource cleanup failed');
  };
  try {
    const parameters = own(new seal.EncryptionParameters(seal.SchemeType.bfv));
    parameters.setPolyModulusDegree(policy.parameters.polyModulusDegree);
    parameters.setCoeffModulus(own(seal.CoeffModulus.BFVDefault(policy.parameters.polyModulusDegree, seal.SecLevelType.tc128)));
    parameters.setPlainModulus(own(seal.PlainModulus.Batching(policy.parameters.polyModulusDegree, policy.parameters.plainModulusBits)));
    const context = own(new seal.SEALContext(parameters, true, seal.SecLevelType.tc128));
    if (!context.parametersSet()) throw Error('SEAL rejected experiment parameters');
    return { seal, context, own, close };
  } catch (error) { close(); throw error; }
}
