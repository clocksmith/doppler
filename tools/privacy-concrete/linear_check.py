"""Secret random-projection checks on decoded encrypted products.

Research integrity mechanism for the hybrid route, where the client knows its
own integer input and the public coefficient tile. It replaces a full encrypted
replay of every tile with a few secret integer projections of the decoded
result, plus exact replay of a secret sample of tiles. The argument, its
assumptions and its limits are in INTEGRITY.md. No independent cryptographic
review has occurred and no production backend is admitted.
"""
import hashlib
import hmac
import json
import math
from pathlib import Path
import numpy as np

POLICY=json.loads(Path(__file__).with_name('linear-check-policy.json').read_text())
POLICY_ID=hashlib.sha256(json.dumps(POLICY,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_policy(policy):
    sample=policy.get('replaySample',{})
    if (policy.get('schema')!='doppler.linear-check-policy/v1' or policy.get('securityQualified') is not False
            or policy.get('productionBackendAvailable') is not False
            or type(policy.get('checkVectors')) is not int or not 16<=policy['checkVectors']<=64
            or policy.get('weightBits')!=16 or type(policy.get('falseRejectLog2')) is not int or not 20<=policy['falseRejectLog2']<=80
            or type(sample.get('numerator')) is not int or type(sample.get('denominator')) is not int
            or not 1<=sample['numerator']<=sample['denominator']<=64 or policy.get('vectorScope')!='tile-step-row/v1'
            or policy.get('rejection')!='deferred-until-route-close/v1' or policy.get('rejectedResponseSubstitute')!='zero'):
        raise ValueError('Unsupported installed linear-check policy')
    return policy


def noise_model(parameter_profile,policy=POLICY):
    value=validate_policy(policy)['noise'].get(parameter_profile)
    if (not isinstance(value,dict) or any(type(value.get(key)) not in (int,float) or not math.isfinite(value[key]) for key in ['varianceBase','variancePerOutput','margin'])
            or value['varianceBase']<0 or value['variancePerOutput']<=0 or not 1<=value['margin']<=4):
        raise ValueError('No measured honest-noise model is installed for this parameter profile')
    return dict(value)


def deviation_quantile(policy=POLICY):
    """z with exp(-z*z/2) = 2^-(falseRejectLog2+1)."""
    return math.sqrt(2*math.log(2)*(policy['falseRejectLog2']+1))


def honest_norm_bound(outputs,noise,policy=POLICY):
    """Assumed bound on the l2 norm of an honest tile's decoded error (integer units)."""
    sigma=noise['margin']*math.sqrt(noise['varianceBase']+noise['variancePerOutput']*outputs)
    return sigma*(math.sqrt(outputs)+deviation_quantile(policy))


def tolerance(outputs,noise,policy=POLICY):
    if type(outputs) is not int or outputs<=0:raise ValueError('Explicit output width required')
    weight=2**(policy['weightBits']-1)
    return math.ceil(weight*honest_norm_bound(outputs,noise,policy)*deviation_quantile(policy))


def acceptance_bound(outputs,deviation,noise,policy=POLICY):
    """Upper bound on the probability that every check accepts a response whose
    largest coordinate deviation is `deviation` integer units (INTEGRITY.md, Claim 1)."""
    if deviation<=0:return 1.0
    weight=2**(policy['weightBits']-1)
    single=min(1.0,(2*tolerance(outputs,noise,policy)/deviation+1)/(2*weight))
    return single**policy['checkVectors']


def detection_threshold(outputs,noise,policy=POLICY,log2_bound=40):
    """Smallest max-coordinate deviation whose acceptance bound is at most 2^-log2_bound."""
    weight=2**(policy['weightBits']-1)
    per=2.0**(-log2_bound/policy['checkVectors'])
    if per*2*weight<=1:raise ValueError('Requested bound is below the weight-range floor')
    return math.ceil(2*tolerance(outputs,noise,policy)/(per*2*weight-1))


class CheckKey:
    """Attempt-local secret. Never serialized into a frame, plan, report or cache name."""
    def __init__(self,seed,parameter_profile,computation_bits,quantization_bits,policy=POLICY):
        if not isinstance(seed,bytes) or len(seed)!=32:raise ValueError('Fresh 256-bit attempt-local check secret required')
        self._seed=seed;self.policy=validate_policy(policy);self.noise=noise_model(parameter_profile,policy)
        self.count=policy['checkVectors'];self.weight=2**(policy['weightBits']-1)
        if type(computation_bits) is not int or type(quantization_bits) is not int or not 2<=quantization_bits<computation_bits<=40:
            raise ValueError('Declared integer domains required')
        self.computation_bits,self.quantization_bits=computation_bits,quantization_bits
        self.retired=False

    def retire(self):
        self.retired=True;self._seed=None

    def _require(self):
        if self.retired or self._seed is None:raise ValueError('Check secret is retired')

    def vectors(self,tile_id,outputs,step,rows):
        """Independent vectors for every tile, route step and input row: (rows, outputs, count)."""
        self._require()
        if (not isinstance(tile_id,str) or type(outputs) is not int or outputs<=0 or type(step) is not int or step<0
                or type(rows) is not int or rows<=0):raise ValueError('Explicit tile identity, width, step and row count required')
        raw=hashlib.shake_256(b'doppler.linear-check.vectors/v2\0'+self._seed+tile_id.encode()+step.to_bytes(8,'big')+rows.to_bytes(8,'big')).digest(rows*outputs*self.count*2)
        return (np.frombuffer(raw,dtype='<u2').astype(np.int64)-self.weight).reshape(rows,outputs,self.count)

    def sampled(self,tile_id,step):
        self._require()
        sample=self.policy['replaySample']
        if type(step) is not int or step<0:raise ValueError('Explicit route step required')
        raw=hmac.new(self._seed,b'doppler.linear-check.replay/v1\0'+tile_id.encode()+step.to_bytes(8,'big'),hashlib.sha256).digest()
        return int.from_bytes(raw[:8],'big')%sample['denominator']<sample['numerator']

    def expectation(self,tile_id,matrix,step,rows):
        """matrix @ vectors for each row, exact: (rows, inputs, count). matrix is the signed public tile."""
        signed=np.asarray(matrix)
        if signed.ndim!=2 or signed.dtype.kind!='i':raise ValueError('Signed public coefficient tile required')
        inputs,outputs=signed.shape
        limit=2**(self.quantization_bits-1)
        if signed.size==0 or int(signed.min())<-limit or int(signed.max())>=limit:raise ValueError('Public coefficient exceeds its declared precision')
        # Largest |entry| is limit*weight*outputs; keep the float64 product exact
        # and the later int64 input product inside 62 bits.
        if limit*self.weight*outputs>=2**52 or 2**self.quantization_bits*limit*self.weight*outputs*inputs>=2**62:
            raise ValueError('Linear check would exceed exact integer arithmetic')
        vectors=self.vectors(tile_id,outputs,step,rows).transpose(1,0,2).reshape(outputs,rows*self.count)
        product=signed.astype(np.float64)@vectors.astype(np.float64)
        result=product.astype(np.int64)
        if not np.array_equal(result.astype(np.float64),product):raise ValueError('Inexact linear-check expectation')
        return np.ascontiguousarray(result.reshape(inputs,rows,self.count).transpose(1,0,2))

    def deviations(self,tile_id,inputs,decoded,expectation,step):
        """Signed integer <r, decoded_row> - <input_row, expectation> per row and check, or None outside the domain."""
        inputs=np.asarray(inputs);decoded=np.asarray(decoded)
        if (inputs.ndim!=2 or decoded.ndim!=2 or inputs.dtype!=np.int64 or decoded.dtype!=np.int64 or inputs.shape[0]!=decoded.shape[0]
                or decoded.size==0 or expectation.shape!=(inputs.shape[0],inputs.shape[1],self.count) or expectation.dtype!=np.int64):
            raise ValueError('Exact check geometry required')
        if int(inputs.min())<0 or int(inputs.max())>=2**self.quantization_bits:raise ValueError('Client input exceeds its declared domain')
        rows,outputs=decoded.shape;domain=2**(self.computation_bits-1)
        if domain*self.weight*outputs>=2**62:raise ValueError('Linear check would exceed exact integer arithmetic')
        # Compare the extremes directly: abs() of the most negative int64 is negative.
        if int(decoded.min())<-domain or int(decoded.max())>=domain:return None
        vectors=self.vectors(tile_id,outputs,step,rows)
        return np.einsum('ro,rok->rk',decoded,vectors)-np.einsum('ri,rik->rk',inputs,expectation)

    def passes(self,tile_id,inputs,decoded,expectation,step):
        value=self.deviations(tile_id,inputs,decoded,expectation,step)
        if value is None:return False
        limit=tolerance(decoded.shape[1],self.noise,self.policy)
        return int(value.max())<=limit and int(value.min())>=-limit
