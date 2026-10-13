"""Exact dyadic operand analysis; never replaces the approved model reference."""
from pathlib import Path
from fractions import Fraction
import json,math,hashlib
ROOT=Path(__file__).resolve().parent
raw=(ROOT/'projection-outlier-result.json').read_bytes();r=json.loads(raw)
results=[]
for row in r['outliers']:
    assert len(row['input'])==2*row['K'] and len(row['weights'])==row['K']
    exact=Fraction(0)
    for k,w in enumerate(row['weights']):
        exact+=(Fraction(row['input'][2*k])+Fraction(row['input'][2*k+1]))*Fraction(w)
    exact*=Fraction(row['alpha'])
    actual=Fraction(row['actualPair'][0])+Fraction(row['actualPair'][1])
    sequential=Fraction(row['sequentialF64'])
    denominator=max(Fraction(1),abs(exact))
    results.append({'dtype':row['dtype'],'row':row['row'],'column':row['col'],'K':row['K'],
      'exactNumerator':str(exact.numerator),'exactDenominator':str(exact.denominator),
      'actualPair':row['actualPair'],'sequentialF64':row['sequentialF64'],
      'gpuRelativeError':float(abs(actual-exact)/denominator),
      'sequentialF64RelativeError':float(abs(sequential-exact)/denominator),
      'originalDiagnosticRelativeError':row['error'],
      'withinOriginalThresholdAgainstExactArithmetic':abs(actual-exact)/denominator <= Fraction(r['tolerance'])})
out={'schema':'doppler.projection-outlier-exact-analysis/v1','sourceSha256':hashlib.sha256(raw).hexdigest(),
     'scope':'Exact rational analysis of retained synthetic operands; original diagnostic and approved full-model references unchanged',
     'originalDiagnosticPassed':r['passed'],'originalTolerance':r['tolerance'],'results':results}
(ROOT/'projection-outlier-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
