import Doppler.ExecutionContractFixtures
import Doppler.ExecutionRulesFixtures
import Doppler.ExecutionV0ContractFixtures
import Doppler.ExecutionV0GraphFixtures
import Doppler.KernelPathFixtures
import Doppler.LayerPatternFixtures
import Doppler.MergeSemanticsFixtures
import Doppler.QuantizationFixtures
import Doppler.RequiredInferenceFieldsFixtures

def renderCheck (entry : String × Bool) : String :=
  let status := if entry.2 then "pass" else "fail"
  s!"{entry.1}: {status}"

def renderedChecks : List String :=
  (
    executionContractChecks
      ++ executionRulesChecks
      ++ executionV0ContractChecks
      ++ executionV0GraphChecks
      ++ kernelPathChecks
      ++ layerPatternChecks
      ++ mergeSemanticsChecks
      ++ quantizationChecks
      ++ requiredInferenceFieldsChecks
  ).map renderCheck

#eval renderedChecks
