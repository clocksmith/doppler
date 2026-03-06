import Doppler.ExecutionContractFixtures
import Doppler.KernelPathFixtures

def renderCheck (entry : String × Bool) : String :=
  let status := if entry.2 then "pass" else "fail"
  s!"{entry.1}: {status}"

def renderedChecks : List String :=
  (executionContractChecks ++ kernelPathChecks).map renderCheck

#eval renderedChecks
