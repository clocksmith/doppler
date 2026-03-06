import Doppler.ExecutionContractFixtures

def renderCheck (entry : String × Bool) : String :=
  let status := if entry.2 then "pass" else "fail"
  s!"{entry.1}: {status}"

def renderedChecks : List String :=
  executionContractChecks.map renderCheck

#eval renderedChecks
