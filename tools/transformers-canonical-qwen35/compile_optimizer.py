"""Bounded explicit ONNX IR passes, avoiding quadratic generic pattern matching."""
from pathlib import Path
import json


def optimize(program):
    import onnx_ir as ir
    import onnx_ir.passes.common as common
    from onnxscript.optimizer import _constant_folding

    policy = json.loads(Path(__file__).with_name("export-compiler.json").read_text())
    if policy["genericPatternRewriter"] is not False:
        raise ValueError("generic pattern rewriting has not been qualified for the full graph")

    def should_fold(node):
        return False if node.op_type in policy["skipConstantOperators"] else None

    passes = [common.InlinePass()]
    for iteration in range(policy["constantFoldIterations"]):
        passes.extend([_constant_folding.FoldConstantsPass(shape_inference=True,
            input_size_limit=policy["constantInputLimitBytes"], output_size_limit=policy["constantOutputLimitBytes"],
            should_fold=should_fold), common.RemoveUnusedNodesPass()])
    passes.extend([common.RemoveUnusedFunctionsPass(), common.RemoveUnusedOpsetsPass(),
                   common.LiftConstantsToInitializersPass(lift_all_constants=True, size_limit=0),
                   common.LiftSubgraphInitializersToMainGraphPass(), common.DeduplicateInitializersPass(),
                   common.CommonSubexpressionEliminationPass(), common.OutputFixPass(), common.NameFixPass()])
    result = ir.passes.Sequential(*passes)(program.model)
    program.model = result.model
