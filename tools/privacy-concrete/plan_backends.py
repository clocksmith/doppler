"""Installed research plan loaders. Advertisements cannot register or replace executable profiles."""
DEFAULT_PROFILE='qwen-all-linear'
QWEN_PROFILES=(DEFAULT_PROFILE,'qwen-all-linear-ntt','qwen-all-linear-ntt-48','qwen-all-linear-ntt12')
PROFILES=(*QWEN_PROFILES,'flash-expert0','flash-dense-moe','flash-mlp-hyper','flash-mlp-block','flash-first-delta','flash-first-qsa-ntt','flash-first-ple-ntt')


def load_backend_plan(directory,profile):
    if profile=='qwen-all-linear':
        from full_plan import load_plan
    elif profile=='qwen-all-linear-ntt12':
        from full_ntt12_plan import load_plan
    elif profile=='qwen-all-linear-ntt-48':
        from full_ntt48_plan import load_plan
    elif profile=='qwen-all-linear-ntt':
        from full_ntt_plan import load_plan
    elif profile=='flash-expert0':
        from flash_expert_plan import load_plan
    elif profile=='flash-dense-moe':
        from flash_moe_plan import load_plan
    elif profile=='flash-mlp-hyper':
        from flash_hyper_plan import load_plan
    elif profile=='flash-mlp-block':
        from flash_mlp_plan import load_plan
    elif profile=='flash-first-ple-ntt':
        from flash_ple_plan import load_plan
    elif profile=='flash-first-qsa-ntt':
        from flash_qsa_plan import load_plan
    elif profile=='flash-first-delta':
        from flash_delta_plan import load_plan
    else:raise ValueError('Unsupported installed encrypted plan backend')
    return load_plan(directory)


def provider_arguments(arguments):
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('directory');parser.add_argument('plan_id')
    parser.add_argument('--profile',choices=PROFILES,default=DEFAULT_PROFILE)
    parser.add_argument('--ssh-route',nargs=2)
    return parser.parse_args(arguments)


def qwen_backend(profile):
    if profile==DEFAULT_PROFILE:
        from full_plan import FULL_POLICY,load_plan,implementation_versions
        from protocol import POLICY
        import copy
        return copy.deepcopy(FULL_POLICY),load_plan,implementation_versions,copy.deepcopy({key:POLICY[key] for key in ['dependencies','parameters','evaluationBackend']})
    if profile=='qwen-all-linear-ntt':
        from full_ntt_plan import POLICY_NTT,load_plan
        from native_runtime import ntt_parameters,ntt_versions
        import copy
        return copy.deepcopy(POLICY_NTT),load_plan,ntt_versions,ntt_parameters()
    if profile=='qwen-all-linear-ntt-48':
        from full_ntt48_plan import POLICY_NTT48,load_plan,parameters
        from native_runtime import ntt_versions
        import copy
        return copy.deepcopy(POLICY_NTT48),load_plan,ntt_versions,parameters()
    if profile=='qwen-all-linear-ntt12':
        from full_ntt12_plan import POLICY_NTT12,load_plan,parameters
        from native_runtime import ntt_versions
        import copy
        return copy.deepcopy(POLICY_NTT12),load_plan,ntt_versions,parameters()
    raise ValueError('Unsupported installed full-Qwen research backend')
