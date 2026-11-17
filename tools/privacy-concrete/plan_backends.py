"""Installed research plan loaders. Advertisements cannot register or replace executable profiles."""
DEFAULT_PROFILE='qwen-all-linear'
QWEN_PROFILES=(DEFAULT_PROFILE,'qwen-all-linear-ntt','qwen-all-linear-ntt-48','qwen-all-linear-ntt12','qwen-all-linear-ntt12-check')
MOE_PROFILES=('flash-dense-moe','flash-dense-moe-ntt')
PROFILES=(*QWEN_PROFILES,'flash-expert0',*MOE_PROFILES,'flash-mlp-hyper','flash-mlp-block','flash-first-delta','flash-first-qsa-ntt','flash-first-ple-ntt','flash-text-all-linear-research','flash-text-all-linear-ntt12-check')


def load_backend_plan(directory,profile):
    if profile=='flash-text-all-linear-research':
        from flash_text_plan import load_plan
    elif profile=='flash-text-all-linear-ntt12-check':
        from flash_text_ntt12_check_plan import load_plan
    elif profile=='qwen-all-linear':
        from full_plan import load_plan
    elif profile=='qwen-all-linear-ntt12':
        from full_ntt12_plan import load_plan
    elif profile=='qwen-all-linear-ntt12-check':
        from full_ntt12_check_plan import load_plan
    elif profile=='qwen-all-linear-ntt-48':
        from full_ntt48_plan import load_plan
    elif profile=='qwen-all-linear-ntt':
        from full_ntt_plan import load_plan
    elif profile=='flash-expert0':
        from flash_expert_plan import load_plan
    elif profile=='flash-dense-moe':
        from flash_moe_plan import load_plan
    elif profile=='flash-dense-moe-ntt':
        from flash_moe_ntt_plan import load_plan
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
    if profile=='qwen-all-linear-ntt12-check':
        from full_ntt12_check_plan import POLICY_NTT12_CHECK,load_plan,parameters
        from native_runtime import ntt_versions
        import copy
        return copy.deepcopy(POLICY_NTT12_CHECK),load_plan,ntt_versions,parameters()
    raise ValueError('Unsupported installed full-Qwen research backend')


def moe_backend(profile):
    if profile=='flash-dense-moe':
        from flash_moe_plan import CONFIG,load_plan,versions
    elif profile=='flash-dense-moe-ntt':
        from flash_moe_ntt_plan import CONFIG_NTT as CONFIG,load_plan,versions
    else:raise ValueError('Unsupported installed all-expert MoE backend')
    import copy
    return copy.deepcopy(CONFIG),load_plan,versions
