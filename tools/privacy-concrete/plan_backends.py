"""Installed research plan loaders. Advertisements cannot register or replace executable profiles."""
DEFAULT_PROFILE='qwen-all-linear'
PROFILES=(DEFAULT_PROFILE,'flash-expert0','flash-dense-moe','flash-mlp-hyper','flash-mlp-block','flash-first-delta')


def load_backend_plan(directory,profile):
    if profile=='qwen-all-linear':
        from full_plan import load_plan
    elif profile=='flash-expert0':
        from flash_expert_plan import load_plan
    elif profile=='flash-dense-moe':
        from flash_moe_plan import load_plan
    elif profile=='flash-mlp-hyper':
        from flash_hyper_plan import load_plan
    elif profile=='flash-mlp-block':
        from flash_mlp_plan import load_plan
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
