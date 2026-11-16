"""Installed research plan loaders. Advertisements cannot register or replace executable profiles."""
PROFILES=('qwen-all-linear','flash-expert0')


def load_backend_plan(directory,profile):
    if profile=='qwen-all-linear':
        from full_plan import load_plan
    elif profile=='flash-expert0':
        from flash_expert_plan import load_plan
    else:raise ValueError('Unsupported installed encrypted plan backend')
    return load_plan(directory)
