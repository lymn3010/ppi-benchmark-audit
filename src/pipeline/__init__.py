"""Public reader APIs, imported on demand so verification needs no NLP stack."""
from importlib import import_module

_EXPORTS = {
    'Pipeline': 'pipeline', 'init_default_pipeline': 'pipeline',
    'RunConfig': 'config', 'RunIdentity': 'run_identity',
    'run_dataset': 'runs', 'run_sentence': 'runs',
    'PreparedEntry': 'dataset_runner', 'WorkflowStats': 'dataset_runner',
    'run_pipeline_on_dataset': 'dataset_runner', 'write_workflow_stats': 'dataset_runner',
    'find_cached_run': 'run_registry', 'get_latest_run': 'run_registry', 'list_runs': 'run_registry',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f'{__name__}.{_EXPORTS[name]}'), name)
    globals()[name] = value
    return value
