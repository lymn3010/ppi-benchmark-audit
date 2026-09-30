"""Run the paper experiment into an isolated, provenance-bearing directory."""
from __future__ import annotations
import hashlib
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from .config import MANIFEST, MANIFEST_PATH, REPO_ROOT, SOURCE_PAPER_DIR


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(module, args, env, log_path):
    command = [sys.executable, '-m', 'src.reproduction.' + module, *args]
    with log_path.open('w') as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{module} failed (exit {result.returncode}); see {log_path}')


def _full_reader(output, env, state):
    """Build exact run manifests; downstream analyses cannot choose another run."""
    from src.pipeline import RunConfig, run_dataset
    from src.analysis.stats_ledgers import export_statistics_ledgers
    from src.parsing import build_parser_backend
    # Fail at the native boundary before attempting a corpus with missing models.
    parser_config = MANIFEST['parser']
    build_parser_backend(
        'stanza', stanza_package=parser_config['package'],
        use_coref=parser_config['coreference'], use_nominalization=parser_config['nominalization'],
        qanom_threshold=parser_config['qanom_threshold'], use_gpu=parser_config['gpu'],
    ).parse('PROTEIN0 binds PROTEIN1.', 'reproduction-smoke')
    profiles = [('reader', parser_config), *MANIFEST['ablations'].items()]
    for profile, settings in profiles:
        coref, nominal = settings['coreference'], settings['nominalization']
        paths = {}
        for corpus in MANIFEST['corpora']:
            run_dir = output / profile / corpus
            cfg = RunConfig(dataset=corpus, split='full', use_coref=coref,
                            use_nominalization=nominal, force_rerun=True,
                            stanza_package=parser_config['package'],
                            qanom_threshold=parser_config['qanom_threshold'], use_gpu=parser_config['gpu'])
            result = run_dataset(cfg, output_dir=run_dir)
            if result is None or not (run_dir / 'db/events.db').exists():
                raise RuntimeError(f'Full reader failed for {profile}/{corpus}')
            # Partial parser runs cannot be called successful reproductions.
            import sqlite3
            from src.datasets import get_dataset_path
            from src.utils import load_dataset_json
            expected = len(load_dataset_json(get_dataset_path(corpus, 'full')))
            with sqlite3.connect(run_dir / 'db/events.db') as conn:
                count = conn.execute('SELECT COUNT(*) FROM sentences').fetchone()[0]
            if count != expected:
                raise RuntimeError(f'Incomplete {profile}/{corpus}: {count}/{expected} sentences')
            paths[corpus] = str(run_dir / 'db/events.db')
            export_statistics_ledgers(run_dir / 'db/events.db', output / 'ledgers' / profile / corpus, corpus=corpus)
        path = output / (profile + '-runs.json')
        path.write_text(json.dumps(paths, indent=2))
        state['steps'].append({'id': profile, 'status': 'recomputed', 'runs': paths})
        if profile == 'reader':
            env['PPI_REPRODUCTION_RUNS'] = str(path)
            env['PPI_PAPER_LEDGER_ROOT'] = str(output / 'ledgers/reader')
        else:
            _run('cross_corpus', [str(output / 'ledgers/noqanom' / c) for c in MANIFEST['corpora']] + ['--out-dir', str(output / 'paper/results/ablation')], env, output / 'logs/noqanom.log')
    # Coverage uses paths and structural rows from the same saved reader DBs.
    _run('coverage', ['--db-source', 'frozen'], env, output / 'logs/coverage.log')
    state['steps'].append({'id':'coverage', 'status':'recomputed'})


def reproduce(mode: str, output: Path | None = None) -> Path:
    from .materials import check_materials

    check_materials()
    if mode not in {'verify', 'statistics', 'full'}:
        raise ValueError(f'Unknown reproduction mode: {mode}')
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = REPO_ROOT / 'output/reproduction' / f'{mode}-{stamp}'
    output = output.resolve()
    if any(output.is_relative_to(path) or path.is_relative_to(output)
           for path in (REPO_ROOT / 'papers', SOURCE_PAPER_DIR)):
        raise ValueError('Reproduction output must be separate from frozen paper packages')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}')
    if not SOURCE_PAPER_DIR.is_dir():
        raise FileNotFoundError(f'Missing fixed paper inputs: {SOURCE_PAPER_DIR}')
    (output / 'logs').mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('PPI_REPRODUCTION_RUNS', None)
    env.pop('PPI_PAPER_OUTPUT', None)
    env['PPI_PAPER_LEDGER_ROOT'] = str(SOURCE_PAPER_DIR / 'ledgers')
    paper = SOURCE_PAPER_DIR if mode == 'verify' else output / 'paper'
    env['PPI_REPRODUCTION_PAPER'] = str(paper)
    state = {'schema':'ppi_reproduction_run_v1','mode':mode,'paper_source':str(SOURCE_PAPER_DIR),
             'manifest_sha256':_hash(MANIFEST_PATH),'status':'running','steps':[],
             'retained_evidence':MANIFEST['retained_evidence'],
             'note':'Human ratings are retained observations, not regenerated judgments.'}
    from src.pipeline.phase_hash import compute_phase_hashes
    state['phase_hashes'] = compute_phase_hashes()
    state['reproduction_code_sha256'] = {str(p.relative_to(REPO_ROOT)):_hash(p) for p in sorted((REPO_ROOT / 'src/reproduction').rglob('*.py'))}
    state['input_sha256'] = {str(p.relative_to(SOURCE_PAPER_DIR)):_hash(p) for p in sorted(SOURCE_PAPER_DIR.rglob('*')) if p.is_file() and p.suffix in {'.json','.jsonl','.csv','.tex'}}
    state['environment'] = {'python': platform.python_version()}
    for package in ('stanza', 'qanom', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'openpyxl'):
        try:
            state['environment'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            state['environment'][package] = None
    report = output / 'reproduction.json'
    try:
        if mode != 'verify':
            shutil.copytree(SOURCE_PAPER_DIR, paper)
            state['steps'].append({'id':'coverage_and_parser_ablation','status':'verified_retained' if mode=='statistics' else 'pending_rebuild'})
            if mode == 'full':
                _full_reader(output, env, state)
                state['steps'][0]['status'] = 'recomputed'
            fields = {'paper':str(paper),'seed':str(MANIFEST['seeds']['statistics']),
                      'within_iters':str(MANIFEST['iterations']['within_corpus_null']),
                      'matched_iters':str(MANIFEST['iterations']['clustered_matched_gap_null']),
                      'document_bootstrap':str(MANIFEST['iterations']['document_bootstrap']),
                      'matched_null_iters':str(MANIFEST['iterations']['matched_gap_null']),
                      'family_seed':str(MANIFEST['seeds']['family_bootstrap']),
                      'matched_seed':str(MANIFEST['seeds']['matched_gap'])}
            for step in MANIFEST['steps']:
                print(f"[reproduce] {step['id']}", flush=True)
                args = []
                for value in step['args']:
                    args.extend([str(Path(env['PPI_PAPER_LEDGER_ROOT']) / c) for c in MANIFEST['corpora']] if value=='{ledgers}' else [value.format(**fields)])
                current = {'id':step['id'],'status':'running',
                           'command':[sys.executable, '-m', 'src.reproduction.' + step['module'], *args]}
                state['steps'].append(current)
                report.write_text(json.dumps(state, indent=2))
                before = {name: (paper / name).stat().st_mtime_ns if (paper / name).exists() else None for name in step['outputs']}
                _run(step['module'], args, env, output / 'logs' / (step['id'] + '.log'))
                for name, previous in before.items():
                    artifact = paper / name
                    if not artifact.is_file() or artifact.stat().st_size == 0 or artifact.stat().st_mtime_ns == previous:
                        raise RuntimeError(f"{step['id']} did not regenerate required artifact: {name}")
                current['outputs'] = {name: _hash(paper / name) for name in step['outputs']}
                current['status'] = step.get('status','recomputed')
        _run('check_numbers', ['--paper-dir',str(paper)], env, output / 'logs/number-gate.log')
        state['steps'].append({'id':'paper_number_gate','status':'verified'})
        state['status'] = 'passed'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = str(exc)
        for step in state['steps']:
            if step['status'] == 'running': step['status'] = 'failed'
        raise RuntimeError(f"{exc}; report: {report}") from exc
    finally:
        report.write_text(json.dumps(state, indent=2))
    print(f'[reproduce] {mode} passed: {report}', flush=True)
    return report
