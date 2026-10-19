"""Paired runtime benchmark using the unchanged STRATUS artifact implementation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
EXPECTED = 'c4e4b81c948e28c80e0ba50c84e118e54081d5257cb3c5b7a2a9d1e85e467525'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reference', type=Path, required=True,
                        help='Local clean_reference_segments.csv (not redistributed).')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory. Defaults to results/runtime_reproduced or runtime_smoke.')
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('--repeats must be at least 2')
    # Control BLAS parallelism before importing numerical libraries.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    print('Loading numerical libraries...', flush=True)
    import numpy as np
    import pandas as pd
    import sklearn
    from threadpoolctl import threadpool_limits, threadpool_info
    print('Libraries ready.', flush=True)
    ref = args.reference.expanduser().resolve()
    if not ref.is_file():
        parser.error(f'Reference CSV not found: {ref}')
    actual_hash = hashlib.sha256(ref.read_bytes()).hexdigest()
    if actual_hash != EXPECTED:
        parser.error(f'Reference SHA-256 mismatch: {actual_hash} (expected {EXPECTED})')
    # Use the original published runner already tracked in this repository.
    runner_path = REPOSITORY_ROOT / 'scripts' / 'run_hybrid_experiment.py'
    split_path = REPOSITORY_ROOT / 'results' / 'tables' / 'participant_split.csv'
    for required in (runner_path, split_path):
        if not required.is_file():
            parser.error(f'Repository file missing: {required}')
    runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    expected_runner_hash = '414958e7966fdf206f97cb3a9222c354703d0c28a3ac4e3dd258692a2dc50f0c'
    if runner_hash != expected_runner_hash:
        parser.error('Runner SHA-256 differs from the October 2026 benchmark version: ' + runner_hash)
    spec = importlib.util.spec_from_file_location('stratus_original_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    out = args.output or (ROOT / 'results' / ('runtime_smoke' if args.smoke else 'runtime_reproduced'))
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(ref)
    runner.validate_reference(reference)
    clean = runner.build_clean_cases(reference)
    split = pd.read_csv(split_path, dtype=str)
    ids = lambda label: {(r.dataset, runner.normalize_id(r.participant_id))
                         for r in split.itertuples() if r.split == label}
    repetitions = 2 if args.smoke else args.repeats
    rows = []
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        print('Preparing training cases...', flush=True)
        training = runner.generate_prepared_cases(
            clean, ids('train'), (0,) if args.smoke else runner.TRAIN_SEEDS,
            ('medium',) if args.smoke else runner.SEVERITIES, runner.CORRUPTIONS,
            10 if args.smoke else 120, runner.RANDOM_SEED+2)
        preparation = time.perf_counter()-start
        # Fixed published C=1 and transition strength=1; tuning is excluded.
        fits = []
        for repeat in range(2 if args.smoke else 5):
            print('Training repetition', repeat+1, flush=True)
            start = time.perf_counter()
            model = runner.fit_hybrid(training, 1.0, 5000 if args.smoke else 75000,
                                      runner.RANDOM_SEED+3)
            fits.append({'repeat': repeat, 'seconds': time.perf_counter()-start})
        heldout = [c for c in clean.values()
                   if (str(c.dataset.iloc[0]), runner.normalize_id(c.participant_id.iloc[0])) in ids('test')]
        if args.smoke:
            heldout = [next(c for c in heldout if c.dataset.iloc[0] == ds)
                       for ds in ('ETDD70', 'Autism')]
        for trace in heldout:
            for seconds in (2, 4, 8):
                chunk = trace[trace.time_s <= trace.time_s.iloc[0]+seconds].copy()
                for corruption in (('mixed',) if args.smoke else runner.CORRUPTIONS):
                    degraded = runner.inject_corruption(chunk, corruption, 'medium', 0)
                    ds = str(chunk.dataset.iloc[0])
                    meta = runner.DEFAULT_METADATA[ds]
                    fs = runner.estimate_sampling_rate_hz(chunk, meta['sampling_rate_hz'])
                    def process(strength):
                        diag = runner.compute_diagnostics(degraded, meta['width'], meta['height'])
                        prediction = model.predict(diag, fs, strength)
                        result = runner.apply_policy(diag, prediction, expanded=True)
                        assert len(result) == len(chunk)
                    process(0.0)
                    process(1.0)
                    for repeat in range(repetitions):
                        order = [('Hybrid-P', 0.), ('STRATUS-H', 1.)]
                        if repeat % 2:
                            order.reverse()
                        for method, strength in order:
                            start = time.perf_counter_ns()
                            process(strength)
                            elapsed = (time.perf_counter_ns()-start)/1e6
                            rows.append(dict(dataset=ds, participant=str(chunk.participant_id.iloc[0]),
                                case_segment=str(chunk.case_segment.iloc[0]), duration_setting_s=seconds,
                                samples=len(chunk), corruption=corruption, repeat=repeat,
                                method=method, milliseconds=elapsed))
            print('Completed', chunk.case_segment.iloc[0], flush=True)
        metadata = dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
            sklearn=sklearn.__version__, platform=platform.platform(), processor=platform.processor(),
            logical_cpus=os.cpu_count(), threadpools=threadpool_info(), reference_sha256=EXPECTED,
            training_preparation_seconds=preparation, training_cases=len(training), C=1., strength=1.,
            smoke=args.smoke, training_shared_by_methods=True,
            scope='diagnostics + features + prediction + policy; excludes IO, corruption, metrics, tuning',
            runner_sha256=runner_hash)
    pd.DataFrame(fits).to_csv(out/'training.csv', index=False)
    measured = pd.DataFrame(rows)
    measured.to_csv(out/'timings.csv', index=False)
    measured.groupby(['dataset','duration_setting_s','method']).milliseconds.agg(
        ['count','median','min','max']).to_csv(out/'summary.csv')
    (out/'environment.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Saved', out)


if __name__ == '__main__':
    main()
