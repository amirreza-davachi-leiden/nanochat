"""Task 3: two training stages, benchmark evaluation, and a loss-mask example.

Adapted from nanochat/scripts/chat_sft.py. Nanochat's original files stay unchanged.
Read make_dataset(), training_batches(), and train() first.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TASK2 = ROOT.parent / 'task2'
NANOCHAT = ROOT.parent if (ROOT.parent / 'nanochat/gpt.py').is_file() else ROOT.parent / 'nanochat'
CACHE = ROOT / 'cache'
ARTIFACTS = ROOT / 'artifacts/checkpoints'
BENCHMARKS = ('ARC-Easy', 'ARC-Challenge', 'GSM8K')
sys.dont_write_bytecode = True


def read_json(path):
    """Read settings or a saved experiment record."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    """Save an experiment record in a readable format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha256(path):
    """Identify exactly which checkpoint, tokenizer, or data file we used."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_name(name):
    """Keep experiment names simple and inside their output folders."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Run names must contain only letters, digits, underscores, or hyphens.')
    return name


def setup_environment(device, compiled=False):
    """Reuse nanochat and the Task 2 tokenizer, keeping new caches in Task 3."""
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This small Task 3 runner supports one device; do not launch it with torchrun.')
    source = TASK2 / 'artifacts/tokenizer'
    if not (source / 'tokenizer.pkl').is_file():
        raise FileNotFoundError(f'Missing Task 2 tokenizer: {source}')
    CACHE.mkdir(parents=True, exist_ok=True)
    link = CACHE / 'tokenizer'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(source, target_is_directory=True)
    if not link.exists() or sha256(link / 'tokenizer.pkl') != sha256(source / 'tokenizer.pkl'):
        raise ValueError('Task 3 must use the same tokenizer as Task 2.')
    os.environ.update(NANOCHAT_BASE_DIR=str(CACHE), HF_HOME=str(CACHE / 'huggingface'),
                      TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'),
                      TORCH_COMPILE_DISABLE='0' if compiled else '1',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    if device == 'cpu':
        os.environ['NANOCHAT_DTYPE'] = 'float32'
    sys.path.insert(0, str(NANOCHAT))


def checkpoint(stage, run, config, preview=False):
    """Locate the base model or a completed Task 3 checkpoint of the required stage."""
    if stage == 'base':
        return TASK2 / 'artifacts/base_checkpoints' / check_name(config['base_run']), config['base_step']
    run = check_name(run)
    record_path = ROOT / 'results' / run / 'run.json'
    if preview and not record_path.exists():
        return ARTIFACTS / run, None
    record = read_json(record_path)
    if record['status'] != 'complete' or record['stage'] != stage:
        raise ValueError(f'{run} must be a completed {stage} run.')
    return ARTIFACTS / run, record['checkpoint_step']


def make_dataset(stage, settings):
    """Choose only MMLU + GSM-8K for mid-training, or only SmolTalk for SFT."""
    from tasks.common import TaskMixture
    from tasks.mmlu import MMLU
    from tasks.gsm8k import GSM8K
    from tasks.smoltalk import SmolTalk
    if stage == 'mid':
        specs = [
            ('MMLU', 'cais/mmlu', 'all', 'auxiliary_train', settings['mmlu_repetitions'],
             MMLU(subset='all', split='auxiliary_train')),
            ('GSM8K', 'openai/gsm8k', 'main', 'train', settings['gsm8k_repetitions'],
             GSM8K(subset='main', split='train')),
        ]
    else:
        specs = [('SmolTalk', 'HuggingFaceTB/smol-smoltalk', 'default', 'train', 1, SmolTalk(split='train'))]
    tasks, counts = [], []
    for name, repo, subset, split, repetitions, task in specs:
        tasks.extend([task] * repetitions)
        counts.append(dict(dataset=name, repo=repo, subset=subset, split=split,
                           rows=len(task), repetitions=repetitions, mixture_entries=len(task) * repetitions))
    return TaskMixture(tasks), counts


def data_files(specs):
    """Record the exact cached Parquet files behind the selected dataset splits."""
    import pyarrow.parquet as pq
    records = []
    for spec in specs:
        directory = CACHE / 'task_data' / spec['repo'].replace('/', '--') / spec['subset'] / spec['split']
        for filename in read_json(directory / 'manifest.json'):
            path = directory / filename
            records.append(dict(dataset=spec['dataset'], file=str(path.relative_to(ROOT)),
                                bytes=path.stat().st_size, rows=pq.read_metadata(path).num_rows, sha256=sha256(path)))
    return records


def training_batches(dataset, tokenizer, length, batch_size, stats):
    """Turn conversations into fixed-size batches with assistant-only targets."""
    import torch
    if not len(dataset):
        raise ValueError('The training mixture is empty.')
    inputs, targets = [], []
    for index in itertools.cycle(range(len(dataset))):
        ids, mask = tokenizer.render_conversation(dataset[index], max_tokens=None)
        stats['conversations_opened'] += 1
        stats['long_conversations'] += int(len(ids) > length + 1)
        # Adjacent windows overlap by one context token, so no target is duplicated.
        # This keeps long conversations instead of silently losing their answers.
        for start in range(0, len(ids) - 1, length):
            window = ids[start:start + length + 1]
            target_mask = mask[start + 1:start + len(window)]
            if not any(target_mask):
                stats['context_only_windows_skipped'] += 1
                continue
            x = window[:-1]
            y = [token if keep else -1 for token, keep in zip(window[1:], target_mask)]
            padding = length - len(x)
            inputs.append(x + [tokenizer.get_bos_token_id()] * padding)
            targets.append(y + [-1] * padding)
            stats['windows_prepared'] += 1
            stats['padding_positions'] += padding
            if len(inputs) == batch_size:
                yield torch.tensor(inputs, dtype=torch.int32), torch.tensor(targets, dtype=torch.int64)
                inputs, targets = [], []
        if index == len(dataset) - 1 and not stats['windows_prepared']:
            raise ValueError('No assistant targets were found in this dataset.')


def lr_multiplier(step, steps, settings):
    """Warm up, hold the learning rate, then reduce it near the end of training."""
    warmup = min(settings['warmup_steps'], steps // 10)
    if step <= warmup:
        return step / warmup
    progress = (step - 1) / max(steps - 1, 1)
    decay = max(0.0, (progress - (1 - settings['warmdown_fraction'])) / settings['warmdown_fraction'])
    return 1 - decay * (1 - settings['final_lr_fraction'])


def train(args, config):
    """Run one requested stage, recording losses and saving its own checkpoints."""
    settings = dict(config['training'])
    settings['device'] = args.device or settings['device']
    settings['compile'] = settings['compile'] and settings['device'] == 'cuda'
    if args.steps is not None:
        settings['steps'] = args.steps
    if settings['steps'] < 1:
        raise ValueError('--steps must be positive.')
    name = check_name(args.name or f'{args.stage}-d2')
    parent_stage = 'base' if args.stage == 'mid' else 'mid'
    source, source_step = checkpoint(parent_stage, args.from_run, config, preview=args.dry_run)
    output = ARTIFACTS / name
    results = ROOT / 'results' / name
    plan = dict(stage=args.stage, source_stage=parent_stage, source_directory=str(source),
                source_step=source_step, output_directory=str(output), settings=settings,
                datasets=['MMLU', 'GSM8K'] if args.stage == 'mid' else ['SmolTalk'])
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if output.exists() or results.exists():
        raise ValueError('This run name already exists. Use --name with a new name to preserve results.')
    inspection = ROOT / 'results/inspection.json'
    if not inspection.is_file():
        raise FileNotFoundError('First finish the data-inspection notebook, including its save cell.')
    device_type = args.device or settings['device']
    setup_environment(device_type, compiled=settings['compile'] and device_type == 'cuda')
    import torch
    from nanochat.common import compute_init, compute_cleanup, COMPUTE_DTYPE
    from nanochat.checkpoint_manager import build_model, save_checkpoint
    if device_type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment. Run on a GPU machine, or explicitly use --device cpu.')
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, meta = build_model(str(source), source_step, device, phase='train')
    if model.config.n_layer != 2 or tokenizer.get_vocab_size() != 32768:
        raise ValueError('This experiment expects the depth-2 model and the 32,768-token tokenizer.')
    if settings['sequence_length'] != model.config.sequence_len:
        raise ValueError('Keep sequence_length equal to the source checkpoint context length.')
    dataset, counts = make_dataset(args.stage, settings)
    stats = dict(conversations_opened=0, long_conversations=0, context_only_windows_skipped=0,
                 windows_prepared=0, padding_positions=0)
    batches = training_batches(dataset, tokenizer, settings['sequence_length'], settings['batch_size'], stats)
    # Full fine-tuning: nanochat's optimizer includes every model parameter.
    # Start fresh optimizer moments at each stage; model weights continue from the previous stage.
    optimizer = model.setup_optimizer(embedding_lr=settings['embedding_lr'],
                                      unembedding_lr=settings['unembedding_lr'],
                                      matrix_lr=settings['matrix_lr'], weight_decay=0.0)
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr'] * settings['initial_lr_fraction']
    initial_groups = [dict(kind=g['kind'], learning_rate=g['initial_lr'], weight_decay=g['weight_decay'])
                      for g in optimizer.param_groups]
    train_model = torch.compile(model, dynamic=False) if settings['compile'] and device_type == 'cuda' else model
    scaler = torch.amp.GradScaler(enabled=COMPUTE_DTYPE == torch.float16 and device_type == 'cuda')
    results.mkdir(parents=True)
    record = dict(plan, status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device_type == 'cuda' else 'CPU',
                  compute_dtype=str(COMPUTE_DTYPE), seed=42, optimizer_warm_start=False,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  source_sha256=sha256(source / f'model_{source_step:06d}.pt'),
                  tokenizer_sha256=sha256(CACHE / 'tokenizer/tokenizer.pkl'),
                  runner_sha256=sha256(Path(__file__)), inspection_sha256=sha256(inspection),
                  nanochat_commit=subprocess.check_output(['git', '-C', str(NANOCHAT), 'rev-parse', 'HEAD'], text=True).strip(),
                  dataset_counts=counts, dataset_files=data_files(counts), optimizer_groups=initial_groups,
                  batch_token_positions=settings['batch_size'] * settings['sequence_length'] * settings['gradient_accumulation'])
    write_json(results / 'run.json', record)
    started, supervised_tokens, optimizer_updates = time.perf_counter(), 0, 0
    fields = ['step', 'assistant_loss_nats', 'assistant_target_tokens', 'lr_multiplier', 'optimizer_update_applied', 'elapsed_seconds']
    try:
        with (results / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step in range(1, settings['steps'] + 1):
                microbatches = [next(batches) for _ in range(settings['gradient_accumulation'])]
                target_count = sum(int((y >= 0).sum()) for _, y in microbatches)
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                multiplier = lr_multiplier(step, settings['steps'], settings)
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * multiplier
                    if group['kind'] == 'muon':
                        group['momentum'] = 0.85 + 0.10 * min((step - 1) / 300, 1)
                for x, y in microbatches:
                    loss = train_model(x.to(device), y.to(device), loss_reduction='sum')
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite training loss at step {step}.')
                    # Weight by actual assistant tokens, rather than by padded batch size.
                    scaler.scale(loss / target_count).backward()
                    loss_sum += loss.detach().item()
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= previous_scale
                optimizer_updates += int(updated)
                supervised_tokens += target_count
                row = dict(step=step, assistant_loss_nats=loss_sum / target_count,
                           assistant_target_tokens=target_count, lr_multiplier=multiplier,
                           optimizer_update_applied=updated, elapsed_seconds=time.perf_counter() - started)
                writer.writerow(row)
                handle.flush()
                if step == 1 or step % settings['log_every'] == 0 or step == settings['steps']:
                    print(f"{args.stage} step {step}/{settings['steps']} | assistant loss {row['assistant_loss_nats']:.4f}", flush=True)
                if step % settings['checkpoint_every'] == 0 or step == settings['steps']:
                    saved_meta = dict(step=step, stage=args.stage, model_config=asdict(model.config),
                                      user_config=settings, source_directory=str(source), source_step=source_step,
                                      max_seq_len=settings['sequence_length'], device_batch_size=settings['batch_size'],
                                      total_batch_size=record['batch_token_positions'], data_stats=dict(stats))
                    save_checkpoint(str(output), step, model.state_dict(), optimizer.state_dict(), saved_meta)
                    record['checkpoint_step'] = step
                    write_json(results / 'run.json', record)
        record.update(status='complete', checkpoint_sha256=sha256(output / f'model_{step:06d}.pt'))
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      training_seconds=time.perf_counter() - started, data_stats=stats,
                      supervised_target_tokens=supervised_tokens, optimizer_updates=optimizer_updates)
        write_json(results / 'run.json', record)
        compute_cleanup()
    print(f'Saved checkpoint and training records: {results}')


def benchmark_table(directory):
    """Combine evaluations that used the same protocol into a report table."""
    records = [read_json(path) for path in sorted(directory.glob('*.json'))]
    records.sort(key=lambda r: (['base', 'mid', 'sft'].index(r['stage']), r['run']))
    lines = ['| Stage | Run | ARC-Easy | ARC-Challenge | GSM-8K |', '|---|---|---:|---:|---:|']
    with (directory / 'scores.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['stage', 'run', *BENCHMARKS])
        for row in records:
            scores = [row['scores'][task] for task in BENCHMARKS]
            writer.writerow([row['stage'], row['run'], *scores])
            lines.append(f"| {row['stage']} | {row['run']} | " + ' | '.join(f'{100*s:.2f}%' for s in scores) + ' |')
    protocol = records[0]['protocol']
    scope = 'Full test splits' if protocol['max_problems'] is None else f"Subset check: at most {protocol['max_problems']} problems per task"
    text = scope + '. Missing stages have not been evaluated.\n\n' + '\n'.join(lines) + '\n'
    (directory / 'scores.md').write_text(text, encoding='utf-8')
    print(text)


def evaluate(args, config):
    """Use scripts/chat_eval.py to measure the same three benchmarks at each stage."""
    run = args.run or (config['base_run'] if args.stage == 'base' else f'{args.stage}-d2')
    source, step = checkpoint(args.stage, run, config)
    protocol = dict(config['evaluation'])
    if args.max_problems is not None:
        if args.max_problems < 1:
            raise ValueError('--max-problems must be positive; omit it for full evaluation.')
        protocol['max_problems'] = args.max_problems
    device_type = args.device or config['training']['device']
    if args.dry_run:
        print(json.dumps(dict(stage=args.stage, checkpoint=str(source), step=step, device=device_type, tasks=BENCHMARKS, protocol=protocol), indent=2))
        return
    setup_environment(device_type)
    from nanochat.common import compute_init, compute_cleanup, COMPUTE_DTYPE
    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine
    from scripts.chat_eval import run_chat_eval
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = build_model(str(source), step, device, phase='eval')
    engine = Engine(model, tokenizer)
    started = time.perf_counter()
    scores = {}
    try:
        for task in BENCHMARKS:
            scores[task] = run_chat_eval(task, model, tokenizer, engine, **protocol)
    finally:
        compute_cleanup()
    # Hash the exact test files as well as the generation settings for fair comparisons.
    specs = [dict(dataset=task, repo='allenai/ai2_arc', subset=task, split='test') for task in BENCHMARKS[:2]]
    specs.append(dict(dataset='GSM8K', repo='openai/gsm8k', subset='main', split='test'))
    files = data_files(specs)
    protocol.update(tasks=list(BENCHMARKS), seed=42, compute_dtype=str(COMPUTE_DTYPE),
                    tokenizer_sha256=sha256(CACHE / 'tokenizer/tokenizer.pkl'),
                    test_files=[dict(file=f['file'], sha256=f['sha256']) for f in files])
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / 'results/benchmarks' / key
    available = {task: sum(f['rows'] for f in files if f['dataset'] == task) for task in BENCHMARKS}
    evaluated_counts = {task: min(n, protocol['max_problems']) if protocol['max_problems'] is not None else n
                        for task, n in available.items()}
    record = dict(stage=args.stage, run=run, checkpoint_directory=str(source), checkpoint_step=step,
                  checkpoint_sha256=sha256(source / f'model_{step:06d}.pt'), protocol=protocol,
                  scores=scores, evaluated_examples=evaluated_counts, seconds=time.perf_counter() - started, evaluated_utc=datetime.now(timezone.utc).isoformat())
    destination = directory / f'{args.stage}-{check_name(run)}.json'
    if destination.exists():
        raise FileExistsError(f'This evaluation is already recorded: {destination}')
    write_json(destination, record)
    benchmark_table(directory)
    print(f'Saved benchmark evidence: {directory}')


def mask_example(args, config):
    """Show exactly which next-token targets contribute to the training loss."""
    setup_environment('cpu')
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    conversation = {'messages': [
        {'role': 'user', 'content': 'What is 2 + 2?'},
        {'role': 'assistant', 'content': [
            {'type': 'python', 'text': '2+2'},
            {'type': 'python_output', 'text': '4'},
            {'type': 'text', 'text': 'The answer is 4.'},
        ]},
    ]}
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=None)
    rows = [dict(position=i, input_piece=tokenizer.decode([ids[i - 1]]),
                 target_piece=tokenizer.decode([ids[i]]), token_id=ids[i],
                 loss_mask=mask[i], masked_target_id=ids[i] if mask[i] else -1) for i in range(1, len(ids))]
    path = ROOT / 'results/loss_mask_example.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"target={row['target_piece']!r:24} id={row['token_id']:5} mask={row['loss_mask']} training_target={row['masked_target_id']}")
    print(f'Without masking: average loss over {len(rows)} next-token targets.')
    print(f'With masking: average loss over {sum(mask[1:])} assistant/tool-call targets; ignore targets marked -1.')
    print(f'Saved {path}; this command does not load or train a model.')



def validate_config(config):
    """Reject settings that would produce empty batches or an invalid schedule."""
    settings = config['training']
    for key in ('steps', 'sequence_length', 'batch_size', 'gradient_accumulation',
                'mmlu_repetitions', 'gsm8k_repetitions', 'log_every', 'checkpoint_every'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if settings['sequence_length'] < 2 or settings['warmup_steps'] < 0:
        raise ValueError('Use sequence_length >= 2 and warmup_steps >= 0.')
    if not 0 < settings['warmdown_fraction'] <= 1 or not 0 <= settings['final_lr_fraction'] <= 1:
        raise ValueError('Learning-rate schedule fractions are outside their valid ranges.')
    if settings['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda.')
    for key in ('embedding_lr', 'unembedding_lr', 'matrix_lr', 'initial_lr_fraction'):
        if settings[key] <= 0:
            raise ValueError(f'{key} must be positive.')
    if config['evaluation']['num_samples'] != 1 or config['evaluation']['temperature'] != 0:
        raise ValueError('Keep greedy, one-answer evaluation for comparable assignment scores.')
    if config['evaluation']['max_problems'] is not None:
        raise ValueError('Keep full evaluation in config; use --max-problems for explicit subset checks.')


def main():
    """Choose one action explicitly; displaying help never starts training."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train', help='Train Stage 1 (mid) or Stage 2 (sft).')
    train_parser.add_argument('--stage', choices=['mid', 'sft'], required=True)
    train_parser.add_argument('--name', help='New output run name; defaults to mid-d2 or sft-d2.')
    train_parser.add_argument('--from-run', default='mid-d2', help='Completed mid run used to initialize SFT.')
    train_parser.add_argument('--steps', type=int, help='Override the number of optimizer-step attempts.')
    train_parser.add_argument('--device', choices=['cpu', 'cuda'])
    train_parser.add_argument('--dry-run', action='store_true', help='Print the plan without importing PyTorch or training.')
    train_parser.set_defaults(action=train)
    eval_parser = sub.add_parser('evaluate', help='Benchmark the base, mid, or SFT checkpoint.')
    eval_parser.add_argument('--stage', choices=['base', 'mid', 'sft'], required=True)
    eval_parser.add_argument('--run', help='Task 3 run name; defaults to mid-d2 or sft-d2.')
    eval_parser.add_argument('--max-problems', type=int, help='Use a small subset for a quick check; omit for full test sets.')
    eval_parser.add_argument('--device', choices=['cpu', 'cuda'])
    eval_parser.add_argument('--dry-run', action='store_true')
    eval_parser.set_defaults(action=evaluate)
    sub.add_parser('mask', help='Save a concrete token-level loss-mask example.').set_defaults(action=mask_example)
    args = parser.parse_args()
    config = read_json(ROOT / 'config.json')
    validate_config(config)
    args.action(args, config)


if __name__ == '__main__':
    main()
