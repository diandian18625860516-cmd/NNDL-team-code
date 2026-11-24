"""统一追加训练：数据→基础目标→额外目标→更新→记录与保存。"""
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import imgaug
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from .contract import fingerprint, load_pairs, make_schedule, read_json, safe_path, sha, weight_content
from .losses import BaseCompositionLoss, composition_score, extra_losses

REFERENCE = Path(__file__).parent / 'author_snapshot'
sys.path.insert(0, str(REFERENCE))
import ae
import unet
import salad_dataset as original_data

RGB = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
JITTER = transforms.RandomChoice([transforms.ColorJitter(brightness=.2),
                                 transforms.ColorJitter(contrast=.2), transforms.ColorJitter(saturation=.2)])
CORRUPTIONS = [original_data.create_draem_anomalies, original_data.change_label_same_img,
               original_data.copy_label_diff_img, original_data.copy_and_change]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    imgaug.seed(seed % (2**31))


def load_models(contract, path, device):
    allowed = [getattr(module, name) for module, names in (
        (ae, 'AutoEncoder ConvBlock Decoder DownBlock Downsample Encoder UpBlock Upsample'),
        (unet, 'DownBlock Downsample ResidualBlock SkipConnection UNet UpBlock Upsample'),
        (torch.nn, 'ReLU SiLU ModuleList Sequential Conv2d ConvTranspose2d Dropout Identity GroupNorm AvgPool2d MaxPool2d Upsample')
    ) for name in names.split()]
    models = {}
    for member in contract['weights']['members']:
        data = weight_content(path, member['archive_member'])
        if hashlib.sha256(data).hexdigest() != member['sha256']:
            raise ValueError('初始化权重在核对后发生变化')
        key = Path(member['archive_member']).name.removesuffix('_final.pth')
        with torch.serialization.safe_globals(allowed):
            models[key] = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
        models[key].to(device).train(key != 'teacher')
    models['teacher'].requires_grad_(False)
    return models


def labels(path):
    with Image.open(path) as image:
        if image.mode not in ('P', 'L') or image.size != (256, 256):
            raise ValueError('组成图必须为 256×256 的 P/L 标签索引图: ' + str(path))
        values = np.array(image, dtype=np.int64)
    if values.min() < 0 or values.max() > 5:
        raise ValueError('组成图标签不在作者 0..5 范围内: ' + str(path))
    return torch.from_numpy(values)


def one_hot(values):
    return torch.nn.functional.one_hot(values, 6).permute(2, 0, 1).float()


def reviewed_tensors(pair, root, source_labels):
    valid = labels(safe_path(root, pair['valid_path']))
    invalid = labels(safe_path(root, pair['invalid_path']))
    with Image.open(safe_path(root, pair['anomaly_mask_path'])) as image:
        if image.mode != 'L' or image.size != (256, 256):
            raise ValueError('异常区域掩码需为 256×256 L 图')
        mask_values = np.array(image)
    if not set(np.unique(mask_values)).issubset({0, 255}):
        raise ValueError('掩码只接受 0=背景、255=异常监督区域')
    mask = torch.from_numpy(mask_values == 255)
    changed = invalid != source_labels
    if not changed.any() or not mask.any() or (changed & ~mask).any():
        raise ValueError('违规编辑为空，或掩码未覆盖所有改变的标签像素')
    if torch.equal(valid, source_labels) or torch.equal(valid, invalid):
        raise ValueError('合法编辑必须非空且区别于违规版')
    # 这里只检查编码/像素对应；合法性由 reviewer 与 rule_reference 的人工审核保证。
    return one_hot(valid), one_hot(invalid), mask.float()


def prepare_batch(contract, entry, args, pairs, pair_root):
    source = entry['source_path']
    seed_all(entry['rgb_seed'])
    with Image.open(safe_path(args.rgb_root, source)) as image:
        image = image.convert('RGB')
        rgb, rgb_ae = RGB(image), RGB(JITTER(image))
    original_labels = labels(safe_path(args.maps_root, source))
    original = one_hot(original_labels)
    modified, mask = original.clone(), torch.zeros((256, 256))
    candidates = pairs.get(source, []) if entry['corrupt'] else []
    pair = candidates[entry['step'] % len(candidates)] if candidates else None
    valid = invalid = None
    kind = 'normal'
    if pair:
        valid, invalid, reviewed_mask = reviewed_tensors(pair, pair_root, original_labels)
    seed_all(entry['corruption_seed'])
    if entry['corrupt']:
        if pair and contract['variant']['sample_mode'] == 'reviewed':
            modified, mask, kind = invalid, reviewed_mask, 'reviewed_invalid'
        else:
            donor = one_hot(labels(safe_path(args.maps_root, entry['donor_path'])))
            fn = CORRUPTIONS[entry['corruption_index']]
            mask, _, modified = fn(modified, donor)
            kind = fn.__name__
    return {'rgb': rgb[None], 'rgb_ae': rgb_ae[None], 'original': original[None],
            'modified': modified[None], 'mask': mask[None],
            'valid': valid[None] if valid is not None else None,
            'invalid': invalid[None] if invalid is not None else None,
            'pair_id': pair['pair_id'] if pair else None, 'corruption': kind}


@torch.no_grad()
def teacher_statistics(teacher, sources, rgb_root, device):
    total = squares = None
    count = 0
    for i, source in enumerate(sources):
        with Image.open(safe_path(rgb_root, source)) as image:
            output = teacher(RGB(image.convert('RGB'))[None].to(device)).double()
        part_sum, part_square = output.sum((0, 2, 3)), output.square().sum((0, 2, 3))
        total = part_sum if total is None else total + part_sum
        squares = part_square if squares is None else squares + part_square
        count += output.shape[0] * output.shape[2] * output.shape[3]
        if i % 25 == 0:
            print(f'统计教师输出：{i + 1}/{len(sources)}', flush=True)
    mean = total / count
    variance = (squares - total.square() / count) / (count - 1)
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError('教师输出存在无效或零方差通道')
    return mean.float()[None, :, None, None], variance.sqrt().float()[None, :, None, None]


def objective(models, batch, teacher_mean, teacher_std, criterion, contract):
    teacher, student, autoencoder = (models[k] for k in ('teacher', 'student', 'autoencoder'))
    with torch.no_grad():
        teacher_st = (teacher(batch['rgb']) - teacher_mean) / teacher_std
        teacher_ae = (teacher(batch['rgb_ae']) - teacher_mean) / teacher_std
    channels = teacher_st.shape[1]
    distance = (teacher_st - student(batch['rgb'])[:, :channels]).square()
    hard = distance[distance >= torch.quantile(distance, .999)].mean()
    reconstruction = autoencoder(batch['rgb_ae'])
    ae_loss = (teacher_ae - reconstruction).square().mean()
    stae_loss = (reconstruction - student(batch['rgb_ae'])[:, channels:]).square().mean()
    input_maps = torch.cat((batch['modified'], batch['original']))
    targets = torch.cat((batch['original'], batch['original'])).argmax(1)
    masks = torch.cat((batch['mask'], torch.zeros_like(batch['mask'])))
    recon = models['comp_autoencoder'](input_maps).softmax(1)
    logits = models['comp_unet'](torch.cat((input_maps, recon), 1)).squeeze(1)
    recon_loss, mask_loss = criterion(recon, targets, logits, masks)
    cons = rank = hard.new_zeros(())
    variant = contract['variant']
    if batch['pair_id'] and (variant['spatial_weight'] or variant['rank_weight']):
        # 关闭 dropout 的随机性，让目标衡量图片变换；eval 模式仍然保留梯度。
        composition_models = [models['comp_autoencoder'], models['comp_unet']]
        modes = [model.training for model in composition_models]
        try:
            for model in composition_models:
                model.eval()
            s_valid = composition_score(models, batch['valid'])
            s_original = composition_score(models, batch['original']) if variant['spatial_weight'] else s_valid
            s_invalid = composition_score(models, batch['invalid']) if variant['rank_weight'] else s_valid
            cons, rank = extra_losses(s_original, s_valid, s_invalid, contract['settings']['rank_margin'])
        finally:
            for model, mode in zip(composition_models, modes):
                model.train(mode)
    base = hard + ae_loss + stae_loss + recon_loss + mask_loss
    total = base + variant['spatial_weight'] * cons + variant['rank_weight'] * rank
    return total, {'base': base, 'appearance_hard': hard, 'appearance_ae': ae_loss,
                   'appearance_stae': stae_loss, 'composition_recon': recon_loss,
                   'composition_mask': mask_loss, 'consistency': cons, 'ranking': rank}


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def train(contract, args):
    cfg = contract['settings']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device(args.device)
    if device.type != 'cuda' and not args.smoke:
        raise ValueError('V1 正式实验统一使用 CUDA；CPU 仅用于 --smoke 工程验证')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('未找到可用 CUDA；请在 GPU 环境运行，或用 --device cpu 做短训练验证')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    seed_all(cfg['seed'])
    pairs, pair_root = load_pairs(args.pairs, contract)
    total_steps = cfg['smoke_steps'] if args.smoke else cfg['steps']
    schedule = make_schedule(contract, total_steps)
    eligible = sum(bool(pairs.get(e['source_path'])) and e['corrupt'] for e in schedule)
    if contract['group'] != 'SB0' and not eligible:
        raise ValueError('本次采样没有审核配对，不能冒充新增目标已训练；请补齐配对或更换正式配置版本')
    code_hashes = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    signature_inputs = {'contract': contract, 'schedule': schedule, 'device': str(device),
                        'smoke': args.smoke, 'pairs': fingerprint(pairs), 'code': code_hashes,
                        'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                        'cuda_runtime': torch.version.cuda, 'numpy': str(np.__version__),
                        'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else None,
                        'rgb_hashes': {p: sha(safe_path(args.rgb_root, p)) for p in contract['roles']['optimization']}}
    signature = fingerprint(signature_inputs)
    out = args.out.resolve()
    if args.resume is None and out.exists() and any(out.iterdir()):
        raise ValueError('输出目录已有文件，请另选目录或明确指定 --resume')
    if args.resume is not None and args.resume.resolve().parent != out:
        raise ValueError('恢复时 --out 必须是检查点所在目录，防止混合记录')
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(contract, args.weights, device)
    appearance = list(models['student'].parameters()) + list(models['autoencoder'].parameters())
    composition = list(models['comp_autoencoder'].parameters()) + list(models['comp_unet'].parameters())
    parameters = appearance + composition
    optimizer = torch.optim.Adam([{'params': appearance, 'lr': cfg['appearance_lr']},
                                  {'params': composition, 'lr': cfg['composition_lr']}], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg['scheduler_step'], gamma=cfg['scheduler_gamma'])
    criterion = BaseCompositionLoss()
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint['signature'] != signature:
            raise ValueError('恢复条件不一致：组别、数据、代码、环境或正式/短训模式已变化')
        for key, model in models.items():
            model.load_state_dict(checkpoint['models'][key])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start = checkpoint['next_step']
        teacher_mean, teacher_std = checkpoint['teacher_mean'], checkpoint['teacher_std']
        criterion.dice_weight = checkpoint['dice_weight'].to(checkpoint['dice_weight_device'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
        # 每步重新设置 Python/NumPy/imgaug/torch 种子，恢复不依赖丢失的全局采样状态。
        log = out / 'steps.jsonl'
        if log.exists():
            retained = []
            for line in log.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break  # 掉电可能留下最后一行的不完整写入。
                if record['step'] <= start:
                    retained.append(line)
            log.write_text('\n'.join(retained) + ('\n' if retained else ''))
    else:
        sources = sorted(contract['roles']['optimization'])
        if args.smoke:
            sources = sources[:cfg['smoke_statistics_sources']]
        teacher_mean, teacher_std = teacher_statistics(models['teacher'], sources, args.rgb_root, device)
        atomic_json(out / 'resolved_protocol.json', signature_inputs)
        atomic_json(out / 'schedule.json', schedule)
    if start >= total_steps:
        raise ValueError('该检查点已完成全部约定更新，无需再次训练')
    stop = min(total_steps, start + args.stop_after) if args.stop_after else total_steps

    def save(next_step):
        checkpoint = {'signature': signature, 'next_step': next_step,
                      'models': {k: m.state_dict() for k, m in models.items()},
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'teacher_mean': teacher_mean, 'teacher_std': teacher_std,
                      'dice_weight': criterion.dice_weight, 'dice_weight_device': str(criterion.dice_weight.device),
                      'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        temp = out / 'checkpoint_latest.pt.tmp'
        torch.save(checkpoint, temp)
        temp.replace(out / 'checkpoint_latest.pt')

    atomic_json(out / 'status.json', {'state': 'running', 'group': contract['group'], 'smoke': args.smoke,
                                    'next_step': start, 'total_steps': total_steps, 'eligible_pair_steps': eligible})
    try:
        with (out / 'steps.jsonl').open('a', buffering=1) as log:
            for i in range(start, stop):
                started = time.perf_counter()
                batch = prepare_batch(contract, schedule[i], args, pairs, pair_root)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                seed_all(schedule[i]['model_seed'])
                optimizer.zero_grad(set_to_none=True)
                total, losses = objective(models, batch, teacher_mean, teacher_std, criterion, contract)
                if not torch.isfinite(total):
                    raise FloatingPointError('损失不是有限值')
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
                if grad_norm.item() == 0:
                    raise FloatingPointError('训练参数梯度全部为零')
                optimizer.step()
                scheduler.step()
                record = {'step': i + 1, 'source_path': schedule[i]['source_path'], 'pair_id': batch['pair_id'],
                          'corruption': batch['corruption'], 'total': float(total.detach()),
                          **{k: float(v.detach()) for k, v in losses.items()}, 'gradient_norm': float(grad_norm),
                          'seconds': time.perf_counter() - started,
                          'lr': [g['lr'] for g in optimizer.param_groups],
                          'peak_gpu_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0}
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                if (i + 1) % cfg['log_every'] == 0 or i == start:
                    print(f"{contract['group']} {i + 1}/{total_steps} loss={record['total']:.6f} pair={batch['pair_id']}", flush=True)
                if (i + 1) % cfg['checkpoint_every'] == 0 or i + 1 == stop:
                    save(i + 1)
        state = 'completed' if stop == total_steps else 'stopped_at_requested_checkpoint'
        if stop == total_steps:
            # 导出结构兼容的五个模型；这些是小组追加训练权重，不是作者原始权重。
            export = out / 'trained_models'
            export.mkdir(exist_ok=True)
            for key, model in models.items():
                torch.save(model, export / (key + '_final.pth'))
            atomic_json(export / 'TRAINING_PROVENANCE.json', {'group': contract['group'], 'stage': 'continued_training',
                        'smoke_only': args.smoke, 'steps': total_steps, 'protocol_sha256': signature,
                        'evaluation_completed': False})
        atomic_json(out / 'status.json', {'state': state, 'group': contract['group'], 'smoke': args.smoke,
                                        'next_step': stop, 'total_steps': total_steps, 'eligible_pair_steps': eligible})
    except BaseException as error:
        # 不保存可能只更新了一部分参数的现场；只从上一次完整保存点恢复。
        atomic_json(out / 'status.json', {'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                        'error': str(error), 'resume_from': 'checkpoint_latest.pt if present'})
        raise
