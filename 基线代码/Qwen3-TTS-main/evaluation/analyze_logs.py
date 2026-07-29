import re, json, csv
from collections import defaultdict

# Read both log files
with open('/Users/wlf/Downloads/all_experiments.log') as f:
    text1 = f.read()
with open('/Users/wlf/Downloads/remaining_experiments.log') as f:
    text2 = f.read()
log = text1 + text2

# Parse per-experiment loss series
data = {}
current_exp = None
for line in log.split('\n'):
    # Match "实验: r1_00_baseline" at line start
    m = re.search(r'\s+实验:\s+([a-z0-9_]+)', line)
    if m:
        current_exp = m.group(1)
        if current_exp not in data:
            data[current_exp] = {'losses': defaultdict(list), 'total_steps': 0}
        continue

    # Match loss
    m2 = re.match(r'Epoch (\d+) \| Step (\d+) \| Loss: ([\d.]+)', line)
    if m2 and current_exp:
        ep = int(m2.group(1)); lo = float(m2.group(3))
        data[current_exp]['losses'][ep].append(lo)
        data[current_exp]['total_steps'] += 1
        continue

    # "[完成] r1_00_baseline" marks end
    m3 = re.search(r'\[完成\]\s+(r\d_|mixed)', line)
    if m3:
        current_exp = None

print(f'Parsed {len(data)} experiments: {sorted(data.keys())}')

# Load SIM summary
sim_data = {}
with open('/Users/wlf/Desktop/讯飞实训营/智能语音课题资料包/基线代码/Qwen3-TTS-main/data/reports/sim_summary.csv',
          encoding='utf-8-sig') as f:
    sim_data = {r['experiment']: float(r['sim_mean']) for r in csv.DictReader(f)}

print('===== 训练 Loss vs SIM 全量对比 =====')
print(f'{"实验":<25s} {"步/epoch":>7s} {"起始Loss":>8s} {"终止Loss":>8s} {"降幅":>6s} {"SIM":>8s} {"备注"}')
print('-' * 100)

for exp_name in sorted(data.keys()):
    if 'mixed' in exp_name:
        continue
    d = data[exp_name]
    if not d['losses']:
        continue

    first_loss = d['losses'][0][0]
    last_ep = max(d['losses'].keys())
    last_loss = d['losses'][last_ep][-1]
    reduction = (first_loss - last_loss) / first_loss * 100
    epochs = len(d['losses'])
    steps = d['total_steps']
    steps_per_ep = steps / epochs

    sim = sim_data.get(exp_name, 0)

    flags = ''
    if last_loss > 9:
        flags += 'LOSS_高'
    if first_loss < 14:
        flags += ' 起始低(复用?)'

    print(f'{exp_name:<25s} {steps_per_ep:>7.0f} {first_loss:>8.2f} {last_loss:>8.2f} {reduction:>5.0f}% {sim:>8.4f} {flags}')

# ---- Critical analysis ----
print()
print('=' * 80)
print('=======  关键发现  =======')
print('=' * 80)

# Check: Did any experiment's e5 loss converge?
print('\n[1] Loss 收敛检查：')
for exp_name in sorted(data.keys()):
    if 'mixed' in exp_name: continue
    d = data[exp_name]
    losses = d['losses']
    end_losses = [losses[e][-1] for e in sorted(losses.keys())]
    end_steps = len(end_losses)
    if end_steps >= 2:
        delta_last = end_losses[-1] - end_losses[-2]
        status = '↓' if delta_last < 0 else '↑'
        print(f'  {exp_name:<25s} e{end_steps-2}→e{end_steps-1}: {end_losses[-2]:.2f}→{end_losses[-1]:.2f} ({delta_last:+.2f}) {status}')

# Check: R2 vs R1 loss comparison
print('\n[2] R1 vs R2 最终 Loss 对比：')
r1_end = []; r2_end = []
for exp_name in sorted(data.keys()):
    if 'mixed' in exp_name: continue
    d = data[exp_name]
    last_ep = max(d['losses'].keys())
    end_loss = d['losses'][last_ep][-1]
    if exp_name.startswith('r1'): r1_end.append(end_loss)
    elif exp_name.startswith('r2'): r2_end.append(end_loss)
if r1_end and r2_end:
    print(f'  R1 avg final loss: {sum(r1_end)/len(r1_end):.2f}')
    print(f'  R2 avg final loss: {sum(r2_end)/len(r2_end):.2f}')
    print(f'  R1 较低说明模型拟合得更好（预期之中，因为文本更简单）')

# Check: aug50 specifically - large dataset performance
print('\n[3] aug50 (410条) 详细 Loss 曲线：')
for exp_name in ['r1_04_aug50', 'r2_14_aug50']:
    if exp_name not in data: continue
    d = data[exp_name]
    print(f'  {exp_name}:')
    for ep in sorted(d['losses'].keys()):
        ls = d['losses'][ep]
        print(f'    e{ep}: {ls[0]:.2f} → {ls[-1]:.2f} (avg={sum(ls)/len(ls):.2f}, n={len(ls)} steps)')

# Check: mixed experiment crash
print('\n[4] Mixed 实验错误原因：')
for line in log.split('\n'):
    if 'Sizes of tensors must match' in line:
        print(f'  {line.strip()}')
        print('  → 原因: v1和v2用了不同ref_audio, ref_mels维度不同')
        print('  → 解决方案: 混合实验必须统一ref_audio, 或分别提取audio_codes后手动合并')

# Check: FlashAttention status
print('\n[5] 训练环境：')
for line in log.split('\n')[:50]:
    if 'flash-attn' in line.lower():
        print(f'  {line.strip()}')
        break
print('  → GPU上也没有FlashAttention！比假设的更差')
