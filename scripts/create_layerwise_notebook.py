"""Generate the checked-in Colab notebook without notebook-library dependencies."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip() + '\n')
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell('markdown', '''
# ThinkingViT: Layer-wise Drop (H1 후속 실험)
기준: `experiment/h1-diagnosis`의 `5d53816`. 추가 실험: `codex/layer-wise-drop`.
**런타임 → 런타임 유형 변경 → Python 3 / T4 GPU**를 선택한 뒤 위에서부터 실행합니다.
유료 GPU를 구매할 필요는 없습니다. 무료 GPU 할당이 없으면 할당 가능한 시점에 재시도합니다.

6H의 각 Block **진입 전** 토큰을 실제로 제거합니다. 3H 마지막 CLS attention 순위를 고정해
스케줄 효과를 비교하며, 제거한 토큰은 복원하지 않습니다. CLS는 항상 남깁니다.
기존 고정 Drop과 동일 출력인지 확인하고, 같은 이미지에서 정확도/해석적 MACs/실제 시간을 비교합니다.
이 노트북은 **실행 전에는 추가 실험의 성능 결과를 제공하지 않습니다.**
''')
cell('code', '''
import os, sys, subprocess, json, shutil
from pathlib import Path

PROJECT = Path('/content/OnDevice')
BRANCH = 'codex/layer-wise-drop'
OFFICIAL_COMMIT = '9254c4950d3159937ad682592d58a2be69708c42'

def run(*command, **kwargs):
    command = [str(x) for x in command]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, **kwargs) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)

if not PROJECT.exists():
    run('git', 'clone', '--branch', BRANCH, 'https://github.com/hee4040/OnDevice.git', PROJECT)
os.chdir(PROJECT)
print('Project revision:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
# Separate Python 3.10 avoids changing the Colab kernel or its preinstalled packages.
run(sys.executable, '-m', 'pip', 'install', 'uv==0.9.8')
PYTHON = Path('/content/ondevice-env/bin/python')
if not PYTHON.exists():
    run(sys.executable, '-m', 'uv', 'venv', '--python', '3.10', '/content/ondevice-env')
run(sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON, '-r', 'requirements-colab.txt')
official = PROJECT / 'third_party/ThinkingViT'
if not official.exists():
    run('git', 'clone', 'https://github.com/ds-kiel/ThinkingViT.git', official)
run('git', '-C', official, 'checkout', OFFICIAL_COMMIT)
run(PYTHON, '-c', 'import torch; print(torch.__version__, torch.cuda.get_device_name() if torch.cuda.is_available() else "NO GPU"); assert torch.cuda.is_available(), "Select GPU runtime first"')
run(PYTHON, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_layerwise_drop.py', '-v')
run(PYTHON, 'scripts/prepare_weights.py')
run(PYTHON, 'baseline/check_layerwise_model.py')
''')
cell('markdown', '''
## Drive 저장 위치
아래 연결 창은 직접 승인합니다. 검증 데이터만 받습니다(train 128만 장은 받지 않습니다).
검증 이미지·캐시·결과를 위한 Drive 여유 공간을 확인하세요(최소 15GB 권장).
이미지는 Drive에 영구 저장하고 실험 때 Colab 로컬 디스크로 복사합니다.
Drive 연결 권한 화면은 사용자가 직접 처리해야 합니다.
''')
cell('code', '''
from google.colab import drive
drive.mount('/content/drive')
STORE = Path('/content/drive/MyDrive/OnDevice')
DATA_DRIVE = STORE / 'data/imagenet1k'
RESULTS = STORE / 'results/layerwise-drop-v1'
RESULTS.mkdir(parents=True, exist_ok=True)
shutil.copy2(PROJECT / 'results/layerwise-model-check.json', RESULTS / 'model-check.json')
print('Persistent data:', DATA_DRIVE)
print('Persistent results:', RESULTS)
print('Drive free GiB:', round(shutil.disk_usage(STORE).free / 1024**3, 1))
''')
cell('markdown', '''
## ImageNet 접근 인증
1. [ImageNet-1K 이용 조건/접근 신청](https://huggingface.co/datasets/ILSVRC/imagenet-1k)을 완료합니다.
2. 승인 계정의 [읽기 토큰](https://huggingface.co/settings/tokens)을 아래 숨김 입력칸에 입력합니다.
토큰을 채팅, 코드, Git, 출력에 쓰지 마세요. Colab Secrets에 `HF_TOKEN`으로 넣고
노트북 접근을 직접 허용해도 됩니다. 입력한 토큰은 현재 런타임의 환경에만 둡니다.
접근 승인이 없으면 다운로드가 401/403으로 중단됩니다. 승인 후 같은 셀을 다시 실행합니다.
''')
cell('code', '''
from getpass import getpass
from google.colab import userdata
try:
    token = userdata.get('HF_TOKEN')
except Exception:
    token = getpass('ImageNet 승인 계정의 Hugging Face 읽기 토큰: ')
if not token.strip():
    raise ValueError('토큰이 입력되지 않았습니다.')
os.environ['HF_TOKEN'] = token.strip()
del token
os.environ['HF_HOME'] = str(PROJECT / '.cache/huggingface')
run(PYTHON, 'scripts/prepare_imagenet.py', '--splits', 'validation', '--output', DATA_DRIVE)
''')
cell('code', '''
# Reuse the persistent Drive copy after runtime resets. Do not evaluate over Drive FUSE.
DATA = PROJECT / 'data/imagenet1k'
shutil.copytree(DATA_DRIVE, DATA, dirs_exist_ok=True)
run(PYTHON, '-c', "import sys; sys.path.insert(0, 'scripts'); from prepare_imagenet import check_counts; from pathlib import Path; check_counts(Path('data/imagenet1k/validation'), 50000); print('50,000 images / 1,000 classes verified')")
print('Local validation data ready')
''')
cell('markdown', '''
## 1. Smoke test (64장)
공식 full 6H, 새 100% 경로, 기존 고정 Drop과 새 고정 스케줄의 logits를 비교합니다.
이 표본의 정확도는 성능 결론에 쓰지 않습니다. 각 배치를 원자적으로 Drive에 저장하므로
중단되면 같은 설정으로 재실행해 이어 갑니다. 소스·환경·설정이 바뀌면 새 결과 폴더를 사용하세요.
''')
cell('code', '''
COMMON = [PYTHON, 'baseline/evaluate_layerwise_drop.py', '--data-dir', DATA,
          '--batch-size', '32', '--workers', '2', '--seed', '42', '--threshold', '1.0']
run(*COMMON, '--samples', '64', '--output', RESULTS / 'smoke')
''')
cell('markdown', '''
## 2. 본실험 (validation 50,000장, 학습 없음)
고정 25/50/75%, 단계적/선형/후반 감소, 최근접 GMACs의 고정 유지율 대조군을 모두 비교합니다.
3H와 routing은 동일합니다. 어려운 이미지의 정확도와 τ=1 전체 정확도를 구분합니다.
마지막 토큰 비율이 같은 비교와 **연산량이 가까운** 비교는 서로 다른 질문입니다.
오차가 남는 최근접 연산량 비교에는 `uniform_cost_delta_percent`를 함께 기록합니다.
동일 validation에서 여러 스케줄을 탐색하므로 최고 점수를 독립적인 검증 성능으로 해석하지 마세요.
''')
cell('code', '''
run(*COMMON, '--samples', '50000', '--output', RESULTS / 'accuracy')
''')
cell('markdown', '''
## 3. 실제 전체 추론 시간 (batch 1, 동일 1,000장)
3H, drop용 attention 수집, entropy gate 동기화, 정렬/gather, 필요한 6H를 포함합니다.
이미지 읽기/전처리/GPU 전송은 제외합니다. MACs 감소가 GPU 시간 감소를 보장하지 않습니다.
사전에 정한 full/고정 50%/단계적 50%/그 연산량 대조군을 측정합니다.
''')
cell('code', '''
config_rows = json.loads((RESULTS / 'accuracy/summary.json').read_text())['rows']
control = next(r['matched_uniform'] for r in config_rows if r['name'] == 'staged_50')
LATENCY_CONFIGS = ['full', 'uniform_50', 'staged_50', control]
run(*COMMON, '--mode', 'latency', '--samples', '1000', '--warmup', '20', '--repeats', '3',
    '--configs', *LATENCY_CONFIGS, '--output', RESULTS / 'latency-gpu')
# CPU comparison is optional; uncomment to run the same images on one CPU thread.
# run(*COMMON, '--mode', 'latency', '--device', 'cpu', '--threads', '1', '--samples', '1000',
#     '--warmup', '5', '--repeats', '3', '--configs', *LATENCY_CONFIGS, '--output', RESULTS / 'latency-cpu1')
''')
cell('code', '''
import pandas as pd
accuracy = pd.read_csv(RESULTS / 'accuracy/summary.csv')
display(accuracy[['name', 'top1', 'top1_routed', 'avg_gmacs', 'matched_uniform',
                  'uniform_gain_pp', 'uniform_cost_delta_percent']])
display(pd.read_csv(RESULTS / 'latency-gpu/summary.csv'))
run(PYTHON, 'baseline/plot_layerwise_drop.py', '--input', RESULTS / 'accuracy', '--output', RESULTS / 'figures')
from IPython.display import Image, display
display(Image(filename=str(RESULTS / 'figures/pareto.png')))
display(Image(filename=str(RESULTS / 'figures/schedules.png')))
# Share summary/environment/figures, not credentials or the ImageNet images.
print('Saved:', RESULTS)
''')

notebook = dict(cells=cells, metadata=dict(colab=dict(name='layerwise_drop.ipynb'),
                kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                accelerator='GPU'), nbformat=4, nbformat_minor=5)
for i, item in enumerate(cells):
    item['id'] = f'layerwise-{i:02d}'
target = ROOT / 'notebooks/layerwise_drop.ipynb'
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(target)
