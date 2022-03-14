import os
print('move pretrained weights...')
try:
    # v100 machines
    if not os.path.exists('~/.cache/torch/hub/checkpoints/'):
        os.makedirs('~/.cache/torch/hub/checkpoints/')
    os.system(
        'cp pretrained/*.pth ~/.cache/torch/hub/checkpoints/.')
except Exception as e:
    print(e)
try:
    # v100 machines
    if not os.path.exists('/usr/local/app/.cache/torch/hub/checkpoints/'):
        os.makedirs('/usr/local/app/.cache/torch/hub/checkpoints')
    os.system(
        'cp pretrained/*.pth /usr/local/app/.cache/torch/hub/checkpoints/.')
except Exception as e:
    print(e)
try:
    # a100 machines
    if not os.path.exists('/root/.cache/torch/hub/checkpoints/'):
        os.makedirs('/root/.cache/torch/hub/checkpoints/')
    os.system(
        'cp pretrained/*.pth /root/.cache/torch/hub/checkpoints/.')
    print('move finished...')
except Exception as e:
    print(e)