##  The pytorch implementation of DCGAN

The code add residual block and simplify code in the basic of source code. If you want to visualize the training process , you need to type `python -m visdom.server` to open visdom. 

Since the code add residual block, you can use `data_augment.py` to augment data, which can get better results.

In addition, this code uses one GPU . If you want to use two GPUs, you need to ues the following code to perform parallel operations

```python
# netg = t.nn.DataParallel(netg, device_ids=[0, 1])
# netd = t.nn.DataParallel(netd, device_ids=[0, 1])
```

Paper:  [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks ](https://arxiv.org/abs/1511.06434 ) 

data: 5w animation images dataset [80K/13GB] [baidu pandownload password: g5qa](https://pan.baidu.com/s/1eSifHcA ).

## Requirement

```
pip install -r requirements.txt
```



## Usage

train examples:

```
CUDA_VISIBLE_DEVICES=3,5  python main.py train --gpu=True --vis=True
```

if you only have one gpu, you can run :

```
python main.py train --gpu=True --vis=True
```

test examples:

```
python main.py generate  --vis=False \
            --netd-path=checkpoints/netd_299.pth \
            --netg-path=checkpoints/netg_299.pth \
            --gen-img=result.png \
            --gen-num=64
```

complete paremeters:

```
data_path='./data'
num_workers = 4
image_size = 96 # the size of image
batch_size = 256 # the batch size of dataset
max_epoch = 50 # the max iteration
lr = 1e-4 # learning rate of Adam
beta = 0.5 # the first parameter of optimizer
gpu = True # use gpu
nz = 100 # the dim of noise
ngf = 64 # channels of generator feature
ndf = 64 # channels of discriminator feature
save_path = 'imgs/'  # the path of saving image

vis = True  # use visdom
plot_every = 10  # every 10 step , visiual once

d_every = 1  # every 1 step , train discriminator once
g_every = 5  # every 10 step , train generator once
netd_path = None  # 'checkpoints/netd_.pth' 
netg_path = None  # 'checkpoints/netg_211.pth'
```



## Result

<img src="http://m.qpic.cn/psb?/V12kySKV4IhBFe/Rj3FI7O2PPxAsjKOQmgPhcB8o0l3F6SQk2*NAhlIhVY!/b/dL8AAAAAAAAA&bo=EgMSAwAAAAADZ0I!&rf=viewer_4" width="80%" height="80%">

