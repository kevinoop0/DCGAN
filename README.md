##  The pytorch implementation of DCGAN

The code add residual block and simplify code in the basic of source code. If you want to visualize the training process , you need to type `python -m visdom.server` to open visdom. In addition, this code uses two GPUs to perform parallel operations. If you want to use one GPU, you need to notes the following code

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

test examples:

```
python main.py generate  --vis=False \
            --netd-path=checkpoints/netd_299.pth \
            --netg-path=checkpoints/netg_299.pth \
            --gen-img=result.png \
            --gen-num=64
```



## Result

<img src="http://m.qpic.cn/psb?/V12kySKV4IhBFe/Rj3FI7O2PPxAsjKOQmgPhcB8o0l3F6SQk2*NAhlIhVY!/b/dL8AAAAAAAAA&bo=EgMSAwAAAAADZ0I!&rf=viewer_4" width="80%" height="80%">

