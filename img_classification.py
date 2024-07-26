import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
from numpy.random import RandomState
from nnabla.models.imagenet import MobileNetV2,ResNet18
import nnabla.utils.data_iterator as d
import nnabla.solvers as S
import os
import nnabla.utils.save as save
from _checkpoint_nnp_util import save_checkpoint, load_checkpoint, save_nnp

#損失関数を返す関数
def loss_function(pred,label):
    loss = F.mean(F.softmax_cross_entropy(pred,label))
    return loss

model = MobileNetV2()
batch_size = 8
#訓練用データ
data = d.data_iterator_csv_dataset("./train.csv",batch_size,True,rng=RandomState(1223))
#テスト用データ
vdata = d.data_iterator_csv_dataset("./test.csv",batch_size,False)

import matplotlib.pyplot as plt
#１バッチ分データを取得
images,labels = data.next()
sample_image,sample_label = images[0],labels[0]

#データを一つ取り出して表示
# plt.imshow(sample_image.transpose(1,2,0))
# plt.show()
# print("image_shape: {}".format(sample_image.shape))
# print("lable_id: {}".format(sample_label))
channels,image_height,image_width = sample_image.shape

#テスト用の入力Variable
image_valid = nn.Variable([batch_size,channels,image_height,image_width])
label_valid = nn.Variable([batch_size,1])
input_image_valid = {"image": image_valid, "label":label_valid}

#訓練用の入力Variable
image_train = nn.Variable([batch_size,channels,image_height,image_width])
label_train = nn.Variable([batch_size,1])
input_image_train = {"image":image_train,"label":label_train}

#訓練用
#gloval_average_pooling層までの出力
y_train = model(image_train,force_global_pooling=True,use_up_to="pool",training=True)

#最後の全結合層を作成、出力を6クラスにする
with nn.parameter_scope("fineturning_fc"):
    pred_train = PF.affine(y_train,6)

#テスト用
#gloval_average_pooling層までの出力
y_valid = model(image_valid,force_global_pooling=True,use_up_to="pool",training=True)

#最後の全結合層を作成、出力を6クラスにする
with nn.parameter_scope("fineturning_fc"):
    pred_valid = PF.affine(y_valid,6)

#テスト用のloss
loss_valid = loss_function(pred_valid,label_valid)
#一番大きいエラー
top_1_error_valid = F.mean(F.top_n_error(pred_valid,label_valid))

#訓練用のloss
loss_train = loss_function(pred_train,label_train)
#一番大きいエラー
top_1_error_train = F.mean(F.top_n_error(pred_train,label_train))

#lossの最適化、アルゴリズムはAdamを使用
solver = S.Adam()
solver.set_parameters(nn.get_parameters())

print(nn.get_parameters())

num_epoch = 3
one_epoch = data.size // batch_size #1エポックを計算
max_iter = num_epoch * one_epoch
val_iter = data.size // batch_size

def run_validation(pred_valid,loss_valid,top_1_error_valid,
                   input_image_valid,data_iterator_valid,
                   with_visualized=False,num_visualized=3):
    assert num_visualized < pred_valid.shape[0], "too_many images to plot"
    val_iter = data.size // pred_valid.shape[0] #反復回数
    ve = 0
    vloss = 0
    for j in range(val_iter):
        v_image,v_label = data.next()
        input_image_valid["image"].d =v_image
        input_image_valid["label"].d = v_label
        #loss_validとtop_1_error_validを出力
        nn.forward_all([loss_valid,top_1_error_valid],clear_no_need_grad=True)
        vloss += loss_valid.d.copy()
        ve += top_1_error_valid.d.copy()
    
    #lossとerrorの平均を計算
    vloss /= val_iter
    ve /= val_iter

    if with_visualized:
        ind = 1
        random_start = np.random.randint(pred_valid.shape[0] - num_visualized)
        fig = plt.figure(figsize=(12.,12.))
        for n in range(random_start,random_start + num_visualized):
            sample_image,sample_label = v_image[n],v_label[n]
            ax = fig.add_subplot(1,num_visualized,ind)
            ax.imshow(sample_image.transpose(1,2,0))
            with nn.auto_forward():
                predicted_id = np.argmax(F.softmax(pred_valid)[n].d)
            result = "true label_id: {} -predicted as {}".format(str(sample_label[0]),str(predicted_id))
            ax.set_title(result)
            ind += 1
        fig.show()
        plt.show()
    return ve,vloss

# _, _ = run_validation(pred_valid,loss_valid,top_1_error_valid,input_image_valid,vdata,with_visualized=True)

from nnabla.monitor import Monitor,MonitorSeries
monitor = Monitor("tmo.monitor")
monitor_loss = MonitorSeries("Training loss",monitor,interval=200)
monitor_err = MonitorSeries("Training error",monitor,interval=200)
monitor_vloss = MonitorSeries("Test loss",monitor,interval=200)
monitor_verr = MonitorSeries("Test error",monitor,interval=200)

for i in range(max_iter):
    image,label = data.next()
    input_image_train["image"].d = image
    input_image_train["label"].d = label
    #lossとerrorを出力
    nn.forward_all([loss_train,top_1_error_train],clear_no_need_grad=True)

    monitor_loss.add(i,loss_train.d.copy())
    monitor_err.add(i,top_1_error_train.d.copy())
    
    print(f"loss: {loss_train.d}, error: {top_1_error_train.d}")

    solver.zero_grad()
    #逆伝播
    loss_train.backward(clear_buffer=True)

    solver.weight_decay(3e-4)
    solver.update()

    if i % 200 == 0: #200回ごとにテストしている
        ve,vloss = run_validation(pred_valid, loss_valid, top_1_error_valid,
                                   input_image_valid, vdata,
                                   with_visualized=False, num_visualized=3)
        monitor_vloss.add(i,vloss)
        monitor_verr.add(i,ve)
    
_,_ = run_validation(pred_valid, loss_valid, top_1_error_valid,
                                input_image_valid, vdata,
                                with_visualized=True, num_visualized=3)


parameter_file = os.path.join(
        "output", '{}_params_{:06}.h5'.format("mobileNetV2", max_iter))
nn.save_parameters(parameter_file)

# save_nnp_lastepoch
contents = save_nnp({'x': image_valid}, {'y': y_valid}, batch_size)
save.save(os.path.join("output",
                           '{}_result.nnp'.format("mobileNetV2")), contents)
    

















