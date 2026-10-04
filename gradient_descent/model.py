
"""
模型
包含数据导入（8000条 30+1）

目标 使用逻辑回归，建立负对数似然+L2正则化

梯度

Hessian (二阶导数矩阵，对梯度再度求导)

"""
from pathlib import Path
import numpy as np

data_path=Path(__file__).with_name("equipment_failure_risk.csv")#定位同目录下的csv文件
REGULARIZATION=1e-2

def load_data(path=data_path,test_cycle_count=2):
    #按逗号分割 names=True第一行作列名，dtypy=None让numpy自动判断数据类型
    #读取完, raw 类似一维数组,每个元素是一个record, record包含多个字段,如{'2026-01-01 10:00', 12.5, 30.2, 0,...}
    raw=np.genfromtxt(path,delimiter=",",names=True,dtype=None,encoding="utf-8")
    names=raw.dtype.names

    #提取 前缀sensor_ 得到[sensor_1,sensor_2,...]
    sensor_names=[name for name in names if name.startswith("sensor_")]

    #raw[name] 提取name列, 然后按列合并-->x
    features=np.column_stack([raw[name] for name in sensor_names]).astype(float) #x
    labels=np.asarray(raw["fault_state"],dtype=float) #y 
    cycles=np.asarray(raw["maintenance_cycle"],dtype=int)
    timestamps=np.asarray(raw["timestamp"],dtype="datetime64[m]")

    order=np.argsort(timestamps)#返回按timestamps排序的索引
    features=features[order]
    labels=labels[order]
    cycles=cycles[order]
    timestamps=timestamps[order]

    unique_cycles=np.unique(cycles)#取出不重复 周期号
    if len(unique_cycles)<=test_cycle_count:
        raise ValueError("not enough to train and test")

    train_cycles=unique_cycles[:-test_cycle_count]
    test_cycles=unique_cycles[-test_cycle_count:]
    train_mask=np.isin(cycles,train_cycles) 
    test_mask=np.isin(cycles,test_cycles)

    x_train_raw=features[train_mask]
    x_test_raw=features[test_mask]
    mean=x_train_raw.mean(axis=0) #axis=0:rowNums 行压缩,求列平均
    scale=x_train_raw.std(axis=0)
    scale[scale==0.0]=1.0 #防止除0
    x_train=(x_train_raw-mean)/scale
    x_test=(x_test_raw-mean)/scale

    x_train=np.column_stack((np.ones(len(x_train)),x_train)) #[1,x1,x2,...] 方便把截距b放进w
    x_test=np.column_stack((np.ones(len(x_test)),x_test))

    metadata={
        "sensor":len(sensor_names),
        "train_cycles":train_cycles,
        "train_start":str(timestamps[train_mask][0]),
        "train_end":str(timestamps[train_mask][-1]),
        "test_start":str(timestamps[test_mask][0]),
        "test_end":str(timestamps[test_mask][-1])
    }
    return x_train,labels[train_mask],x_test,labels[test_mask],metadata


"""
sigmoid(z)=1/(1+e^{-z})

为了避免e^{-z} 在z<0溢出,利用等价 1/(1+e^{-z})=e^z/(1+e^z)
"""
def sigmoid(z):
    positive=z>=0.0
    result=np.empty_like(z,dtype=float)
    result[positive]=1.0/(1.0+np.exp(-z[positive]))
    exp_z=np.exp(z[~positive])
    result[~positive]=exp_z/(1.0+exp_z)
    return result

# 对数似然函数+L2正则
def penalized_log_likelihood(weights,
                             features,labels,regularization=REGULARIZATION):
    scores=features @ weights  #z=w^Tx z=[z_1,z_2,...]
    # 下面括号内(逐元素相乘-[log(1+e^z1)+...]) 使用logaddexp(a,b)稳定计算log(e^a+e^b)
    data_term=np.mean(labels*scores-np.logaddexp(0.0,scores))
    penalty=0.5*regularization*np.dot(weights[1:],weights[1:])
    return data_term-penalty

def objective(weights,features,labels,regularization=REGULARIZATION):
    return -penalized_log_likelihood(weights,features,labels,regularization)

# x^T(\sigmoid(z)-labels)+\lambda R w  这里是普通梯度,即批量梯度
def gradient(weights,features,labels,regularization=REGULARIZATION):
    residual=sigmoid(features @ weights)-labels
    result=features.T @ residual/ len(labels)  #各样本梯度的和 求平均
    result[1:]+=regularization*weights[1:]
    return result

# H=1/n sum( x_i e^ z_i x_i/(1+ e^ z_i)^2 )+ \lambda R
def hessian(weights,features,labels,regularization=REGULARIZATION):
    prob=sigmoid(features @ weights)
    cur=prob*(1.0-prob) #e^ z_i /(1+ e^ z_i)^2
    result=features.T @ (cur[:,None]*features)/len(labels)
    result[1:,1:]+=regularization*np.eye(features.shape[1]-1)
    return result

def make_record(method,weights,features,labels,steps,elapsed):
    grad=gradient(weights,features,labels)
    return {
        "method":method,
        "steps":steps,
        "objective":objective(weights,features,labels),
        "grad_norm":np.linalg.norm(grad),
        "seconds":elapsed,
        "weights":weights
    }


"""
TP 实际正例(label=1)  预测正例 predictions[i]=True
FP 实际反例           预测正例
FN 实际正例           预测反例
TN 实际反例           预测反例
"""
def classification_metrics(weights, features, labels):
    prob = sigmoid(features @ weights)
    predictions = prob >= 0.5  # 预测为正例为True TP= np.sum(predictions & (labels == 1.0))
    tp = np.sum(predictions & (labels == 1.0))
    fp = np.sum(predictions & (labels == 0.0))
    tn = np.sum((~predictions) & (labels == 0.0))
    fn = np.sum((~predictions) & (labels == 1.0))

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0 

    log_loss = np.mean(np.logaddexp(0.0, features @ weights) - labels * (features @ weights))

    return accuracy,precision,recall,log_loss

    

