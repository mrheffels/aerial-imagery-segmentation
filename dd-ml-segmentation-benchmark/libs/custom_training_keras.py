from keras import optimizers, metrics
from libs import datasets_keras
from libs.config import LABELMAP
from libs.util_keras import FBeta
import numpy as np
from keras import backend as K

import wandb
from wandb.keras import WandbCallback


def categorical_focal_loss(alpha, gamma=2.):
    """
    Credit: https://github.com/umbertogriffo/focal-loss-keras

    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def train_model(dataset, model):
    epochs = 40
#     epochs = 0
    lr     = 1e-4
    size   = 300
    wd     = 1e-2
    bs     = 8 # reduce this if you are running out of GPU memory
    pretrained = True
    """For Focal Loss 
    we need alpha and gamma initialized
    """
    alpha = [[.25, .25, .25, .25, .25, .25]]
    gamma = 2

    config = {
        'epochs' : epochs,
        'lr' : lr,
        'size' : size,
        'wd' : wd,
        'bs' : bs,
        'pretrained' : pretrained,
    }

    wandb.config.update(config)

    model.compile(
        optimizer=optimizers.Adam(lr=lr),
        loss=[categorical_focal_loss(alpha, gamma)],
        metrics=[
            metrics.Precision(top_k=1, name='precision'),
            metrics.Recall(top_k=1, name='recall'),
            FBeta(name='f_beta')
        ]
    )

    train_data, valid_data = datasets_keras.load_dataset(dataset, bs)
    _, ex_data = datasets_keras.load_dataset(dataset, 10)
    model.fit_generator(
        train_data,
        validation_data=valid_data,
        epochs=epochs,
        callbacks=[
            WandbCallback(
                input_type='image',
                output_type='segmentation_mask',
                validation_data=ex_data[0]
            )
        ]
    )
