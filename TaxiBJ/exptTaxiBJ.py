# @Time    : 2018/12/10 19:15
# @Email  : wangchengo@126.com
# @File   : exptTaxiBJ.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import os, sys
import logging

sys.path.append('../')
from data.TaxiBJ.TaxiBJ import load_data
from models.STResNet import STResNet

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format=' %(levelname)s - %(message)s')
    nb_epoch = 5000000  # number of epoch at training stage
    batch_size = 32  # batch size
    T = 48  # number of time intervals in one day
    len_closeness = 3  # length of closeness dependent sequence
    len_period = 1  # length of  peroid dependent sequence
    len_trend = 1  # length of trend dependent sequence
    days_test = 7 * 4
    len_test = T * days_test
    map_height, map_width = 32, 32  # grid size
    nb_flow = 2
    lr = 0.0002  # learning rate
    nb_residual_unit = 4
    path_model = 'MODEL'

    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = \
        load_data(len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test)

    model = STResNet(
        learning_rate=lr,
        epoches=nb_epoch,
        batch_size=batch_size,
        model_path=path_model,
        len_closeness=len_closeness,
        len_trend=len_trend,
        external_dim=external_dim,
        map_heigh=map_height,
        map_width=map_width,
        nb_flow=nb_flow,
        nb_residual_unit=nb_residual_unit
    )
    model.train(X_train,Y_train)
    # model.evaluate(mmn, X_test, Y_test)
