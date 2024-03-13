#!/bin/bash
./start_setfit_optuna_model_body.sh $1
./start_setfit_train_model_body.sh $1
./start_setfit_optuna_end_to_end.sh $1
./start_setfit_train_end_to_end.sh $1
