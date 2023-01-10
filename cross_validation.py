from cross_validation_localization import cross_validate as cvl
from cross_validation_recognition import cross_validate as cvr

def cross_validation(file_path, hyper_args, localization_hyper_args, test_stage):
    if test_stage == 0:
        cvl(file_path, hyper_args)
    if test_stage == 1:
        cvr(file_path, hyper_args, localization_hyper_args)