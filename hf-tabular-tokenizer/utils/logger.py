############################################################################
##
## Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
##
## NVIDIA Sample Code
##
## Please refer to the NVIDIA end user license agreement (EULA) associated
## with this source code for terms and conditions that govern your use of
## this software. Any use, reproduction, disclosure, or distribution of
## this software and related documentation outside the terms of the EULA
## is strictly prohibited.
##
###########################################################################

import logging


def print_args(args):
    for arg in vars(args):
        print(f"{arg:20} : {getattr(args, arg)}")


def create_logger(name: str, level=logging.INFO):
    #  grabbed from here for quick setup: https://towardsdatascience.com/how-to-setup-logging-for-your-python-notebooks-in-under-2-minutes-2a7ac88d723d

    # create logger
    logger = logging.getLogger(name)
    # set log level for all handlers to debug
    logger.setLevel(level)

    # create console handler and set level to debug
    # best for development or debugging
    consoleHandler = logging.StreamHandler()

    consoleHandler.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    consoleHandler.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(consoleHandler)
    return logger
