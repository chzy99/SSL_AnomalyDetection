'''
Created: 2022-03-01 15:20:10
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Graduation Design
'''

import logging

class Logger:
    def __get_formatter(self):
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return fmt

    def __get_handler(self):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        fmt = self.__get_formatter()
        ch.setFormatter(fmt)
        return ch

    def getLogger(self, module_name):
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.__get_handler())
        return logger