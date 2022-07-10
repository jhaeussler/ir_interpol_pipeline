from enum import Enum
import datetime

"""
python logging module pollutes my terminal like crazy with weird info in Debug mode. This is much cleaner.
"""


class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    ERROR = 3

    def __eq__(self, other):
        return self.value == other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __le__(self, other):
        return self.value <= other.value


class DeluxLogger:
    level = LogLevel.INFO

    def set_level(self, level):
        self.level = level

    def err(self, msg):
        print(f'[ERROR][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]: {msg}')

    def debug(self, msg):
        if self.level is LogLevel.DEBUG:
            print(f'[DEBUG][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]: {msg}')

    def info(self, msg):
        if self.level <= LogLevel.INFO:
            print(f'[INFO][{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]: {msg}')
