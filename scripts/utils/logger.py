import sys
import logging
import os
import datetime

logger = logging.getLogger("DEFAULT_LOGGER")
logger.setLevel(logging.DEBUG)

# setup logger streams
logFilesDir = os.getcwd() +  "/logs/"

if not os.path.exists(logFilesDir):
    os.makedirs(logFilesDir)

logFilePath = logFilesDir +  f"classification-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
logFileStream = open(logFilePath, "x", encoding="utf-8")

stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fileHandler = logging.StreamHandler(logFileStream)

fmt = logging.Formatter(
    "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s %(message)s"
)

stdoutHandler.setFormatter(fmt)
fileHandler.setFormatter(fmt)

logger.addHandler(stdoutHandler)
logger.addHandler(fileHandler)
