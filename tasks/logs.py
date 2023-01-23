import logging

from tasks.globals import LOG_LEVEL

# # https://www.delftstack.com/howto/python/python-logging-to-file-and-console/
# # Use the logging Module to Print the Log Message to File and Console in Python
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

META_INFO = 18  # extend log level (between Debug and Info level)
logging.addLevelName(META_INFO, 'EXTRA')  # extend log level


def meta_info(self, message, *args, **kws):  # extend log level
    self.log(META_INFO, message, *args, **kws)  # extend log level


logging.Logger.meta_info = meta_info  # extend log level

if LOG_LEVEL == 'PRODUCTION':
    # Use the logging Module to Print the Log Message to Console in Python
    logging.basicConfig(
        level=logging.ERROR,
        format="[%(levelname)s] %(message)s")

    extend_logging = logging.getLogger()  # extend log level
    extend_logging.setLevel(logging.ERROR)  # extend log level
else:
    # Use the logging Module to Print the Log Message to Console in Python
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s")

    extend_logging = logging.getLogger()  # extend log level
    extend_logging.setLevel(META_INFO)  # extend log level
