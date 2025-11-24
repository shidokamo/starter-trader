import os
import logging
import json
import traceback

class FormatterJSON(logging.Formatter):
#    def formatException(self, exc_info):
#        result = super().formatException(exc_info)
#        return repr(result)
    def format(self, record):
        json_msg = {'message': record.msg}
        # Cloud functions will add their own timestamp so I don't add that in here.
        # record.asctime = self.formatTime(record, self.datefmt)
        # json_msg['time'] = record.asctime
        json_msg['level'] = record.levelname
        json_msg['severity'] = record.levelname
        return json.dumps(json_msg, ensure_ascii=False)

#
# Caution: This will overwrite record itself so if you are using multiple logger, they will be also affected.
#
RESET_SEQ = "\x1b[0m"
class FormatterColor(logging.Formatter):
    def color(self, level):
        match level:
            case 'WARNING':
                return "\x1b[1;43m" + level + RESET_SEQ
            case 'INFO':
                return "\x1b[1;42m" + level + RESET_SEQ
            case 'DEBUG':
                return "\x1b[1;47m" + level + RESET_SEQ
            case 'CRITICAL':
                return "\x1b[1;41m" + level + RESET_SEQ
            case 'ERROR':
                return "\x1b[1;41m" + level + RESET_SEQ
            case _:
                # If it's already colored. Do nothing.
                return level

    def format(self, record):
        record.levelname = self.color(record.levelname)
        return super().format(record)

logger = logging.getLogger('console')
logger.propagate = False  # Some Hyperliquid SDKs use root logger, so we need to disable propagation to avoid duplicate logs.
logger.handlers.clear()  # Clear existing handlers to avoid duplicates

if os.environ.get("PROD"):
    h = logging.StreamHandler()
    fmt = FormatterJSON()
    h.setFormatter(fmt)
    logger.addHandler(h)
else:
    h = logging.StreamHandler()
    fmt = FormatterColor('[%(asctime)s] [%(levelname)s] %(message)s')
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.DEBUG)
