import os

from dotenv import load_dotenv

# ACTIVE_ENV = os.environ["ACTIVE_ENV"]
#
# if ACTIVE_ENV == 'production':
#     load_dotenv('.env.production')
# else:
#     load_dotenv()

load_dotenv() # if uncomment the above, should remove

SEED = int(os.environ["SEED"])
# LXC_DB = eval(os.environ["LXC_DB"])  # boolean
LOG_LEVEL= os.environ["LOG_LEVEL"]