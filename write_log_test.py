from datetime import datetime
from tensorboard_logger import configure, log_value
import os


log_dir = './output'
log_dir = os.path.join(log_dir, datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# init log
configure(log_dir)

epochs = 10

for i in range(epochs):
    log_value('acc', i*10, i)


