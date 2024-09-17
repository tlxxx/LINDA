from torch.utils.tensorboard.writer import SummaryWriter

class Plogger:
    def __init__(self, name):
        self.log_dir = "/home/amax/lgh/LINDA/pymarl-master/results/plog/" + name
        self.summary_writer = SummaryWriter(log_dir = self.log_dir, comment= "114514")

    def log_result(self, attri, value, episode):
        self.summary_writer.add_scalar(attri, value, episode)