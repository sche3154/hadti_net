import os
import time
import warnings
from utils.utils import *
from tensorboardX import SummaryWriter

class Recorder():
    """
    This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a logging file to store training losses, this can be visualized using tensorboard
        """
        self.opt = opt  # cache the option
        log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard')
        mkdirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=opt.name, flush_secs = 20)
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def plot_current_losses(self, current_iters, losses):
        """display the current losses on tensorboard display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v, current_iters)
            warnings.simplefilter(action='ignore', category=FutureWarning)
        self.writer.flush()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """
        print current losses on console and save the losses to the logfile

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.4f, data: %.4f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # losses: same format as |losses| of plot_current_losses
    def print_epoch_losses(self, epoch, iters, losses, time):
        """
        print current losses on console and save the losses to the logfile

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.2f mins) ' % (epoch, iters, time/60)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def close(self):
        self.writer.close()
        warnings.simplefilter(action='ignore', category=FutureWarning)