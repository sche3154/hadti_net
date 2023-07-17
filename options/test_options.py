from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--results_dir', type=str, default='./results', help='results dir')
        parser.add_argument('--save_prediction', type=int, default=4, help='save images')
        
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=20, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')

        self.isTrain = False

        
        return parser