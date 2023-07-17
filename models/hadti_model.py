import torch
from models.base_model import BaseModel
from models.networks import define_net
from models.loss import L1Loss

class HadtiModel(BaseModel): # HADTI
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add model-specific options.
        """

        return parser


    def __init__(self, opt):
        """
        Initialize this mask Inpaint class.
        """
        BaseModel.__init__(self, opt)

        self.model_names = ['U3D']
        self.loss_names = ['l1']
        
        self.net_U3D = define_net(net_name = opt.net, input_nc = opt.input_nc, output_nc = opt.output_nc
        , init_type = opt.init_type, init_gain = opt.init_gain
        , gpu_ids = opt.gpu_ids)

        if self.isTrain:
            self.l1loss = L1Loss(weight = 1)
            self.optimizer_sr = torch.optim.Adam(self.net_U3D.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_sr)


    def set_input(self, input):
        """
        Read the data of input from dataloader then
        """
        self.dwi = input['dwi'].to(self.device)
        self.t1 = input['t1'].to(self.device)

        if self.isTrain:
            self.gt_dti = input['gt_dti'].to(self.device)
            self.wm_mask = input['wm_mask'].to(self.device)


    def forward(self):
        """
        Run forward pass
        """
        self.input = torch.cat((self.dwi, self.t1), dim=1)
        self.sr = self.net_U3D(self.input)

        return self.sr

    
    def backward_sr(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_l1 = self.l1loss(self.sr * self.wm_mask, self.gt_dti * self.wm_mask)
        self.loss_l1.backward()


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        # update optimizer of the inpainting network
        self.optimizer_sr.zero_grad()
        self.backward_sr()
        self.optimizer_sr.step()

