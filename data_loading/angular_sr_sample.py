
class AnuglarSRSample:

    # Create a Sample object
    def __init__(self, index, image, aff=None):

        # Cache data
        self.index = index
        self.dwi = image
        self.shape = self.dwi.shape
        self.aff = aff


    def add_gt_dti(self, gt):
        self.gt_dti = gt


    def add_t1(self, t1):
        self.t1 = t1


    def add_brain_mask(self, mask):
        self.brain_mask = mask


    def add_wm_mask(self, mask):
        self.wm_mask = mask

    
    def add_coords_data(self, coords_data):
        self.coords_data = coords_data


    def add_bb(self, bounding_box, shape_after_boudning = None):
        self.bb = bounding_box
        self.shape_after_boudning = shape_after_boudning


    def add_fa(self, fa):
        self.fa = fa


    def add_md(self, md):
        self.md = md


    def add_roi_1(self, roi):
        self.roi_1 = roi


    def add_roi_2(self, roi):
        self.roi_2 = roi


    def add_roi_3(self, roi):
        self.roi_3 = roi


    def add_l1(self, l1):
        self.l1 = l1

    def add_l2(self, l2):
        self.l2 = l2

    def add_l3(self, l3):
        self.l3 = l3

    def add_gt_dti_mean(self, gt_dti_mean):
        self.gt_dti_mean = gt_dti_mean

    def add_gt_dti_std(self, gt_dti_std):
        self.gt_dti_std = gt_dti_std

    def add_gm_mask(self, gm_mask):
        self.gm_mask = gm_mask

        