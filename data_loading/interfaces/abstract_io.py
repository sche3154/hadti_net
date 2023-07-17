from abc import ABC, abstractmethod

class Abstract_IO(ABC):

    #---------------------------------------------#
    #               load_data                     #
    #---------------------------------------------#
    @abstractmethod
    def load_data(self, path):
        pass


    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    """ Backup the prediction of the image with the index i into the output directory.

        Parameter:
            pred (numpy matrix):    MIScnn computed prediction for the sample index
            index (variable):       An index from the provided indices_list of the initialize function
            output_path (string):   Path to the output directory in which MIScnn predictions are stored.
                                    This directory will be created if not existent
        Return:
            None
    """
    @abstractmethod
    def save_prediction(self, pred, output_path):
        pass
