from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback
from tensorflow_addons.optimizers import CyclicalLearningRate

def get_callbacks(MODEL_PATH):
    """
    
    Saves model if validation IoU increased, 
    applies Cyclic Learning Rate,
    shows TQDM based progress bar.
    
    """
    checkpoint = ModelCheckpoint(MODEL_PATH, 
                                 monitor='val_iou_score', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')


    clr = CyclicalLearningRate(base_lr=0.000001, 
                   max_lr=0.001,
                   step_size=3000, 
                   mode='exp_range') 

    callbacks = [checkpoint,
                 clr,
                 TQDMNotebookCallback(leave_inner=True, 
                                      leave_outer=True)]

    return callbacks
