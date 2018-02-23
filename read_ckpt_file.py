from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("/tmp/pascal_model_scratch/model.ckpt-51", tensor_name='', all_tensors=True)

# print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# # print only tensor v1 in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)
#
# # print only tensor v2 in checkpoint file
# chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)
#
# # tensor_name:  v2
# # [-1. -1. -1. -1. -1.]