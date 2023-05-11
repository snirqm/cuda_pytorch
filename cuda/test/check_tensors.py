from torch import Tensor, allclose, all, where
def assert_tensors(C_res: Tensor, C_target: Tensor):
    if C_res.shape != C_target.shape:
        raise AssertionError(f"Pytorch and CUDA results differ in shape {C_res.shape} != {C_target.shape}")
    if(allclose(C_res, C_target)):
        return
    if(all(C_res < C_target)):
        raise AssertionError(f"Pytorch and CUDA results differ, but it seems like Pytorch computed more than CUDA\n {C_res} < {C_target}")
    elif (all(C_res > C_target)):
        raise AssertionError(f"Pytorch and CUDA results differ, but it seems like Pytorch computed less than CUDA\n {C_res} > {C_target}")
    else:
        raise AssertionError(f"Pytorch and CUDA results differ in indexes {where(C_res != C_target)}")