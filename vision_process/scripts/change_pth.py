import torch
import os
import torch.utils.data
import torchvision.transforms as transforms
from DenseFusion_Lib.network import PoseNet,PoseRefineNet
from SegNet_Lib.segnet import SegNet

def change_pose_pth():
    """
    这里面用于更换torch(从一个zip的到原始的pth)
    :return:
    """
    model_path="sapien_posemodel_0.042.pth"
    python_path=os.path.dirname(os.path.abspath(__file__))
    estimator = PoseNet(num_points=1000, num_obj=79)
    estimator.cuda()
    estimator.load_state_dict(torch.load(python_path+"/"+model_path))
    torch.save(estimator.state_dict(), 'output_model.pth',_use_new_zipfile_serialization=False)
    print("Saved")


def change_segnet_pth():
    """
    这里面用于更换torch(从一个zip的到原始的pth)
    :return:
    """
    model_path="model_58_0.0054804237359868625.pth"
    python_path=os.path.dirname(os.path.abspath(__file__))
    estimator = SegNet(input_nbr=3,label_nbr=80)
    estimator.cuda()
    estimator.load_state_dict(torch.load(python_path+"/"+model_path))
    torch.save(estimator.state_dict(), 'output_model.pth',_use_new_zipfile_serialization=False)
    print("Saved")

if __name__ == '__main__':
    # change_pose_pth()
    change_segnet_pth()