from .upernet import UPerHead
from .segformer import SegFormerHead
from .sfnet import SFHead
from .fpn import FPNHead
from .fapn_CVPR import FaPNHead, FaPNCBAMHead
from .AFAM import FAMHead
__all__ = ['UPerHead', 'FAMHead', 'SegFormerHead', 'SFHead', 'FPNHead', 'FaPNHead', 'FaPNCBAMHead']
