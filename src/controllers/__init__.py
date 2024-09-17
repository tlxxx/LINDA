REGISTRY = {}

from .basic_controller import BasicMAC
from .LINDA_controller import LINDAMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["LINDA_mac"] = LINDAMAC