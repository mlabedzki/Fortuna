from .analytics import *
from .loaders import *
from .ml import *
from .spectral import *
from .strategies import *
from .utils import *

# import importlib
# import pkgutil

# # Automatically find and import all submodules in this directory
# __all__ = []
# for _, module_name, _ in pkgutil.walk_packages(__path__):
#     # Import the submodule dynamically
#     submodule = importlib.import_module(f"{__name__}.{module_name}")
    
#     # Expose everything from the submodule to the top package level
#     for attribute_name in dir(submodule):
#         if not attribute_name.startswith('_'):
#             globals()[attribute_name] = getattr(submodule, attribute_name)
#             __all__.append(attribute_name)