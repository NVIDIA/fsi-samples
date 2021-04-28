from .node import *  # noqa: F401,F403
from .taskSpecSchema import *  # noqa: F401,F403
from .taskGraph import *  # noqa: F401,F403
from .portsSpecSchema import *  # noqa: F401,F403
from .metaSpec import *  # noqa: F401,F403
import sys
try:
    # For python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # prior to python 3.8 need to install importlib-metadata
    import importlib_metadata

# load all the plugins from entry points
for entry_point in \
        importlib_metadata.entry_points().get('greenflow.plugin', ()):
    mod = entry_point.load()
    name = entry_point.name
    sys.modules[name] = mod
