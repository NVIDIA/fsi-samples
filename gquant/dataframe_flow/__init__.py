from .node import *  # noqa: F401,F403
from .taskSpecSchema import *  # noqa: F401,F403
from .taskGraph import *  # noqa: F401,F403
from .portsSpecSchema import *  # noqa: F401,F403
import sys
import pkg_resources

# load all the plugins from entry points
for entry_point in pkg_resources.iter_entry_points('gquant.plugin'):
    mod = entry_point.load()
    name = entry_point.name
    sys.modules[name] = mod
