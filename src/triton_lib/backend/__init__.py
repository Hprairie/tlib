from .register import register_for_module, register, get, backends
from .base import Backend, get_default, InvalidBackendException

from . import _triton as triton