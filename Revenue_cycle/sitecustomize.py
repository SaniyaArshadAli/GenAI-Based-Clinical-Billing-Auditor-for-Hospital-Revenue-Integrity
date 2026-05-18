"""Project-local Python startup fixes for this Windows laptop.

Python 3.12 can hang in ``platform._wmi_query`` when Windows WMI is unhealthy.
Streamlit imports ``platform.system()`` during startup, so skip WMI and let
Python use its built-in fallback based on ``sys.getwindowsversion()``.
"""

import platform


def _raise_wmi_not_supported(*_args, **_kwargs):
    raise OSError("WMI disabled for this project runtime")


if hasattr(platform, "_wmi_query"):
    platform._wmi_query = _raise_wmi_not_supported
