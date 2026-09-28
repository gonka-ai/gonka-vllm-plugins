# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-export of the engine's ``PoCParams``.

The class must be the one the engine annotates ``EngineCoreRequest.poc_params``
with: msgspec encodes it by field NAME and silently drops keys the annotated
class does not declare, so a local copy that drifts by one field loses that
field crossing the process boundary — with no error. Import, never duplicate.
"""

from vllm.poc.poc_params import PoCParams

__all__ = ["PoCParams"]
