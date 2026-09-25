# SPDX-License-Identifier: Apache-2.0
"""Mixed-PoC (chat + PoC in one forward).

Depends one-way on the gonka_poc core: the core never imports gonka_poc.mixed.

The engine calls into this package from the residual seams: the scheduler asks
``poc_step_tokens`` how many tokens a PoC row takes this step (its whole prompt
at prefill, one token per decode step), and the model runner drives
``PoCRunnerBridge`` around the forward. Scheduling itself is vLLM's: PoC rows
are admitted, budgeted, allocated and preempted like chat (ADR-0017).
"""

from gonka_poc.mixed.admission import poc_step_tokens  # noqa: E402
from gonka_poc.mixed.bridge import PoCRunnerBridge  # noqa: E402
