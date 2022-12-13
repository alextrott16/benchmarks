from composer.optim.scheduler import ComposerScheduler, _convert_time, LinearScheduler
from composer.core import State, Time, TimeUnit
from typing import Union

__all__ = ['InverseSquareRootWithWarmupScheduler']

class InverseSquareRootWithWarmupScheduler(ComposerScheduler):
    def __init__(self,
                 t_warmup: Union[str, Time],
                 alpha_max: float = 0.01,
                 scale: float = 1.0,
                 scale_warmup: bool = False,
                 ):
        self.t_warmup = t_warmup
        self.alpha_max = alpha_max
        assert 0 < self.alpha_max <= 1.0
        self.scale = scale
        assert self.scale > 0.0
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)

        self.scale_warmup = scale_warmup

    def __call__(self, state: State, ssr: float = 1.0):
        t_warmup = _convert_time(self.t_warmup, state)

        warmup_coeff = 1.0
        if state.timestamp < t_warmup:
            if self.scale_warmup:
                warmup_coeff = self.warmup_scheduler(state, ssr)
            warmup_coeff = self.warmup_scheduler(state)

        current_time = state.timestamp.get(TimeUnit.BATCH).value / (ssr * self.scale)

        isq_factor = min(self.alpha_max, 1.0 / ((current_time + 1e-12)**0.5))
        current_factor = isq_factor * warmup_coeff

        return current_factor