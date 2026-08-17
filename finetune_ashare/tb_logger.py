class TBLogger:
    def __init__(self, log_dir: str, enabled: bool = True):
        self._writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
