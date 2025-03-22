class ComputeMetrics:
    def __init__(self) -> None:
        pass

    def __call__(self, p):
        metric = self.compute_metrics(p=p)
        return metric

    def compute_metrics(self, p):
        pass
