try:
    from . import config
    from .anfis_model import ANFISInference
except ImportError:
    import config
    from anfis_model import ANFISInference

class ThresholdManager:

    def __init__(self, use_anfis=False, weights_path=None):
        self.default_lt = config.LOW_THRESHOLD
        self.default_ut = config.HIGH_THRESHOLD
        self.lt = self.default_lt
        self.ut = self.default_ut
        self.model = None

        if use_anfis:
            if weights_path and __import__('os').path.isfile(weights_path):
                self.model = ANFISInference.from_file(weights_path)
            else:
                print(f"[ThresholdManager] ANFIS weights not found: {weights_path}, "
                      f"falling back to fixed thresholds")

    @property
    def active(self):
        return self.model is not None

    def get(self):
        return self.lt, self.ut

    def update(self, s, d, u):
        if self.model is None:
            return
        lt, ut = self.model.predict(s, d, u)

        lt = max(0.01, min(lt, 0.45))
        ut = max(lt + 0.1, min(ut, 0.95))
        self.lt = lt
        self.ut = ut

    def reset(self):
        self.lt = self.default_lt
        self.ut = self.default_ut
