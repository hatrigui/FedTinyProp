class EarlyStoppingMonitor:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_acc = 0.0
        self.best_round = 0
        self.counter = 0
        self.history = []

    def step(self, acc, round_number):
        self.history.append((round_number, acc))
        if acc > self.best_acc + self.delta:
            self.best_acc = acc
            self.best_round = round_number
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def summary(self):
        return {
            "best_acc": self.best_acc,
            "best_round": self.best_round,
            "history": self.history
        }
