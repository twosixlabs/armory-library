from torch import tensor
from torchmetrics import Metric


class TextClassificationAccuracy(Metric):
    """A rough whack to measure accuracy on T/F questions.
    The first word of the response is checked against a few affirmative and negative words.
    Responses that don't begin with one of these words are counted as incorrect
    (since the model was instructed to respond with "true" or "false" anyway...)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, y, y_target):
        assert len(y) == 1  # batch size 1 for now
        self.total += 1
        y = y[0].lower().strip()
        y_target = y_target[0]  # Is 0 or 1

        if y_target == 0:
            # if y[:5] == "false" or y[:2] == "no":
            if y.startswith("false") or y.startswith("no") or y.startswith("incorrect"):
                self.correct += 1

        elif y_target == 1:
            if y.startswith("true") or y.startswith("yes") or y.startswith("correct"):
                # if y[:4] == "true" or y[:3] == "yes":
                self.correct += 1

        else:
            # incorrect, or not answered adequately
            pass

    # Alternative update:
    # responses that don't begin with yes, no, true, or false are ignored in the count

    # def update(self, y, y_target):
    #     assert len(y) == 1  # batch size 1 for now
    #     y = y[0].lower().strip()
    #     y_target = y_target[0]  # Is 0 or 1

    #     # if y[:5] == "false" or y[:2] == "no":
    #     if y.startswith("false") or y.startswith("no") or y.startswith("incorrect"):
    #         self.total += 1
    #         if y_target == 0:
    #             self.correct += 1

    #     # elif y[:4] == "true" or y[:3] == "yes":
    #     if y.startswith("true") or y.startswith("yes") or y.startswith("correct"):
    #         self.total += 1
    #         if y_target == 1:
    #             self.correct += 1

    #     else:
    #         # responses that don't begin with yes, no, true, or false are ignored in the count
    #         pass

    def compute(self):
        return self.correct.float() / self.total
