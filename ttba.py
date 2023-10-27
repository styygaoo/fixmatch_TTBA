from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit


class TTBA(nn.Module):
    """TTBA adapts a model by fixmatch during testing.

    Once TTBAed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"  # if not steps >=0, then trigger error
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory

        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, image, weak, strong):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(image, weak, strong, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(image, weak, strong, model, optimizer):
    """Forward and adapt model on batch of data.
       take gradients, and update params.
    """
    # forward
    predictions_weaks = model(weak)
    predictions_strongs = model(strong)
    # predictions_weaks = predictions_weaks.detach()   # detach the target before computing the loss  https://stackoverflow.com/questions/72590591/the-derivative-for-target-is-not-implemented

    # adapt
    loss = torch.nn.L1Loss()
    loss = loss(predictions_strongs, predictions_weaks)
    print("loss: ", loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    prediction = model(image)
    return prediction


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

