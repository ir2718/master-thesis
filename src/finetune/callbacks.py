class BaseEpochCallback:
    def __init__(self):
        pass

    def on_epoch_begin(self): 
        pass

    def on_epoch_end(self): 
        pass


# unfreezes the backbone at unfreeze epoch, at the beggining pooler and cls get unfrozen
# unfreeze epoch is a number in {1, 2, ..., num_epochs - 1} 
# 0 doesnt make sense, just dont use the callback
class FrozenHeadCallback(BaseEpochCallback):
    def __init__(self, model, num_epochs, unfreeze_epoch):
        super().__init__()
        self.model = model
        self.num_epochs = num_epochs
        self.curr_epoch = None
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_begin(self): 
        if self.curr_epoch is None:
            self.curr_epoch = 0 
            
            for p in self.model.model.parameters():
                p.requires_grad = False

            for p in self.model.pooler.parameters():
                p.requires_grad = True

            for p in self.model.cls.parameters():
                p.requires_grad = True

        else:
            self.curr_epoch += 1
            
            if self.curr_epoch == self.unfreeze_epoch:

                for p in self.model.model.parameters():
                    p.requires_grad = True


class GradualUnfreezingHeadCallback(BaseEpochCallback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.curr_epoch = 0
        self.curr_layer_to_unfreeze = 12

    def on_epoch_begin(self):
        if self.curr_epoch == 0:

            for p in self.model.parameters():
                p.requires_grad = False

            if hasattr(self.model, "pooler"):
                for p in self.pooler.parameters():
                    p.requires_grad = False

            for p in self.cls.parameters():
                p.requires_grad = True

        elif self.curr_epoch == 1:

            if hasattr(self.model, "pooler"):
                for p in self.pooler.parameters():
                    p.requires_grad = True

        else:
            
            if self.curr_layer_to_unfreeze > 0:

                for p in self.model.encoder.layer[self.curr_layer_to_unfreeze]:
                    p.requires_grad = True

                for p in self.model.encoder.layer[self.curr_layer_to_unfreeze-1]:
                    p.requires_grad = True

                self.curr_layer_to_unfreeze -= 2

            else:
                
                for p in self.model.embeddings:
                    p.requires_grad = True

        self.curr_epoch += 1
