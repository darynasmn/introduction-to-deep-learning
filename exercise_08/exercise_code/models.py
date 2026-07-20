import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim 

class Encoder(nn.Module):

    def __init__(self, hparams, input_size= (1, 28, 28), latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################
        C, H, W = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(C, self.hparams["enc_channels"][0], kernel_size= self.hparams['enc_kernel_size'], padding=self.hparams['enc_kernel_size']//2),
            nn.BatchNorm2d(self.hparams["enc_channels"][0]),
            nn.ReLU(),
            #nn.Dropout2d(self.hparams['enc_dropout']),
            nn.MaxPool2d(self.hparams['enc_pool_kernel']),

            nn.Conv2d(self.hparams["enc_channels"][0], self.hparams["enc_channels"][1], kernel_size= self.hparams['enc_kernel_size'], padding=self.hparams['enc_kernel_size']//2),
            nn.BatchNorm2d(self.hparams["enc_channels"][1]),
            nn.ReLU(),
            #nn.Dropout2d(self.hparams['enc_dropout']),
            nn.MaxPool2d(self.hparams['enc_pool_kernel']),

            nn.Conv2d(self.hparams["enc_channels"][1], self.hparams["enc_channels"][2], kernel_size=self.hparams['enc_kernel_size'], padding=self.hparams['enc_kernel_size']//2),
            nn.BatchNorm2d(self.hparams["enc_channels"][2]),
            nn.ReLU(),
            nn.MaxPool2d(self.hparams['enc_pool_kernel']),

            nn.Flatten(),
            nn.Linear(self.hparams["enc_channels"][2] * (H//8) * (W//8), latent_dim)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hparams['dec_channels'][0]),
            nn.ReLU(),
            nn.Linear(self.hparams['dec_channels'][0], self.hparams['dec_channels'][1]),
            nn.ReLU(),
            nn.Linear(self.hparams['dec_channels'][1], self.output_size),
            nn.Unflatten(1, (1, 28, 28)),
            nn.Sigmoid(),
            ) 

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################
        if x.dim()  == 2: 
            x = x.view(x.size(0), 1 , 28, 28)
        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'],  weight_decay= self.hparams['weight_decay'] )
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################

        self.train()

        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(self.device) 

        self.optimizer.zero_grad() 

        reconstruction = self(images)
        
        if reconstruction.dim() == 2: 
            target = images.view(images.shape[0], -1)

        else:
            target = images        

             
        loss = loss_func(reconstruction, target) 
        loss.backward() 
        self.optimizer.step()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        # and "with torch.no_grad()" wrapper.
        self.eval()
        loss = 0
    
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(self.device)

        with torch.no_grad():
            reconstruction = self(images)

            if reconstruction.dim() == 2: 
                target = images.view(images.shape[0], -1)

            else:
                target = images        

            loss = loss_func(reconstruction, target)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder

        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = nn.Linear(hparams['latent_dim'], hparams['num_class'])        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################
        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams['lr'],  weight_decay= self.hparams['weight_decay'] )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
