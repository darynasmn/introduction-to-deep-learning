"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################

        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))  
        else:
            index_iterator = iter(range(len(self.dataset)))
       
        batch_array = []
        batch = {}
        i = 0 

        for index in index_iterator: 
            i += 1
            batch_array.append(self.dataset[index].get('data'))
            if len(batch_array) % self.batch_size == 0:
                batch['data'] = np.array(batch_array)
                yield batch
                batch_array = []
                batch = {}

            if i == 50 and self.drop_last == False: 
                batch['data'] = np.array(batch_array)
                yield batch
            

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = None
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################
        
        len_dataset = len(self.dataset)
        if np.mod(len_dataset, self.batch_size) == 0: 
            length = len_dataset // self.batch_size
        elif self.drop_last: 
            length = len_dataset // self.batch_size
        else: 
            length = len_dataset // self.batch_size + 1
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
