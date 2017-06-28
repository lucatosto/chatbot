import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, source_file, target_file):
        # Read lines from source and target
        with open(source_file) as f:
            source_lines = [x.strip() for x in f]
        with open(target_file) as f:
            target_lines = [x.strip() for x in f]
        # Convert each line into sequence
        source_lines = [[int(x) for x in line.split(' ')] for line in source_lines]
        target_lines = [[int(x) for x in line.split(' ')] for line in target_lines]
        # TODO check same number of items in the lists
        # Find vocabulary length (just from source, should be the same)
        self.vocab_len = max([x for seq in source_lines for x in seq]) + 1 + 2 # SOS, EOS symbols
        self.sos_idx = self.vocab_len - 2
        self.eos_idx = self.vocab_len - 1
        # Find max seq length (from both sources and target; shorter sequences will be padded)
        self.seq_len = max([len(seq) for seq in source_lines] + [len(seq) for seq in target_lines]) + 2 # SOS + EOS symbol
        # Get dataset length
        self.size = len(source_lines)
        # Generate one-hot vectors   #il dataset va salvato in txt e qui lo riconverto in tensore
        self.source_data = torch.ByteTensor(self.size, self.seq_len, self.vocab_len).zero_()
        self.target_data = torch.ByteTensor(self.size, self.seq_len, self.vocab_len).zero_()
        for i in range(0, self.size):
            # Set start token
            self.source_data[i, 0, self.sos_idx] = 1
            self.target_data[i, 0, self.sos_idx] = 1
            for j in range(1, self.seq_len):
                # Set source
                if j-1 < len(source_lines[i]):
                    self.source_data[i, j, source_lines[i][j-1]] = 1
                else:
                    self.source_data[i, j, self.eos_idx] = 1
                # Set target
                if j-1 < len(target_lines[i]):
                    self.target_data[i, j, target_lines[i][j-1]] = 1
                else:
                    self.target_data[i, j, self.eos_idx] = 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        # Return data
        # NOTE: target includes start token! it must be stripped when computing the loss
        return self.source_data[i, :, :].float(), self.target_data[i, :, :].float()

if __name__ == "__main__":
    dataset = Dataset(source_file = "data/train/sources.txt", target_file = "data/train/targets.txt")
    print(dataset[0][0][0])
