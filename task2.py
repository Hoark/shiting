import json
import torch, torchaudio, torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import random
import os
import utils
import numpy as np
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
from statistics import mode

# Globals
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
g_sample_ms = 500 # 500ms
g_sample_len = utils.g_sr // 1000 * g_sample_ms  # Floor division

#Audio Processing Utilities
class AudioUtil:
    # Utilities for Audio processing
    @staticmethod
    def audioload(filepath):
        #Load audio file
        waveform, sample_rate = torchaudio.load(filepath)
        return waveform, sample_rate

    @staticmethod
    def rechannel(audio, nchannel):
        # Convert the given audio to the desired number of channels
        waveform, sample_rate = audio
        if (waveform.shape[0] == nchannel):
            # Do nothing
            return audio
        elif (nchannel == 1):
            # Convert from stereo to mono by selecting only the first channel
            rechannelled = waveform[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            rechannelled= torch.cat([waveform, waveform])
        return ((rechannelled, sample_rate))

    @staticmethod
    def resample(aud, newsr):
        # Resample audio. Since Resample applies to a single channel, we resample one channel at a time
        wf, sr = aud
        if (sr == newsr):
            # Nothing to do
            return aud
        num_channels = wf.shape[0]
        # Resample first channel
        rewf = torchaudio.transforms.Resample(sr, newsr)(wf[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(wf[1:, :])
            resig = torch.cat([rewf, retwo])
        return ((rewf, newsr))

    @staticmethod
    def extract_sample(aud, sample_ms):
      # Extract a random segment of length sample_ms
      sig, sr = aud
      sample_len = sr // 1000 * sample_ms # Floor division
      num_rows, sig_len = sig.shape
      rnd = random.randint(0,sig_len-sample_len)
      sig = sig[:, rnd:rnd + sample_len]
      return (sig, sr)

    @staticmethod
    def pad_trunc(aud, max_ms):
        # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
        sig, sr = aud
        num_rows, sig_len = sig.shape #shape: (numChannels, lenAudio)
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            # Randomly Truncate the signal to the given length
            rnd = random.randint(0, sig_len - max_len)
            sig = sig[:, rnd:max_len+rnd]
        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len) # We use random here to prevent that the silence falls in the same position
            pad_end_len = max_len - sig_len - pad_begin_len
            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit):
        # Shifts the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of the transformed signal.
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mfcc=20, n_fft=1024, hop_len=None):
        # Generate a Spectrogram. Spectrograms are better for NN than audio data.
        sig, sr = aud
        top_db = 80
        spec = torchaudio.transforms.MFCC(sr, n_mfcc=n_mfcc)(sig) # Use MFCC, better suited for human speech
        return (spec)

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        # Augment the Spectrogram by masking out some sections of it in both the frequency
        # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
        # overfitting and to help the model generalise better. The masked sections are
        # replaced with the mean value.
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

def label_to_index(searchlabel):
  for index, label in utils.ID_dict.items():
    if label == searchlabel:
      return (index-1)
  return 1

def index_to_label(index):
  return utils.ID_dict[index+1]

class myTrainDataset(Dataset):
  def __init__(self, path):
    self.labels, self.audio_samples = self.Data_Load(path)
    self.sr = utils.g_sr  # Global Sample Rate
    self.seg_dur = g_sample_ms # 500 ms
    self.shift_pct = g_sample_ms*0.1  # Time shift. We do not use it (for now)

  def Data_Load(self, path):
    # Load data. Data will be augmented in getitem, so no need to do that here.
    labels = list();
    audio_samples = list()
    for label in os.listdir(f'{path}'):
      for filename in os.listdir(f'{path}/{label}'):
        filepath = f'./{path}/{label}/{filename}'
        audio = utils.read_audio(filepath)
        for i in range(audio.shape[0] // g_sample_len):
          labels.append(label)
          audio_samples.append(audio[i * g_sample_len:(i + 1) * g_sample_len, :])
    return labels, audio_samples

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    # Get the Class ID
    target = label_to_index(self.labels[idx])

    # Load audio file
    aud =  self.audio_samples[idx]
    aud = torch.transpose(torch.tensor(aud),0,1)
    #aud_segment = AudioUtil.pad_trunc([aud, utils.g_sr], self.seg_dur)
    # Generalise data
    #dur_aud = AudioUtil.pad_trunc([aud, utils.g_sr], self.duration) # Extract a random 4 sec segment out of the entire audio file

    # Augmentations
    #shift_aud = AudioUtil.time_shift(aud_segment, self.shift_pct) # Not Needed. Speech has time-correlation a time shift could mess that up
    sgram = AudioUtil.spectro_gram([aud, utils.g_sr]) # Generate a spectrogram
    rnd_freq = random.randint(0,2+1) #randint(a,b+1) picks random int from a<=N<=b
    rnd_time = random.randint(0,2+1)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=rnd_freq, n_time_masks=rnd_time) # Not Needed (for now)

    end_sgram = aug_sgram
    return end_sgram, target

class myTestDataset(Dataset):
    def __init__(self, path):
      self.labels, self.data_path = self.Data_Load(path)
      self.seg_dur = g_sample_ms  # 500 ms

    def Data_Load(self, path):
      # Load data.
      labels = list(); data_path = list()
      with open('./test_offline/task2_gt.json', 'r') as f:
        task2_gt = json.load(f)
        for file_name in os.listdir(path):
          labels.append(task2_gt[file_name])
          filepath = f'./{path}/{file_name}'
          data_path.append(filepath)
      return labels, data_path

    def __len__(self):
      return len(self.labels)

    def __getitem__(self, idx):
      # Get the Class ID
      target = label_to_index(self.labels[idx])

      # Load audio file
      audio_path = self.data_path[idx]
      audio = utils.read_audio(audio_path)
      audio = torch.transpose(torch.tensor(audio),0,1)
      sgrams = list()
      for i in range(audio.shape[1] // g_sample_len):
        aud_segment = (audio[:,i * g_sample_len:(i + 1) * g_sample_len])
        sgrams.append(AudioUtil.spectro_gram([aud_segment, utils.g_sr]))
      #aud_segment = AudioUtil.pad_trunc([aud, utils.g_sr], self.seg_dur)
      #sgram = AudioUtil.spectro_gram(aud)  # Generate a spectrogram
      # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2) # Not Needed (for now)

      end_sgram = sgrams
      return end_sgram, target

# Audio Classification Model
class AudioClassifier(nn.Module):
    # Build the model architecture. Based on https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
    # with small modifications made to make sure that the sizes and outputs are correct.
    def __init__(self):
        super().__init__()
        conv_layers = []
        self.layer = [4,8,16,32,64,64,128,128,128]

        in_channels = 1
        for v in self.layer:
          conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          bn2d = nn.BatchNorm2d(v)
          relu2d = nn.ReLU()
          init.kaiming_normal_(conv2d.weight, a=0.1)
          conv2d.bias.data.zero_()
          conv_layers += [conv2d, bn2d, relu2d]
          in_channels = v

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # Change in_features when changing conv. layers
        self.lin = nn.Linear(in_features=self.layer[-1], out_features=20) #Input is an array of size 64, and we have 20 possible outputs (The bird species)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # Forward pass computations
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        # Linear layer
        x = self.lin(x)
        # Final output
        return x

# Training Loop
def training(model, train_dl):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=1,
                                                    anneal_strategy='linear')

    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

        # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      scheduler.step()

      # Keep stats for Loss and Accuracy
      running_loss += loss.item()

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs, 1)
       # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]

    # Save stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction / total_prediction
    return avg_loss, acc

# Testing Loop
def testing(model, val_dl):
  correct_prediction = 0; total_prediction = 0
  running_loss = 0; criterion = nn.CrossEntropyLoss()
  # Disable gradient updates
  with torch.no_grad():
    for i, data in enumerate(val_dl):
      predictions = list()
      # Get the input features and target labels, and put them on the GPU
      inlist, label = data[0], data[1].to(device)
      for inputs in inlist:
        inputs = inputs.to(device)
        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Get predictions
        outputs = model(inputs)
        loss = criterion(outputs, label)
        running_loss += loss.item()
        # Get the predicted class with the highest score
        _, predict= torch.max(outputs, 1)
        predict = predict.item()
        predictions.append(predict)
      prediction = mode(predictions)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == label.item())
      total_prediction += 1

  acc = correct_prediction / total_prediction
  num_batches = len(val_dl)
  avg_loss = running_loss / num_batches
  return avg_loss, acc

# Main
def run(train_path, test_path):
  print('Start DataSet and DataLoader')
  # Dataset
  train_ds = myTrainDataset(train_path)
  test_ds = myTestDataset(test_path)
  # DataLoader
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
  test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
  print('Finish DataSet and DataLoader')

  '''
  myModel = AudioClassifier()
  myModel = myModel.to(device)
  '''
  model = torchvision.models.resnet18(pretrained=True)
  model.conv1 = nn.Conv2d(1, model.conv1.out_channels,
                          kernel_size=model.conv1.kernel_size[0],
                          stride=model.conv1.stride[0],
                          padding=model.conv1.padding[0])
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 20)
  myModel = model
  myModel = myModel.to(device)

  #myModel = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)

  fail_cnt = 0; fail_tresh = 10; fail_loss = 99
  nEpoch = 0
  print('Started Training and Testing')
  while True:
    # Train and test
    train_loss, train_acc = training(myModel, train_dl)
    test_loss, test_acc = testing(myModel, test_dl)
    print(f'Epoch:{nEpoch}, Train Loss:{train_loss:.2f}, Train Accuracy:{train_acc:.2f}, Test Loss:{test_loss:.2f}, Test Accuracy:{test_acc:.2f}')

    # Counter Overfitting
    if test_loss>fail_loss:
      if fail_cnt == fail_tresh:
        print('end 1')
        break
      fail_cnt += 1
    else:
      fail_loss = test_loss
      fail_cnt = 0
      # Save Current Best Model
      torch.save(myModel.state_dict(), 'mymodel_param.pt')
      torch.save(myModel, 'mymodel.pt')
      # Save Test statistics
      f = open('test_stats', 'w')
      f.write(f'Test Accuracy:{test_acc}\nTrain Accuraccy:{train_acc} \nEpoch:{nEpoch}')
      f.close()

    nEpoch += 1

  return test_acc