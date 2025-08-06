    #https://www.youtube.com/watch?v=q_HS4s1L8UI

def arrange(lookback):
    import pandas as pd
    import numpy as np
    from copy import deepcopy as dc

    data=pd.read_csv('dataset.csv') #import dataset
    data=data[['timestamp', 'value']] #filter out unneeded data columns
    data['timestamp']=pd.to_datetime(data['timestamp']) #organize x-axis data

    def prepare_dataframe_for_lstm(df, n_steps):
        df = dc(df)
        df.set_index('timestamp', inplace=True)

        df[[f'value(t-{i})' for i in range(1, n_steps + 1)]] = df['value'].shift(periods=list(range(1, n_steps + 1)), axis=0)

        df.dropna(inplace=True)
        return df
    shifted_df = prepare_dataframe_for_lstm(data, lookback) #function to arrange previous data points

    shifted_df_as_np = shifted_df.to_numpy()

    return shifted_df_as_np

def LSTMmodel(lookback, hidlayers, recurse, lrate, num_epochs, batch_size, dat, shifted_df_as_np):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score

    from copy import deepcopy as dc
    import random #for setting the seed
    import math #for error analysis (square root function)

    seed=42 #seed
    futuresteps=200 #futurestep
    outputs=1 #outputs

    learning_rate=lrate/1000000
    splt = dat/100

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ferror=0 #forecasting error

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' #use graphics card instead of cpu

    #scaled data from between 0 to 1
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(shifted_df_as_np)

    x=scaled_df[:, 1:] #lookback data
    y=scaled_df[:, 0] #actual data point
    x=dc(np.flip(x, axis=1)) #flip the lookback data points t-1→t-15 to t-15→t-1

    #split into 3 categories
    dataL=len(y) #length of dataset

    xtrain=x[:int(dataL*splt)]
    xtest=x[int(dataL*splt):dataL-futuresteps]
    xverify=x[dataL-futuresteps:]

    ytrain=y[:int(dataL*splt)]
    ytest=y[int(dataL*splt):dataL-futuresteps]
    yverify=y[dataL-futuresteps:]

    #add dimension to match pytorch
    xtrain=xtrain.reshape((-1, lookback, 1))
    xtest=xtest.reshape((-1, lookback, 1))

    ytrain=ytrain.reshape((-1, 1))
    ytest=ytest.reshape((-1, 1))

    # Ensure ytrain and ytest are float32 in NumPy
    ytrain = ytrain.astype(np.float32)
    ytest = ytest.astype(np.float32)

    #convert to Pytorch tensors
    Xtrain=torch.from_numpy(xtrain).float()
    Xtest=torch.from_numpy(xtest).float()
    Ytrain=torch.from_numpy(ytrain).float()
    Ytest=torch.from_numpy(ytest).float()

    #make dataset to train LSTM in Pytorch
    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    train_dataset = TimeSeriesDataset(Xtrain, Ytrain)
    test_dataset = TimeSeriesDataset(Xtest, Ytest)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        break

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers):
            super().__init__()
            self.hidden_size = hidden_size #number of hidden layers in LSTM
            self.num_stacked_layers = num_stacked_layers #stacking LSTM models to boost complexity

            self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            batch_size = x.size(0)
            h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    model = LSTM(outputs, hidlayers, recurse) #(number of features, number of hidden layers, number of LSTMs)
    model.to(device)

    #function for model in training mode
    def train_one_epoch():
        model.train(True)
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                running_loss = 0.0

    #function for model in testing mode
    def validate_one_epoch():
        model.train(False)
        running_loss = 0.0

        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch()
        validate_one_epoch()

    #plotting training results
    with torch.no_grad():
        predicted=model(Xtrain.to(device)).to('cpu').numpy()

    #invert the scale of the the training data
    trainpred=predicted.flatten()
    empty=np.zeros((Xtrain.shape[0], lookback+1))
    empty[:, 0]= trainpred
    empty=scaler.inverse_transform(empty)
    trainpred=dc(empty[:, 0])

    empty=np.zeros((Xtrain.shape[0], lookback+1))
    empty[:, 0]= Ytrain.flatten()
    empty=scaler.inverse_transform(empty)
    nytrain=dc(empty[:, 0])

    #repeat simialr process with test data
    test_predictions = model(Xtest.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((Xtest.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((Xtest.shape[0], lookback+1))
    dummies[:, 0] = Ytest.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])

    #forecast
    future=x[len(x)-1,:]
    forecasted=np.array([])

    for i in range(futuresteps):
      xval=future[len(future)-lookback:len(future)] #take last lookback# of elements of future

      xval=xval.reshape((-1, lookback, 1)) #reshape xval for Pytorch

      Xval=torch.from_numpy(xval).float() #convert to Pytorch tensor
      yval = model(Xval.to(device)).detach().cpu().numpy().flatten() #send xval to model
      #add output to forecasted and future
      future=np.append(future, yval)
      forecasted=np.append(forecasted, yval)

    #inverse scale forecasted
    fmodel=forecasted.flatten()
    empty=np.zeros((futuresteps, lookback+1))
    empty[:, 0]= fmodel
    empty=scaler.inverse_transform(empty)
    fmodel=dc(empty[:, 0])

    #inverse scale forecasted
    verify=yverify.flatten()
    empty=np.zeros((futuresteps, lookback+1))
    empty[:, 0]= verify
    empty=scaler.inverse_transform(empty)
    verify=dc(empty[:, 0])

    #error analysis
    for e in range(len(fmodel)):
      hold = abs(fmodel[e]-verify[e])
      hold = hold*hold
      hold = math.sqrt(hold)
      ferror=ferror+hold
    ferror=ferror/len(fmodel)

    #print parameters to file
    with open('parameters.txt', 'a') as f:
        f.write(f'Lookback:{str(lookback)}\n')
        f.write(f'Hidden Layers:{str(hidlayers)}\n')
        f.write(f'Stacks:{str(recurse)}\n')
        f.write(f'Learning Rate:{str(learning_rate)}\n')
        f.write(f'Epochs:{str(num_epochs)}\n')
        f.write(f'Data Split:{str(splt)}\n')
        f.write(f'Error:{str(ferror)}\n\n')

    plt.plot(verify, label='Actual') #make plot comparing forecast with real data
    plt.plot(fmodel, label='Predicted')
    plt.legend()
    plt.savefig('forecast_result.png') #saves plot result to png file

    return int(ferror)