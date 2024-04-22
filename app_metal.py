import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries

import requests
import json
from email_metal import email

from metalpriceapi.client import Client

from dotenv import load_dotenv
load_dotenv()

# api_key = '6600ef8939cd5025b77e8909bb3d1a6d'
# client = Client(api_key)

# data = client.timeframe(start_date='2024-02-05', end_date='2024-02-09', base='USD', currencies=['ALU'])
# print(data)
# exit()

# # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
# url = 'https://www.alphavantage.co/query?function=ALUMINUM&interval=monthly&apikey=demo'
# rl = 'https://www.alphavantage.co/query?function=ALUMINUM&interval=monthly&apikey=demo'
# r = requests.get(url)
# data = r.json()



# from alpha_vantage.timeseries import TimeSeries

# key = 'your_api_key'
# ts = TimeSeries(key, output_format='pandas')
# data, meta_data = ts.get_daily_adjusted('ALUMINUM', outputsize='full')


# metalpriceapi.com key - 6600ef8939cd5025b77e8909bb3d1a6d

# print(data)

print("All libraries loaded")

# FREE - 8368BHI5PDJQKUOV
# PREMIUM - O43FB23TXRWT1Q0C
config = {
    "api": {
        "key": "6600ef8939cd5025b77e8909bb3d1a6d", # Claim your free API key here: https://www.alphavantage.co/support/#api-key
        # "symbol": "LME-ALU",
        "symbol": "ALU",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close",
        "refresh": False,
        "notification": True,
        "mail_to": ["garrett@fazer.tech","bnichols@natalloys.com","twalters@natalloys.com"]
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 100,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

def convert_kilogram_to_tonne(price):
    print(f'price0: {price}')
    
    price = price/1000
    price = 1/price
    print(f'price_1: {price}')
    
    
    # price = 1/price
    # print(f'price1: {price}')
    # price = price * 0.001
    # print(f'price2: {price}')
    # price = 1/price
    # print(f'price3: {price}')
    
    return price

def download_data(config):
    
    
    
    
    
    ticker = config["api"]["symbol"]
    filename = f'data_{ticker}.json'

    try:
        # Load the JSON data into a Python dictionary
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, create a new file and write the data
        with open(filename, "w") as f:
            json_data = json.dumps({})
            f.write(json_data)
        with open(filename, 'r') as f:
            data = json.load(f)
    
    # client = Client(config["api"]["key"])

    if config["api"]["refresh"] == True:
        year = 2024
        
        client = Client(config["api"]["key"])

        new_data = client.timeframe(start_date=f'{year}-01-01', end_date=f'{year}-12-31', base='USD', currencies=[config["api"]["symbol"]], unit="kilogram")
        new_data = new_data['rates']

        # Get rid of any data that is empty
        new_data = {k: v for k, v in new_data.items() if config["api"]["symbol"] in v}
        
        print('new_data')
        print(new_data)

        # Check if the keys exist and if not add
        for x in new_data:
            print(new_data[x])
            if x not in data:
                print(f'Adding {x}')
                data[x] = new_data[x]
            else:
                if new_data[x] != "":
                    print(f'Updating {x}')
                    data[x].update(new_data[x])
        # Sort Data
        data = dict(sorted(data.items(), reverse=True))
        # Save Data
        json_data = json.dumps(data)
        with open(filename, "w") as f:
            f.write(json_data)
            

    
    
    
    
    
    
    ### SORT DATA SO MUST RECENT IS FIRST
    # data = dict(sorted(data.items(), reverse=True))
    # print(data)
    # exit()
    
    
    
    
    # ts = TimeSeries(key=config["alpha_vantage"]["key"])
    # data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    
    
    
    
    # data = client.timeframe(start_date='2023-01-01', end_date='2023-12-31', base='USD', currencies=['ALU'])
    # data = data['rates']
    # print(data)
    # exit()
    
    

    # data = {'success': True, 'base': 'USD', 'start_date': '2024-02-05', 'end_date': '2024-02-09', 'rates': {'2024-02-05': {'ALU': 16.0020091086}, '2024-02-06': {'ALU': 15.8242934562}, '2024-02-07': {'ALU': 15.8940972073}, '2024-02-08': {'ALU': 15.8754349179}, '2024-02-09': {'ALU': 15.9278735741}}}
    
    
    
    # print(data)
    # exit()

    # '2003-01-13': {'1. open': '88.31', '2. high': '88.95', '3. low': '87.35', '4. close': '87.51', '5. adjusted close': '46.3373710065631', '6. volume': '10499000', '7. dividend amount': '0.0000', '8. split coefficient': '1.0'}, 

    data_date = [date for date in data.keys()]
    data_date.reverse()

    # print(data_date)

    # data_close_price = [float(data[date][config["api"]["symbol"]]) for date in data.keys()]
    data_close_price = [float(convert_kilogram_to_tonne(data[date][config["api"]["symbol"]])) for date in data.keys()]
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    # data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    # data_close_price.reverse()
    # data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range = download_data(config)








# plot

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close price for " + config["api"]["symbol"] + ", " + display_date_range)
plt.grid(which='major', axis='y', linestyle='--')

plt.savefig(f'static//{config["api"]["symbol"]}_1.png', dpi=300, transparent=False)
if config["api"]["notification"]:
    plt.close()
else:
    plt.show()

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize
scaler = Normalizer()
print(data_close_price)
normalized_data_close_price = scaler.fit_transform(data_close_price)

def prepare_data_x(x, window_size):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]


def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    return output

print(normalized_data_close_price)
data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

# split dataset

split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

# prepare data for plotting

to_plot_data_y_train = np.zeros(num_data_points)
to_plot_data_y_val = np.zeros(num_data_points)

to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

## plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close prices for " + config["api"]["symbol"] + " - showing training and validation data")
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()

plt.savefig(f'static//{config["api"]["symbol"]}_2.png', dpi=300, transparent=False)
if config["api"]["notification"]:
    plt.close()
else:
    plt.show()


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x,
                           2)  # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))


# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize

predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))

# prepare data for plotting

to_plot_data_y_train_pred = np.zeros(num_data_points)
to_plot_data_y_val_pred = np.zeros(num_data_points)

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

# plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Compare predicted prices to actual prices")
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()

plt.savefig(f'static//{config["api"]["symbol"]}_3.png', dpi=300, transparent=False)
if config["api"]["notification"]:
    plt.close()
else:
    plt.show()

# prepare data for plotting the zoomed in view of the predicted prices vs. actual prices

to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
to_plot_predicted_val = scaler.inverse_transform(predicted_val)
to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

# plots

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
plt.title("Zoom in to examine predicted price on validation data portion")
xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
xs = np.arange(0,len(xticks))
plt.xticks(xs, xticks, rotation='vertical')
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()

plt.savefig(f'static//{config["api"]["symbol"]}_4.png', dpi=300, transparent=False)
if config["api"]["notification"]:
    plt.close()
else:
    plt.show()

# predict the closing price of the next trading day

model.eval()

x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
prediction = model(x)
prediction = prediction.cpu().detach().numpy()

# prepare plots

plot_range = 10
to_plot_data_y_val = np.zeros(plot_range)
to_plot_data_y_val_pred = np.zeros(plot_range)
to_plot_data_y_test_pred = np.zeros(plot_range)

to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)

to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

# plot

plot_date_test = data_date[-plot_range+1:]
plot_date_test.append("tomorrow")

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
plt.title("Predicting the close price of the next trading day")
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()

plt.savefig(f'static//{config["api"]["symbol"]}_5.png', dpi=300, transparent=False)
if config["api"]["notification"]:
    plt.close()

    subject = f'${round(to_plot_data_y_test_pred[plot_range-1], 2)} - {config["api"]["symbol"]} - Prediction'
    body = f'{config["api"]["symbol"]} is predicted to close at ${round(to_plot_data_y_test_pred[plot_range-1], 2)} on the next trading day.  Please see the attached images for more details.'
    resp_bool = email().send(config["api"]["mail_to"], subject, body, config["api"]["symbol"])
else:
    plt.show()

print("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))
