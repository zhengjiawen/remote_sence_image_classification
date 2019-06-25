
import pandas as pd
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable

from tqdm import tqdm
from utils.utils import *
from config import config
from model.resnet import getResnet
from dataset.remote_dataloader import RemoteDataLoader, get_files, collate_fn
from torch.utils.data import DataLoader


def test(test_path, model):
    model.cuda()

    test_files = get_files(test_path, "test")
    test_loader = DataLoader(RemoteDataLoader(test_files, test=True), batch_size=1, shuffle=False, pin_memory=False)

    # 3.1 confirm the model converted to cuda
    csv_map = OrderedDict({"filename": [], "probability": []})
    model.cuda()
    model.eval()
    with open("./submit/baseline.txt", "w", encoding="utf-8") as f:
        submit_results = []
        for i, (input, filepath) in enumerate(tqdm(test_loader)):
            # 3.2 change everything to cuda and get only basename
            filepath = [os.path.basename(x) for x in filepath]
            with torch.no_grad():
                image_var = Variable(input).cuda()
                # 3.3.output
                # print(filepath)
                # print(input,input.shape)
                y_pred = model(image_var)
                # print(y_pred.shape)
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
            # 3.4 save probability to csv files
            csv_map["filename"].extend(filepath)
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)
        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x: [float(i) for i in x.split(";")])
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])
            result_str = '{} {}\r\n'.format(row['filename'], pred_label)

            f.writelines(result_str)

if __name__ == '__main__':
    best_model = torch.load(config.test_model_path)

    model = getResnet(config.test_model_name)

    model.load_state_dict(best_model["state_dict"])

    test(config.test_data_path, model)



