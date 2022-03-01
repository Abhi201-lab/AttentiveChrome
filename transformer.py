import torch
import collections
import pdb
import requests
import torch.utils.data
import kfserving
import argparse
from typing import List, Dict
import csv
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
import logging
import io
import base64
import sys
from pdb import set_trace as stop
import numpy as np

#DEFAULT_MODEL_NAME = "model"
batch_size = 16
data_root = "data/"
n_bins = 100
#predictor_host = 
model_name = 'model'

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument(
    "--model_name",
    default=DEFAULT_MODEL_NAME,
    help="The name that the model is served under.",
)
parser.add_argument(
    "--predictor_host", help="The URL for the model predict function", required=True
)

args, _ = parser.parse_known_args()

filename = "/tmp/temp.csv"

class Transformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def loadData(filename,windows):
        with open(filename) as fi:
            csv_reader=csv.reader(fi)
            data=list(csv_reader)

            ncols=(len(data[0]))
        fi.close()
        nrows=len(data)
        ngenes=nrows/windows
        nfeatures=ncols-1
        print("Number of genes: %d" % ngenes)
        print("Number of entries: %d" % nrows)
        print("Number of HMs: %d" % nfeatures)

        count=0
        attr=collections.OrderedDict()

        for i in range(0,nrows,windows):
            hm1=torch.zeros(windows,1)
            hm2=torch.zeros(windows,1)
            hm3=torch.zeros(windows,1)
            hm4=torch.zeros(windows,1)
            hm5=torch.zeros(windows,1)
            for w in range(0,windows):
                hm1[w][0]=int(data[i+w][2])
                hm2[w][0]=int(data[i+w][3])
                hm3[w][0]=int(data[i+w][4])
                hm4[w][0]=int(data[i+w][5])
                hm5[w][0]=int(data[i+w][6])
            geneID=str(data[i][0].split("_")[0])

            thresholded_expr = int(data[i+w][7])

            attr[count]={
                'geneID':geneID,
                'expr':thresholded_expr,
                'hm1':hm1,
                'hm2':hm2,
                'hm3':hm3,
                'hm4':hm4,
                'hm5':hm5
            }
            count+=1

        return attr

    class HMData(Dataset):
        # Dataset class for loading data
        def __init__(self,data_cell1,transform=None):
            self.c1=data_cell1
        def __len__(self):
            return len(self.c1)
        def __getitem__(self,i):
            final_data_c1=torch.cat((self.c1[i]['hm1'],self.c1[i]['hm2'],self.c1[i]['hm3'],self.c1[i]['hm4'],self.c1[i]['hm5']),1)
            label=self.c1[i]['expr']
            geneID=self.c1[i]['geneID']
            sample={'geneID':geneID,
                'input':final_data_c1,
                'label':label,
                }
            return sample

    def preprocess(self, inputs: Dict) -> Dict:
        print("==>loading train data")
        logging.info("prep =======> %s", json.dumps(inputs))
        del inputs["instances"]
        logging.info("prep =======> %s", str(type(inputs)))
        logging.info("prep =======> %s", json.dumps(inputs))
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({"error": "Recieved invalid json"})
        data = json_data["signatures"]["inputs"][0][0]["data"]
        with open(filename, "w") as f:
            f.write(data)
        cell_train_dict1=loadData(filename,n_bins)
        train_inputs = HMData(cell_train_dict1)
        Train = torch.utils.data.DataLoader(train_inputs, batch_size=batch_size, shuffle=True)

        for idx, Sample in enumerate(Train):
            
                inputs_1 = Sample['input']

                temp = inputs_1.type(dtype)
        
        payload = {"instances": inputs_1.tolist(), "token": inputs["token"]}
        logging.info("token =======> %s", str(inputs["token"]))
        return payload


    def postprocess(self, predictions: List) -> List:
        logging.info("prep =======> %s", str(type(predictions)))
        preds = predictions["predictions"]
        res = f'Your predictions would be: ${round(preds[0],2)}'
        return {"result": res}

if __name__ == "__main__":
    transformer = Transformer(model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])
