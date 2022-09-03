# GNN Drugnames Extraction from Prescription


***@nabang1010***

## Folder Structure

```
├── data_GNN
│   ├── data_csv_test               <csv test>
│   │   ├── VAIPE_P_TEST_0.csv
│   │   ├── VAIPE_P_TEST_1.csv
│   │   ├── VAIPE_P_TEST_10.csv
│   │   ├── VAIPE_P_TEST_100.csv
│   │   ├── VAIPE_P_TEST_1000.csv
│   ├── data_csv_train              <csv train>
│   │   ├── VAIPE_P_TRAIN_0.csv
│   │   ├── VAIPE_P_TRAIN_1.csv
│   │   ├── VAIPE_P_TRAIN_10.csv
│   │   ├── VAIPE_P_TRAIN_100.csv
│   │   ├── VAIPE_P_TRAIN_1000.csv
│   ├── data_image_test_new         <test image>
│   │   ├── VAIPE_P_TEST_0.png
│   │   ├── VAIPE_P_TEST_1.png
│   │   ├── VAIPE_P_TEST_10.png
│   │   ├── VAIPE_P_TEST_100.png
│   │   ├── VAIPE_P_TEST_1000.png
│   ├── data_image_train            <train/val image>
│   │   ├── VAIPE_P_TRAIN_0.png
│   │   ├── VAIPE_P_TRAIN_1.png
│   │   ├── VAIPE_P_TRAIN_10.png
│   │   ├── VAIPE_P_TRAIN_100.png
│   │   ├── VAIPE_P_TRAIN_1000.png
│   ├── data_image_train            <train/val json>
│   │   ├── VAIPE_P_TRAIN_0.json
│   │   ├── VAIPE_P_TRAIN_1.json
│   │   ├── VAIPE_P_TRAIN_10.json
│   │   ├── VAIPE_P_TRAIN_100.json
│   │   ├── VAIPE_P_TRAIN_1000.json
│   ├── dataset                     <trainset/valset/testset>
│   │   ├── train_data.dataset
│   │   ├── val_data.dataset
│   │   ├── test_data.dataset
│   ├── output_folder               <results predict>
│   │   ├── VAIPE_P_TEST_0.png
│   │   ├── VAIPE_P_TEST_1.png
│   │   ├── VAIPE_P_TEST_10.png
│   │   ├── VAIPE_P_TEST_100.png
│   │   ├── VAIPE_P_TEST_1000.png
├── weight
│   ├── weight_2000.pt
```

## Traning

### Generate Data Train Graph
`json_to_csv.ipynb`

`<file json> --> <file csv>`


### Data prepare

`gen_graph_data_train.ipynb`
* input:
*  * image: `data_GNN/data_image_train`
*  * csv  : `data_GNN/data_csv`
* output:
*  * train set: `dataset/train_data.dataset`
*  * val set: `dataset/val_data.dataset`

### Train

`training_GNN.ipynb`


##  Test
`test_GNN.ipynb`

`<scene text box> + <context> ==> <drugname box>`

#### Generate Data Test Graph
`pres_to_csv.ipynb`

`<prescription> --> <scene text detection> --> <scene text recognition> --> <file csv>`

#### Data prepare

`gen_graph_data_test.ipynb`
* input:
*  * image: `data_GNN/data_image_test_new`
*  * csv  : `data_GNN/data_test_new_csv`
* output:
*  * test set: `dataset/test_data.dataset`





