# Graph net traffic data

Our repository for the bachelor thesis of AI at Radboud University 2020-2021.

## Generate manhattan data set:
```
python gen_data.py \
   -c cityflow_configs/manhattan_16_3_cfg.json \
   -o generated_data/manhattan_16_3_data.json 
```

Disclaimer: First make the `generated_data` folder before trying to write to it, otherwise it will not work.

## Train on manhattan data set:
```
python training.py \
    -d generated_data/manhattan_16_3_data.json \
    -r sample-code/data/manhattan_16x3/roadnet_16_3.json \
    -f models/model.pt \
    -n 350 \
    -b 50 \ 
    -R results
```

## Test model on manhattan data set:
```
python testing.py \
    -d generated_data/manhattan_16_3_data.json \
    -r sample-code/data/manhattan_16x3/roadnet_16_3.json \
    -f models/model.pt \
    -R results
```

