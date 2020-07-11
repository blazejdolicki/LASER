# LASER for cross-lingual sentiment analysis
This is a fork of the original LASER [repository](https://github.com/facebookresearch/LASER) used for my [bachelor thesis](https://github.com/blazejdolicki/multilingual-analysis). We add scripts to use CLS and 110k DBRD datasets, merge them into CLS+ and evaluate the supervised and zero-shot learning performance in Dutch.

## Reproducing the results
### Initial steps
1. Clone this repository.
2. Create and activate a new conda environment for LASER (Optional, but highly recommended).
3. Make the following installations
```
>> conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
>> pip install faiss-cpu --no-cache`
>> pip numpy==1.15.4
>> pip cython==0.29.6
>> pip transliterate==1.10.2
>> pip jieba==0.39
```
4. Following the [installation instructions](https://github.com/facebookresearch/LASER#installation) from the original repo.

### Make predictions on CLS+
1. Download 110k DBRD and preprocess it.
```
>> cd LASER/tasks/cls
>> bash prepare_110kDBRD_laser.sh
```
2. Download CLS, train and evaluate (you can choose which languages to train on and evaluate by changing the variable in `cls.sh`, the default is English and Dutch, but you can use it for all other CLS languages).
```
>> cd LASER/tasks/cls
>> bash cls.sh
```



### Pseudolabels (for Multifit)
To create pseudolabels with LASER needed for zero-shot learning with Multifit run:
```
>> cd LASER/tasks/cls
>> bash cls.sh create_labels
>> cd ../../../multifit
>> python get_labels.py
```

After running these commands you should obtain outputs similar to [those](https://github.com/blazejdolicki/multilingual-analysis/blob/master/results/).


