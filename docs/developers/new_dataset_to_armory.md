# How to add a new dataset into armory
I will use the RAIT dataset as an example. It is an object detection dataset.

## Step 1 download dataset and locate all file in a folder

Use the RAITE dataset as an example. The train/test json files are loaded in which contain the image name, label, bboxes, and etc.
```python 
with open('/home/chris/make_dataset/raite/train_dataset.json') as f:
    dataset_train = json.load(f)
with open('/home/chris/make_dataset/raite/test_dataset.json') as f:
    dataset_test = json.load(f)
```
## Step 2 loop through the dictionary of the images to create a dataset in the COCO format
For the RAIT dataset there were 4 keys in the train/test json files: 'info', 'categories', 'images',and 'annotations'. 'images' contains the list of images for that dataset split, the width/height of that image, and the image id that corresponds to label in 'annotations'. 'annotations' contains all the bbox objects for all images in that dataset split. It contains same image id, object id, bbox area, bbox, and category label.

Here I create an annotations DataFrame which contains all the objects. I don't want to look through this dataset since it is longer than the dataset['images'] dictionary. Also I define where the actual images folder is located on my computer.
```python
new_train = pd.DataFrame.from_dict(dataset_train['annotations']) 
new_test = pd.DataFrame.from_dict(dataset_test['annotations']) 
val_string = '/mnt/c/Users/Christopher Honaker/Downloads/archive_fixed/dataset/frames/'
```

Here I create a custom loop that will efficiently create a final dataframe with the RAITE dataset in COCO format. COCO means it has the following columns: 'image_id', 'image', 'width', 'height', 'objects'. The objects column is more complex since it is a dictionary with all objects in that image. That dictionary contains: 'id','area','bbox', 'category'. Each row is a separate image in the dataset. For other datasets, different types of data manipulation can be preformed here.

```python
df_final = pd.DataFrame()
LIST =[]; i = 0
for values in dataset_train['images']:
    df_append = pd.DataFrame(index=range(1),columns=['image_id','image','width','height','objects'])
    df_append.at[0,'image_id']=values['id']
    df_append.at[0,'image'] = val_string + values['file_name']
    df_append.at[0,'width'] = values['width']
    df_append.at[0,'height'] = values['height']
    contents = new_train[new_train.image_id == values['id']]
    
    df_append.at[0,'objects'] = dict({
        'id': contents['id'].tolist(),
        'area': contents['area'].tolist(),
        'bbox': contents['bbox'].tolist(),
        'category': contents['category_id'].tolist()
    })
    
    LIST.append(df_append)
    if len(LIST) > 20:
        df_concat = pd.concat(LIST)
        df_final = pd.concat([df_final,df_concat])
        LIST = []
        print('finished with ' + str(i))
    i += 1
if len(LIST) > 0:
    df_concat = pd.concat(LIST)
    df_final = pd.concat([df_final,df_concat])
    
df_train_final = df_final.reset_index()
del df_final, df_concat, df_append
```
Lastly I preform the same loop structure for the test dataset dictionary


## Step 3 Convert DataFrame into HuggingFace Dataset
I do this since DataFrame are easier and more efficient to create the COCO dataset structure. Converting from DataFrame to HugginFace Dataset is very simple and fast.

Here I convert each DataFrame into a huggingface dataset. I then cast the column with the image path into the Image object in the datasets library. I found this to be a lot more efficient than doing it inside the DataFrame loop creation. I do this for both train and test data, then I create a final dataset with both train and test in the corresponding places.

```python
from datasets import Dataset
from datasets import Image 

hg_dataset_train = Dataset.from_pandas(df_train_final)
dataset_train = datasets.DatasetDict({"train":hg_dataset_train})
newdata_train = dataset_train['train'].cast_column("image", Image())


hg_dataset_test = Dataset(pa.Table.from_pandas(df_test_final))
dataset_test = datasets.DatasetDict({"train":hg_dataset_test})
newdata_test = dataset_test['train'].cast_column("image", Image())

NewDataset = datasets.DatasetDict({"train":newdata_train,"test": newdata_test})
```

The final object NewDataset will look like this:
```python
DatasetDict({
    train: Dataset({
        features: ['index', 'image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 21078
    })
    test: Dataset({
        features: ['index', 'image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 4170
    })
})
```


## Step 4 Saving to Disk or Uploading to S3 Bucket
To save the dataset to disk run the following line of code.


```python
#To save to disk
NewDataset.save_to_disk("raite_dataset.hf")

#To load after saving to disk
from datasets import load_from_disk
NewDataset = load_from_disk("raite_dataset.hf")

#Or if uploading to s3 bucket is preferred
#To upload dataset
from datasets.filesystems import S3FileSystem
s3 = S3FileSystem(anon=False)
NewDataset.save_to_disk('s3://armory-library-data/raite_dataset/', max_shard_size="1GB",fs=s3)

#To dowload Dataset
from datasets import load_from_disk
s3 = S3FileSystem(anon=False)
dataset = load_from_disk('s3://armory-library-data/raite_dataset/',fs=s3)    
```

Next you can load the dataset from disk and run the armory code.