# How to add a new dataset into armory

This file presents two examples of how to add new datasets into armory-library.

## Torchvision

The [SAMPLE (Synthetic and Measured Paired Labeled Experiment)](https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public) dataset consists of measured SAR imagery from the MSTAR collection (Moving and Stationary Target Acquisition and Recognition) paired with synthetic SAR imagery. 

The MSTAR dataset contains SAR imagery of 10 types of military vehicles illustrated in the figure below.

![MSTAR classes](./assets/MSTAR-classes.png)

[Anas, H., Majdoulayne, H., Chaimae, A., & Nabil, S. M. (2020). Deep learning for sar image classification. In Intelligent Systems and Applications: Proceedings of the 2019 Intelligent Systems Conference (IntelliSys) Volume 1 (pp. 890-898). Springer International Publishing.](https://link.springer.com/chapter/10.1007/978-3-030-29516-5_67)

The SAMPLE dataset is organized according to the `ImageFolder` pattern. The imagery is split into two normalizations -- decibel and quarter power magnitude (QPM).
For each normalization type, real and synthetic SAR gray-scale imagery is partitioned into folders according to vehicle type.
```
git clone https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public $1

 |-SAMPLE_dataset_public
 | |-png_images
 | | |-qpm
 | | | |-real
 | | | | |-m1
 | | | | |-t72
 | | | | |-btr70
 | | | | |-m548
 | | | | |-zsu23
 | | | | |-bmp2
 | | | | |-m35
 | | | | |-m2
 | | | | |-m60
 | | | | |-2s1
```

For a Torchvision dataset, we load the dataset using the `ImageFolder` dataset class, which automatically infers
the class labels based on the directory names. The `transform` parameter applies a chain of transformations
that resize, normalize and ouput the images as numpy arrays.

```python
import numpy as np
import torchvision as tv
from tv import transforms as T

tmp_dir = Path('/tmp')
sample_dir = tmp_dir / Path('SAMPLE_dataset_public')
data_dir = sample_dir / Path("png_images", "qpm", "real")

tv_dataset = tv.datasets.ImageFolder(
    root=data_dir,
    transform=T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.ToTensor(),  # HWC->CHW and scales to 0-1
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                T.Lambda(np.asarray),
            ]
        ),
)
```

Next, we use scikit-learn's [`train_test_split`](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.train_test_split.html)
function to generate stratified train and test splits based on the dataset target classes.
```python
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# generate indices: instead of the actual data we pass in integers instead
train_indices, test_indices, _, _ = train_test_split(
    range(len(tv_dataset)),
    tv_dataset.targets,
    stratify=tv_dataset.targets,
    test_size=0.25,
)

# generate subset based on indices
train_split = Subset(tv_dataset, train_indices)
test_split = Subset(tv_dataset, test_indices)
```

Next, we wrap the training split into an armory-library dataset with the `TupleDataset` class.
```python
armory_dataset = armory.dataset.TupleDataset(train_split, ("image", "label"))
```

Finally, we use the tuple dataset above to define an `ImageClassificationDataLoader` and
evaluation dataset. Note that the armory-library `normalized_scale` must match the normalization
transform defined by the Torchvision dataset.
```python
normalized_scale = armory.data.Scale(
    dtype=armory.data.DataType.FLOAT,
    max=1.0,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)

batch_size = 16
shuffle = False

dataloader = armory.dataset.ImageClassificationDataLoader(
    armory_dataset,
    dim=armory.data.ImageDimensions.CHW,
    scale=normalized_scale,
    image_key="image",
    label_key="label",
    batch_size=batch_size,
    shuffle=shuffle,
)

evaluation_dataset = armory.evaluation.Dataset(
    name="food-101",
    dataloader=dataloader,
)
```

## Hugging Face

To demonstrate a new Hugging Face dataset, we load the [VisDrone2019 dataset](https://github.com/VisDrone/VisDrone-Dataset) object detection dataset.
The VisDrone2019 dataset, created by the AISKYEYE team at Tianjin University, China, includes 288 video clips and 10,209 images from various drones,
providing a comprehensive benchmark with over 2.6 million manually annotated bounding boxes for objects like pedestrians and vehicles across diverse
conditions and locations.

As a first step, we download the [validation split](https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing) to a temporary directory.
Note that we do not need to unzip the archive for processing as a Hugging Face dataset.
```python
tmp_dir = Path('/tmp')
visdrone_dir = tmp_dir / Path('visdrone_2019')
visdrone_dir.mkdir(exist_ok=True)

visdrone_val_zip = visdrone_dir / Path('VisDrone2019-DET-val.zip')
```
The VisDrone 2019 Task 1 dataset is organized as parallel folders of images and annotations containing pairs of image and annotation files, respectively.
We then need to designate the object categories and name the fields in the annotation files.
```python
CATEGORIES = [
    'ignored',
    'pedestrian',
    'people',
    'bicycle',
    'car',
    'van',
    'truck',
    'tricycle',
    'awning-tricycle',
    'bus',
    'motor',
    'other'
]

ANNOTATION_FIELDS = [
    'x',
    'y',
    'width',
    'height',
    'score',
    'category_id',
    'truncation',
    'occlusion'
]
```

Next, we define the hierarchical features of the dataset by instantiating a [`datasets.Features`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Features) object -- each feature is named and a Hugging Face data type provided.
```python
features = datasets.Features(
    {
        'image_id': datasets.Value('int64'),
        'file_name': datasets.Value('string'),
        'image': datasets.Image(),
        'objects': datasets.Sequence(
            {
                'id': datasets.Value('int64'),
                'bbox': datasets.Sequence(datasets.Value('float32'), length=4),
                'category': datasets.ClassLabel(num_classes=len(CATEGORIES), names=CATEGORIES),
                'truncation': datasets.Value('int32'),
                'occlusion': datasets.Value('int32'),
            }
        )
    }
)
```

We additionally need to define functions `load_annotations` and `generate_examples`. The `load_annotations` function takes a reader for an annotation file, parses an image description into a dictionary and returns the dictionary of image features. The `generate_examples` generator function uses the specified file reader to iterate over the image in dataset archive. For each image, the generator reads the image file bytes and parses
the associated annotation.

```python
def load_annotations(f: io.BufferedReader) -> List[Dict]:
    reader = csv.DictReader(io.StringIO(f.read().decode('utf-8')), fieldnames=ANNOTATION_FIELDS)
    annotations = []
    for idx, row in enumerate(reader):
        category_id = int(row['category_id'])
        annotation = {
            'id': idx,
            'bbox': list(map(float, [row[k] for k in ANNOTATION_FIELDS[:4]])),
            'category': category_id,
            'truncation': row['truncation'],
            'occlusion': row['occlusion']
        }
        annotations.append(annotation)
    return annotations

def generate_examples(files: Iterator[Tuple[str, io.BufferedReader]], annotation_file_ext:str ='.txt') -> Iterator[Dict[str, object]]:
    annotations = {}
    images = {}
    for path, f in files:
        file_name, _ = os.path.splitext(os.path.basename(path))
        if path.endswith(annotation_file_ext):
            annotation = load_annotations(f)
            annotations[file_name] = annotation
        else:
            images[file_name] = {'path': path, 'bytes': f.read()}
    for idx, (file_name, annotation) in enumerate(annotations.items()):
        example = {
            'image_id': idx,
            'file_name': file_name,
            'image': images[file_name],
            'objects': annotation,
        }
        yield example
```

We can now create the validation dataset by calling [`datasets.Dataset.from_generator`](https://huggingface.co/docs/datasets/v2.19.0/en/package_reference/main_classes#datasets.Dataset.from_generator) with the generator above.
```python
visdrone_val_files = datasets.DownloadManager().iter_archive(visdrone_val_zip)

visdrone_dataset = datasets.Dataset.from_generator(
    generate_examples,
    gen_kwargs={
    "files": visdrone_val_files,
    },
    features=features,
    
)
```
