## Folder structure
Please organize the file structure of this folder as following:

data
|-- shape_binvox
|   |-- airplane-02691156
|   |-- car-02958343
|   `-- guitar-03467517
|-- shape_list
|   |-- airplane-02691156
|   |-- car-02958343
|   `-- guitar-03467517
`-- shape_reference
    |-- depth
    |   |-- airplane-02691156
    |   |-- car-02958343
    |   `-- guitar-03467517
    `-- rgb
        |-- airplane-02691156
        |-- car-02958343
        `-- guitar-03467517

**shape_binvox:** the voxlized groundtruth data of each category stored in .binvox file
**shape_list:** the list of shape names of each category with demonstration/train/test split
**shape_reference:** the shape reference of each cateogry

## Data download
To directly train the network without modification, you can download the [data.zip](https://drive.google.com/file/d/1inwGXugUEB_vbmTjl33gfWWPhAw594Fv/view?usp=sharing) and use them to replace this folder.
