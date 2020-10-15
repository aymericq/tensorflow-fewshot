# Tensorflow Few Shot

A simple library to train, evaluate and make predictions using few shot models.

# Installation

Install and update using pip:

```pip install -U tensorflow-fewshot```

# Getting started on mini-ImageNet 5-way 5-shot

```python
import tensorflow_fewshot as tf_fs
 
meta_train_ds = tf_fs.datasets.MetaDatasetFromDisk('./mini-imagenet/')


encoder = tf_fs.models.utils.create_standardized_CNN(input_shape=(84, 84, 3))
protonet = tf_fs.models.PrototypicalNetwork(encoder)


def task_generator(n_task = 32):
  for task in range(n_task):
      yield meta_train_ds.get_one_episode()


protonet.meta_train(task_generator)
 
train_set = np.load('./train_set.npy')
protonet.fit(train_set.data, train_set.labels)
 
eval_set = np.load('./eval_set.npy')
predictions = protonet.predict(eval_set.data)
 
print('Accuracy is', np.mean(predictions == eval_set.labels)*100, '%.')
>>> Accuracy is 96 %.
```
