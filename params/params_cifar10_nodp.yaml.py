test_batch_size: 100
batch_size: 256
num_microbatches: 256
lr: 0.01
momentum: 0.5
decay: 0
dp: False
epochs: 200
save_on_epochs: [10, 50, 100, 150, 199]

mu: 1.0

ds_size: 5000

# Note: 0 = airplane, 3 = cat, 5 = dog, 8 = ship
minority_group_keys: [3, 5]  # Should be a subset of the positive and/or negative class keys.
positive_class_keys: [0, 3]  # These keys are grouped into the class with label 1.
negative_class_keys: [5, 8]  # These keys are grouped into the class with label 0.
fixed_n_train: 5000
number_of_entries_test: 1000
optimizer: SGD

save_model: True
dataset: cifar10
model: FlexiNet
scheduler: False

count_norm_cosine_per_batch: False

csigma: 0.9