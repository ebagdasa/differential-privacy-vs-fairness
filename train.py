import logging

from models.word_model import RNNModel
from text_helper import TextHelper

logger = logging.getLogger('logger')

from datetime import datetime
import argparse
from scipy import ndimage
from collections import defaultdict
from tensorboardX import SummaryWriter
import torchvision.models as models
from models.mobilenet import MobileNetV2
from image_helper import ImageHelper
from models.densenet import DenseNet
from models.simple import Net, FlexiNet, reseed, RegressionNet
from models.resnet import get_resnet_extractor, get_pretrained_resnet
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm
import yaml
from utils.text_load import *
from utils.utils import dict_html, create_table, plot_confusion_matrix
from inception import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# These are datasets that yield tuples of (images, idxs, labels) instead of
# (images,labels).
TRIPLET_YIELDING_DATASETS = ('dif', 'celeba', 'lfw')

# These are datasets where we explicitly track performance according to some majority/minority
# attribute defined in the params.
MINORITY_PERFORMANCE_TRACK_DATASETS = ('celeba', 'lfw')


def get_number_of_entries_train(args, params):
    """Get the number of entries for the minority group in the training set."""
    assert not (args.get('alpha')
                and (args.number_of_entries_train or params.get('number_of_entries'))
                ), "Can only specify alpha or number_of_entries[_train], not both."
    import ipdb;ipdb.set_trace()
    if args.get('alpha'):  # Case: alpha is specified, use it.
        num_entries_train = int(1 - args.alpha * params['dataset_size'])
    elif args.number_of_entries_train:  # Case: command-line arg overrides params.
        num_entries_train = args.number_of_entries_train
        print("[INFO] overriding number of entries in parameters file; "
              "using %s entries" % num_entries_train)
    else:  # Case: use params value.
        num_entries_train = params['number_of_entries']
    return num_entries_train


def get_helper(params, d, name):
    if params.get('model', False) == 'word':
        helper = TextHelper(current_time=d, params=params, name='text')

        helper.corpus = torch.load(helper.params['corpus'])
        logger.info(helper.corpus.train.shape)
    else:
        helper = ImageHelper(current_time=d, params=params, name=name)
    return helper


def get_optimizer(helper):
    if helper.params['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                              weight_decay=decay)
    elif helper.params['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
    else:
        raise Exception('Specify `optimizer` in params.yaml.')
    return optimizer


def get_net(helper, num_classes):
    if helper.params['model'] == 'densenet':
        net = DenseNet(num_classes=num_classes, depth=helper.params['densenet_depth'])
    elif helper.params['model'] == 'resnet':
        logger.info(f'Model size: {num_classes}')
        net = models.resnet18(num_classes=num_classes)
    elif helper.params['model'] == 'PretrainedRes':
        net = get_pretrained_resnet(num_classes,
                                    helper.params['freeze_pretrained_weights'])
        net = net.cuda()
    elif helper.params['model'] == 'PretrainedResExtractor':
        net = get_resnet_extractor(num_classes,
                                   helper.params['freeze_pretrained_weights'])
    elif helper.params['model'] == 'FlexiNet':
        net = FlexiNet(3, num_classes)
    elif helper.params['model'] == 'dif_inception':
        net = inception_v3(pretrained=True, dif=True)
        net.fc = nn.Linear(768, num_classes)
        net.aux_logits = False
    elif helper.params['model'] == 'inception':
        net = inception_v3(pretrained=True)
        net.fc = nn.Linear(2048, num_classes)
        net.aux_logits = False
        # model = torch.nn.DataParallel(model).cuda()
    elif helper.params['model'] == 'mobilenet':
        net = MobileNetV2(n_class=num_classes, input_size=64)
    elif helper.params['model'] == 'word':
        net = RNNModel(rnn_type='LSTM', ntoken=helper.n_tokens,
                       ninp=helper.params['emsize'], nhid=helper.params['nhid'],
                       nlayers=helper.params['nlayers'],
                       dropout=helper.params['dropout'],
                       tie_weights=helper.params['tied'])
    elif helper.params['model'] == 'regressionnet':
        net = RegressionNet(output_dim=1)
    else:
        net = Net(output_dim=num_classes)
    logger.info(
        'Total number of params for model {}: {}'.format(
            helper.params["model"],
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        ))
    return net


def get_criterion(helper):
    # For DP training, no loss reduction is used; otherwise, use default (mean) reduction.
    if helper.params.get('criterion') == 'mse':  # Case: MSE objective.
        print('[DEBUG] using MSE loss')
        if dp:
            criterion = nn.MSELoss(reduction='none')
        else:
            criterion = nn.MSELoss()
    else:  # Case: not MSE; use cross-entropy objective.
        if dp:
            criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            criterion = nn.CrossEntropyLoss()
    return criterion

def load_data(helper, params):
    classes_to_keep = None
    true_labels_to_binary_labels = None
    if helper.params['dataset'] == 'inat':
        helper.load_inat_data()
        helper.balance_loaders()
    elif helper.params['dataset'] == 'word':
        helper.load_data()
    elif helper.params['dataset'] == 'dif':
        helper.load_dif_data()
        helper.get_unbalanced_faces()
    elif helper.params['dataset'] == 'celeba':
        helper.load_celeba_data()
    elif helper.params['dataset'] == 'lfw':
        helper.load_lfw_data()
    else:
        if helper.params.get('binary_mnist_task'):
            # Labels are assigned in order of index in this array; so minority_key has
            # label 0, majority_key has label 1.
            assert len(helper.params['key_to_drop']) == 1
            classes_to_keep = (args.majority_key, helper.params['key_to_drop'][0])
            true_labels_to_binary_labels = {
                label: i for i, label in enumerate(classes_to_keep)}
        elif helper.params.get('grouped_mnist_task'):
            classes_to_keep = helper.params['positive_class_keys'] + \
                              helper.params['negative_class_keys']
            true_labels_to_binary_labels = {
                label: int(label in helper.params['positive_class_keys'])
                for label in classes_to_keep}
        else:
            raise ValueError
        helper.load_cifar_or_mnist_data(dataset=params['dataset'],
                                        classes_to_keep=classes_to_keep)
        logger.info('before loader')
        helper.create_loaders()
        logger.info('after loader')

        keys_to_drop = params.get('key_to_drop')
        if not isinstance(keys_to_drop, list):
            keys_to_drop = list(keys_to_drop)
        # Create a unique DataLoader for each class
        helper.sampler_per_class()
        logger.info('after sampler')
        number_of_entries_train = get_number_of_entries_train(args, params)
        helper.sampler_exponential_class(mu=mu, total_number=params['ds_size'],
                                         keys_to_drop=keys_to_drop,
                                         number_of_entries=number_of_entries_train)
        logger.info('after sampler expo')
        helper.sampler_exponential_class_test(mu=mu, keys_to_drop=keys_to_drop,
                                              number_of_entries_test=params[
                                                  'number_of_entries_test'])
        logger.info('after sampler test')
        import ipdb;ipdb.set_trace()
    return true_labels_to_binary_labels, classes_to_keep

def mean_of_tensor_list(lst):
    lst_nonempty = [x for x in lst if x.numel() > 0]
    if len(lst_nonempty):
        return torch.mean(torch.stack(lst_nonempty))
    else:
        return None


def compute_channelwise_mean(dataset):
    means = defaultdict(list)
    sds = defaultdict(list)
    for (i, batch) in enumerate(dataset):
        x, _, _ = batch
        # batch is a set of images of shape [b, c, h, w]
        means[0].append(torch.mean(x[:, 0, ...]))
        sds[0].append(torch.std(x[:, 0, ...]))
        means[1].append(torch.mean(x[:, 1, ...]))
        sds[1].append(torch.std(x[:, 1, ...]))
        means[2].append(torch.mean(x[:, 2, ...]))
        sds[2].append(torch.std(x[:, 2, ...]))

    # We ignore the last batch in case it is incomplete.
    print("Channel 0 mean: %f" % mean_of_tensor_list(means[0][:-1]))
    print("Channel 1 mean: %f" % mean_of_tensor_list(means[1][:-1]))
    print("Channel 2 mean: %f" % mean_of_tensor_list(means[2][:-1]))
    print("Channel 0 sd: %f" % mean_of_tensor_list(sds[0][:-1]))
    print("Channel 1 sd: %f" % mean_of_tensor_list(sds[1][:-1]))
    print("Channel 2 sd: %f" % mean_of_tensor_list(sds[2][:-1]))
    return


def add_pos_and_neg_summary_images(data_loader, max_images=64):
    images, idxs, labels = next(iter(data_loader))
    attr_labels = data_loader.dataset.get_attribute_annotations(idxs)
    pos_attr_idxs = idx_where_true(attr_labels == 1)
    neg_attr_idxs = idx_where_true(attr_labels == 0)
    pos_label_images = images[labels == 1]
    neg_label_images = images[labels == 0]
    pos_attr_images = images[pos_attr_idxs]
    neg_attr_images = images[neg_attr_idxs]
    writer.add_images('pos_label_images', pos_label_images[:max_images, ...])
    writer.add_images('neg_label_images', neg_label_images[:max_images, ...])
    writer.add_images('pos_attr_images', pos_attr_images[:max_images, ...])
    writer.add_images('neg_attr_images', neg_attr_images[:max_images, ...])
    return


def make_uid(params, number_of_entries_train:int=None):
    # If number_of_entries_train is provided, it overrides the params file. Otherwise,
    # fetch the value from the params file.
    if number_of_entries_train is None:
        number_of_entries_train = params.get('number_of_entries')
    uid = "{dataset}-S{S}-z{z}-sigma{sigma}-alpha-{alpha}-ada{adaptive_sigma}-dp{dp}-n{n}-{model}".format(
        dataset=params['dataset'],
        S=params.get('S'),
        z=params.get('z'),
        sigma=params.get('sigma'), alpha=params.get('alpha'),
        adaptive_sigma=params.get('adaptive_sigma', False),
        dp=str(params['dp']),
        n=number_of_entries_train,
        model=params['model'])
    if params.get('positive_class_keys') and params.get('negative_class_keys'):
        pos_keys = [str(i) for i in params['positive_class_keys']]
        neg_keys = [str(i) for i in params['negative_class_keys']]
        pos_keys_str = '-'.join(pos_keys)
        neg_keys_str = '-'.join(neg_keys)
        keys_str = pos_keys_str + '-vs-' + neg_keys_str
        uid += '-' + keys_str
    if params.get('target_colname'):
        uid += '-' + params['target_colname']
    if params.get('attribute_colname'):
        uid += '-' + params['attribute_colname']
    if params.get('train_attribute_subset') is not None:
        uid += '-trattrsub' + str(params['train_attribute_subset'])
    if params.get('label_threshold'):
        uid += '-' + str(params['label_threshold'])
    if params.get('freeze_pretrained_weights'):
        uid += '-freezept'
    return uid


def plot(x, y, name):
    if y is not None:
        writer.add_scalar(tag=name, scalar_value=y, global_step=x)


def compute_norm(model, norm_type=2):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def compute_mse(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    assert outputs.shape == labels.shape, \
        "Expected outputs and labels same shape, got shapes {} and {}".format(
            outputs.shape, labels.shape
        )
    mse = (outputs - labels) ** 2
    return torch.mean(mse)


def per_class_mse(outputs, labels, target_class, grouped_label=None) -> torch.Tensor:
    per_class_idx = labels == target_class
    per_class_outputs = outputs[per_class_idx]
    if grouped_label is not None:
        # Create a new labels tensor, with all values equal to grouped_label
        per_class_labels = torch.full_like(per_class_outputs,
                                           fill_value=grouped_label, dtype=torch.float32)
    else:
        # Use the existing labels tensor, with all values equal to target_class
        per_class_labels = labels[per_class_idx]
    mse_per_class = compute_mse(per_class_outputs, per_class_labels)
    return mse_per_class


def idx_where_true(ary):
    return np.ravel(np.argwhere(ary.values))

def test(net, epoch, name, testloader, vis=True, mse: bool = False,
         labels_mapping: dict = None):
    net.eval()
    running_metric_total = 0
    running_ce_loss_total = 0
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    n_test = 0
    i = 0
    correct_labels = []
    predict_labels = []
    pos_class_losses = []
    neg_class_losses = []
    pos_attr_losses = []
    neg_attr_losses = []
    metric_name = 'accuracy' if not mse else 'mse'
    with torch.no_grad():
        for data in tqdm(testloader):
            if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
                inputs, idxs, labels = data
            else:
                inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            n_test += labels.size(0)
            if not mse:
                _, predicted = torch.max(outputs.data, 1)
                predict_labels.extend([x.item() for x in predicted])
                correct_labels.extend([x.item() for x in labels])
                running_metric_total += (predicted == labels).sum().item()
                main_test_metric = 100 * running_metric_total / n_test
                batch_ce_loss = ce_loss(outputs, labels)
                running_ce_loss_total += torch.mean(batch_ce_loss).item()
                pos_class_losses.extend(batch_ce_loss[labels == 1])
                neg_class_losses.extend(batch_ce_loss[labels == 0])
                if helper.params['dataset'] in MINORITY_PERFORMANCE_TRACK_DATASETS:
                    # batch_attr_labels is an array of shape [batch_size] where the
                    # ith entry is either 1/0/nan and correspond to the attribute labels
                    # of the ith element in the batch.
                    try:
                        batch_attr_labels = helper.test_dataset.get_attribute_annotations(idxs)
                        pos_attr_losses.extend(batch_ce_loss[idx_where_true(batch_attr_labels == 1)])
                        neg_attr_losses.extend(batch_ce_loss[idx_where_true(batch_attr_labels == 0)])
                    except Exception as e:
                        print("[WARNING] exception when computing"
                              "attribute-level loss: {}".format(e))
                        import ipdb;ipdb.set_trace()
            else:
                assert labels_mapping, "provide labels_mapping to use mse."
                pos_labels = [k for k, v in labels_mapping.items() if v == 1]
                binarized_labels_tensor = binarize_labels_tensor(labels, pos_labels)

                running_metric_total += compute_mse(outputs, binarized_labels_tensor)
                main_test_metric = running_metric_total / n_test

    if vis:
        plot(epoch, main_test_metric, metric_name)
        metric_list = list()
        metric_dict = dict()
        if not mse:
            fig, cm = plot_confusion_matrix(correct_labels, predict_labels,
                                            labels=helper.labels, normalize=True)
            writer.add_figure(figure=fig, global_step=epoch, tag='tag/normalized_cm')
            avg_test_loss = running_ce_loss_total / n_test
            plot(epoch, avg_test_loss, 'test_crossentropy_loss')
            plot(epoch, mean_of_tensor_list(pos_class_losses), 'test_loss_per_class/1')
            plot(epoch, mean_of_tensor_list(pos_attr_losses), 'test_loss_per_attr/1')
            plot(epoch, mean_of_tensor_list(neg_class_losses), 'test_loss_per_class/0')
            plot(epoch, mean_of_tensor_list(neg_attr_losses), 'test_loss_per_attr/0')
        for i, class_name in enumerate(helper.labels):
            if not mse:
                metric_value = cm[i][i] / cm[i].sum() * 100
                fig, cm = plot_confusion_matrix(correct_labels, predict_labels,
                                                labels=helper.labels, normalize=False)
                cm_name = f'{helper.params["folder_path"]}/cm_{epoch}.pt'
                torch.save(cm, cm_name)
                writer.add_figure(figure=fig, global_step=epoch,
                                  tag='tag/unnormalized_cm')
            else:
                metric_value = per_class_mse(
                    outputs, labels, class_name, grouped_label=labels_mapping[class_name]
                ).cpu().numpy()
            metric_dict[class_name] = metric_value
            logger.info(f'Class: {i}, {class_name}: {metric_value}')
            plot(epoch, metric_value, name=f'{metric_name}_per_class/class_{class_name}')
            metric_list.append(metric_value)

        fig2 = helper.plot_acc_list(metric_dict, epoch, name='per_class',
                                    accuracy=main_test_metric)
        writer.add_figure(figure=fig2, global_step=epoch, tag='tag/per_class')
        torch.save(metric_dict,
                   f"{helper.folder_path}/test_{metric_name}_class_{epoch}.pt")

        plot(epoch, np.var(metric_list),
             name=f'{metric_name}_per_class/{metric_name}_var')
        plot(epoch, np.max(metric_list),
             name=f'{metric_name}_per_class/{metric_name}_max')
        plot(epoch, np.min(metric_list),
             name=f'{metric_name}_per_class/{metric_name}_min')
        plot(epoch, np.max(metric_list) - np.min(metric_list),
             name=f'{metric_name}_intra_class_max_diff/'
                  f'{metric_name}_intra_class_max_diff')

    return main_test_metric


def binarize_labels_tensor(labels: torch.Tensor, pos_labels: list):
    binary_labels = torch.zeros_like(labels, dtype=torch.float32)
    for l in pos_labels:
        is_l = (labels == l)
        binary_labels += is_l.type(torch.float32)
    assert torch.max(binary_labels) <= 1., "Sanity check on binarized grouped labels."
    return binary_labels


def train_dp(trainloader, model, optimizer, epoch, sigma, alpha, labels_mapping=None,
             adaptive_sigma=False):
    norm_type = 2
    model.train()
    running_loss = 0.0
    label_norms = defaultdict(list)
    ssum = 0
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
            inputs, idxs, labels = data
        else:
            inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        if labels_mapping:
            labels = labels.float()
            pos_labels = [k for k, v in labels_mapping.items() if v == 1]
            binarized_labels_tensor = binarize_labels_tensor(labels, pos_labels)
            loss = criterion(outputs, binarized_labels_tensor)
        else:
            loss = criterion(outputs, labels)

        batch_loss = torch.mean(loss).item()
        running_loss += batch_loss

        losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)
        grad_vecs = list()
        for pos, j in enumerate(losses):
            j.backward(retain_graph=True)

            grad_vec = helper.get_grad_vec(model, device)
            grad_vecs.append(grad_vec)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), S)
            label_norms[int(labels[pos])].append(total_norm)

            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    new_grad = tensor.grad
                    # logger.info('new grad: ', new_grad)
                    saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        # Compute average norm and the sigma value (if adaptive)
        grad_norms = [torch.norm(x, p=2) for x in grad_vecs]
        avg_grad_norm = torch.mean(torch.stack(grad_norms))
        if adaptive_sigma:
            # Case: Use adaptive noise
            sigma_dp = alpha * avg_grad_norm
        else:
            # Case: Do not use adaptive noise
            sigma_dp = sigma

        for tensor_name, tensor in model.named_parameters():
            if tensor.grad is not None:
                # Sometimes we use dp training even when sigma is set to zero (to get
                #  gradient magnitudes); we do not add noise when sigma==0.
                if sigma_dp > 0:
                    if device.type == 'cuda':
                        saved_var[tensor_name].add_(
                            torch.cuda.FloatTensor(tensor.grad.shape).normal_(0,
                                                                              sigma_dp))
                    else:
                        saved_var[tensor_name].add_(
                            torch.FloatTensor(tensor.grad.shape).normal_(0, sigma_dp))
                tensor.grad = saved_var[tensor_name] / num_microbatches

        optimizer.step()

        if i > 0 and i % 100 == 0:
            logger.info('[epoch %d, batch %5d] loss: %.3f' %
                        (epoch + 1, i + 1, batch_loss))
            plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0
    print(ssum)
    plot(epoch, avg_grad_norm, "norms/avg_grad_norm")
    for pos, norms in sorted(label_norms.items(), key=lambda x: x[0]):
        logger.info(f"{pos}: {torch.mean(torch.stack(norms))}")
        plot(epoch, torch.mean(torch.stack(norms)), f'norms/class_{pos}')


def train(trainloader, model, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), leave=True):
        # get the inputs
        if helper.params['dataset'] in TRIPLET_YIELDING_DATASETS:
            inputs, idxs, labels = data
        else:
            inputs, labels = data

        if helper.params.get('key_to_drop'):
            keys_input = labels == helper.params['key_to_drop']

            inputs[keys_input] = torch.tensor(
                ndimage.filters.gaussian_filter(inputs[keys_input].numpy(),
                                                sigma=helper.params['csigma']))

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # logger.info statistics
        running_loss += loss.item()
        if i > 0 and i % 20 == 0:
            #             logger.info('[%d, %5d] loss: %.3f' %
            #                   (epoch + 1, i + 1, running_loss / 2000))
            plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
            running_loss = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument("--majority_key", default=3, type=int,
                        help="Optionally specify the majority group key (e.g. '1').")
    parser.add_argument("--alpha", default=None, type=float,
                        help="Fractoin of samples to take from majority class. Minority "
                             "class will be downsampled if necessary.")
    parser.add_argument("--number_of_entries_train", default=None, type=int,
                        help="Optional number of minority class entries/size to "
                             "downsample to; if provided, this value overrides value in "
                             ".yaml parameters.")
    parser.add_argument("--logdir", default="./runs",
                        help="Location to write TensorBoard logs.")
    parser.add_argument("--train_attribute_subset", default=None, type=int,
                        help="Optional argument to the train_attribute_subset param; this"
                        "overrides any value which may be present for that field in the"
                        "parameters yaml file.")
    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')

    with open(args.params) as f:
        params = yaml.load(f)

    if args.train_attribute_subset is not None:
        print("[INFO] overriding train_attribute_subset with value from command: {}"
              .format(args.train_attribute_subset))
        params['train_attribute_subset'] = args.train_attribute_subset
    name = make_uid(params, number_of_entries_train=args.number_of_entries_train)

    writer = SummaryWriter(log_dir=os.path.join(args.logdir, name))

    helper = get_helper(params, d, name)
    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'experiment uid: {name}')
    logger.info(f'current path: {helper.folder_path}')
    batch_size = int(helper.params['batch_size'])
    num_microbatches = int(helper.params['num_microbatches'])
    lr = float(helper.params['lr'])
    momentum = float(helper.params['momentum'])
    decay = float(helper.params['decay'])
    epochs = int(helper.params['epochs'])
    z = helper.params.get('z')
    # If clipping bound S is not specified, it is set to inf.
    S = float(helper.params['S']) if helper.params.get('S') else None
    if helper.params.get('S') and (helper.params.get('S') != 'inf'):
        # Case: clipping bound S is specified; use this to compute sigma.
        sigma = z * S
    else:
        # Case: clipping bound S is not specified (no clipping);
        # sigma must be set explicitly in the params.
        sigma = helper.params.get('sigma')
    alpha = helper.params.get('alpha')
    adaptive_sigma = helper.params.get('adaptive_sigma', False)
    dp = helper.params['dp']
    mu = helper.params['mu']

    reseed(5)

    true_labels_to_binary_labels, classes_to_keep = load_data(helper, params)
    num_classes = helper.get_num_classes(classes_to_keep)

    if dp and sigma != 0:
        helper.compute_rdp()
    print('[DEBUG] num_classes is %s' % num_classes)
    reseed(5)
    net = get_net(helper, num_classes)

    if helper.params.get('multi_gpu', False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    if helper.params.get('resumed_model', False):
        logger.info('Resuming training...')
        loaded_params = torch.load(f"saved_models/{helper.params['resumed_model']}")
        net.load_state_dict(loaded_params['state_dict'])
        helper.start_epoch = loaded_params['epoch']
        # helper.params['lr'] = loaded_params.get('lr', helper.params['lr'])
        logger.info(f"Loaded parameters from saved model: LR is"
                    f" {helper.params['lr']} and current epoch is {helper.start_epoch}")
    else:
        helper.start_epoch = 1

    criterion = get_criterion(helper)

    # Write sample images, for the image classification tasks
    if helper.params['dataset'] in ('lfw', 'celeba'):
        add_pos_and_neg_summary_images(helper.unnormalized_test_loader)
        compute_channelwise_mean(helper.train_loader)

    optimizer = get_optimizer(helper)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * epochs,
                                                                 0.75 * epochs],
                                                     gamma=0.1)

    table = create_table(helper.params)
    writer.add_text('Model Params', table)
    logger.info(table)
    logger.info(helper.labels)
    epoch = 0
    metric_name = 'mse' if helper.params.get('criterion') == 'mse' else 'accuracy'
    for epoch in range(helper.start_epoch,
                       epochs):  # loop over the dataset multiple times
        if dp:
            train_dp(helper.train_loader, net, optimizer, epoch,
                     labels_mapping=true_labels_to_binary_labels,
                     sigma=sigma, alpha=alpha, adaptive_sigma=adaptive_sigma)
        else:
            assert true_labels_to_binary_labels is None, \
                "Label binarization is not implemented for non-DP training."
            train(helper.train_loader, net, optimizer, epoch)
        if helper.params['scheduler']:
            scheduler.step()
        test_loss = test(net, epoch, name, helper.test_loader,
                         mse=metric_name == 'mse',
                         labels_mapping=true_labels_to_binary_labels)
        unb_acc_dict = dict()

        helper.save_model(net, epoch, test_loss)
    logger.info(
        f"Finished training for model: {helper.current_time}. Folder: {helper.folder_path}")
