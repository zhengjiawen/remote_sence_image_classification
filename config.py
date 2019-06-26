class DefaultConfigs(object):
    train_data_path = '/data/home/zjw/dataset/rssrai2019_scene_classification/train_all/'
    val_data_path = ''
    test_data_path = '/data/home/dengjinhong/dataset/rssrai2019_scene_classification/test/'
    relation_path = '/data/home/zjw/dataset/rssrai2019_scene_classification/ClsName2id.txt'

    test_model_name = 'resnext50'
    test_model_path = './checkpoints/best_model/model.pth'
    seed = 123456

    img_weight = 640
    img_height = 640


    model_name = "resnext50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "3,4,6,7"

    epochs = 40
    batch_size = 40

    num_classes = 45
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4



config = DefaultConfigs()