class DefaultConfigs(object):
    train_data_path = ''
    val_data_path = ''
    test_data_path = ''
    relation_path = 'D:/rssrai2019_scene_classification/ClsName2id.txt'
    seed = 123456

    img_weight = 640
    img_height = 640

    #1.string parameters

    model_name = "resnet50"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "1"

    #2.numeric parameters
    epochs = 40
    batch_size = 8

    num_classes = 59
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4



config = DefaultConfigs()