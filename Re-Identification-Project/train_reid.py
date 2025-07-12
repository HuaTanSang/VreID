import sys
import os
sys.path.append(
    os.path.dirname(__file__)
)
import argparse
import torchreid
import custom_dataset
from utils import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse argument")
    parser.add_argument('--root_data', type=str,default='./Custom_ReID_Dataset', help='Reid dataset format')
    parser.add_argument('--model_path', type=str,default='./models/best.pt', help='Path to yolo model')
    args = parser.parse_args()
    if not os.path.exists(args.root_data):
        make_data_for_train_reid(model_path=args.model_path)
    datamanager = torchreid.data.ImageDataManager(
        root=args.root_data,
        sources='dataset_0',
        height=256, width=128,
        batch_size_train=64, batch_size_test=64,
        transforms=['color_aug', 'random_erase']
    )


    model = torchreid.models.build_model(
        name='osnet_ain_x1_0',
        num_classes=datamanager.num_train_pids,
        loss='triplet'
    )
    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model, optim='adam', lr=0.0003
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler='single_step', stepsize=20
    )


    engine = torchreid.engine.ImageTripletEngine(
        datamanager=datamanager,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        margin=0.2,        
        weight_t=1,       
        weight_x=1,        
        use_gpu=True,
        label_smooth=True
    )

    engine.run(
        save_dir='./model_reid',
        max_epoch=100,
        eval_freq=10,
        print_freq=20,
        test_only=False
    )
