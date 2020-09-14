To train model
python train.py --config-file configs/vgg_ssd300_voc07.yaml

#Detect on images given to the model #don't forget to select GPU or CPU in config file. (Can be automated)
python demo.py --config-file configs/vgg_ssd300_voc07.yaml --ckpt outputs/vgg_ssd300_voc07/model_097500.pth --score_threshold 0.6 --images_dir ../../dataset/samples/samplesBDD/ --output_dir samplesBDD

# test model with evaluation metric
python test.py --config-file configs/vgg_ssd300_voc07.yaml --ckpt outputs/vgg_ssd300_voc07trainValTest/model_last.pth --output_dir samplesVOC 
