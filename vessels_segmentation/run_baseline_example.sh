#Main directory to save experiments
main_dir=segmentation_experiments 
mkdir $main_dir

#Dataset directory
train_data_dir=data/drive_training
test_data_dir=data/drive_test

#Experiments configuration
experiments_file=splits/experiments15.csv
epoch_size=15
optimizer=Adam
init_lr=1e-4
min_lr=1e-5
patience=100

#Subdirectory to save experiments
results_folder=15samples_adam_1e-4_1e-5_e15_p100_unet_fs

#Run experiments for selected seeds (from 0 to 9)
for n in 0
do

#Training
python3 src/train.py --path_dataset $train_data_dir\
                     --main_path $main_dir \
                     --results_path $results_folder \
                     --seeds_list $n \
                     --seed $n \
                     --csv_file $experiments_file \
                     --init_lr $init_lr \
                     --min_lr $min_lr \
                     --patience $patience \
                     --epoch_size $epoch_size \
                     --optimizer $optimizer

#Evaluation
folder=$main_dir/$results_folder/$n
python3 src/evaluate_v.py --model_file $folder/generator_last.pth --results_folder $folder/results_last --path_dataset $test_data_dir --save_images --compute_pr
python3 src/evaluate_v.py --model_file $folder/generator_best.pth --results_folder $folder/results_best --path_dataset $test_data_dir --save_images --compute_pr

done
