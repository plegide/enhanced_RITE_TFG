#Main directory to save experiments
main_dir=segmentation_experiments 
mkdir $main_dir

#Dataset directory
train_data_dir=data/drive_training
test_data_dir=data/drive_test

# Generate geometric maps before training
# python3 src/generate_geometric_maps.py --input_dir $train_data_dir \
#                                      --method fmm_subpixel

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
                     --optimizer $optimizer \
                     --pre_output_chs 4

#Evaluation
folder=$main_dir/$results_folder/$n
# Synthesis method
synthesis_method=distance

# Synthesis step

# python3 src/evaluate_v.py --model_file $folder/generator_last.pth \
#                          --results_folder $folder/results_last \
#                          --path_dataset $test_data_dir \
#                          --save_images --compute_pr \
#                          --synthesis_method $synthesis_method

# python3 src/evaluate_v.py --model_file $folder/generator_best.pth \
#                          --results_folder $folder/results_best \
#                          --path_dataset $test_data_dir \
#                          --save_images --compute_pr \
#                          --synthesis_method $synthesis_method

# Generate geometric maps for test (images 1-20)
# python3 src/generate_geometric_maps.py --input_dir $test_data_dir \
#                                      --method fmm_subpixel \
#                                      --start_idx 1 --end_idx 21


# python3 src/evaluate_new.py --model_file $folder/generator_best.pth \
#                            --results_folder $folder/results_new_training \
#                            --path_dataset $test_data_dir \
#                            --save_images --compute_pr \
#                            --synthesis_method $synthesis_method \
#                            --save_error_maps

# Generate graphics for image metrics
python3 src/evaluate_synthesis.py --analysis_methods fmm_subpixel \
                                --synthesis_methods distance \
                                --test_data_dir $test_data_dir \
                                --pred_data_dir $folder/results_new_training            

done
