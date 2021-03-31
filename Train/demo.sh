#image enhancement
python main.py --template LPNet --save LPNet --scale 1 --reset --save_results --patch_size 96 --ext sep_reset


# Test your own images
python main.py --template LPNet --data_test MyImage --scale 1 --model LPNet --pre_train ../model/mit5k_baseline.pt --test_only --save_results --save "mit5k" --testpath ../LR/LRBI --testset mit5k

