python eval_analyze.py --model_path outputs/edm_qm9 --n_samples 10_000;
python eval_sample.py --model_path outputs/edm_qm9 --n_samples 10_000;

python main_qm9.py --exp_name test_filter0.0 --n_epochs 2 --batch_size 1 --nf 2 --n_layers 1 --filter_n_atoms 6 --model egnn_dynamics --remove_h --n_stability_samples 100 --test_epochs 1 --diffusion_steps 100;

python main_qm9.py --exp_name test_quantum3.2 --n_epochs 2 --batch_size 1 --nf 2 --n_layers 1 --filter_n_atoms 5 --model qegnn_dynamics --remove_h --n_stability_samples 3 --test_epochs 1 --diffusion_steps 10;

# origin
python main_qm9.py --n_epochs 3000 --exp_name origin --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999

# quantum4.0
python main_qm9.py --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --lr 1e-4 --normalize_factors [1,4,10] --ema_decay 0.9999 --n_epochs 2 --test_epochs 1 --diffusion_steps 100 --nf 256 --n_layers 9 --model qegnn_dynamics --n_stability_samples 5 --exp_name test_quantum4.0 --batch_size 16 --filter_n_atoms 9;

# classical4.0
python main_qm9.py --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --lr 1e-4 --normalize_factors [1,4,10] --ema_decay 0.9999 --n_epochs 2 --test_epochs 1 --diffusion_steps 100 --nf 256 --n_layers 9 --model egnn_dynamics --n_stability_samples 5 --exp_name test_classical4.0 --batch_size 16 --filter_n_atoms 9



# origin_test_second_half
python main_qm9.py --diffusion_noise_schedule polynomial_2 --lr 1e-4 --normalize_factors [1,4,10] --n_stability_samples 20 --diffusion_steps 200 --batch_size 128 --nf 16 --n_layers 2  --n_epochs 1000 --test_epochs 100 --exp_name origin_test_second_half --dataset qm9_second_half;
python eval_analyze.py --model_path outputs/origin_test_second_half --n_samples 100;
python eval_sample.py --model_path outputs/origin_test_second_half --n_samples 100;

# origin_test_full_19
python main_qm9.py --diffusion_noise_schedule polynomial_2 --lr 1e-4 --normalize_factors [1,4,10] --n_stability_samples 20 --diffusion_steps 200 --batch_size 128 --nf 16 --n_layers 2  --n_epochs 3000 --test_epochs 100 --exp_name origin_test_full_19 --filter_n_atoms 19;
python eval_analyze.py --model_path outputs/origin_test_full_19 --n_samples 100;
python eval_sample.py --model_path outputs/origin_test_full_19 --n_samples 100;


# origin_test_without_h
python main_qm9.py --diffusion_noise_schedule polynomial_2 --lr 1e-4 --normalize_factors [1,4,10] --n_stability_samples 20 --diffusion_steps 200 --batch_size 128 --nf 16 --n_layers 2  --n_epochs 3000 --test_epochs 100 --exp_name origin_test_without_h --remove_h;
python eval_analyze.py --model_path outputs/origin_test_without_h --n_samples 100;
python eval_sample.py --model_path outputs/origin_test_without_h --n_samples 100;


# test_quantum_second_half #还没做
python main_qm9.py --model qegnn_dynamics --diffusion_noise_schedule polynomial_2 --lr 1e-4 --normalize_factors [1,4,10] --n_stability_samples 20 --diffusion_steps 200 --batch_size 128 --nf 16 --n_layers 2  --n_epochs 2 --test_epochs 1 --exp_name test_quantum_second_half --dataset qm9_second_half;
python eval_analyze.py --model_path outputs/test_quantum_second_half --n_samples 100;
python eval_sample.py --model_path outputs/test_quantum_second_half --n_samples 100;


# test_quantum_full_19
python main_qm9.py --model qegnn_dynamics --diffusion_noise_schedule polynomial_2 --lr 1e-4 --normalize_factors [1,4,10] --n_stability_samples 20 --diffusion_steps 200 --batch_size 128 --nf 16 --n_layers 2  --n_epochs 2 --test_epochs 1 --exp_name test_quantum_full_19 --filter_n_atoms 19;
python eval_analyze.py --model_path outputs/test_quantum_full_19 --n_samples 100;
python eval_sample.py --model_path outputs/test_quantum_full_19 --n_samples 100;


# test_quantum_without_h
python main_qm9.py --model qegnn_dynamics --diffusion_noise_schedule polynomial_2 --lr 1e-4 --normalize_factors [1,4,10] --n_stability_samples 20 --diffusion_steps 200 --batch_size 128 --nf 16 --n_layers 2  --n_epochs 2 --test_epochs 1 --exp_name test_quantum_without_h --remove_h;
python eval_analyze.py --model_path outputs/test_quantum_without_h --n_samples 100;
python eval_sample.py --model_path outputs/test_quantum_without_h --n_samples 100;

