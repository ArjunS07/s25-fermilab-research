POD_UID="local-run-pod-$(uuidgen)"
UNIQUE_RUN_ID="$(date +%F_%H-%M-%S)--$POD_UID"
OUTPUT_PATH="local_out/$UNIQUE_RUN_ID"
mkdir $OUTPUT_PATH

python3 data.py --num_particles=30 --output_path=${OUTPUT_PATH} --jet_types g
python3 jet_attr_model.py --output_path=${OUTPUT_PATH} --batch_size=1024 --num_epochs=10
python3 train.py --batch_size=20 --num_epochs=2 --n_samples=5000 --n_layers=2 --n_train_samples=50 --n_samples=100 --output_path=${OUTPUT_PATH} --n_viz_samples=100 --integration_steps=4 --num_particles=30 --jet_types g