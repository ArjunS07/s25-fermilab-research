POD_UID="local-run-pod-$(uuidgen)"
UNIQUE_RUN_ID="$(date +%F_%H-%M-%S)--$POD_UID"
OUTPUT_PATH="out/$UNIQUE_RUN_ID"
mkdir $OUTPUT_PATH

python3 data.py --jet_types 'g' --num_particles=30 --output_path=${OUTPUT_PATH}
python3 jet_attr_model.py --output_path=${OUTPUT_PATH} --batch_size=1024 --num_epochs=10
python3 train.py --batch_size=20 --num_epochs=2 --n_samples=5000 --n_layers=3 --jet_types 'g' --num_particles=30 --output_path=${OUTPUT_PATH}