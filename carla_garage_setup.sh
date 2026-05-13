carla setup
cd ./carla_garage
./setup_carla.sh
conda env create -f environment.yml
conda activate garage_2
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128



# get pretrained models
wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip
apt install unzip
unzip ./pretrained_models.zip -d .
rm ./pretrained_models.zip
cd ..