# install package
pip install -e .

# curl for dataset, download if not already downloaded
if [ -d "neopolyp" ]; then
  echo "neopolyp already downloaded"
else
  echo "neopolyp not downloaded, downloading now"
  curl -L "https://drive.google.com/uc?id=1NIQU5s2maodQeJW8UEzizB24pE2p-Ww5&export=download" > "TrainDataset"
  unzip -q "TrainDataset.zip" -d "TrainDataset"
fi
