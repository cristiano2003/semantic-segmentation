# install package
pip install -e .

# curl for dataset, download if not already downloaded

echo "polyp not downloaded, downloading now"
curl -L "https://drive.google.com/uc?id=1NIQU5s2maodQeJW8UEzizB24pE2p-Ww5&export=download" > "TrainDataset.zip"
unzip -q "TrainDataset.zip" -d "TrainDataset"
fi
